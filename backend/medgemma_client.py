"""
MedGemma API Client Module
Connects to the MedGemma multimodal API for clinical entity extraction,
ICD-10 code suggestion, and fraud detection.
"""

import os
import json
import time
import logging
import base64
from datetime import datetime
from typing import Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# LOGGING SETUP
# =========================================================
log_level = os.getenv("LOG_LEVEL", "INFO")
# Compute logs dir relative to this file so it works both locally
# (backend/logs/) and in Docker (/app/logs/).
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_LOG_DIR, "medgemma_api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("medgemma_client")

# =========================================================
# CONFIGURATION
# =========================================================
MEDGEMMA_API_KEY = os.getenv("MEDGEMMA_API_KEY", "")
MEDGEMMA_ENDPOINT = os.getenv("MEDGEMMA_ENDPOINT", "")
MEDGEMMA_MODEL = os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it")
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 60


# =========================================================
# PROMPT TEMPLATES
# =========================================================

IMAGE_ANALYSIS_SYSTEM = (
    "You are an expert radiologist reviewing a medical image. "
    "Examine ONLY the image — ignore any clinical notes or context. "
    "List all radiographic findings you directly observe. "
    "Return JSON only. No preamble."
)

IMAGE_ANALYSIS_USER = """Analyze this medical image. Return a COMPACT SINGLE-LINE JSON only (no indentation, no extra text):
{"raw_findings":["finding1","finding2"],"image_quality":"good","view":"PA"}

Use underscore_format for findings. Common findings: lobar_consolidation, cardiomegaly, \
enlarged_cardiac_silhouette, pleural_effusion, pneumothorax, air_bronchograms, \
hyperinflation, flattened_diaphragms, pulmonary_edema, kerley_b_lines, normal_lungs, \
atelectasis, consolidation, ground_glass_opacity, interstitial_markings.

Observe ONLY the image. Do not infer from any clinical text."""

ENTITY_EXTRACTION_SYSTEM = (
    "You are a clinical entity extraction system. "
    "Extract medical entities only. Return JSON only. No explanation."
)

ENTITY_EXTRACTION_USER = """From the following clinical note, \
extract and return ONLY a valid JSON object with these exact keys:
- "symptoms": JSON array of strings, each symptom uses underscore_for_spaces
- "severity": exactly one of: "mild", "moderate", "severe"
- "body_system": exactly one of: "respiratory", "cardiovascular", "gastrointestinal", "other"

{image_context}

Example: {{"symptoms": ["fever", "productive_cough", "dyspnea"], "severity": "moderate", "body_system": "respiratory"}}

Clinical Note: {clinical_note}"""

ICD_SUGGESTION_SYSTEM = (
    "You are a clinical coding assistant. "
    "You must ONLY suggest codes from the provided candidate list. "
    "Never suggest codes outside this list. "
    "Return JSON only. No preamble."
)

ICD_SUGGESTION_USER = """Clinical Note: {clinical_note}
Image Findings: {image_findings}
Existing Code: {existing_code}
Valid Candidate Codes: {candidate_codes}

Return JSON with:
- suggested_code: (from candidate list only)
- confidence: 0.0-1.0
- reasoning: (clinical explanation)
- mismatch_detected: true/false
- mismatch_reason: (if mismatch, explain what conflicts)"""

CAUSAL_MATCH_SYSTEM = (
    "You are a billing fraud auditor. Be VERY CONSERVATIVE. "
    "Only mark findings as contradicting a code if they provide DIRECT evidence AGAINST "
    "the billed condition — specifically: (1) an explicit negation finding present "
    "(e.g. 'no_cardiomegaly' or 'no_consolidation'), OR (2) the code requires active "
    "pathology but ALL findings are normal/absent/clear. "
    "Do NOT contradict based on: related conditions, diagnostic subtypes, "
    "ambiguous findings, or findings that are in the same organ system. "
    "When in doubt, answer 'contradicted': false. "
    "Return JSON only. No preamble."
)

CAUSAL_MATCH_USER = """ICD-10 Code under audit: {code} — {description}

What this code requires:
{causal_necessity}

Expected supporting findings:
required_any (at least one): {required_any}
required_all (all needed): {required_all}

Observed image findings:
{findings}

FRAUD DETECTION QUESTION:
Do the observed findings provide DIRECT EVIDENCE AGAINST {code}?

Answer "contradicted": true ONLY IF:
  a) Any finding EXPLICITLY NEGATES a required feature (e.g. "no_cardiomegaly" for I51.7, "no_consolidation" for J18.1), OR
  b) Findings indicate a completely UNRELATED organ system (e.g. bone fractures for a pneumonia code)

Answer "contradicted": false IF:
  - Findings are non-specific, ambiguous, or incomplete
  - Findings suggest a related/variant condition in the same organ system
  - Findings are from the same anatomical region as the code
  - Findings include synonyms or anatomical variants of required features

Return compact single-line JSON:
{{"contradicted": true/false, "confidence": 0.0-1.0, "reason": "brief evidence"}}"""

FRAUD_DETECTION_SYSTEM = (
    "You are a medical coding auditor detecting billing fraud. "
    "Be precise and evidence-based. Return JSON only."
)

FRAUD_DETECTION_USER = """Clinical Note: {clinical_note}
Image shows: {image_findings}
Submitted Code: {existing_code} ({existing_description})
Correct Code: {suggested_code} ({suggested_description})
Reimbursement Difference: ${financial_gap}

Return JSON with:
- fraud_risk: none/low/medium/high
- risk_type: upcoding/downcoding/unrelated/none
- evidence: (specific evidence from note and image)
- recommendation: (what auditor should do)"""


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _log_api_call(call_type: str, success: bool, duration_ms: float, detail: str = ""):
    """Log API call metadata to file and console."""
    status = "SUCCESS" if success else "FAILURE"
    logger.info(
        f"API_CALL | type={call_type} | status={status} | "
        f"duration={duration_ms:.1f}ms | {detail}"
    )


def _build_multimodal_payload(
    system_prompt: str,
    user_prompt: str,
    image_base64: Optional[str] = None,
    max_tokens: int = 1024
) -> dict:
    """
    Build the request payload for MedGemma API (OpenAI-compatible format).
    Supports text-only and multimodal (text + image) inputs.
    Endpoint: https://router.huggingface.co/featherless-ai/v1/chat/completions
    """
    messages = [{"role": "system", "content": system_prompt}]

    if image_base64:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        })
    else:
        messages.append({
            "role": "user",
            "content": user_prompt
        })

    return {
        "model": MEDGEMMA_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1
    }


def _call_api(payload: dict, call_type: str) -> dict:
    """
    Make API call to MedGemma with retry logic.
    Returns parsed JSON response or raises exception after max retries.
    """
    if not MEDGEMMA_ENDPOINT or not MEDGEMMA_API_KEY:
        logger.warning("MedGemma API not configured. Returning mock response.")
        return _mock_response(call_type, payload)

    headers = {
        "Authorization": f"Bearer {MEDGEMMA_API_KEY}",
        "Content-Type": "application/json"
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        start_time = time.time()
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = client.post(
                    MEDGEMMA_ENDPOINT,
                    headers=headers,
                    json=payload
                )
                duration_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    response_data = response.json()
                    # Extract text content from response
                    content = _extract_content(response_data)
                    parsed = _parse_json_response(content)
                    _log_api_call(call_type, True, duration_ms)
                    return parsed
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.warning(
                        f"Attempt {attempt}/{MAX_RETRIES} failed: {error_msg}"
                    )
                    last_error = error_msg
                    _log_api_call(call_type, False, duration_ms, error_msg)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            last_error = str(e)
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} exception: {e}")
            _log_api_call(call_type, False, duration_ms, str(e))

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    raise RuntimeError(
        f"MedGemma API failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


def _extract_content(response_data: dict) -> str:
    """Extract text content from API response structure."""
    # Handle Anthropic-style response
    if "content" in response_data:
        content = response_data["content"]
        if isinstance(content, list):
            return " ".join(
                c.get("text", "") for c in content
                if c.get("type") == "text"
            )
        return str(content)

    # Handle OpenAI-style response
    if "choices" in response_data:
        choices = response_data["choices"]
        if choices and "message" in choices[0]:
            return choices[0]["message"].get("content", "")

    # Handle Google-style response
    if "candidates" in response_data:
        candidates = response_data["candidates"]
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            return " ".join(p.get("text", "") for p in parts)

    return str(response_data)


def _parse_json_response(content: str) -> dict:
    """
    Parse JSON from model response, handling markdown code blocks.
    """
    content = content.strip()

    # Strip markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last lines (``` markers)
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

    # Try direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

    logger.error(f"Failed to parse JSON from response: {content[:200]}")
    return {"error": "Failed to parse model response", "raw": content[:500]}


def _mock_response(call_type: str, payload: dict) -> dict:
    """
    Returns realistic mock responses when API is not configured.
    Used for development and testing without API keys.
    """
    logger.info(f"Returning mock response for call_type={call_type}")

    if call_type == "image_analysis":
        # Pass 1 mock: what the image shows — independent of clinical note
        return {
            "raw_findings": ["lobar_consolidation", "air_bronchograms", "right_lower_lobe_opacity"],
            "image_quality": "good",
            "view": "PA"
        }

    elif call_type == "entity_extraction":
        # Pass 2 mock: symptoms from clinical note (image_findings overridden by Pass 1)
        return {
            "symptoms": ["fever", "productive_cough", "dyspnea", "chest_pain"],
            "severity": "moderate",
            "body_system": "respiratory"
        }
    elif call_type == "icd_suggestion":
        # Extract first candidate from payload if available
        messages = payload.get("messages", [{}])
        content = messages[0].get("content", "") if messages else ""
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        suggested = "J18.1"
        if "candidate_codes" in content:
            # Try to extract first code from the candidates list
            import re
            codes = re.findall(r'[A-Z]\d{2}\.\d+', content)
            if codes:
                suggested = codes[0]

        return {
            "suggested_code": suggested,
            "confidence": 0.87,
            "reasoning": (
                "The clinical note describes fever, productive cough, and dyspnea. "
                "The image findings of lobar consolidation and air bronchograms are "
                "pathognomonic for lobar pneumonia (J18.1). This is consistent with "
                "the clinical presentation."
            ),
            "mismatch_detected": True,
            "mismatch_reason": (
                "The submitted code J06.9 (Common Cold) is inconsistent with the "
                "radiographic evidence of lobar consolidation, which indicates a "
                "more severe lower respiratory tract infection."
            )
        }
    elif call_type == "fraud_detection":
        return {
            "fraud_risk": "high",
            "risk_type": "downcoding",
            "evidence": (
                "The chest X-ray demonstrates lobar consolidation with air bronchograms "
                "in the right lower lobe, indicative of bacterial pneumonia. "
                "The clinical note confirms fever (39.2°C), productive cough with "
                "purulent sputum, and oxygen saturation of 92%. Coding as J06.9 "
                "(Common Cold, reimbursement $180) instead of J18.1 (Lobar Pneumonia, "
                "reimbursement $2,400) represents a $2,220 revenue loss."
            ),
            "recommendation": (
                "Flag for immediate review. Recode to J18.1 (Lobar Pneumonia). "
                "Initiate internal audit of similar cases from this provider. "
                "Consider if systemic downcoding pattern exists."
            )
        }
    elif call_type == "counterfactual_verification":
        return {
            "answer": "NO",
            "explanation": (
                "Clinical findings (lobar consolidation, air bronchograms) are "
                "inconsistent with the submitted code under the alternative simpler "
                "diagnosis. The radiographic evidence requires a more specific and "
                "severe classification, making the submitted code medically unjustifiable."
            )
        }

    elif call_type == "causal_match":
        # Mock: contradiction-based audit — default is NOT contradicted (innocent)
        return {
            "contradicted": False,
            "confidence": 0.80,
            "reason": "Observed findings are compatible with the submitted code.",
            "alternative_findings": []
        }

    return {"error": f"Unknown call type: {call_type}"}


# =========================================================
# PUBLIC API FUNCTIONS
# =========================================================

def analyze_image_only(image_base64: str) -> dict:
    """
    Pass 1 of two-pass multimodal analysis.
    Runs CheXNet (Tier 2) + MedGemma in ensemble, then merges findings.

    CheXNet (DenseNet-121, NIH): structured standardized labels (14 conditions)
      → fixes terminology mismatch: "bilateral_infiltrates" → "consolidation"
      → fixes visual encoder gaps: detects cardiomegaly when MedGemma misses it

    MedGemma: free-form image description
      → catches findings outside CheXNet's 14 labels (rarer conditions)

    Both outputs are normalized via medical_normalizer (Tier 1 RadLex map)
    before being merged into a single deduplicated findings list.
    """
    from medical_normalizer import normalize_findings
    from chexnet_client import analyze_with_chexnet

    logger.info("Pass 1 [image-only]: Running CheXNet + MedGemma ensemble")

    # ── CheXNet: structured, standardized analysis ──────────────────────
    chexnet_result = analyze_with_chexnet(image_base64)
    chexnet_ke_findings = chexnet_result.get("findings", [])
    chexnet_available = chexnet_result.get("available", False)

    if chexnet_available:
        logger.info(
            f"CheXNet labels: {chexnet_result.get('positive_labels', [])} "
            f"→ {chexnet_ke_findings}"
        )

    # ── MedGemma: free-form analysis ────────────────────────────────────
    payload = _build_multimodal_payload(
        system_prompt=IMAGE_ANALYSIS_SYSTEM,
        user_prompt=IMAGE_ANALYSIS_USER,
        image_base64=image_base64,
        max_tokens=512
    )
    try:
        mg_result = _call_api(payload, "image_analysis")
        mg_result.setdefault("raw_findings", [])
        mg_result.setdefault("image_quality", "unknown")
        mg_result.setdefault("view", "unknown")
        val = mg_result.get("raw_findings")
        if isinstance(val, str) and val:
            mg_result["raw_findings"] = [
                v.strip().replace(" ", "_") for v in val.split(",") if v.strip()
            ]
        medgemma_raw = mg_result["raw_findings"]
        image_quality = mg_result.get("image_quality", "unknown")
        image_view = mg_result.get("view", "unknown")
    except Exception as e:
        logger.error(f"MedGemma image analysis failed: {e}")
        medgemma_raw = []
        image_quality = "unknown"
        image_view = "unknown"

    # ── Tier 1: Normalize MedGemma findings via RadLex synonym map ──────
    normalized_medgemma = normalize_findings(medgemma_raw)

    # ── Merge: CheXNet (structured) ∪ normalized MedGemma ───────────────
    merged = sorted(set(chexnet_ke_findings + normalized_medgemma))

    logger.info(
        f"Pass 1 complete: {len(merged)} merged findings "
        f"(CheXNet: {len(chexnet_ke_findings)}, MedGemma norm: {len(normalized_medgemma)})"
        f" — {merged}"
    )

    return {
        "raw_findings": merged,
        "image_quality": image_quality,
        "view": image_view,
        "chexnet_available": chexnet_available,
        "chexnet_labels": chexnet_result.get("positive_labels", []),
        "chexnet_findings": chexnet_ke_findings,
        "medgemma_raw": medgemma_raw,
    }


def extract_entities(
    clinical_note: str,
    image_base64: Optional[str] = None
) -> dict:
    """
    Two-pass clinical entity extraction that prevents text-bias.

    Pass 1 (image present): Image-only → raw_image_findings
        MedGemma reads the X-ray before seeing any clinical text.
    Pass 2 (always): Clinical note + pass-1 findings as context → symptoms, severity, body_system
        Text-only call; image_findings come from Pass 1, not inferred from the note.

    Without two-pass: a misleading note ("common cold") causes MedGemma to ignore
    pneumothorax on the X-ray. With two-pass: image findings are locked in first.
    """
    raw_image_findings: list = []
    image_quality = "unknown"
    image_view = "unknown"

    # ── Pass 1: image-only (no clinical note) ──────────────────────────────
    if image_base64:
        image_result = analyze_image_only(image_base64)
        raw_image_findings = image_result.get("raw_findings", [])
        image_quality = image_result.get("image_quality", "unknown")
        image_view = image_result.get("view", "unknown")

    # ── Pass 2: text-only, with pass-1 findings as context ─────────────────
    logger.info(
        "Pass 2 [text extraction]: Clinical note analysis"
        + (f" with {len(raw_image_findings)} image findings as context" if image_base64 else "")
    )

    if image_base64 and raw_image_findings:
        image_context = (
            f"Radiographic findings from image analysis ({image_view} view, "
            f"quality: {image_quality}): {', '.join(raw_image_findings)}"
        )
    elif image_base64:
        image_context = f"Image was analyzed ({image_view} view, quality: {image_quality}) but no significant findings were detected."
    else:
        image_context = "No image provided — base diagnosis on clinical note only."

    user_prompt = ENTITY_EXTRACTION_USER.format(
        clinical_note=clinical_note,
        image_context=image_context
    )

    # Pass 2 is intentionally text-only — image findings already captured in Pass 1
    payload = _build_multimodal_payload(
        system_prompt=ENTITY_EXTRACTION_SYSTEM,
        user_prompt=user_prompt,
        image_base64=None
    )

    try:
        result = _call_api(payload, "entity_extraction")

        # Authoritative image_findings = Pass 1 output (unbiased by clinical note)
        result["image_findings"] = raw_image_findings
        result["image_confidence"] = (
            f"high - direct image analysis ({image_quality} quality, {image_view} view)"
            if image_base64 else "low - no image provided"
        )

        result.setdefault("symptoms", [])
        result.setdefault("severity", "moderate")
        result.setdefault("body_system", "other")

        # Normalize symptoms: if model returns a string, split on commas
        val = result.get("symptoms")
        if isinstance(val, str) and val:
            result["symptoms"] = [
                v.strip().replace(" ", "_") for v in val.split(",") if v.strip()
            ]

        return result

    except Exception as e:
        logger.error(f"Entity extraction (Pass 2) failed: {e}")
        return {
            "symptoms": [],
            "image_findings": raw_image_findings,   # still return Pass 1 findings
            "severity": "unknown",
            "body_system": "unknown",
            "error": str(e)
        }


def suggest_icd_code(
    clinical_note: str,
    image_base64: Optional[str],
    candidate_codes: list,
    existing_code: Optional[str] = None
) -> dict:
    """
    Second pass call to MedGemma.
    Uses candidate_codes from context graph to constrain output.
    Returns: suggested_code, confidence, explanation,
             mismatch_detected, mismatch_reason
    """
    logger.info(
        f"Suggesting ICD code from {len(candidate_codes)} candidates. "
        f"Existing code: {existing_code or 'None'}"
    )

    # Format candidate codes for prompt
    candidate_list = "\n".join(
        f"  - {c['code']}: {c.get('description', '')} "
        f"(${c.get('avg_reimbursement_usd', 0):,})"
        for c in candidate_codes[:10]
    )

    # Extract image findings descriptions
    image_finding_desc = "No image provided"
    if image_base64:
        image_finding_desc = "See attached medical image"

    user_prompt = ICD_SUGGESTION_USER.format(
        clinical_note=clinical_note,
        image_findings=image_finding_desc,
        existing_code=existing_code or "None provided",
        candidate_codes=candidate_list
    )

    payload = _build_multimodal_payload(
        system_prompt=ICD_SUGGESTION_SYSTEM,
        user_prompt=user_prompt,
        image_base64=image_base64
    )

    try:
        result = _call_api(payload, "icd_suggestion")

        # Validate suggested code is in candidate list
        candidate_code_ids = [c["code"] for c in candidate_codes]
        if result.get("suggested_code") not in candidate_code_ids:
            logger.warning(
                f"Model suggested code outside candidate list: "
                f"{result.get('suggested_code')}. "
                f"Falling back to top candidate."
            )
            if candidate_codes:
                result["suggested_code"] = candidate_codes[0]["code"]
                result["confidence"] = max(0.0, float(result.get("confidence", 0.5)) - 0.2)
                result["reasoning"] = (
                    result.get("reasoning", "") +
                    " [Note: Code constrained to candidate list.]"
                )

        result.setdefault("suggested_code", candidate_codes[0]["code"] if candidate_codes else "")
        result.setdefault("confidence", 0.5)
        result.setdefault("reasoning", "Unable to determine reasoning.")
        result.setdefault("mismatch_detected", False)
        result.setdefault("mismatch_reason", "")

        return result

    except Exception as e:
        logger.error(f"ICD code suggestion failed: {e}")
        fallback_code = candidate_codes[0]["code"] if candidate_codes else ""
        return {
            "suggested_code": fallback_code,
            "confidence": 0.3,
            "reasoning": f"Suggestion failed due to API error: {str(e)}",
            "mismatch_detected": False,
            "mismatch_reason": "",
            "error": str(e)
        }


def detect_fraud_risk(
    clinical_note: str,
    image_base64: Optional[str],
    existing_code: str,
    suggested_code: str,
    existing_description: str = "",
    suggested_description: str = "",
    financial_gap: int = 0
) -> dict:
    """
    Third pass — focused fraud detection.
    Returns: fraud_risk (none/low/medium/high),
             risk_type (upcoding/downcoding/unrelated),
             explanation, financial_impact_usd
    """
    logger.info(
        f"Detecting fraud risk: {existing_code} → {suggested_code} "
        f"(gap: ${financial_gap:,})"
    )

    # Extract image findings for prompt (use description if no image)
    image_desc = "No image provided — analysis based on clinical note only"
    if image_base64:
        image_desc = "See attached medical image (analyzed by MedGemma vision)"

    user_prompt = FRAUD_DETECTION_USER.format(
        clinical_note=clinical_note,
        image_findings=image_desc,
        existing_code=existing_code,
        existing_description=existing_description,
        suggested_code=suggested_code,
        suggested_description=suggested_description,
        financial_gap=abs(financial_gap)
    )

    payload = _build_multimodal_payload(
        system_prompt=FRAUD_DETECTION_SYSTEM,
        user_prompt=user_prompt,
        image_base64=image_base64
    )

    try:
        result = _call_api(payload, "fraud_detection")

        result.setdefault("fraud_risk", "none")
        result.setdefault("risk_type", "none")
        result.setdefault("evidence", "No specific evidence identified.")
        result.setdefault("recommendation", "No action required.")

        # Add financial impact calculation
        result["financial_impact_usd"] = abs(financial_gap)

        return result

    except Exception as e:
        logger.error(f"Fraud detection failed: {e}")

        # Heuristic fallback based on financial gap
        abs_gap = abs(financial_gap)
        if abs_gap == 0:
            risk = "none"
        elif abs_gap < 500:
            risk = "low"
        elif abs_gap < 2000:
            risk = "medium"
        else:
            risk = "high"

        return {
            "fraud_risk": risk,
            "risk_type": "upcoding" if financial_gap < 0 else "downcoding",
            "evidence": f"API error occurred. Heuristic: ${abs_gap:,} reimbursement difference detected.",
            "recommendation": "Manual review recommended due to API error.",
            "financial_impact_usd": abs_gap,
            "error": str(e)
        }


def evaluate_causal_match(
    code: str,
    description: str,
    causal_necessity: str,
    required_any: list,
    required_all: list,
    visual_findings: list,
) -> dict:
    """
    Ask MedGemma whether the observed visual findings semantically satisfy
    the causal requirements for a given ICD-10 code.

    Replaces brittle string-matching (exact keyword comparison) with LLM
    semantic reasoning. MedGemma understands that:
      - "right_pneumothorax" satisfies the requirement for "pneumothorax"
      - "bibasilar_airspace_disease" satisfies "consolidation" / "airspace_disease"
      - "enlarged_cardiac_silhouette" satisfies "cardiomegaly"
      - laterality, severity, and anatomical-zone prefixes are handled naturally

    Returns:
        {
          "satisfied": bool,
          "confidence": float (0.0–1.0),
          "evidence": str,
          "missing": list[str]
        }
    """
    findings_str = ", ".join(visual_findings) if visual_findings else "none"
    req_any_str  = ", ".join(required_any)  if required_any  else "none"
    req_all_str  = ", ".join(required_all)  if required_all  else "none"

    prompt = CAUSAL_MATCH_USER.format(
        code=code,
        description=description,
        causal_necessity=causal_necessity,
        required_any=req_any_str,
        required_all=req_all_str,
        findings=findings_str,
    )
    payload = _build_multimodal_payload(
        system_prompt=CAUSAL_MATCH_SYSTEM,
        user_prompt=prompt,
        max_tokens=256,
    )

    try:
        result = _call_api(payload, "causal_match")
        # Contradiction-based: code is CLEAN unless findings actively CONTRADICT it.
        # Default contradicted=False → innocent until proven guilty.
        contradicted = bool(result.get("contradicted", False))
        confidence   = float(result.get("confidence", 0.0))
        evidence     = result.get("reason", result.get("evidence", ""))
        missing      = result.get("alternative_findings", result.get("missing", []))
        satisfied    = not contradicted
        logger.info(
            f"Causal match [{code}]: contradicted={contradicted}, satisfied={satisfied}, "
            f"confidence={confidence:.2f}, findings={findings_str[:60]}"
        )
        return {
            "satisfied": satisfied,
            "confidence": confidence,
            "evidence": evidence,
            "missing": missing,
        }
    except Exception as e:
        logger.error(f"Causal match API error for {code}: {e}")
        # Conservative fallback: insufficient evidence → do NOT flag fraud on API error
        return {
            "satisfied": None,
            "confidence": 0.0,
            "evidence": f"API error: {e}",
            "missing": [],
        }


if __name__ == "__main__":
    # Quick test with mock responses
    print("Testing MedGemma client (mock mode)...")

    test_note = """
    Patient is a 58-year-old male presenting with 3 days of high fever (39.4°C),
    productive cough with yellow-green sputum, right-sided pleuritic chest pain,
    and progressive dyspnea. SpO2 88% on room air. CXR shows right lower lobe
    lobar consolidation with air bronchograms.
    """

    # Test entity extraction
    print("\n1. Entity Extraction:")
    entities = extract_entities(test_note)
    print(json.dumps(entities, indent=2))

    # Test ICD suggestion
    print("\n2. ICD Code Suggestion:")
    candidates = [
        {"code": "J18.1", "description": "Lobar pneumonia", "avg_reimbursement_usd": 2400},
        {"code": "J18.9", "description": "Unspecified pneumonia", "avg_reimbursement_usd": 2000},
        {"code": "J06.9", "description": "Common cold", "avg_reimbursement_usd": 180},
    ]
    suggestion = suggest_icd_code(test_note, None, candidates, existing_code="J06.9")
    print(json.dumps(suggestion, indent=2))

    # Test fraud detection
    print("\n3. Fraud Detection:")
    fraud = detect_fraud_risk(
        clinical_note=test_note,
        image_base64=None,
        existing_code="J06.9",
        suggested_code="J18.1",
        existing_description="Acute upper respiratory infection, unspecified",
        suggested_description="Lobar pneumonia, unspecified organism",
        financial_gap=2220
    )
    print(json.dumps(fraud, indent=2))
