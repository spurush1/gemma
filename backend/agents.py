"""
MedGemma Causal-Audit Agent (MCAA) — LangGraph Workflow
=========================================================
Symbolic-Neural Hybrid: MedGemma (neural) + Knowledge Engine (symbolic)

5-Node StateGraph:
  START
    → VoiceAnalyst       (google/medasr: transcribe patient audio)
    → VisualAnalyst      (MedGemma: extract image/note findings)
    → KnowledgeRetriever (MedicalKnowledgeGraph: get causal requirements)
    → CausalAuditor      (Three-Way Match: Voice ↔ Image ↔ Code)
    → CounterfactualVerifier  [conditional: only if causal_score <= 0.8]
  END

Three-Way Match in CausalAuditor:
  - Voice vs Image: did patient describe findings consistent with radiology?
  - Voice vs Code: does patient statement support the billing code?
  - Image vs Code: do visual findings causally support the billed diagnosis?
"""

import os
import sys
import operator
import logging
from typing import Optional, Annotated, TypedDict

# Ensure backend dir is importable
sys.path.insert(0, os.path.dirname(__file__))

from langgraph.graph import StateGraph, START, END

import medgemma_client as mg
from audio_client import (
    transcribe_audio,
    extract_voice_symptoms,
    compare_voice_to_image,
    compare_voice_to_code,
)
from knowledge_engine import get_knowledge_graph
from icd10_data import ICD10_DATA

logger = logging.getLogger("agents")

# =========================================================
# STATE DEFINITION
# =========================================================

class AuditState(TypedDict):
    # --- Inputs ---
    clinical_note: str
    image_base64: Optional[str]
    audio_bytes: Optional[bytes]     # patient voice recording (.wav)
    existing_code: str

    # --- Set by VoiceAnalyst ---
    patient_statement: str           # raw transcript
    voice_symptoms: list             # normalized tokens from transcript

    # --- Set by VisualAnalyst ---
    visual_findings: list            # normalized from MedGemma image analysis (Pass 1)
    clinical_symptoms: list          # symptoms extracted from clinical note text (Pass 2)

    # --- Set by KnowledgeRetriever ---
    knowledge_requirements: Optional[dict]   # serialized CausalRequirement

    # --- Set by CausalAuditor ---
    causal_score: Optional[float]   # None when no image evidence available
    found_features: list
    missing_features: list
    fraud_flagged: bool
    three_way_match: dict            # voice_vs_image, voice_vs_code, image_vs_code

    # --- Set by CounterfactualVerifier (conditional) ---
    counterfactual_result: Optional[dict]

    # --- Accumulator: each node appends trace lines ---
    agent_trace: Annotated[list, operator.add]

    # --- Final output ---
    final_verdict: Optional[dict]
    upcoding_financial_impact_usd: int
    litigation_risk: dict


# =========================================================
# NODE 0: VoiceAnalyst
# =========================================================

def voice_analyst_node(state: AuditState) -> dict:
    """
    Transcribe patient audio using google/medasr.
    Skipped gracefully if no audio provided.
    """
    audio_bytes = state.get("audio_bytes")

    if not audio_bytes:
        return {
            "patient_statement": "",
            "voice_symptoms": [],
            "agent_trace": [
                "Step 0 [VoiceAnalyst]: No audio provided. Voice channel skipped. "
                "Two-way match (Image ↔ Code) will be used instead of three-way."
            ],
        }

    transcription = transcribe_audio(audio_bytes, audio_format="wav")
    transcript = transcription.get("transcript", "")
    voice_symptoms = transcription.get("voice_symptoms", [])
    model_used = transcription.get("model_used", "unknown")
    is_mock = transcription.get("mock", False)

    trace_msg = (
        f"Step 0 [VoiceAnalyst]: Transcribed {len(audio_bytes)//1024:.1f}KB audio "
        f"using {model_used}{'(mock)' if is_mock else ''}. "
        f"Transcript: '{transcript[:80]}...'. "
        f"Extracted {len(voice_symptoms)} voice symptoms: {voice_symptoms}"
    )

    return {
        "patient_statement": transcript,
        "voice_symptoms": voice_symptoms,
        "agent_trace": [trace_msg],
    }


# =========================================================
# NODE 1: VisualAnalyst
# =========================================================

def visual_analyst_node(state: AuditState) -> dict:
    """
    Two-pass visual analysis to eliminate text-bias in MedGemma.

    Pass 1 (inside extract_entities): Image-only → MedGemma reads the X-ray
      before seeing the clinical note. Prevents the model from ignoring image
      findings when the note contradicts them (e.g., "common cold" note on a
      pneumothorax X-ray).
    Pass 2 (inside extract_entities): Clinical note + Pass-1 findings as text
      context → extracts symptoms, severity, body_system. image_findings come
      from Pass 1, never inferred from the clinical note text.
    """
    has_image = bool(state.get("image_base64"))
    entities = mg.extract_entities(
        clinical_note=state["clinical_note"],
        image_base64=state.get("image_base64")
    )

    raw_findings = entities.get("image_findings", [])
    # Normalize image findings to lowercase_underscore
    normalized = [
        f.lower().replace(" ", "_").replace("-", "_")
        for f in raw_findings
        if f
    ]

    # Extract clinical symptoms (Pass 2 output) — used as text-based fallback
    raw_symptoms = entities.get("symptoms", [])
    if isinstance(raw_symptoms, str):
        raw_symptoms = [s.strip() for s in raw_symptoms.split(",") if s.strip()]
    clinical_symptoms = [
        s.lower().replace(" ", "_").replace("-", "_")
        for s in raw_symptoms
        if s
    ]

    image_confidence = entities.get("image_confidence", "low - no image provided")
    if has_image:
        trace_msg = (
            f"Step 1 [VisualAnalyst]: Two-pass image analysis. "
            f"Pass 1 (image-only): MedGemma read the X-ray without clinical note — "
            f"found {len(normalized)} findings: {', '.join(normalized) if normalized else 'none'}. "
            f"Pass 2 (text): extracted {len(clinical_symptoms)} symptoms from clinical note. "
            f"Image confidence: {image_confidence}. "
            f"Severity: {entities.get('severity', 'unknown')}. "
            f"Body system: {entities.get('body_system', 'unknown')}."
        )
    else:
        trace_msg = (
            f"Step 1 [VisualAnalyst]: Text-only analysis (no image provided). "
            f"Extracted {len(clinical_symptoms)} symptoms from clinical note: "
            f"{', '.join(clinical_symptoms[:5]) if clinical_symptoms else 'none'}. "
            f"Severity: {entities.get('severity', 'unknown')}. "
            f"Body system: {entities.get('body_system', 'unknown')}. "
            f"Text-based causal check will be used (no X-ray evidence)."
        )

    return {
        "visual_findings": normalized,
        "clinical_symptoms": clinical_symptoms,
        "agent_trace": [trace_msg],
    }


# =========================================================
# NODE 2: KnowledgeRetriever
# =========================================================

def knowledge_retriever_node(state: AuditState) -> dict:
    """
    Fetch CausalRequirement for the submitted code from MedicalKnowledgeGraph.
    """
    kg = get_knowledge_graph()
    code = state["existing_code"].upper()
    req = kg.get_requirements(code)

    if req is None:
        serialized = {
            "code": code,
            "visual_required_any": [],
            "visual_required_all": [],
            "visual_supporting": [],
            "causal_necessity": f"No causal requirements defined for {code}.",
            "alternative_diagnoses": [],
            "fraud_differential": "",
            "minimum_causal_score": 0.5,
            "voice_keywords": [],
        }
        trace_msg = (
            f"Step 2 [KnowledgeRetriever]: Code {code} not in knowledge base. "
            f"Using neutral requirements (no causal gating)."
        )
    else:
        serialized = {
            "code": req.code,
            "description": req.description,
            "visual_required_any": req.visual_required_any,
            "visual_required_all": req.visual_required_all,
            "visual_supporting": req.visual_supporting,
            "causal_necessity": req.causal_necessity,
            "alternative_diagnoses": req.alternative_diagnoses,
            "fraud_differential": req.fraud_differential,
            "minimum_causal_score": req.minimum_causal_score,
            "voice_keywords": req.voice_keywords,
        }
        trace_msg = (
            f"Step 2 [KnowledgeRetriever]: Retrieved requirements for {code} "
            f"({req.description}). "
            f"Requires ANY of: {req.visual_required_any or 'none'}. "
            f"Requires ALL of: {req.visual_required_all or 'none'}. "
            f"Causal necessity: '{req.causal_necessity[:80]}...'"
        )

    return {
        "knowledge_requirements": serialized,
        "agent_trace": [trace_msg],
    }


# =========================================================
# NODE 3: CausalAuditor (Three-Way Match)
# =========================================================

def causal_auditor_node(state: AuditState) -> dict:
    """
    Core audit node — performs THREE-WAY MATCH:
      1. Image vs Code (causal_score from knowledge engine)
      2. Voice vs Image (did patient describe what image shows?)
      3. Voice vs Code (does patient statement support the billing code?)

    Sets fraud_flagged, three_way_match, financial metrics, litigation risk.
    """
    kg = get_knowledge_graph()
    code = state["existing_code"].upper()
    visual_findings = state.get("visual_findings", [])
    voice_symptoms = state.get("voice_symptoms", [])
    clinical_symptoms = state.get("clinical_symptoms", [])

    # --- Match 1: Image vs Code (or text fallback when no image) ---
    if visual_findings:
        evaluation = kg.evaluate_findings(code, visual_findings)
        check_mode = "image"
    else:
        # No image: fall back to text-symptom vs voice_keywords check
        evaluation = kg.evaluate_text_symptoms(code, clinical_symptoms)
        check_mode = "text"

    causal_score = evaluation["causal_score"]  # may be None if insufficient evidence
    found_features = evaluation["found_features"]
    missing_features = evaluation["missing_features"]

    # Display helpers for None-safe formatting
    causal_score_display = f"{causal_score:.2f}" if causal_score is not None else "N/A"
    causal_pct_display = f"{causal_score:.0%}" if causal_score is not None else "N/A"

    # --- Match 2: Voice vs Image ---
    if voice_symptoms:
        vvi = compare_voice_to_image(voice_symptoms, visual_findings)
    else:
        vvi = {
            "match_score": 1.0,
            "mismatch_detected": False,
            "mismatch_severity": "none",
            "explanation": "No voice data — voice vs image check skipped.",
            "agreed": [],
            "voice_only": [],
            "image_only": [],
        }

    # --- Match 3: Voice vs Code ---
    if voice_symptoms:
        vvc = compare_voice_to_code(voice_symptoms, code, kg)
    else:
        vvc = {
            "match_score": 1.0,
            "mismatch_detected": False,
            "explanation": "No voice data — voice vs code check skipped.",
            "found_keywords": [],
            "missing_keywords": [],
        }

    # --- Overall fraud assessment ---
    image_code_mismatch = evaluation["fraud_flagged"]
    voice_image_mismatch = vvi.get("mismatch_detected", False)
    voice_code_mismatch = vvc.get("mismatch_detected", False)

    fraud_count = sum([image_code_mismatch, voice_image_mismatch, voice_code_mismatch])
    fraud_flagged = fraud_count >= 1

    three_way_match = {
        "image_vs_code": {
            "causal_score": causal_score,
            "mismatch_detected": image_code_mismatch,
            "found_features": found_features,
            "missing_features": missing_features,
            "explanation": evaluation.get("causal_necessity", ""),
        },
        "voice_vs_image": vvi,
        "voice_vs_code": vvc,
        "mismatches_detected": fraud_count,
        "overall_fraud_flagged": fraud_flagged,
    }

    # --- Financial and litigation metrics ---
    # Check if this is upcoding or downcoding
    # We need a "suggested correct code" to compare — use fraud_differential from KG
    req = state.get("knowledge_requirements", {})
    fraud_differential = req.get("fraud_differential", "")

    upcoding_impact = 0
    litigation_risk = {"score": 0, "label": "none", "factors": [], "applies": False}

    if fraud_differential and fraud_differential in ICD10_DATA:
        submitted_data = ICD10_DATA.get(code, {})
        correct_data = ICD10_DATA.get(fraud_differential, {})
        sub_reimb = submitted_data.get("avg_reimbursement_usd", 0)
        cor_reimb = correct_data.get("avg_reimbursement_usd", 0)
        financial_gap = sub_reimb - cor_reimb  # positive = submitted > correct = upcoding

        if financial_gap > 0:
            # Upcoding
            upcoding_impact = financial_gap
        elif financial_gap < 0:
            # Downcoding — compute litigation risk
            body_system = submitted_data.get("category", "other").lower()
            body_system_name = (
                "cardiovascular" if any(c in body_system for c in ["i5", "i2", "i1"])
                else "respiratory" if any(c in body_system for c in ["j1", "j2", "j4", "j9"])
                else "gastrointestinal" if "k" in body_system
                else "other"
            )
            litigation_risk = kg.compute_litigation_risk(
                submitted_code=code,
                correct_code=fraud_differential,
                financial_gap=financial_gap,
                body_system=body_system_name
            )

    # --- Build preliminary verdict ---
    if fraud_count == 0:
        overall_risk = "none"
        verdict_text = (
            f"All three checks pass. Image findings support {code}. "
            f"Causal score: {causal_pct_display}. No fraud detected."
        )
    elif fraud_count == 1:
        overall_risk = "medium"
        verdict_text = (
            f"One of three checks failed. Causal score: {causal_pct_display}. "
            f"Possible coding error — review recommended."
        )
    elif fraud_count == 2:
        overall_risk = "high"
        verdict_text = (
            f"Two of three checks failed. Causal score: {causal_pct_display}. "
            f"Strong fraud indication — audit required."
        )
    else:
        overall_risk = "critical"
        verdict_text = (
            f"ALL THREE CHECKS FAILED. Causal score: {causal_pct_display}. "
            f"Clinical findings, voice statement, and billing code are inconsistent. "
            f"Flag for immediate legal review."
        )

    final_verdict = {
        "fraud_detected": fraud_flagged,
        "overall_risk": overall_risk,
        "confidence": causal_score,
        "mismatches_count": fraud_count,
        "explanation": verdict_text,
        "counterfactual_ran": False,
        "upcoding_financial_impact_usd": upcoding_impact,
        "litigation_risk": litigation_risk,
    }

    # --- Trace ---
    vvi_score = vvi.get('match_score')
    vvc_score = vvc.get('match_score')
    check_label = "Image" if check_mode == "image" else "Text (no image)"
    trace_msg = (
        f"Step 3 [CausalAuditor]: Three-Way Match results:\n"
        f"  {check_label} ↔ Code ({code}): score={causal_score_display}, "
        f"mismatch={'YES' if image_code_mismatch else 'NO'}. "
        f"Missing: {missing_features or 'none'}.\n"
        f"  Voice ↔ Image: score={f'{vvi_score:.2f}' if vvi_score is not None else 'N/A'}, "
        f"mismatch={'YES' if voice_image_mismatch else 'NO'}.\n"
        f"  Voice ↔ Code: score={f'{vvc_score:.2f}' if vvc_score is not None else 'N/A'}, "
        f"mismatch={'YES' if voice_code_mismatch else 'NO'}.\n"
        f"  Overall: {fraud_count}/3 mismatches → {overall_risk.upper()} risk."
    )

    if upcoding_impact > 0:
        trace_msg += f"\n  Upcoding exposure: ${upcoding_impact:,} overbilling detected."
    if litigation_risk.get("applies"):
        trace_msg += (
            f"\n  Litigation risk: {litigation_risk['label'].upper()} "
            f"(score {litigation_risk['score']}/100)."
        )

    return {
        "causal_score": causal_score,
        "found_features": found_features,
        "missing_features": missing_features,
        "fraud_flagged": fraud_flagged,
        "three_way_match": three_way_match,
        "final_verdict": final_verdict,
        "upcoding_financial_impact_usd": upcoding_impact,
        "litigation_risk": litigation_risk,
        "agent_trace": [trace_msg],
    }


# =========================================================
# CONDITIONAL EDGE
# =========================================================

def should_run_counterfactual(state: AuditState) -> str:
    """
    Run CounterfactualVerifier only when causal auditor already suspects fraud.
    Skips when: no image evidence (score=None), clean causal verdict (not fraud_flagged),
    or high causal score with no mismatches.
    This prevents the counterfactual from creating false positives on borderline-clean cases.
    """
    score = state.get("causal_score")  # may be None if no image findings
    fraud_flagged = state.get("fraud_flagged", False)
    three_way = state.get("three_way_match", {})
    any_mismatch = three_way.get("overall_fraud_flagged", False)

    # Skip when there's no fraud suspicion from causal auditor
    if not fraud_flagged and not any_mismatch:
        return "skip"
    # Skip when no image evidence available
    if score is None:
        return "skip"
    # Skip when high confidence clean result
    if score > 0.8:
        return "skip"
    return "verify"


# =========================================================
# NODE 4: CounterfactualVerifier (conditional)
# =========================================================

def counterfactual_verifier_node(state: AuditState) -> dict:
    """
    Runs a counterfactual MedGemma call:
    "If patient had [alternative] instead of [current], would [current] still be necessary?"

    This forces causal reasoning beyond correlation.
    """
    kg = get_knowledge_graph()
    code = state["existing_code"].upper()
    req_data = state.get("knowledge_requirements", {})
    fraud_differential = req_data.get("fraud_differential", "")

    if not fraud_differential:
        result = {
            "ran": False,
            "answer": "N/A",
            "explanation": "No fraud differential defined for this code.",
            "skipped_reason": "no_alternative",
        }
        trace_msg = (
            "Step 4 [CounterfactualVerifier]: No fraud differential defined. "
            "Counterfactual skipped."
        )
        return {
            "counterfactual_result": result,
            "agent_trace": [trace_msg],
        }

    # Build counterfactual prompt
    visual_findings = state.get("visual_findings", [])
    cf_prompt = kg.get_counterfactual_prompt(
        code=code,
        alternative_code=fraud_differential,
        clinical_note=state["clinical_note"],
        visual_findings=visual_findings,
    )

    # Call MedGemma with the counterfactual prompt
    payload = mg._build_multimodal_payload(
        system_prompt=(
            "You are a clinical coding auditor performing a counterfactual analysis. "
            "Answer with YES or NO followed by one sentence of reasoning. "
            "Return JSON only with keys 'answer' and 'explanation'."
        ),
        user_prompt=cf_prompt,
        image_base64=state.get("image_base64"),
    )

    try:
        cf_response = mg._call_api(payload, "counterfactual_verification")
        cf_answer = str(cf_response.get("answer", "UNKNOWN")).strip().upper()
        cf_explanation = cf_response.get("explanation", "No explanation provided.")
    except Exception as e:
        logger.warning(f"CounterfactualVerifier API call failed: {e}")
        cf_answer = "ERROR"
        cf_explanation = str(e)

    result = {
        "ran": True,
        "answer": cf_answer,
        "explanation": cf_explanation,
        "alternative_code": fraud_differential,
        "prompt_used": cf_prompt[:200],
    }

    # Update final_verdict with counterfactual context
    existing_verdict = state.get("final_verdict", {})
    cf_implies_fraud = cf_answer == "NO"  # "NO" = code not needed → fraud
    updated_verdict = {
        **existing_verdict,
        "counterfactual_ran": True,
        "counterfactual_answer": cf_answer,
        "fraud_detected": existing_verdict.get("fraud_detected", False) or cf_implies_fraud,
        "explanation": (
            existing_verdict.get("explanation", "") +
            f" Counterfactual: If patient had {fraud_differential} instead of "
            f"{code} — would {code} still be necessary? Answer: {cf_answer}. "
            f"{cf_explanation}"
        ),
    }

    trace_msg = (
        f"Step 4 [CounterfactualVerifier]: "
        f"If patient had {fraud_differential} instead of {code}, "
        f"would {code} still be medically necessary? "
        f"MedGemma answer: {cf_answer}. "
        f"Reasoning: {cf_explanation[:120]}"
    )

    return {
        "counterfactual_result": result,
        "final_verdict": updated_verdict,
        "agent_trace": [trace_msg],
    }


# =========================================================
# GRAPH ASSEMBLY
# =========================================================

def _build_audit_graph():
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(AuditState)

    # Register nodes
    graph.add_node("VoiceAnalyst", voice_analyst_node)
    graph.add_node("VisualAnalyst", visual_analyst_node)
    graph.add_node("KnowledgeRetriever", knowledge_retriever_node)
    graph.add_node("CausalAuditor", causal_auditor_node)
    graph.add_node("CounterfactualVerifier", counterfactual_verifier_node)

    # Linear edges
    graph.add_edge(START, "VoiceAnalyst")
    graph.add_edge("VoiceAnalyst", "VisualAnalyst")
    graph.add_edge("VisualAnalyst", "KnowledgeRetriever")
    graph.add_edge("KnowledgeRetriever", "CausalAuditor")

    # Conditional: run CounterfactualVerifier only when fraud is suspected
    graph.add_conditional_edges(
        "CausalAuditor",
        should_run_counterfactual,
        {
            "verify": "CounterfactualVerifier",
            "skip": END,
        }
    )

    graph.add_edge("CounterfactualVerifier", END)

    return graph.compile()


# Module-level compiled graph (built once on import)
_audit_graph = None


def get_audit_graph():
    """Return the compiled audit graph (singleton)."""
    global _audit_graph
    if _audit_graph is None:
        _audit_graph = _build_audit_graph()
    return _audit_graph


def run_audit(
    clinical_note: str,
    existing_code: str,
    image_base64: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
) -> dict:
    """
    Convenience wrapper to run the full causal audit workflow.

    Args:
        clinical_note: Clinical text
        existing_code: ICD-10 code submitted for billing
        image_base64: Base64-encoded X-ray image (optional)
        audio_bytes: Raw .wav bytes of patient voice recording (optional)

    Returns: Final AuditState dict with all results
    """
    initial_state: AuditState = {
        "clinical_note": clinical_note,
        "image_base64": image_base64,
        "audio_bytes": audio_bytes,
        "existing_code": existing_code.upper(),
        "patient_statement": "",
        "voice_symptoms": [],
        "visual_findings": [],
        "clinical_symptoms": [],
        "knowledge_requirements": None,
        "causal_score": None,
        "found_features": [],
        "missing_features": [],
        "fraud_flagged": False,
        "three_way_match": {},
        "counterfactual_result": None,
        "agent_trace": [],
        "final_verdict": None,
        "upcoding_financial_impact_usd": 0,
        "litigation_risk": {"score": 0, "label": "none", "factors": [], "applies": False},
    }
    return get_audit_graph().invoke(initial_state)


# =========================================================
# STANDALONE TEST
# =========================================================

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("MCAA — LangGraph Causal Audit Agent Test")
    print("=" * 60)

    # Test Case: Classic fraud — Common Cold billed for Lobar Pneumonia
    note = (
        "Patient is a 58-year-old male with 3 days of high fever (39.4°C), "
        "productive cough with yellow-green sputum, right-sided pleuritic chest pain, "
        "and progressive dyspnea. SpO2 88% on room air. CXR shows right lower lobe "
        "lobar consolidation with air bronchograms."
    )

    print("\nRunning audit: J06.9 submitted (should be J18.1)...")
    print("(Uses mock MedGemma responses if API not configured)\n")

    result = run_audit(
        clinical_note=note,
        existing_code="J06.9",
    )

    print("\n--- Agent Trace ---")
    for line in result.get("agent_trace", []):
        print(f"\n{line}")

    print("\n--- Final Verdict ---")
    verdict = result.get("final_verdict", {})
    print(json.dumps({
        "fraud_detected": verdict.get("fraud_detected"),
        "overall_risk": verdict.get("overall_risk"),
        "causal_score": result.get("causal_score"),
        "mismatches": verdict.get("mismatches_count"),
        "upcoding_exposure": result.get("upcoding_financial_impact_usd"),
        "litigation_risk": result.get("litigation_risk", {}).get("label"),
    }, indent=2))

    print("\n--- Three-Way Match Summary ---")
    twm = result.get("three_way_match", {})
    print(f"  Image vs Code: {twm.get('image_vs_code', {}).get('mismatch_detected')}")
    print(f"  Voice vs Image: {twm.get('voice_vs_image', {}).get('mismatch_detected')}")
    print(f"  Voice vs Code: {twm.get('voice_vs_code', {}).get('mismatch_detected')}")
    print(f"  Overall mismatches: {twm.get('mismatches_detected', 0)}/3")
