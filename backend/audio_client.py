"""
MedASR Audio Client
===================
Transcribes patient voice recordings using google/medasr (medical-grade ASR).
Falls back to openai/whisper-large-v3 if medasr is unavailable.

Workflow:
  audio (.wav bytes) → transcribe_audio() → patient statement + voice symptoms
  voice symptoms + image findings → compare_voice_to_image() → mismatch detection
  voice symptoms + billing code → compare_voice_to_code() → code validity check
"""

import os
import io
import logging
import time
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("audio_client")

# =========================================================
# CONFIGURATION
# =========================================================

MEDGEMMA_API_KEY = os.getenv("MEDGEMMA_API_KEY", "")
MEDASR_MODEL = os.getenv("MEDASR_MODEL", "google/medasr")
WHISPER_FALLBACK = "openai/whisper-large-v3"

# HF Inference API for ASR (pipeline-based, not chat completions)
HF_ASR_BASE = "https://router.huggingface.co/hf-inference/models"
REQUEST_TIMEOUT = 60

# Symptom keyword map: spoken phrases → normalized underscore tokens
# Helps bridge natural speech → graph node names
SYMPTOM_KEYWORD_MAP = {
    "chest pain": "chest_pain",
    "chest pressure": "chest_pain",
    "shortness of breath": "shortness_of_breath",
    "can't breathe": "shortness_of_breath",
    "difficulty breathing": "shortness_of_breath",
    "fever": "fever",
    "high temperature": "fever",
    "cough": "cough",
    "productive cough": "productive_cough",
    "coughing up": "productive_cough",
    "sputum": "productive_cough",
    "wheeze": "wheezing",
    "wheezing": "wheezing",
    "leg swelling": "leg_edema",
    "ankle swelling": "leg_edema",
    "swollen legs": "leg_edema",
    "palpitations": "palpitations",
    "irregular heartbeat": "palpitations",
    "rapid heartbeat": "tachycardia",
    "fast heart": "tachycardia",
    "rib pain": "rib_pain",
    "broke my rib": "rib_fracture",
    "broken rib": "rib_fracture",
    "stomach pain": "epigastric_pain",
    "belly pain": "abdominal_pain",
    "nausea": "nausea",
    "vomiting": "vomiting",
    "vomiting blood": "hematemesis",
    "throwing up blood": "hematemesis",
    "black stool": "melena",
    "dark stool": "melena",
    "blood in stool": "hematochezia",
    "dizzy": "dizziness",
    "confusion": "altered_mental_status",
    "can't sleep lying flat": "orthopnea",
    "wake up gasping": "paroxysmal_nocturnal_dyspnea",
    "orthopnea": "orthopnea",
    "fatigue": "fatigue",
    "weakness": "weakness",
    "weakness one side": "focal_weakness",
    "face drooping": "facial_droop",
    "can't speak": "aphasia",
    "headache": "headache",
    "runny nose": "rhinorrhea",
    "sore throat": "pharyngitis",
    "sneezing": "sneezing",
    "cold": "upper_respiratory_symptoms",
    "flu": "influenza_symptoms",
    "body aches": "myalgia",
    "joint pain": "arthralgia",
    "back pain": "back_pain",
    "radiating to back": "back_pain",
    "trauma": "trauma",
    "accident": "trauma",
    "fell": "fall_injury",
    "jaundice": "jaundice",
    "yellow skin": "jaundice",
    "swollen abdomen": "abdominal_distension",
}

# =========================================================
# CORE TRANSCRIPTION
# =========================================================

def transcribe_audio(audio_bytes: bytes, audio_format: str = "wav") -> dict:
    """
    Transcribe patient audio using google/medasr via HuggingFace Inference API.
    Falls back to openai/whisper-large-v3 if medasr returns 404.
    Falls back to mock response if no API key configured.

    Args:
        audio_bytes: Raw bytes of the audio file (.wav recommended)
        audio_format: File format hint (wav, mp3, ogg)

    Returns:
        {
            "transcript": str,          # full medical transcription text
            "voice_symptoms": list[str], # normalized symptom tokens
            "model_used": str,          # which model was used
            "confidence": float,        # 0.0-1.0 (mock: 0.9)
            "mock": bool                # True if no API key
        }
    """
    if not MEDGEMMA_API_KEY or not audio_bytes:
        return _mock_transcription(audio_bytes)

    headers = {
        "Authorization": f"Bearer {MEDGEMMA_API_KEY}",
        "Content-Type": f"audio/{audio_format}",
    }

    # Try google/medasr first, then whisper fallback
    models_to_try = [MEDASR_MODEL, WHISPER_FALLBACK]

    for model in models_to_try:
        url = f"{HF_ASR_BASE}/{model}"
        start = time.time()
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.post(url, headers=headers, content=audio_bytes)
                duration_ms = (time.time() - start) * 1000

                if response.status_code == 200:
                    data = response.json()
                    transcript = _extract_transcript(data)
                    logger.info(
                        f"ASR success with {model} in {duration_ms:.0f}ms: "
                        f"'{transcript[:80]}...'"
                    )
                    voice_symptoms = extract_voice_symptoms(transcript)
                    return {
                        "transcript": transcript,
                        "voice_symptoms": voice_symptoms,
                        "model_used": model,
                        "confidence": 0.9,
                        "mock": False,
                    }
                elif response.status_code in (404, 503):
                    logger.warning(
                        f"Model {model} unavailable ({response.status_code}), "
                        f"trying fallback..."
                    )
                    continue
                else:
                    logger.warning(
                        f"ASR error {response.status_code}: {response.text[:200]}"
                    )
                    continue

        except Exception as e:
            logger.warning(f"ASR attempt with {model} failed: {e}")
            continue

    # All models failed — use mock
    logger.warning("All ASR models failed. Using mock transcription.")
    return _mock_transcription(audio_bytes)


def _extract_transcript(response_data: dict) -> str:
    """Extract text from HuggingFace ASR response."""
    # HF ASR returns: {"text": "..."}
    if "text" in response_data:
        return response_data["text"].strip()
    # Some models return a list of chunks
    if isinstance(response_data, list) and response_data:
        texts = [item.get("text", "") for item in response_data if isinstance(item, dict)]
        return " ".join(texts).strip()
    return str(response_data)


def _mock_transcription(audio_bytes: Optional[bytes] = None) -> dict:
    """
    Return a realistic mock transcription when API is unavailable.
    Simulates a patient describing respiratory symptoms.
    """
    logger.info("Using mock ASR transcription")
    transcript = (
        "I have been having chest pain and shortness of breath for about three days. "
        "I also have a fever and I'm coughing up yellow sputum. "
        "I can't breathe very well when I lie flat."
    )
    return {
        "transcript": transcript,
        "voice_symptoms": extract_voice_symptoms(transcript),
        "model_used": "mock",
        "confidence": 0.9,
        "mock": True,
    }


# =========================================================
# SYMPTOM EXTRACTION FROM TRANSCRIPT
# =========================================================

def extract_voice_symptoms(transcript: str) -> list:
    """
    Extract normalized symptom tokens from a voice transcript.
    Uses keyword matching against SYMPTOM_KEYWORD_MAP.

    Returns list of underscore_normalized symptom strings.
    """
    if not transcript:
        return []

    text_lower = transcript.lower()
    found = set()

    for phrase, token in SYMPTOM_KEYWORD_MAP.items():
        if phrase in text_lower:
            found.add(token)

    # Also extract basic single words that might be symptoms
    basic_symptoms = [
        "fever", "cough", "wheeze", "vomiting", "nausea", "fatigue",
        "trauma", "jaundice", "confusion", "weakness"
    ]
    for word in basic_symptoms:
        if word in text_lower and word not in found:
            found.add(word)

    return sorted(list(found))


# =========================================================
# THREE-WAY MATCH: VOICE vs IMAGE
# =========================================================

def compare_voice_to_image(voice_symptoms: list, image_findings: list) -> dict:
    """
    Detects mismatch between what patient SAID vs what the image SHOWS.

    Logic:
    - voice_symptoms and image_findings are both normalized underscore lists
    - Compute Jaccard-like overlap between the two sets
    - Check for explicit contradictions (e.g., voice: "normal", image: "consolidation")

    Returns:
        {
            "match_score": float,        # 0.0 = no overlap, 1.0 = full agreement
            "voice_only": list,          # symptoms only in voice, not in image
            "image_only": list,          # findings only in image, not in voice
            "agreed": list,              # both confirm
            "mismatch_detected": bool,
            "mismatch_severity": str,    # none/low/medium/high
            "explanation": str
        }
    """
    if not voice_symptoms and not image_findings:
        return {
            "match_score": 1.0,
            "voice_only": [],
            "image_only": [],
            "agreed": [],
            "mismatch_detected": False,
            "mismatch_severity": "none",
            "explanation": "No voice or image data to compare.",
        }

    if not voice_symptoms:
        return {
            "match_score": 0.5,
            "voice_only": [],
            "image_only": image_findings,
            "agreed": [],
            "mismatch_detected": False,
            "mismatch_severity": "none",
            "explanation": "No voice data provided. Cannot compare voice vs. image.",
        }

    # Map voice symptoms → related image findings (bridge vocabulary gap)
    voice_image_bridge = {
        "chest_pain": ["lobar_consolidation", "cardiomegaly", "pleural_effusion",
                       "pneumothorax"],
        "shortness_of_breath": ["lobar_consolidation", "cardiomegaly", "pulmonary_edema",
                                 "pleural_effusion", "pneumothorax", "hyperinflation"],
        "productive_cough": ["lobar_consolidation", "bronchial_thickening",
                              "pulmonary_infiltrates"],
        "fever": ["lobar_consolidation", "pulmonary_infiltrates", "consolidation"],
        "leg_edema": ["cardiomegaly", "pleural_effusions", "kerley_b_lines"],
        "orthopnea": ["cardiomegaly", "pulmonary_edema", "kerley_b_lines"],
        "wheezing": ["hyperinflation", "air_trapping", "flattened_diaphragms"],
        "rib_pain": ["rib_fracture_visible", "cortical_disruption", "single_rib_fracture"],
        "rib_fracture": ["rib_fracture_visible", "cortical_disruption", "multiple_rib_fractures"],
        "hematemesis": ["free_air_under_diaphragm", "no_acute_chest_findings"],
        "melena": ["no_acute_chest_findings", "normal_chest"],
        "upper_respiratory_symptoms": ["normal_chest", "clear_lung_fields"],
    }

    # Expand voice symptoms to potential image findings
    voice_as_image_terms = set()
    for vs in voice_symptoms:
        related = voice_image_bridge.get(vs, [vs])
        voice_as_image_terms.update(related)

    image_set = set(image_findings)
    agreed = list(voice_as_image_terms & image_set)
    image_only = list(image_set - voice_as_image_terms)
    voice_only = list(voice_as_image_terms - image_set)

    # Score based on overlap
    total = len(voice_as_image_terms | image_set)
    match_score = len(agreed) / total if total > 0 else 1.0
    match_score = round(match_score, 3)

    # Contradiction check: voice says "normal" but image shows pathology
    contradictions = []
    negative_voice = {"upper_respiratory_symptoms", "cold", "mild_symptoms"}
    if any(v in negative_voice for v in voice_symptoms):
        serious_findings = [f for f in image_findings
                            if f not in ("normal_chest", "clear_lung_fields",
                                         "no_acute_findings")]
        if serious_findings:
            contradictions.append(
                f"Voice describes mild/normal symptoms, but image shows: "
                f"{', '.join(serious_findings)}"
            )

    mismatch_detected = match_score < 0.3 or bool(contradictions)
    if contradictions:
        severity = "high"
    elif match_score < 0.2:
        severity = "high"
    elif match_score < 0.4:
        severity = "medium"
    elif match_score < 0.6:
        severity = "low"
    else:
        severity = "none"

    explanation = (
        f"Voice-Image match score: {match_score:.0%}. "
        f"Agreed findings: {agreed or 'none'}. "
        f"Image-only: {image_only or 'none'}. "
    )
    if contradictions:
        explanation += f"Contradictions: {'; '.join(contradictions)}"

    return {
        "match_score": match_score,
        "voice_only": list(voice_as_image_terms - image_set),
        "image_only": image_only,
        "agreed": agreed,
        "mismatch_detected": mismatch_detected,
        "mismatch_severity": severity,
        "explanation": explanation,
    }


# =========================================================
# THREE-WAY MATCH: VOICE vs CODE
# =========================================================

def compare_voice_to_code(voice_symptoms: list, code: str, kg) -> dict:
    """
    Detects mismatch between patient's spoken symptoms and the billing code.
    e.g., patient says "I have chest pain" but code is S22.4 (Multiple rib fractures).

    Uses knowledge_engine's voice_keywords to check alignment.

    Args:
        voice_symptoms: normalized tokens from extract_voice_symptoms()
        code: ICD-10 code string (e.g., "S22.4")
        kg: MedicalKnowledgeGraph instance

    Returns:
        {
            "match_score": float,
            "voice_matches_code": bool,
            "expected_voice_keywords": list,
            "found_keywords": list,
            "missing_keywords": list,
            "mismatch_detected": bool,
            "explanation": str
        }
    """
    req = kg.get_requirements(code)

    if req is None:
        return {
            "match_score": 0.5,
            "voice_matches_code": True,
            "expected_voice_keywords": [],
            "found_keywords": [],
            "missing_keywords": [],
            "mismatch_detected": False,
            "explanation": f"No voice keywords defined for code {code}. Cannot verify.",
        }

    expected = req.voice_keywords
    if not expected:
        return {
            "match_score": 1.0,
            "voice_matches_code": True,
            "expected_voice_keywords": [],
            "found_keywords": [],
            "missing_keywords": [],
            "mismatch_detected": False,
            "explanation": "No expected voice keywords for this code.",
        }

    voice_set = set(voice_symptoms)
    expected_set = set(expected)

    # Partial keyword matching: check if ANY expected keyword appears in voice
    found = list(voice_set & expected_set)
    missing = list(expected_set - voice_set)

    # Also check for semantic matches (e.g., "chest_pain" matches "chest_pain" keyword)
    semantic_matches = []
    for exp in expected:
        for v in voice_set:
            if exp in v or v in exp:
                if exp not in found:
                    semantic_matches.append(exp)

    all_found = list(set(found + semantic_matches))

    # Score: what fraction of expected keywords does voice cover?
    match_score = len(all_found) / len(expected) if expected else 1.0
    match_score = round(min(1.0, match_score), 3)

    voice_matches_code = match_score >= 0.3 or bool(all_found)
    mismatch_detected = not voice_matches_code

    if mismatch_detected:
        explanation = (
            f"MISMATCH: Patient's voice describes "
            f"'{', '.join(voice_symptoms[:4]) if voice_symptoms else 'nothing'}' "
            f"but code {code} ({req.description}) expects patient to report: "
            f"'{', '.join(expected[:4])}'. "
            f"This could indicate a fraudulent billing code."
        )
    else:
        explanation = (
            f"Voice-Code alignment: {match_score:.0%}. "
            f"Voice confirms: {all_found or 'some expected symptoms'}."
        )

    return {
        "match_score": match_score,
        "voice_matches_code": voice_matches_code,
        "expected_voice_keywords": expected,
        "found_keywords": all_found,
        "missing_keywords": [m for m in missing if m not in semantic_matches],
        "mismatch_detected": mismatch_detected,
        "explanation": explanation,
    }


# =========================================================
# STANDALONE TEST
# =========================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from knowledge_engine import get_knowledge_graph

    print("=== Audio Client Tests ===\n")

    # Test 1: Symptom extraction
    transcript = (
        "I have chest pain and shortness of breath. "
        "I've also been running a fever and coughing up yellow sputum."
    )
    symptoms = extract_voice_symptoms(transcript)
    print(f"1. Extracted symptoms: {symptoms}\n")

    # Test 2: Mock transcription
    result = _mock_transcription()
    print(f"2. Mock transcription:")
    print(f"   transcript: {result['transcript'][:60]}...")
    print(f"   voice_symptoms: {result['voice_symptoms']}\n")

    # Test 3: Voice vs Image comparison
    voice_syms = ["chest_pain", "shortness_of_breath", "fever"]
    image_findings = ["lobar_consolidation", "air_bronchograms"]
    vvi = compare_voice_to_image(voice_syms, image_findings)
    print(f"3. Voice vs Image (expect good match):")
    print(f"   match_score: {vvi['match_score']} | mismatch: {vvi['mismatch_detected']}")
    print(f"   explanation: {vvi['explanation'][:100]}\n")

    # Test 4: Voice vs Code — fraud scenario (patient says chest pain, code is S22.4)
    kg = get_knowledge_graph()
    vvc = compare_voice_to_code(["chest_pain", "shortness_of_breath"], "S22.4", kg)
    print(f"4. Voice vs Code S22.4 — fraud scenario (chest pain billed as rib fracture):")
    print(f"   match_score: {vvc['match_score']} | mismatch: {vvc['mismatch_detected']}")
    print(f"   explanation: {vvc['explanation']}\n")

    # Test 5: Voice vs Code — correct scenario (J18.1 with cough/fever)
    vvc2 = compare_voice_to_code(["fever", "productive_cough", "chest_pain"], "J18.1", kg)
    print(f"5. Voice vs Code J18.1 — correct scenario:")
    print(f"   match_score: {vvc2['match_score']} | mismatch: {vvc2['mismatch_detected']}")
