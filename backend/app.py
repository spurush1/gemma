"""
FastAPI Backend Application
ICD-10 MedGemma Multimodal Coding Validation System
"""

import asyncio
import base64
import logging
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import context_graph as cg
import medgemma_client as mg
from icd10_data import ICD10_DATA, search_codes
from agents import get_audit_graph

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="ICD-10 MedGemma Validation API",
    description=(
        "Multimodal ICD-10 coding validation using MedGemma. "
        "Detects upcoding/downcoding fraud risks by combining "
        "clinical notes with medical image analysis."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the context graph on startup
_graph = cg.get_graph()
logger.info(
    f"Context graph loaded: {_graph.graph.number_of_nodes()} nodes, "
    f"{_graph.graph.number_of_edges()} edges"
)

# =========================================================
# PYDANTIC MODELS
# =========================================================


class CompareRequest(BaseModel):
    code1: str
    code2: str


class CompareResponse(BaseModel):
    code1: str
    code1_description: str
    code1_reimbursement: int
    code2: str
    code2_description: str
    code2_reimbursement: int
    severity_difference: str
    financial_gap: int
    relationship: str


class AnalyzeResponse(BaseModel):
    extracted_entities: dict
    candidate_codes: list
    suggested_code: str
    confidence: float
    reasoning: str
    mismatch_detected: bool
    mismatch_reason: str = ""
    fraud_risk: str = "none"
    risk_type: str = "none"
    financial_impact_usd: int = 0
    audit_recommendation: str = ""
    evidence: str = ""


class CodeDetailResponse(BaseModel):
    code: str
    description: str
    category: str
    severity: str
    avg_reimbursement_usd: int
    symptoms: list
    image_findings: list
    excludes: list
    related: dict
    exclusions: list


# =========================================================
# ENDPOINTS
# =========================================================


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "graph_backend": _graph.backend,        # "neo4j" or "networkx"
        "graph_nodes": _graph.graph.number_of_nodes(),
        "graph_edges": _graph.graph.number_of_edges(),
        "total_icd10_codes": len(ICD10_DATA),
        "total_cms_codes": 98505,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    clinical_note: str = Form(..., description="Clinical note text"),
    image: Optional[UploadFile] = File(None, description="Medical image (X-ray/scan)"),
    existing_code: Optional[str] = Form(None, description="Existing ICD-10 code to validate")
):
    """
    Main analysis endpoint.
    Accepts clinical note + optional image + optional existing code.
    Returns full coding validation with fraud detection.
    """
    logger.info(
        f"Analyze request: existing_code={existing_code}, "
        f"has_image={image is not None}"
    )

    # -------------------------------------------------------
    # 1. Process image if provided
    # -------------------------------------------------------
    image_base64 = None
    if image and image.filename:
        try:
            image_bytes = await image.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            logger.info(f"Image loaded: {len(image_bytes)} bytes")
        except Exception as e:
            logger.warning(f"Image processing failed: {e}")
            image_base64 = None

    # -------------------------------------------------------
    # 2. Extract clinical entities via MedGemma
    # -------------------------------------------------------
    try:
        extracted_entities = mg.extract_entities(clinical_note, image_base64)
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {e}")

    symptoms = extracted_entities.get("symptoms", [])
    image_findings = extracted_entities.get("image_findings", [])

    # -------------------------------------------------------
    # 3. Get candidate codes from context graph
    # -------------------------------------------------------
    candidate_codes = _graph.get_candidate_codes(symptoms, image_findings)
    if not candidate_codes:
        logger.warning("No candidate codes found from graph. Using all codes.")
        candidate_codes = [
            {
                "code": code,
                "description": data["description"],
                "category": data["category"],
                "severity": data["severity"],
                "avg_reimbursement_usd": data["avg_reimbursement_usd"],
                "confidence": 0.1,
                "score": 0.1
            }
            for code, data in list(ICD10_DATA.items())[:10]
        ]

    # -------------------------------------------------------
    # 4. Get suggested code from MedGemma (graph-constrained)
    # -------------------------------------------------------
    try:
        suggestion = mg.suggest_icd_code(
            clinical_note=clinical_note,
            image_base64=image_base64,
            candidate_codes=candidate_codes,
            existing_code=existing_code
        )
    except Exception as e:
        logger.error(f"ICD suggestion error: {e}")
        raise HTTPException(status_code=500, detail=f"ICD suggestion failed: {e}")

    suggested_code = suggestion.get("suggested_code", "")
    confidence = float(suggestion.get("confidence", 0.5))
    reasoning = suggestion.get("reasoning", "")
    mismatch_detected = bool(suggestion.get("mismatch_detected", False))
    mismatch_reason = suggestion.get("mismatch_reason", "") or ""

    # -------------------------------------------------------
    # 5. Detect fraud risk if existing_code provided
    # -------------------------------------------------------
    fraud_risk = "none"
    risk_type = "none"
    financial_impact_usd = 0
    audit_recommendation = "No action required."
    evidence = ""

    if existing_code and existing_code != suggested_code:
        existing_data = ICD10_DATA.get(existing_code, {})
        suggested_data = ICD10_DATA.get(suggested_code, {})

        financial_gap = _graph.calculate_financial_gap(suggested_code, existing_code)

        try:
            fraud_result = mg.detect_fraud_risk(
                clinical_note=clinical_note,
                image_base64=image_base64,
                existing_code=existing_code,
                suggested_code=suggested_code,
                existing_description=existing_data.get("description", existing_code),
                suggested_description=suggested_data.get("description", suggested_code),
                financial_gap=financial_gap
            )
            fraud_risk = fraud_result.get("fraud_risk", "none")
            risk_type = fraud_result.get("risk_type", "none")
            evidence = fraud_result.get("evidence", "")
            audit_recommendation = fraud_result.get("recommendation", "")
            financial_impact_usd = fraud_result.get("financial_impact_usd", abs(financial_gap))
        except Exception as e:
            logger.error(f"Fraud detection error: {e}")
            # Fallback heuristic
            financial_impact_usd = abs(financial_gap)
            if financial_impact_usd > 2000:
                fraud_risk = "high"
            elif financial_impact_usd > 500:
                fraud_risk = "medium"
            elif financial_impact_usd > 0:
                fraud_risk = "low"
            risk_type = "upcoding" if financial_gap < 0 else "downcoding"
            evidence = f"Financial discrepancy of ${financial_impact_usd:,} detected."
            audit_recommendation = "Manual audit recommended."

    elif existing_code and existing_code == suggested_code:
        fraud_risk = "none"
        risk_type = "none"
        audit_recommendation = "Coding appears correct. No action required."
        evidence = "Submitted code matches system recommendation."

    return AnalyzeResponse(
        extracted_entities=extracted_entities,
        candidate_codes=candidate_codes,
        suggested_code=suggested_code,
        confidence=confidence,
        reasoning=reasoning,
        mismatch_detected=mismatch_detected,
        mismatch_reason=mismatch_reason,
        fraud_risk=fraud_risk,
        risk_type=risk_type,
        financial_impact_usd=financial_impact_usd,
        audit_recommendation=audit_recommendation,
        evidence=evidence
    )


@app.get("/validate/{code}", response_model=CodeDetailResponse)
def validate_code(code: str):
    """
    Returns full code details from the knowledge graph.
    Returns 404 if code not found.
    """
    code = code.upper()
    if code not in ICD10_DATA:
        raise HTTPException(status_code=404, detail=f"ICD-10 code {code} not found")

    data = ICD10_DATA[code]
    related = _graph.get_related_codes(code)
    exclusions = _graph.get_exclusions(code)

    return CodeDetailResponse(
        code=code,
        description=data["description"],
        category=data["category"],
        severity=data["severity"],
        avg_reimbursement_usd=data["avg_reimbursement_usd"],
        symptoms=data.get("symptoms", []),
        image_findings=data.get("image_findings", []),
        excludes=data.get("excludes", []),
        related=related,
        exclusions=exclusions
    )


@app.post("/compare", response_model=CompareResponse)
def compare_codes(request: CompareRequest):
    """
    Compares two ICD-10 codes.
    Returns severity difference, financial gap, and relationship.
    """
    code1 = request.code1.upper()
    code2 = request.code2.upper()

    if code1 not in ICD10_DATA:
        raise HTTPException(status_code=404, detail=f"Code {code1} not found")
    if code2 not in ICD10_DATA:
        raise HTTPException(status_code=404, detail=f"Code {code2} not found")

    data1 = ICD10_DATA[code1]
    data2 = ICD10_DATA[code2]

    severity_order = {"mild": 1, "moderate": 2, "severe": 3}
    sev1 = severity_order.get(data1["severity"], 0)
    sev2 = severity_order.get(data2["severity"], 0)

    if sev1 > sev2:
        severity_diff = f"{code1} is more severe than {code2}"
    elif sev2 > sev1:
        severity_diff = f"{code2} is more severe than {code1}"
    else:
        severity_diff = f"{code1} and {code2} have the same severity"

    financial_gap = _graph.calculate_financial_gap(code1, code2)

    # Determine relationship
    validation = _graph.validate_code_combination([code1, code2])
    related = _graph.get_related_codes(code1)

    relationship = "unrelated"
    if not validation["valid"]:
        relationship = "mutually exclusive (excludes)"
    elif related["parent"] and related["parent"]["code"] == code2:
        relationship = f"{code1} is child of {code2}"
    elif any(c["code"] == code2 for c in related["children"]):
        relationship = f"{code2} is child of {code1}"
    elif any(c["code"] == code2 for c in related["siblings"]):
        relationship = "siblings (same parent)"
    elif data1["category"] == data2["category"]:
        relationship = f"same category ({data1['category']})"

    return CompareResponse(
        code1=code1,
        code1_description=data1["description"],
        code1_reimbursement=data1["avg_reimbursement_usd"],
        code2=code2,
        code2_description=data2["description"],
        code2_reimbursement=data2["avg_reimbursement_usd"],
        severity_difference=severity_diff,
        financial_gap=financial_gap,
        relationship=relationship
    )


class CausalAuditResponse(BaseModel):
    causal_score: Optional[float]
    agent_trace: list
    visual_findings: list
    required_features: list
    missing_features: list
    counterfactual_result: dict
    final_verdict: dict
    patient_statement: str
    voice_symptoms: list
    three_way_match: dict
    upcoding_financial_impact_usd: int
    litigation_risk: dict


@app.post("/causal-audit", response_model=CausalAuditResponse)
async def causal_audit(
    clinical_note: str = Form(..., description="Clinical note text"),
    image: Optional[UploadFile] = File(None, description="Medical image (X-ray/scan)"),
    audio: Optional[UploadFile] = File(None, description="Patient voice recording (.wav)"),
    existing_code: str = Form(..., description="ICD-10 code submitted for billing"),
):
    """
    Causal audit endpoint — LangGraph 5-node workflow.
    VoiceAnalyst → VisualAnalyst → KnowledgeRetriever → CausalAuditor → CounterfactualVerifier
    Three-Way Match: Voice Statement ↔ Radiology Image ↔ Billing Code.
    """
    logger.info(
        f"Causal audit: code={existing_code}, "
        f"has_image={image is not None}, has_audio={audio is not None}"
    )

    image_base64 = None
    if image and image.filename:
        try:
            image_bytes = await image.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.warning(f"Image processing failed: {e}")

    audio_bytes = None
    if audio and audio.filename:
        try:
            audio_bytes = await audio.read()
        except Exception as e:
            logger.warning(f"Audio processing failed: {e}")

    initial_state = {
        "clinical_note": clinical_note,
        "image_base64": image_base64,
        "audio_bytes": audio_bytes,
        "existing_code": existing_code.upper(),
        "patient_statement": "",
        "voice_symptoms": [],
        "visual_findings": [],
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

    try:
        audit_graph = get_audit_graph()
        final_state = await asyncio.to_thread(audit_graph.invoke, initial_state)
    except Exception as e:
        logger.error(f"Causal audit workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Causal audit failed: {e}")

    req = final_state.get("knowledge_requirements") or {}
    required_features = (
        req.get("visual_required_all", []) + req.get("visual_required_any", [])
    )

    return CausalAuditResponse(
        causal_score=final_state.get("causal_score"),
        agent_trace=final_state.get("agent_trace", []),
        visual_findings=final_state.get("visual_findings", []),
        required_features=required_features,
        missing_features=final_state.get("missing_features", []),
        counterfactual_result=final_state.get("counterfactual_result") or {},
        final_verdict=final_state.get("final_verdict") or {},
        patient_statement=final_state.get("patient_statement", ""),
        voice_symptoms=final_state.get("voice_symptoms", []),
        three_way_match=final_state.get("three_way_match", {}),
        upcoding_financial_impact_usd=final_state.get("upcoding_financial_impact_usd", 0),
        litigation_risk=final_state.get("litigation_risk") or {},
    )


@app.get("/codes/search")
def search_icd_codes(q: str):
    """
    Search ICD-10 codes by description keyword.
    Returns matching codes with descriptions.
    """
    if not q or len(q) < 2:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 2 characters"
        )

    results = search_codes(q)
    return {
        "query": q,
        "count": len(results),
        "results": results
    }


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
