"""
ICD-10 MedGemma — Trust Dashboard (Streamlit)
================================================
4-Pane Trust Dashboard:
  Col 1: Input (clinical note, X-ray image, voice recording, ICD code)
  Col 2: Agent Trace (5-node LangGraph replay with 0.4s delay)
  Col 3: Three-Way Match table (Voice ↔ Image ↔ Code)
  Col 4: Verdict (fraud risk, causal score, financial/litigation metrics)

Port: 8501 (streamlit run frontend/streamlit_app.py --server.port 8501)
Backend: http://localhost:8000 (or BACKEND_URL env var for Docker)
"""

import os
import time
import json
import requests
import streamlit as st
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================

API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ICD-10 MedGemma Trust Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stColumn > div { padding: 0.5rem; }
    .verdict-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #e94560;
    }
    .verdict-card.safe {
        border-left-color: #00b894;
        background: #0a2e1a;
    }
    .match-table th {
        background: #2d2d44;
        padding: 6px 12px;
    }
    .match-table td { padding: 6px 12px; }
    .agent-step {
        font-family: monospace;
        font-size: 0.82rem;
        background: #0d1117;
        border-radius: 6px;
        padding: 0.6rem;
        margin: 0.3rem 0;
        border-left: 3px solid #58a6ff;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .mismatch-yes { color: #ff4757; font-weight: bold; }
    .mismatch-no  { color: #2ed573; font-weight: bold; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER: API CALL
# =========================================================

def call_causal_audit(
    clinical_note: str,
    existing_code: str,
    image_bytes=None,
    audio_bytes=None,
) -> dict:
    """POST to /causal-audit and return JSON response."""
    files = {}
    if image_bytes:
        files["image"] = ("image.jpg", image_bytes, "image/jpeg")
    if audio_bytes:
        files["audio"] = ("audio.wav", audio_bytes, "audio/wav")

    data = {
        "clinical_note": clinical_note,
        "existing_code": existing_code,
    }

    try:
        resp = requests.post(
            f"{API_BASE}/causal-audit",
            data=data,
            files=files if files else None,
            timeout=180,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Backend error {resp.status_code}: {resp.text[:300]}")
        return {}
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to backend at {API_BASE}. "
            "Start it with: `cd backend && uvicorn app:app --port 8000`"
        )
        return {}
    except Exception as e:
        st.error(f"Request failed: {e}")
        return {}


def check_health() -> dict:
    """Check backend health."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


# =========================================================
# RENDER HELPERS
# =========================================================

def render_match_row(label: str, match_score: float, mismatch: bool, detail: str = "") -> str:
    icon = "⚠️ MISMATCH" if mismatch else "✅ MATCH"
    color_class = "mismatch-yes" if mismatch else "mismatch-no"
    score_pct = f"{match_score:.0%}" if isinstance(match_score, float) else str(match_score)
    return (
        f"<tr>"
        f"<td><b>{label}</b></td>"
        f"<td class='{color_class}'>{icon}</td>"
        f"<td>{score_pct}</td>"
        f"<td style='font-size:0.8rem;color:#aaa;'>{detail[:80]}</td>"
        f"</tr>"
    )


def risk_color(risk: str) -> str:
    return {
        "none": "#2ed573",
        "low": "#ffa502",
        "medium": "#ff6348",
        "high": "#ff4757",
        "critical": "#c0392b",
    }.get(risk.lower() if risk else "none", "#888")


def render_agent_trace(trace_lines: list, placeholder):
    """Replay agent trace lines with typing delay."""
    rendered = ""
    for line in trace_lines:
        rendered += f'<div class="agent-step">{line}</div>'
        placeholder.markdown(rendered, unsafe_allow_html=True)
        time.sleep(0.4)


# =========================================================
# MAIN LAYOUT
# =========================================================

st.title("🏥 ICD-10 MedGemma — Causal Audit Trust Dashboard")
st.caption(
    "Symbolic-Neural Hybrid: MedGemma (perception) + Knowledge Graph (causal rules) + "
    "LangGraph (agentic workflow) + Three-Way Match (Voice ↔ Image ↔ Code)"
)

# Backend health badge
health = check_health()
if health:
    graph_be = health.get("graph_backend", "networkx")
    n_codes = health.get("total_icd10_codes", "?")
    cms_codes = health.get("total_cms_codes", "?")
    st.success(
        f"Backend: {graph_be.upper()} graph | "
        f"{n_codes:,} local codes | {cms_codes:,} CMS ICD-10 codes"
    )
else:
    st.error(f"Backend offline — start with: `cd backend && uvicorn app:app --port 8000`")

st.markdown("---")

# =========================================================
# FOUR COLUMNS
# =========================================================

col_input, col_trace, col_match, col_verdict = st.columns([1.1, 1.3, 0.9, 1.0])

# =========================================================
# COL 1: INPUT
# =========================================================

with col_input:
    st.subheader("📋 Patient Input")

    clinical_note = st.text_area(
        "Clinical Note",
        height=160,
        placeholder=(
            "Patient is a 58-year-old male with 3 days of high fever (39.4°C), "
            "productive cough with yellow-green sputum, right-sided pleuritic chest pain, "
            "and progressive dyspnea. SpO2 88% on room air. CXR shows right lower lobe "
            "lobar consolidation with air bronchograms."
        ),
        help="Paste or type the clinical note here.",
    )

    existing_code = st.text_input(
        "Submitted ICD-10 Code",
        value="J06.9",
        help="The ICD-10 code submitted for billing (to be validated).",
        max_chars=10,
    ).strip().upper()

    image_file = st.file_uploader(
        "X-Ray / Medical Image (optional)",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a chest X-ray or other medical image for visual analysis.",
    )

    if image_file:
        st.image(image_file, caption="Uploaded X-ray", use_container_width=True)

    audio_file = st.file_uploader(
        "Patient Voice Recording (.wav, optional)",
        type=["wav"],
        help="Upload a .wav recording of the patient describing symptoms. Enables Three-Way Match.",
    )

    if audio_file:
        st.audio(audio_file, format="audio/wav")
        st.caption("Voice will be transcribed using google/medasr (medical-grade ASR)")

    st.markdown("---")

    run_btn = st.button(
        "Run Causal Audit",
        type="primary",
        use_container_width=True,
        help="Run the 5-node LangGraph causal audit workflow",
    )

    # Demo presets
    st.markdown("**Quick Presets:**")
    preset_cols = st.columns(2)
    with preset_cols[0]:
        if st.button("Downcoding Demo", use_container_width=True, help="J06.9 billed for pneumonia"):
            st.session_state["preset_note"] = (
                "58M with 3-day fever (39.4°C), productive cough with purulent sputum, "
                "right pleuritic chest pain, SpO2 88%. CXR: right lower lobe lobar "
                "consolidation with air bronchograms."
            )
            st.session_state["preset_code"] = "J06.9"
            st.rerun()

    with preset_cols[1]:
        if st.button("Upcoding Demo", use_container_width=True, help="S22.4 billed for simple rib fracture"):
            st.session_state["preset_note"] = (
                "34F with 2 rib fractures after fall. Breathing normally. SpO2 100%. "
                "No paradoxical motion. Discharged home with analgesics."
            )
            st.session_state["preset_code"] = "S22.4"
            st.rerun()

    # Apply presets
    if "preset_note" in st.session_state:
        clinical_note = st.session_state.pop("preset_note")
        existing_code = st.session_state.pop("preset_code", existing_code)
        st.rerun()

# =========================================================
# COL 2: AGENT TRACE (placeholder)
# =========================================================

with col_trace:
    st.subheader("🤖 Agent Trace")
    trace_placeholder = st.empty()
    trace_placeholder.info("Run the audit to see the 5-node LangGraph workflow replay.")

# =========================================================
# COL 3: THREE-WAY MATCH (placeholder)
# =========================================================

with col_match:
    st.subheader("🔍 Three-Way Match")
    match_placeholder = st.empty()
    match_placeholder.info("Awaiting audit results...")

# =========================================================
# COL 4: VERDICT (placeholder)
# =========================================================

with col_verdict:
    st.subheader("⚖️ Verdict")
    verdict_placeholder = st.empty()
    verdict_placeholder.info("Submit a case to see the fraud verdict.")

# =========================================================
# EXECUTE AUDIT ON BUTTON CLICK
# =========================================================

if run_btn:
    if not clinical_note.strip():
        st.warning("Please enter a clinical note.")
        st.stop()
    if not existing_code:
        st.warning("Please enter an ICD-10 code to audit.")
        st.stop()

    image_bytes = image_file.read() if image_file else None
    audio_bytes = audio_file.read() if audio_file else None

    with st.spinner("Running 5-node causal audit workflow..."):
        result = call_causal_audit(
            clinical_note=clinical_note,
            existing_code=existing_code,
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
        )

    if not result:
        st.stop()

    # Extract fields
    agent_trace = result.get("agent_trace", [])
    three_way_match = result.get("three_way_match", {})
    final_verdict = result.get("final_verdict", {})
    causal_score = result.get("causal_score", 0.0)
    visual_findings = result.get("visual_findings", [])
    required_features = result.get("required_features", [])
    missing_features = result.get("missing_features", [])
    patient_statement = result.get("patient_statement", "")
    voice_symptoms = result.get("voice_symptoms", [])
    upcoding_impact = result.get("upcoding_financial_impact_usd", 0)
    litigation_risk = result.get("litigation_risk", {})
    cf_result = result.get("counterfactual_result", {})

    # ---------------------------------------------------------
    # COL 2: Replay agent trace
    # ---------------------------------------------------------

    with col_trace:
        trace_placeholder.empty()
        if agent_trace:
            rendered = ""
            for line in agent_trace:
                rendered += f'<div class="agent-step">{line}</div>'
                trace_placeholder.markdown(rendered, unsafe_allow_html=True)
                time.sleep(0.35)
        else:
            trace_placeholder.warning("No agent trace returned.")

        if cf_result.get("ran"):
            st.markdown("**Counterfactual Result:**")
            cf_answer = cf_result.get("answer", "?")
            cf_icon = "🔴 Fraud confirmed" if cf_answer == "NO" else "🟢 Code justified"
            st.markdown(f"{cf_icon}: `{cf_answer}` — {cf_result.get('explanation', '')[:200]}")

    # ---------------------------------------------------------
    # COL 3: Three-Way Match table
    # ---------------------------------------------------------

    with col_match:
        match_placeholder.empty()

        img_vs_code = three_way_match.get("image_vs_code", {})
        voice_vs_img = three_way_match.get("voice_vs_image", {})
        voice_vs_code = three_way_match.get("voice_vs_code", {})
        total_mismatches = three_way_match.get("mismatches_detected", 0)

        table_html = (
            "<table class='match-table' style='width:100%;border-collapse:collapse;'>"
            "<thead><tr>"
            "<th>Comparison</th><th>Result</th><th>Score</th><th>Detail</th>"
            "</tr></thead><tbody>"
        )

        table_html += render_match_row(
            "Image ↔ Code",
            img_vs_code.get("causal_score", causal_score),
            img_vs_code.get("mismatch_detected", False),
            f"Found: {', '.join(img_vs_code.get('found_features', []) or ['none'])}"
        )
        table_html += render_match_row(
            "Voice ↔ Image",
            voice_vs_img.get("match_score", 1.0),
            voice_vs_img.get("mismatch_detected", False),
            f"Agreed: {', '.join(voice_vs_img.get('agreed', []) or ['no audio'])}",
        )
        table_html += render_match_row(
            "Voice ↔ Code",
            voice_vs_code.get("match_score", 1.0),
            voice_vs_code.get("mismatch_detected", False),
            f"Found: {', '.join(voice_vs_code.get('found_keywords', []) or ['no audio'])}",
        )
        table_html += "</tbody></table>"

        st.markdown(table_html, unsafe_allow_html=True)

        # Mismatch badge
        if total_mismatches == 0:
            st.success(f"0/3 mismatches — All checks pass")
        elif total_mismatches == 1:
            st.warning(f"1/3 mismatch — Review recommended")
        elif total_mismatches == 2:
            st.error(f"2/3 mismatches — Strong fraud indication")
        else:
            st.error(f"3/3 mismatches — CRITICAL: All checks failed")

        # Feature breakdown
        with st.expander("Feature Details"):
            st.markdown(f"**Visual Findings:** `{', '.join(visual_findings) or 'none'}`")
            st.markdown(f"**Required Features:** `{', '.join(required_features) or 'none'}`")
            st.markdown(f"**Missing Features:** `{', '.join(missing_features) or 'none'}`")

            if patient_statement:
                st.markdown(f"**Voice Transcript:** _{patient_statement[:200]}_")
                st.markdown(f"**Voice Symptoms:** `{', '.join(voice_symptoms) or 'none'}`")

            if voice_vs_img.get("voice_only"):
                st.markdown(f"**Voice Only (not in image):** "
                            f"`{', '.join(voice_vs_img['voice_only'])}`")
            if voice_vs_img.get("image_only"):
                st.markdown(f"**Image Only (not in voice):** "
                            f"`{', '.join(voice_vs_img['image_only'])}`")

    # ---------------------------------------------------------
    # COL 4: Verdict
    # ---------------------------------------------------------

    with col_verdict:
        verdict_placeholder.empty()

        fraud_detected = final_verdict.get("fraud_detected", False)
        overall_risk = final_verdict.get("overall_risk", "none")
        explanation = final_verdict.get("explanation", "")

        # Main verdict banner
        if fraud_detected:
            risk_label = overall_risk.upper()
            col_verdict.error(f"FRAUD DETECTED — {risk_label} RISK")
        else:
            col_verdict.success("No Fraud Detected — Code Appears Valid")

        # Causal score gauge
        score_pct = int(causal_score * 100)
        score_color = "#2ed573" if causal_score >= 0.7 else "#ffa502" if causal_score >= 0.5 else "#ff4757"
        st.markdown(
            f"<div style='margin:8px 0;'>"
            f"<b>Causal Score:</b> "
            f"<span style='color:{score_color};font-size:1.3rem;font-weight:bold;'>"
            f"{score_pct}%</span>"
            f"<br><progress value='{score_pct}' max='100' "
            f"style='width:100%;height:10px;'></progress></div>",
            unsafe_allow_html=True
        )

        st.caption(f"Minimum required: ≥60% for code to be causally supported")

        # Financial metrics
        st.markdown("---")
        if upcoding_impact > 0:
            st.metric(
                "Overbilling Exposure (Upcoding)",
                f"${upcoding_impact:,}",
                help="Amount fraudulently overbilled vs. correct code reimbursement"
            )

        lit_score = litigation_risk.get("score", 0)
        lit_label = litigation_risk.get("label", "none")
        lit_applies = litigation_risk.get("applies", False)

        if lit_applies:
            lit_color = risk_color(lit_label)
            st.markdown(
                f"**Litigation Risk:** "
                f"<span style='color:{lit_color};font-size:1.1rem;'>"
                f"{lit_label.upper()} ({lit_score}/100)</span>",
                unsafe_allow_html=True
            )
            factors = litigation_risk.get("factors", [])
            if factors:
                for f in factors:
                    st.caption(f"• {f}")

            if lit_label == "critical":
                st.error("FLAG FOR LEGAL REVIEW — Possible systematic downcoding")

        # Mismatches count
        st.markdown(f"**Checks Failed:** {total_mismatches}/3")

        # Explanation
        st.markdown("---")
        st.markdown("**Audit Explanation:**")
        st.markdown(
            f"<div style='font-size:0.85rem;color:#ccc;background:#111;padding:8px;"
            f"border-radius:6px;'>{explanation[:400]}</div>",
            unsafe_allow_html=True
        )

        # Raw JSON expander
        with st.expander("Raw Response (JSON)"):
            st.json({
                "causal_score": causal_score,
                "final_verdict": final_verdict,
                "three_way_match": three_way_match,
                "litigation_risk": litigation_risk,
                "upcoding_financial_impact_usd": upcoding_impact,
                "counterfactual_result": cf_result,
            })

# =========================================================
# SIDEBAR: About
# =========================================================

with st.sidebar:
    st.markdown("## About")
    st.markdown(
        "**MedGemma Causal-Audit Agent (MCAA)**\n\n"
        "Symbolic-Neural Hybrid combining:\n"
        "- 🧠 **MedGemma-4B** (Google DeepMind) for clinical perception\n"
        "- 📊 **Medical Knowledge Graph** with causal requirements per code\n"
        "- 🔗 **LangGraph** 5-node agentic workflow\n"
        "- 🎙️ **google/medasr** for medical-grade voice transcription\n\n"
        "**Three-Way Match:**\n"
        "- Voice ↔ Image: did patient describe what image shows?\n"
        "- Voice ↔ Code: does voice support the billing code?\n"
        "- Image ↔ Code: do findings causally support the code?\n\n"
        f"**Backend:** {API_BASE}"
    )

    st.markdown("---")
    st.markdown("**API Endpoints:**")
    st.code(
        f"POST {API_BASE}/causal-audit\n"
        f"POST {API_BASE}/analyze\n"
        f"GET  {API_BASE}/validate/{{code}}\n"
        f"POST {API_BASE}/compare\n"
        f"GET  {API_BASE}/health",
        language="text"
    )

    st.markdown("---")
    st.markdown("**Running Validation:**")
    st.code(
        "cd tests\n"
        "python validate_100_cases.py \\\n"
        "  --dataset sample_dataset/cases.json \\\n"
        "  --endpoint http://localhost:8000",
        language="bash"
    )
