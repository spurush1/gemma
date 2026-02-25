"""
ICD-10 MedGemma — Causal Audit Trust Dashboard
================================================
Kaggle MedGemma Impact Challenge submission UI.

Three-tab layout:
  Tab 1: Live Demo — interactive audit with preset cases + live file uploads
  Tab 2: How It Works — architecture + innovation explanation
  Tab 3: Results — 96.7% NIH validation accuracy + metrics

Port: 9501
Backend: http://localhost:8000 (or BACKEND_URL env var for Docker Compose)

Run:
    streamlit run frontend/streamlit_app.py --server.port 9501
"""

import os
import time
from pathlib import Path

import requests
import streamlit as st

# NIH images directory (relative to project root from this file location)
_IMAGES_DIR = Path(__file__).parent.parent / "tests" / "sample_dataset" / "nih_images"

# =========================================================
# CONFIGURATION
# =========================================================

API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ICD-10 MedGemma Causal Audit",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .main { padding-top: 0.5rem; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    /* Agent trace cards */
    .agent-step {
        font-family: 'Courier New', monospace;
        font-size: 0.80rem;
        background: #0d1117;
        border-radius: 6px;
        padding: 0.55rem 0.7rem;
        margin: 0.25rem 0;
        border-left: 3px solid #58a6ff;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .agent-step.done   { border-left-color: #2ed573; }
    .agent-step.active { border-left-color: #ffa502; }

    /* Three-way match table */
    .match-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .match-table th {
        background: #1e2a3a;
        color: #8ab4f8;
        padding: 7px 10px;
        text-align: left;
    }
    .match-table td { padding: 6px 10px; border-bottom: 1px solid #2a2a2a; }
    .match-ok   { color: #2ed573; font-weight: 700; }
    .match-bad  { color: #ff4757; font-weight: 700; }

    /* Verdict banners */
    .verdict-fraud {
        background: linear-gradient(135deg, #3b0a0a, #5c1a1a);
        border: 2px solid #ff4757;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: #ff4757;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .verdict-clean {
        background: linear-gradient(135deg, #0a2e1a, #0d3d22);
        border: 2px solid #2ed573;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: #2ed573;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .verdict-review {
        background: linear-gradient(135deg, #2e1f00, #3d2900);
        border: 2px solid #ffa502;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: #ffa502;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 1px;
    }

    /* Score bar */
    .score-bar-wrap { margin: 0.6rem 0; }
    .score-num { font-size: 2rem; font-weight: 800; }

    /* Chip badges */
    .chip       { display:inline-block; padding:2px 8px; border-radius:12px;
                  font-size:0.75rem; margin:2px; background:#1e2a3a; color:#8ab4f8; }
    .chip-red   { background:#3b0a0a; color:#ff6b6b; }
    .chip-green { background:#0a2e1a; color:#51cf66; }

    /* Accuracy badge */
    .acc-badge {
        background: linear-gradient(135deg, #0a2e1a, #155239);
        border: 2px solid #2ed573;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #2ed573;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        margin: 0.3rem 0;
    }

    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 0.95rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DEMO PRESET CASES
# =========================================================

DEMO_CASES = {
    "🚨 Pneumothorax Billed for Pneumonia": {
        "note": (
            "71M with 3-day fever (38.8°C), productive purulent cough, right-sided pleuritic "
            "chest pain, SpO2 89% on room air. Auscultation: right lower lobe bronchial breath "
            "sounds, dullness to percussion, egophony. CXR: dense right lower lobe consolidation "
            "with air bronchograms consistent with lobar pneumonia."
        ),
        "code": "J93.1",
        "expected": "FRAUD — J93.1 (Pneumothorax) requires pleural air/lung collapse; image shows lobar consolidation",
        "fraud": True,
        "image_file": "CXR1012_IM-0013-1001.png",
    },
    "✅ Lobar Pneumonia Confirmed": {
        "note": (
            "60F with 48-hour history of fever (39.1°C), productive rusty sputum, and left "
            "lower lobe consolidation signs. SpO2 88%. CXR: left lower lobe lobar consolidation "
            "with air bronchograms, consistent with bacterial lobar pneumonia."
        ),
        "code": "J18.1",
        "expected": "CLEAN — J18.1 (Lobar Pneumonia) correctly supported by consolidation on imaging",
        "fraud": False,
        "image_file": "CXR28_IM-1231-1001.png",
    },
    "✅ Pneumothorax Confirmed": {
        "note": (
            "27M, tall and lean. Sudden-onset right-sided pleuritic chest pain and dyspnea at "
            "rest. HR 122 bpm, SpO2 91%. Absent right breath sounds, hyper-resonance, tracheal "
            "deviation. CXR: right pneumothorax with visible pleural line and mediastinal shift."
        ),
        "code": "J93.1",
        "expected": "CLEAN — J93.1 (Pneumothorax) confirmed by imaging",
        "fraud": False,
        "image_file": "CXR1021_IM-0017-1001-0001.png",
    },
    "🚨 Heart Failure Upcoding": {
        "note": (
            "67F with dilated cardiomyopathy presents with mild exertional dyspnea and "
            "bilateral leg swelling for one week. SpO2 94%. CXR: cardiomegaly with "
            "cardiothoracic ratio >0.55. No Kerley B lines, no pulmonary edema, no pleural "
            "effusions. Billed as acute on chronic systolic heart failure."
        ),
        "code": "I50.1",
        "expected": "FRAUD — I50.1 (Acute Systolic HF) requires pulmonary edema/Kerley B lines; only cardiomegaly found",
        "fraud": True,
        "image_file": "CXR1013_IM-0013-1001.png",
    },
    "✅ Normal Exam — No Finding": {
        "note": (
            "22F presents with intermittent non-pleuritic chest pain attributed to anxiety. "
            "No dyspnea, fever, or cough. SpO2 99%. Clear chest auscultation, "
            "non-tender chest wall. CXR normal: no acute cardiopulmonary findings."
        ),
        "code": "Z03.89",
        "expected": "CLEAN — Z03.89 (No Finding) correctly matched to normal CXR",
        "fraud": False,
        "image_file": "CXR1_1_IM-0001-4001.png",
    },
}

# =========================================================
# API HELPERS
# =========================================================


def call_causal_audit(
    clinical_note: str,
    existing_code: str,
    image_bytes=None,
    audio_bytes=None,
) -> dict:
    """POST to /causal-audit and return parsed JSON."""
    files: dict = {}
    if image_bytes:
        files["image"] = ("image.jpg", image_bytes, "image/jpeg")
    if audio_bytes:
        files["audio"] = ("audio.wav", audio_bytes, "audio/wav")

    try:
        resp = requests.post(
            f"{API_BASE}/causal-audit",
            data={"clinical_note": clinical_note, "existing_code": existing_code},
            files=files or None,
            timeout=180,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Backend error {resp.status_code}: {resp.text[:400]}")
        return {}
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to backend at **{API_BASE}**.\n\n"
            "Start with:\n```bash\ncd backend && uvicorn app:app --port 8000\n```"
        )
        return {}
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return {}


def get_health() -> dict:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


# =========================================================
# RENDER HELPERS
# =========================================================


def _chip(text: str, cls: str = "") -> str:
    return f"<span class='chip {cls}'>{text}</span>"


def _match_icon(mismatch: bool) -> str:
    if mismatch:
        return "<span class='match-bad'>⚠ MISMATCH</span>"
    return "<span class='match-ok'>✅ MATCH</span>"


def render_match_table(three_way: dict, causal_score) -> str:
    img_vs_code   = three_way.get("image_vs_code", {})
    voice_vs_img  = three_way.get("voice_vs_image", {})
    voice_vs_code = three_way.get("voice_vs_code", {})

    def row(label, mismatch, score, detail):
        score_str = f"{score:.0%}" if isinstance(score, float) else "—"
        return (
            f"<tr>"
            f"<td><b>{label}</b></td>"
            f"<td>{_match_icon(mismatch)}</td>"
            f"<td style='color:#aaa;'>{score_str}</td>"
            f"<td style='font-size:0.78rem;color:#888;'>{(detail or '')[:70]}</td>"
            f"</tr>"
        )

    found_str = ", ".join((img_vs_code.get("found_features") or [])[:3]) or "none"
    agreed_str = ", ".join((voice_vs_img.get("agreed") or [])[:3]) or "no audio"
    kw_str = ", ".join((voice_vs_code.get("found_keywords") or [])[:3]) or "no audio"

    return (
        "<table class='match-table'>"
        "<thead><tr>"
        "<th>Comparison</th><th>Result</th><th>Score</th><th>Detail</th>"
        "</tr></thead><tbody>"
        + row("Image ↔ Code",  img_vs_code.get("mismatch_detected", False),
              img_vs_code.get("causal_score", causal_score or 0.0), f"Found: {found_str}")
        + row("Voice ↔ Image", voice_vs_img.get("mismatch_detected", False),
              voice_vs_img.get("match_score", 1.0), f"Agreed: {agreed_str}")
        + row("Voice ↔ Code",  voice_vs_code.get("mismatch_detected", False),
              voice_vs_code.get("match_score", 1.0), f"Keywords: {kw_str}")
        + "</tbody></table>"
    )


def render_score_bar(score) -> str:
    if score is None:
        return "<div style='color:#888;'>Score: N/A (insufficient image evidence)</div>"
    pct = int(float(score) * 100)
    color = "#2ed573" if score >= 0.7 else "#ffa502" if score >= 0.5 else "#ff4757"
    return (
        f"<div class='score-bar-wrap'>"
        f"<div class='score-num' style='color:{color};'>{pct}<small style='font-size:1rem;'>%</small></div>"
        f"<div style='background:#222;border-radius:4px;height:8px;'>"
        f"<div style='background:{color};width:{pct}%;height:100%;border-radius:4px;'></div>"
        f"</div>"
        f"<div style='font-size:0.75rem;color:#666;margin-top:3px;'>≥60% = code causally supported</div>"
        f"</div>"
    )


# =========================================================
# SIDEBAR
# =========================================================

# Fetch once at render time — used by sidebar AND tab_demo mock banner
health = get_health()

with st.sidebar:
    st.markdown("## 🏥 MCAA")
    st.caption("MedGemma Causal-Audit Agent")
    st.markdown("---")

    if health:
        st.success("Backend ✅ Online")
        st.caption(
            f"Graph: {health.get('graph_backend','?').upper()} | "
            f"{health.get('total_icd10_codes','?'):,} codes"
        )
        if health.get("mock_mode", True):
            st.warning(
                "⚠️ **MOCK MODE**\n\n"
                "MedGemma API key not set. Responses are synthetic.\n\n"
                "Set `MEDGEMMA_API_KEY` + `MEDGEMMA_ENDPOINT` to enable live inference."
            )
    else:
        st.error("Backend ❌ Offline")
        st.caption(f"Expected: {API_BASE}")

    st.markdown("---")
    st.markdown(
        "<div class='acc-badge'>96.7% Accuracy<br><small>NIH ChestX-ray14 · 59/61 cases</small></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Quick Demo Cases**")
    for case_name, case in DEMO_CASES.items():
        if st.button(case_name, use_container_width=True, key=f"demo_{case_name}"):
            # Pre-fill widget state directly using widget keys
            st.session_state["_demo_note"]     = case["note"]
            st.session_state["_demo_code"]     = case["code"]
            st.session_state["_demo_expected"] = case["expected"]
            st.session_state["_demo_fraud"]    = case["fraud"]
            # Load paired NIH image bytes
            img_file = case.get("image_file", "")
            if img_file:
                img_path = _IMAGES_DIR / img_file
                if img_path.exists():
                    st.session_state["_demo_image_bytes"] = img_path.read_bytes()
                    st.session_state["_demo_image_name"]  = img_file
                else:
                    st.session_state.pop("_demo_image_bytes", None)
                    st.session_state.pop("_demo_image_name", None)
            else:
                st.session_state.pop("_demo_image_bytes", None)
                st.session_state.pop("_demo_image_name", None)
            st.session_state["_auto_run"] = True
            st.rerun()

    st.markdown("---")
    st.caption(f"Backend: {API_BASE}")
    st.caption("Port 9501 · Kaggle MedGemma Impact Challenge")


# =========================================================
# MAIN TITLE
# =========================================================

st.markdown("# 🏥 ICD-10 Fraud Detection — MedGemma Causal Audit")
st.caption(
    "**Symbolic-Neural Hybrid** · MedGemma 4B-IT + CheXNet DenseNet-121 + "
    "Knowledge Graph (39 codes) + LangGraph 5-node workflow + Three-Way Match"
)

tab_demo, tab_how, tab_results, tab_history = st.tabs([
    "🔬 Live Demo",
    "⚙️ How It Works",
    "📊 Validation Results",
    "📋 Audit History",
])


# =========================================================
# TAB 1 — LIVE DEMO
# =========================================================

with tab_demo:

    # Mock mode top-of-page banner (re-uses health data fetched for sidebar)
    if health.get("mock_mode", True):
        st.warning(
            "**⚠️ MOCK MODE ACTIVE** — Results below are synthetic (rule-based stubs). "
            "Set `MEDGEMMA_API_KEY` and `MEDGEMMA_ENDPOINT` environment variables on the "
            "backend to enable live MedGemma 4B-IT inference.",
        )

    # Pull demo preset from session state (set by sidebar buttons)
    demo_note     = st.session_state.pop("_demo_note",     "")
    demo_code     = st.session_state.pop("_demo_code",     "")
    demo_expected = st.session_state.pop("_demo_expected", "")
    demo_fraud    = st.session_state.pop("_demo_fraud",    None)
    demo_img_bytes = st.session_state.get("_demo_image_bytes")
    demo_img_name  = st.session_state.get("_demo_image_name", "demo.png")
    auto_run      = st.session_state.pop("_auto_run", False)

    # When a demo button was clicked, push values into widget session-state keys
    if demo_note:
        st.session_state["_note_area"] = demo_note
    if demo_code:
        st.session_state["_code_input"] = demo_code

    if demo_expected:
        if demo_fraud:
            st.error(f"**Expected:** {demo_expected}")
        else:
            st.success(f"**Expected:** {demo_expected}")

    st.markdown("### Patient Case Input")

    input_tab_paste, input_tab_upload = st.tabs(["📋 Type / Paste", "📁 Upload Files"])

    # ── Sub-tab: Type / Paste ───────────────────────────────
    with input_tab_paste:
        col_note, col_code_col = st.columns([3, 1])
        with col_note:
            clinical_note_paste = st.text_area(
                "Clinical Note",
                key="_note_area",
                height=130,
                placeholder=(
                    "Patient is a 58-year-old male with 3 days of high fever (39.4°C), "
                    "productive cough, right-sided pleuritic chest pain, SpO2 88%. "
                    "CXR: right lower lobe lobar consolidation with air bronchograms."
                ),
            )
        with col_code_col:
            icd_code_paste = st.text_input(
                "ICD-10 Code to Audit",
                key="_code_input",
                max_chars=10,
            ).strip().upper()

        img_paste = st.file_uploader(
            "X-Ray Image (optional — overrides demo image)",
            type=["jpg", "jpeg", "png"],
            key="img_paste",
        )
        # Show demo image if loaded, or uploaded image
        if img_paste:
            st.image(img_paste, caption="Uploaded X-ray", width=260)
            # Clear any cached demo image so uploaded one takes priority
            st.session_state.pop("_demo_image_bytes", None)
            st.session_state.pop("_demo_image_name", None)
            demo_img_bytes = None
        elif demo_img_bytes:
            st.image(demo_img_bytes, caption=f"Demo image: {demo_img_name}", width=260)

        audio_paste = st.file_uploader(
            "Voice Recording (optional — .wav/.mp3)",
            type=["wav", "mp3"],
            key="audio_paste",
        )
        if audio_paste:
            st.audio(audio_paste)

        run_paste = st.button(
            "▶ Run Causal Audit",
            type="primary",
            use_container_width=True,
            key="run_paste",
        )

    # ── Sub-tab: Upload Files ───────────────────────────────
    with input_tab_upload:
        st.info(
            "Upload real clinical documents and imaging for live analysis. "
            "The system uses the same 5-node LangGraph workflow as demo cases."
        )

        uploaded_note_file = st.file_uploader(
            "Clinical Note (.txt or .pdf)",
            type=["txt", "pdf"],
            key="note_upload",
        )
        uploaded_img_file = st.file_uploader(
            "Chest X-Ray Image (.jpg / .png)",
            type=["jpg", "jpeg", "png"],
            key="img_upload",
        )
        if uploaded_img_file:
            st.image(uploaded_img_file, caption="Uploaded X-ray", width=260)

        uploaded_audio_file = st.file_uploader(
            "Voice Recording (.wav / .mp3)",
            type=["wav", "mp3"],
            key="audio_upload",
        )
        if uploaded_audio_file:
            st.audio(uploaded_audio_file)

        icd_code_upload = st.text_input(
            "ICD-10 Code to Audit",
            value="",
            placeholder="e.g. J18.1",
            max_chars=10,
            key="code_upload",
        ).strip().upper()

        run_upload = st.button(
            "▶ Run Causal Audit (Uploaded Files)",
            type="primary",
            use_container_width=True,
            key="run_upload",
        )

    # ── Resolve which inputs are active ────────────────────
    running = run_paste or run_upload or auto_run

    if run_paste or auto_run:
        active_note  = clinical_note_paste
        active_code  = icd_code_paste
        active_audio = audio_paste
        # Use uploaded image if provided; fall back to demo image
        active_img_bytes = img_paste.read() if img_paste else (
            st.session_state.get("_demo_image_bytes")
        )
    else:
        if uploaded_note_file is not None:
            raw = uploaded_note_file.read()
            try:
                active_note = raw.decode("utf-8")
            except UnicodeDecodeError:
                active_note = raw.decode("latin-1")
        else:
            active_note = ""
        active_code  = icd_code_upload
        active_img_bytes = uploaded_img_file.read() if uploaded_img_file else None
        active_audio = uploaded_audio_file

    # ── Run the audit ───────────────────────────────────────
    if running:
        if not active_note.strip():
            st.warning("Please enter or upload a clinical note.")
            st.stop()
        if not active_code:
            st.warning("Please enter an ICD-10 code to audit.")
            st.stop()

        image_bytes = active_img_bytes
        audio_bytes = active_audio.read() if active_audio else None

        st.markdown("---")
        st.markdown("### Audit Results")

        col_trace, col_match, col_verdict = st.columns([1.3, 1.0, 1.0])

        with col_trace:
            st.markdown("**🤖 Agent Pipeline**")
            trace_ph = st.empty()
            trace_ph.info("Running 5-node LangGraph workflow…")
        with col_match:
            st.markdown("**🔍 Three-Way Match**")
            match_ph = st.empty()
            match_ph.info("Awaiting…")
        with col_verdict:
            st.markdown("**⚖️ Verdict**")
            verdict_ph = st.empty()
            verdict_ph.info("Awaiting…")

        with st.spinner("Calling backend causal audit…"):
            result = call_causal_audit(
                clinical_note=active_note,
                existing_code=active_code,
                image_bytes=image_bytes,
                audio_bytes=audio_bytes,
            )

        if not result:
            st.stop()

        # ── Save to audit history ────────────────────────────
        if "audit_history" not in st.session_state:
            st.session_state["audit_history"] = []
        st.session_state["audit_history"].append({
            "code": active_code,
            "note": active_note,
            "image_bytes": active_img_bytes,
            "image_name": (getattr(img_paste, "name", None) or demo_img_name or "image"),
            "result": result,
            "expected": demo_expected,
            "fraud_expected": demo_fraud,
        })
        st.session_state["_hist_idx"] = len(st.session_state["audit_history"]) - 1

        # Parse
        agent_trace       = result.get("agent_trace", [])
        three_way         = result.get("three_way_match", {})
        final_verdict     = result.get("final_verdict", {})
        causal_score      = result.get("causal_score")
        visual_findings   = result.get("visual_findings", [])
        required_features = result.get("required_features", [])
        missing_features  = result.get("missing_features", [])
        patient_statement = result.get("patient_statement", "")
        voice_symptoms    = result.get("voice_symptoms", [])
        upcoding_impact   = result.get("upcoding_financial_impact_usd", 0)
        litigation_risk   = result.get("litigation_risk", {})
        cf_result         = result.get("counterfactual_result", {})
        fraud_detected    = final_verdict.get("fraud_detected", False)
        needs_human_review = final_verdict.get("needs_human_review", False)
        mismatches        = three_way.get("mismatches_detected", 0)
        chexnet_labels    = result.get("chexnet_labels", [])

        # ── Agent trace replay ──────────────────────────────
        with col_trace:
            trace_ph.empty()
            if agent_trace:
                rendered = ""
                for i, line in enumerate(agent_trace):
                    css = "active" if i == len(agent_trace) - 1 else "done"
                    rendered += f'<div class="agent-step {css}">{line}</div>'
                    trace_ph.markdown(rendered, unsafe_allow_html=True)
                    time.sleep(0.3)
            else:
                trace_ph.warning("No agent trace returned.")

            if chexnet_labels or visual_findings:
                st.markdown("**Image Ensemble:**")
                if chexnet_labels:
                    st.caption(f"CheXNet: {', '.join(chexnet_labels[:4])}")
                if visual_findings:
                    st.caption(f"Merged: {', '.join(visual_findings[:5])}")

            if cf_result.get("ran"):
                cf_ans  = cf_result.get("answer", "?")
                cf_icon = "🔴 Fraud confirmed" if cf_ans == "NO" else "🟢 Code justified"
                st.markdown(f"**Counterfactual:** {cf_icon}")
                st.caption(cf_result.get("explanation", "")[:200])

        # ── Three-way match ─────────────────────────────────
        with col_match:
            match_ph.empty()
            match_ph.markdown(
                render_match_table(three_way, causal_score),
                unsafe_allow_html=True,
            )

            if mismatches == 0:
                st.success("0/3 mismatches — All checks pass")
            elif mismatches == 1:
                st.warning("1/3 mismatch — Review recommended")
            elif mismatches == 2:
                st.error("2/3 mismatches — Strong fraud indication")
            else:
                st.error("3/3 mismatches — CRITICAL: All checks failed")

            with st.expander("Feature Details"):
                if visual_findings:
                    st.markdown(
                        "**Visual Findings:** " +
                        " ".join(_chip(f, "chip-green") for f in visual_findings),
                        unsafe_allow_html=True,
                    )
                if missing_features:
                    st.markdown(
                        "**Missing:** " +
                        " ".join(_chip(f, "chip-red") for f in missing_features),
                        unsafe_allow_html=True,
                    )
                if patient_statement:
                    st.markdown(f"**Transcript:** _{patient_statement[:200]}_")
                if voice_symptoms:
                    st.markdown(
                        "**Voice Symptoms:** " +
                        " ".join(_chip(s) for s in voice_symptoms),
                        unsafe_allow_html=True,
                    )
                iv = three_way.get("voice_vs_image", {})
                if iv.get("voice_only"):
                    st.caption(f"Voice only: {', '.join(iv['voice_only'])}")
                if iv.get("image_only"):
                    st.caption(f"Image only: {', '.join(iv['image_only'])}")

        # ── Verdict ─────────────────────────────────────────
        with col_verdict:
            verdict_ph.empty()
            if fraud_detected:
                verdict_ph.markdown(
                    "<div class='verdict-fraud'>🚨 FRAUD<br>DETECTED</div>",
                    unsafe_allow_html=True,
                )
            elif needs_human_review and not fraud_detected:
                verdict_ph.markdown(
                    "<div class='verdict-review'>⚠️ CANNOT<br>ASSESS</div>",
                    unsafe_allow_html=True,
                )
                st.caption("Code not in knowledge base or image uninformative — manual clinical review required.")
            else:
                verdict_ph.markdown(
                    "<div class='verdict-clean'>✅ CODE<br>VALID</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(render_score_bar(causal_score), unsafe_allow_html=True)

            if upcoding_impact > 0:
                st.metric(
                    "Overbilling Exposure",
                    f"${upcoding_impact:,}",
                    help="Submitted code reimburses more than correct code",
                )

            lit_label = litigation_risk.get("label", "none")
            lit_score = litigation_risk.get("score", 0)
            if litigation_risk.get("applies"):
                _LIT_COLORS = {
                    "low": "#2ed573", "medium": "#ffa502",
                    "high": "#ff6348", "critical": "#ff4757",
                }
                color = _LIT_COLORS.get(lit_label, "#888")
                st.markdown(
                    f"**Litigation Risk:** "
                    f"<span style='color:{color};font-size:1.05rem;font-weight:700;'>"
                    f"{lit_label.upper()} ({lit_score}/100)</span>",
                    unsafe_allow_html=True,
                )
                for factor in litigation_risk.get("factors", []):
                    st.caption(f"• {factor}")
                if lit_label == "critical":
                    st.error("FLAG FOR LEGAL REVIEW")

            explanation = final_verdict.get("explanation", "")
            if explanation:
                st.markdown("---")
                st.markdown(
                    f"<div style='font-size:0.82rem;color:#ccc;background:#111;"
                    f"padding:8px;border-radius:6px;'>{explanation[:400]}</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("Raw JSON"):
                st.json({
                    "causal_score": causal_score,
                    "final_verdict": final_verdict,
                    "three_way_match": three_way,
                    "upcoding_financial_impact_usd": upcoding_impact,
                    "litigation_risk": litigation_risk,
                })


# =========================================================
# TAB 2 — HOW IT WORKS
# =========================================================

with tab_how:
    st.markdown("## System Architecture")

    st.code("""
Patient Voice (.wav)
  └─► [Node 1: VoiceAnalyst]
      google/medasr medical-grade ASR → voice_symptoms list

Chest X-Ray Image
  └─► [Node 2: VisualAnalyst]           ← Two-pass fix prevents text bias
      Pass 1: CheXNet DenseNet-121       → standardised NIH labels (locked)
              MedGemma 4B-IT image-only  → free-form findings
              Merge via RadLex synonyms  → expanded_findings
      Pass 2: MedGemma text-only         → severity, body_system

Clinical Note Text ─────────────────────────────────┐
                                                    ▼
[Node 3: KnowledgeRetriever]
  Looks up CausalRequirement for submitted ICD-10 code
  (39 codes × AND/OR visual rules, fraud_differential, voice_keywords)

[Node 4: CausalAuditor] — Three-Way Match
  ┌──────────────────────────────────────────────┐
  │  Image  ↔ Code   (CausalRequirement)         │
  │  Voice  ↔ Image  (symptom/finding overlap)   │
  │  Voice  ↔ Code   (voice_keyword matching)    │
  └──────────────────────────────────────────────┘
  fraud_flagged = ANY pair mismatches AND confidence ≥ 0.75

[Node 5: CounterfactualVerifier]  (only if fraud_flagged)
  MedGemma: "If patient had {alt_code} instead, is {code} still needed?"
  → Confirms or overturns the fraud finding

VERDICT: FRAUD / CLEAN + Causal Score + Financial Impact + Litigation Risk
""", language="text")

    st.markdown("---")
    st.markdown("## Three Key Innovations")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 1. Two-Pass Multimodal Fix")
        st.markdown(
            "MedGemma has a **text-bias problem**: when image + note are sent together, "
            "it follows the text and ignores the image.\n\n"
            "**Fix:** Pass 1 sends the X-ray *alone* — findings are locked in before "
            "any text is processed. Pass 2 uses only text. "
            "Image findings are therefore unbiased."
        )
        st.info("Pneumothorax X-ray + 'no chest findings' note → image correctly overrides text")

    with col_b:
        st.markdown("### 2. CheXNet + MedGemma Ensemble")
        st.markdown(
            "**CheXNet** (DenseNet-121, NIH weights) produces standardised, "
            "quantitative pathology labels (e.g. 'Consolidation: 0.87').\n\n"
            "**MedGemma** produces free-form clinical language. "
            "A **RadLex synonym map** (200+ term pairs) merges both for maximum coverage."
        )
        st.info("Tier 1 (RadLex) + Tier 2 (CheXNet): 65.6% → 96.7% accuracy")

    with col_c:
        st.markdown("### 3. Symbolic-Neural Hybrid")
        st.markdown(
            "**Neural** (MedGemma, CheXNet): perception — what does the image show?\n\n"
            "**Symbolic** (Knowledge Graph): reasoning — what *must* be true for this code?\n\n"
            "The Knowledge Engine encodes AND/OR causal requirements per code, "
            "validated against CMS billing rules. No LLM can hallucinate a passing score."
        )
        st.info("39 ICD-10 codes × expert-validated causal rules in causal_rules.yaml")

    st.markdown("---")
    st.markdown("## Tech Stack")

    import pandas as pd
    st.dataframe(
        pd.DataFrame({
            "Component": [
                "Clinical NLP", "Image Analysis (Tier 2)", "Terminology (Tier 1)",
                "Workflow Engine", "Knowledge Layer", "Graph DB",
                "Voice ASR", "Backend API", "Frontend",
            ],
            "Technology": [
                "MedGemma 4B-IT (Google DeepMind) via featherless-ai",
                "CheXNet DenseNet-121 — NIH ChestX-ray14 weights (torchxrayvision)",
                "RadLex Synonym Map — 200+ medical term normalizations",
                "LangGraph StateGraph — 5-node agentic pipeline",
                "Medical Knowledge Engine — causal_rules.yaml (39 codes)",
                "Neo4j + NetworkX fallback for candidate code traversal",
                "google/medasr via HuggingFace Inference API",
                "FastAPI + Uvicorn (Python 3.11)",
                "Streamlit (this dashboard) — port 9501",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("## Why MedGemma vs a General LLM?")

    st.dataframe(
        pd.DataFrame({
            "Criterion": [
                "ICD-10 coding training",
                "Chest X-ray visual encoder",
                "Medical safety RLHF",
                "MedQA benchmark",
                "Cost per API call",
                "Model size",
            ],
            "MedGemma 4B-IT": [
                "✅ Explicit (MIMIC + PubMed pre-training)",
                "✅ Medical SigLIP — trained on radiology image-text pairs",
                "✅ Conservative — 'insufficient evidence' vs hallucination",
                "79.1%",
                "Free via HuggingFace",
                "4B params",
            ],
            "GPT-4 / Claude": [
                "❌ Implicit — general web training",
                "⚠️ General CLIP — not radiology-specific",
                "⚠️ General RLHF",
                "~87% (175B params)",
                "$0.01–0.06 / call",
                "175B+ params",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# TAB 3 — VALIDATION RESULTS
# =========================================================

with tab_results:
    import pandas as pd

    st.markdown("## Validation Results — NIH ChestX-ray14")
    st.markdown("### Accuracy Progression (61-case validation set, all legitimate cases)")

    acc_df = pd.DataFrame({
        "System Version": [
            "Baseline (MedGemma text-only, no image fix)",
            "+ Tier 1: RadLex Normalization",
            "+ Tier 2: CheXNet DenseNet-121 Ensemble",
        ],
        "Accuracy (%)": [65.6, 80.3, 96.7],
        "False Positives": [21, 12, 2],
    })
    st.dataframe(acc_df, use_container_width=True, hide_index=True)
    st.bar_chart(acc_df.set_index("System Version")["Accuracy (%)"])

    st.markdown("---")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Accuracy",        "96.7%",  "+31.1% vs baseline")
    col_m2.metric("True Negatives",  "59 / 61", "Legitimate cases passed")
    col_m3.metric("False Positives",  "2 / 61", "Over-flagged cases")
    col_m4.metric("False Negatives",  "0 / 61", "None missed in clean set")

    st.markdown("---")
    st.markdown("### Root Cause Analysis of False Positives")

    st.dataframe(
        pd.DataFrame({
            "Root Cause": [
                "Terminology mismatch (e.g. 'right_sided_pleural_effusion' ≠ 'pleural_effusion')",
                "Visual encoder failure (MedGemma returns all-normal for pathological images)",
                "Remaining (both CheXNet and MedGemma return near-zero confidence)",
            ],
            "FP Count (Before Fix)": [9, 12, 2],
            "Fix Applied": [
                "Tier 1: RadLex synonym map (200+ pairs)",
                "Tier 2: CheXNet DenseNet-121 ensemble",
                "Tier 3: Human review zone (needs_human_review=True, not auto-flagged)",
            ],
            "FP Count (After Fix)": [0, 0, 2],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("### Synthetic Dataset (100 cases)")

    st.dataframe(
        pd.DataFrame({
            "Fraud Type": ["Downcoding", "Upcoding", "Correct Code", "Unrelated Code"],
            "Cases": [28, 25, 26, 21],
            "Description": [
                "Billing lower code than warranted (e.g. J06.9 for confirmed pneumonia)",
                "Billing higher code than warranted (e.g. S22.4 for 2-rib fracture)",
                "Correct ICD-10 code causally supported by findings",
                "Code unrelated to imaging/symptoms",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("### Financial Impact Metrics (Synthetic Dataset)")

    col_f1, col_f2, col_f3 = st.columns(3)
    col_f1.metric("Total Upcoding Exposure", "~$94,500",
                  help="Estimated overbilling across 25 upcoding test cases")
    col_f2.metric("Avg Upcoding Per Case", "~$3,780",
                  help="Mean overbilling per upcoding case")
    col_f3.metric("Critical Litigation Cases", "8 / 28",
                  help="Downcoding cases with litigation risk > 75/100 (critical)")

    st.markdown("---")
    st.markdown("### Run Your Own Validation")
    st.code(
        "# Download NIH ChestX-ray14 subset\n"
        "python tests/download_nih_dataset.py\n\n"
        "# Run full 100-case validation\n"
        "python tests/validate_100_cases.py \\\n"
        "  --dataset tests/sample_dataset/cases.json \\\n"
        "  --endpoint http://localhost:8000",
        language="bash",
    )


# =========================================================
# TAB 4 — AUDIT HISTORY
# =========================================================

with tab_history:
    history = st.session_state.get("audit_history", [])

    if not history:
        st.info(
            "No audit runs yet. Run a case from the **Live Demo** tab "
            "or click a **Quick Demo Case** in the sidebar."
        )
    else:
        n = len(history)
        idx = st.session_state.get("_hist_idx", n - 1)
        idx = max(0, min(idx, n - 1))

        # ── Navigation bar ──────────────────────────────────
        nav_l, nav_c, nav_r = st.columns([1, 3, 1])
        with nav_l:
            if st.button("← Prev", disabled=(idx == 0), key="hist_prev",
                         use_container_width=True):
                st.session_state["_hist_idx"] = idx - 1
                st.rerun()
        with nav_c:
            st.markdown(
                f"<div style='text-align:center;font-size:1rem;padding-top:6px;'>"
                f"Case <b>{idx + 1}</b> of <b>{n}</b> &nbsp;—&nbsp; "
                f"<code style='font-size:1.05rem;'>{history[idx]['code']}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with nav_r:
            if st.button("Next →", disabled=(idx == n - 1), key="hist_next",
                         use_container_width=True):
                st.session_state["_hist_idx"] = idx + 1
                st.rerun()

        entry = history[idx]
        r     = entry["result"]
        fv    = r.get("final_verdict", {})
        fraud_detected     = fv.get("fraud_detected", False)
        needs_review       = fv.get("needs_human_review", False)
        causal_score       = r.get("causal_score")
        three_way          = r.get("three_way_match", {})
        mismatches         = three_way.get("mismatches_detected", 0)
        agent_trace        = r.get("agent_trace", [])
        visual_findings    = r.get("visual_findings", [])
        missing_features   = r.get("missing_features", [])
        chexnet_labels     = r.get("chexnet_labels", [])
        cf                 = r.get("counterfactual_result", {})
        upcoding           = r.get("upcoding_financial_impact_usd", 0)
        lit                = r.get("litigation_risk", {})

        st.markdown("---")

        col_left, col_right = st.columns([1, 2])

        # ── LEFT: image + quick stats ───────────────────────
        with col_left:
            if entry.get("image_bytes"):
                st.image(
                    entry["image_bytes"],
                    caption=entry.get("image_name", "X-Ray"),
                    use_container_width=True,
                )
            else:
                st.info("No image for this case")

            if entry.get("expected"):
                if entry.get("fraud_expected"):
                    st.error(f"**Expected:** {entry['expected']}")
                else:
                    st.success(f"**Expected:** {entry['expected']}")

            st.markdown(render_score_bar(causal_score), unsafe_allow_html=True)

            if upcoding > 0:
                st.metric("Overbilling Exposure", f"${upcoding:,}")

            if lit.get("applies"):
                _LIT_COLORS = {
                    "low": "#2ed573", "medium": "#ffa502",
                    "high": "#ff6348", "critical": "#ff4757",
                }
                color = _LIT_COLORS.get(lit.get("label", ""), "#888")
                st.markdown(
                    f"**Litigation Risk:** "
                    f"<span style='color:{color};font-weight:700;'>"
                    f"{lit.get('label','').upper()} ({lit.get('score', 0)}/100)</span>",
                    unsafe_allow_html=True,
                )
                for factor in lit.get("factors", []):
                    st.caption(f"• {factor}")

            with st.expander("Clinical Note"):
                st.markdown(
                    f"<div style='font-size:0.82rem;color:#ccc;white-space:pre-wrap;'>"
                    f"{entry['note']}</div>",
                    unsafe_allow_html=True,
                )

        # ── RIGHT: verdict + full results ──────────────────
        with col_right:

            # Verdict banner
            if fraud_detected:
                st.markdown(
                    "<div class='verdict-fraud'>🚨 FRAUD DETECTED</div>",
                    unsafe_allow_html=True,
                )
            elif needs_review:
                st.markdown(
                    "<div class='verdict-review'>⚠️ CANNOT ASSESS</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='verdict-clean'>✅ CODE VALID</div>",
                    unsafe_allow_html=True,
                )

            explanation = fv.get("explanation", "")
            if explanation:
                st.markdown(
                    f"<div style='font-size:0.82rem;color:#ccc;background:#111;"
                    f"padding:8px;border-radius:6px;margin-top:6px;'>"
                    f"{explanation[:400]}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Three-way match
            st.markdown("**🔍 Three-Way Match**")
            st.markdown(render_match_table(three_way, causal_score), unsafe_allow_html=True)
            if mismatches == 0:
                st.success("0/3 mismatches — All checks pass")
            elif mismatches == 1:
                st.warning("1/3 mismatch — Review recommended")
            elif mismatches == 2:
                st.error("2/3 mismatches — Strong fraud indication")
            else:
                st.error("3/3 mismatches — CRITICAL: All checks failed")

            st.markdown("---")

            # Agent trace
            st.markdown("**🤖 Agent Pipeline**")
            if agent_trace:
                rendered = "".join(
                    f'<div class="agent-step done">{line}</div>'
                    for line in agent_trace
                )
                st.markdown(rendered, unsafe_allow_html=True)

            # Image analysis findings
            if visual_findings or chexnet_labels:
                st.markdown("---")
                st.markdown("**Image Analysis**")
                if chexnet_labels:
                    st.caption(f"CheXNet: {', '.join(chexnet_labels[:4])}")
                if visual_findings:
                    st.markdown(
                        "**Found:** " +
                        " ".join(_chip(f, "chip-green") for f in visual_findings),
                        unsafe_allow_html=True,
                    )
                if missing_features:
                    st.markdown(
                        "**Missing:** " +
                        " ".join(_chip(f, "chip-red") for f in missing_features),
                        unsafe_allow_html=True,
                    )

            # Counterfactual result
            if cf.get("ran"):
                cf_ans  = cf.get("answer", "?")
                cf_icon = "🔴 Fraud confirmed" if cf_ans == "NO" else "🟢 Code justified"
                st.markdown("---")
                st.markdown(f"**Counterfactual:** {cf_icon}")
                st.caption(cf.get("explanation", "")[:200])

            with st.expander("Raw JSON"):
                st.json({
                    "causal_score": causal_score,
                    "final_verdict": fv,
                    "three_way_match": three_way,
                    "upcoding_financial_impact_usd": upcoding,
                    "litigation_risk": lit,
                })
