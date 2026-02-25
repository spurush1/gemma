# ICD-10 MedGemma — Causal Audit System

**96.7% accuracy on NIH ChestX-ray14** · Symbolic-Neural Hybrid · Kaggle MedGemma Impact Challenge

Detects ICD-10 billing fraud (upcoding/downcoding) by combining:
- **MedGemma 4B-IT** (Google DeepMind) — clinical NLP + radiology image understanding
- **CheXNet DenseNet-121** — NIH-trained chest X-ray pathology classifier
- **LangGraph 5-node workflow** — agentic causal audit pipeline
- **Medical Knowledge Graph** — Neo4j + CausalRequirement rules (39 codes)
- **Three-Way Match** — Voice ↔ Image ↔ Billing Code cross-validation

---

## Quick Start (Docker — Recommended)

```bash
# 1. Configure API keys
cp .env.example .env
# Edit .env — add MEDGEMMA_API_KEY and MEDGEMMA_ENDPOINT
# (Leave blank to run in Mock Mode for development)

# 2. Build and start all services
chmod +x docker-run.sh
./docker-run.sh

# 3. Stop all services
./docker-stop.sh
```

### Services

| Service | URL | Notes |
|---------|-----|-------|
| Trust Dashboard (Streamlit) | http://localhost:9501 | Main UI — causal audit + history |
| Backend API | http://localhost:8000 | FastAPI |
| API Docs (Swagger) | http://localhost:8000/docs | Interactive API explorer |
| HTML Frontend | http://localhost:80 | Lightweight single-page UI |
| Neo4j Browser | http://localhost:7476 | Graph DB — `neo4j` / `medgemma123` |

---

## Architecture

```
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

[Node 5: CounterfactualVerifier]  (only if fraud flagged)
  MedGemma: "If patient had {alt_code}, is {code} still needed?"

VERDICT: FRAUD / CLEAN + Causal Score + Financial Impact + Litigation Risk
```

---

## Project Structure

```
icd10-medgemma/
├── backend/
│   ├── app.py                  # FastAPI — /causal-audit, /analyze, /compare, /health
│   ├── agents.py               # LangGraph 5-node StateGraph workflow
│   ├── medgemma_client.py      # MedGemma 4B-IT API (featherless-ai HuggingFace router)
│   ├── chexnet_client.py       # CheXNet DenseNet-121 (torchxrayvision, NIH weights)
│   ├── audio_client.py         # Voice ASR (google/medasr)
│   ├── context_graph.py        # ICD-10 context graph (Neo4j primary / NetworkX fallback)
│   ├── graph_db.py             # Neo4j Cypher queries + graph seeding
│   ├── icd10_data.py           # CMS ICD-10-CM data + CLINICAL_AUGMENTATION (39 codes)
│   ├── causal_rules.yaml       # CausalRequirement rules per code
│   ├── knowledge_engine.py     # Shim → backend/knowledge/ package
│   ├── knowledge/
│   │   ├── models.py           # CausalRequirement dataclass
│   │   ├── evaluator.py        # Causal score evaluation logic
│   │   ├── litigation.py       # Litigation risk scoring
│   │   └── graph.py            # MedicalKnowledgeGraph
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py        # Trust Dashboard (port 9501)
│   └── index.html              # Lightweight HTML UI (port 80)
├── tests/
│   ├── sample_dataset/
│   │   ├── nih_images/         # NIH ChestX-ray14 subset
│   │   ├── nih_synthetic_notes.json
│   │   └── generate_nih_notes.py
│   ├── validate_nih_synthetic.py
│   └── validate_100_cases.py
├── Dockerfile                  # Backend + Streamlit image
├── Dockerfile.frontend         # Nginx HTML frontend image
├── docker-compose.yml          # All 4 services
├── docker-run.sh               # One-command startup
├── docker-stop.sh              # One-command shutdown
└── .env.example                # Environment template
```

---

## Trust Dashboard Features

- **Live Demo tab** — 5 preset NIH cases with paired X-rays; click to auto-run full audit
- **How It Works tab** — architecture diagram + tech stack comparison
- **Validation Results tab** — accuracy progression (65.6% → 96.7%), root cause analysis
- **Audit History tab** — browse all runs with image, full agent trace, three-way match, and verdict; navigate with Prev / Next

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/causal-audit` | 5-node LangGraph causal audit (main endpoint) |
| POST | `/analyze` | Legacy multimodal analysis |
| GET | `/validate/{code}` | Code details from knowledge graph |
| POST | `/compare` | Compare two ICD-10 codes |
| GET | `/codes/search?q=` | Search codes by keyword |
| GET | `/health` | Health check — reports graph backend + mock mode |

### Example causal audit (curl)

```bash
curl -X POST http://localhost:8000/causal-audit \
  -F "clinical_note=71M with fever, productive cough, right lower lobe consolidation" \
  -F "existing_code=J93.1" \
  -F "image=@chest_xray.jpg"
```

---

## Local Development (Without Docker)

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Configure environment
cp .env.example .env

# 3. Start Neo4j (optional — falls back to NetworkX if unavailable)
docker run -d -p 7476:7474 -p 7689:7687 \
  -e NEO4J_AUTH=neo4j/medgemma123 \
  neo4j:5.18-community

# 4. Start backend
cd backend && uvicorn app:app --port 8000 --reload

# 5. Start Trust Dashboard
streamlit run frontend/streamlit_app.py --server.port 9501

# 6. Run NIH validation
python tests/validate_nih_synthetic.py --limit 20
```

---

## Validation Results

| System Version | Accuracy | False Positives |
|---|---|---|
| Baseline (MedGemma text-only) | 65.6% | 21 |
| + Tier 1: RadLex Normalization | 80.3% | 12 |
| + Tier 2: CheXNet Ensemble | **96.7%** | **2** |

Validated on 61 legitimate NIH ChestX-ray14 cases (synthetic clinical notes).

---

## Mock Mode

Leave `MEDGEMMA_API_KEY` and `MEDGEMMA_ENDPOINT` unset in `.env` to run in Mock Mode.
The system uses rule-based stubs — no API calls, no cost. All UI features work.
A warning banner is shown in the Trust Dashboard when mock mode is active.

---

## Key Design Decisions

**Two-Pass Multimodal Fix** — MedGemma ignores images when clinical text is present (text bias). Fix: Pass 1 sends image only → locks findings. Pass 2 sends text only → extracts symptoms. Confirmed: pneumothorax X-ray + misleading note → Pass 1 correctly finds `right_sided_pneumothorax`.

**Symbolic-Neural Hybrid** — Neural models (MedGemma, CheXNet) handle perception. Symbolic rules (`CausalRequirement` per code) handle reasoning. No LLM can hallucinate a passing causal score.

**Neo4j Knowledge Graph** — Seeded from CMS FY2024 ICD-10-CM data. Used for candidate code traversal, excludes1 validation, hierarchy queries, and financial gap calculation.
