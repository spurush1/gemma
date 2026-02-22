# ICD-10 MedGemma Multimodal Coding Validation System

A multimodal ICD-10 coding validation system that detects billing fraud (upcoding/downcoding) by combining clinical notes with medical image analysis via the MedGemma API and a NetworkX-based knowledge graph.

## How It Works

1. **Clinical note + X-ray image** submitted to system
2. **MedGemma** extracts clinical entities (symptoms, image findings)
3. **Context Graph** (NetworkX) traverses ICD-10 knowledge to find candidate codes
4. **MedGemma** selects the best code from the graph-constrained candidate list
5. **Fraud detection** compares submitted vs suggested code and flags mismatches

## Project Structure

```
icd10-medgemma/
├── backend/
│   ├── app.py                  # FastAPI main app
│   ├── medgemma_client.py      # MedGemma API calls
│   ├── context_graph.py        # ICD-10 knowledge graph
│   ├── icd10_data.py           # ICD-10 codes + rules
│   └── requirements.txt
├── frontend/
│   └── index.html              # Single page UI
├── data/                       # Test fixtures and datasets
├── tests/
│   └── test_cases.py           # 5 validation test cases
├── logs/                       # API call logs
├── .env.example                # Environment template
└── docker-compose.yml          # Docker deployment
```

## Setup (Without Docker)

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env and add your MedGemma API key and endpoint

# 3. Start backend
cd backend && uvicorn app:app --reload

# 4. Open frontend
open frontend/index.html

# 5. Run tests (requires backend running)
python tests/test_cases.py

# 6. Print test cases without API
python tests/test_cases.py --print-cases
```

## Run With Docker (Recommended)

```bash
# 1. Configure API keys
cp .env.example .env
# Edit .env and add your MedGemma API key

# 2. Build and start
chmod +x docker-run.sh
./docker-run.sh

# 3. Open browser
open http://localhost

# 4. Stop
./docker-stop.sh
```

## Run Edge Version (Raspberry Pi / Low RAM)

```bash
chmod +x edge-run.sh
./edge-run.sh
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Full multimodal coding analysis |
| GET | `/validate/{code}` | Get code details |
| POST | `/compare` | Compare two codes |
| GET | `/codes/search?q=` | Search codes by keyword |
| GET | `/health` | Health check |

### Example `/analyze` request (curl)

```bash
curl -X POST http://localhost:8000/analyze \
  -F "clinical_note=Patient presents with fever and productive cough" \
  -F "existing_code=J06.9" \
  -F "image=@chest_xray.jpg"
```

## Test Cases

| Case | Scenario | Submitted | Correct | Impact |
|------|----------|-----------|---------|--------|
| CASE_01 | Downcoding | J06.9 (Cold) | J18.1 (Lobar Pneumonia) | $2,220 loss |
| CASE_02 | Upcoding | J45.50 (Severe Asthma) | J45.20 (Mild Asthma) | $2,210 fraud |
| CASE_03 | Correct | I50.0 (CHF) | I50.0 (CHF) | $0 |
| CASE_04 | Unrelated | J18.9 (Pneumonia) | I50.9 (Heart Failure) | $7,500 gap |
| CASE_05 | Text Only | K25.9 (Ulcer, no bleed) | K25.0 (Hemorrhagic Ulcer) | $7,300 loss |

## Key Features

- **Multimodal**: Processes clinical notes + X-ray images together
- **Context Graph**: Prevents hallucinated codes; constrains AI to valid ICD-10 space
- **Fraud Detection**: Identifies upcoding, downcoding, and unrelated codes
- **Financial Impact**: Calculates dollar difference between submitted and correct codes
- **Explainable**: Full clinical reasoning for every suggestion
- **Mock Mode**: Works without API keys for development/testing

## Why MedGemma vs General LLMs

General LLMs (GPT-4, Claude) can read clinical text but **cannot clinically interpret X-rays**. MedGemma is specifically trained on medical images and understands findings like:
- Lobar consolidation vs ground-glass opacity
- Cardiomegaly (cardiothoracic ratio)
- Kerley B lines (heart failure)
- Air bronchograms (pneumonia)

This multimodal capability is what enables detecting mismatches between text and image — the core fraud detection mechanism.

## Data Sources

- **Synthetic**: Generated immediately via `data/generate_synthetic_cases.py`
- **MTSamples**: Free clinical notes from mtsamples.com
- **NIH Chest X-Ray**: 112k labeled chest X-rays (nihcc.app.box.com)
- **CheXpert**: Stanford paired reports + images (registration required)
- **MIMIC-IV**: Real ICU data with ground-truth ICD codes (physionet.org, 3-5 days)
