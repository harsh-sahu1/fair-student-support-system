# Fair Student-Support Prioritization System

A full-stack, production-grade machine learning decision-support platform designed to help educators allocate limited academic support capacity (strictly 20%, `ceil(0.20 * N)`) while proactively monitoring and mitigating group recall disparities across eligible demographic student cohorts.

---

## 🚀 Quickstart: Running Both Apps

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Node.js 18+ (tested on Node.js 20 LTS)

### Terminal 1: Python FastAPI Backend
```bash
# From workspace root
cd backend
pip install -r requirements.txt

# Run offline training (if artifacts are not already present)
python scripts/train.py

# Start API server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

### Terminal 2: React + Vite Frontend
```bash
# From workspace root
cd frontend
npm install
npm run dev
```
- Frontend Web App: `http://localhost:5173`

---

## 📁 Project Structure

```
student-support-ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, CORS, lifespan startup model load, exception handling
│   │   ├── config.py            # Feature schema, file paths, capacity defaults, CORS origins
│   │   ├── schemas.py           # Strict Pydantic v2 validation models (rejects G3/support_needed)
│   │   ├── api/
│   │   │   └── routes.py        # All endpoints: /health, /overview, /students, /fairness, /predict
│   │   └── ml/
│   │       ├── data_processing.py # UCI dataset preparation, zero-leakage feature extraction
│   │       ├── model.py         # Logistic Regression pipeline (StandardScaler + OneHotEncoder)
│   │       ├── ranking.py       # Deterministic exact ceiling prioritization: ceil(0.20 * N)
│   │       ├── fairness.py      # Eligibility filtering, R_min, and intra-attribute Observed Recall Gap
│   │       ├── explainability.py# Linear feature attribution with non-causal language mapping
│   │       └── evaluation.py    # Calibration Brier score and 10-seed cross-validation audit
│   ├── data/
│   │   └── student-mat.csv      # UCI Student Performance (mathematics) dataset (395 rows)
│   ├── artifacts/
│   │   ├── model.joblib         # Persisted scikit-learn pipeline
│   │   ├── validation_predictions.json # 99 validation cohort records with ranks & explanations
│   │   └── metrics_summary.json # Comprehensive evaluation, fairness, and 10-seed robustness metrics
│   ├── scripts/
│   │   └── train.py             # Offline training runner producing all artifacts
│   ├── tests/
│   │   ├── test_ml_logic.py     # 8 core ML/fairness invariant tests (leakage, ceil, eligibility, etc.)
│   │   └── test_api.py          # FastAPI endpoint integration tests (TestClient, 422 checks)
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── client.ts        # Typed fetch client with base URL configuration
│   │   ├── components/
│   │   │   └── StudentDrawer.tsx# Profile drawer showing raw features, weights, and disclaimer
│   │   ├── pages/
│   │   │   ├── PrioritizedList.tsx # Primary operational roster with search, filter, and ranks
│   │   │   ├── Overview.tsx     # High-level cohort KPIs and capacity indicators
│   │   │   ├── FairnessView.tsx # Group recall comparison chart with Observed Recall Gap bracket
│   │   │   ├── ModelPerformance.tsx # Calibration Brier score and 10-seed cross-validation table
│   │   │   └── PredictSandbox.tsx # Interactive form testing POST /predict and leakage rejection
│   │   ├── App.tsx              # Persistent left navigation rail & connection monitoring
│   │   ├── main.tsx
│   │   └── index.css            # Calm, distraction-free design system tokens
│   ├── .env.example
│   ├── package.json
│   └── README.md
├── RESPONSIBLE_USE.md           # Comprehensive ethical governance & rejected post-hoc experiment details
└── README.md
```

---

## 🧪 Automated Testing

Run full backend test suite covering ML logic invariants and API routes:
```bash
python -m pytest backend/tests -v
```
All 19 tests assert:
1. **Zero Feature Leakage**: `G3`, `support_needed`, `sex`, and `school` are strictly barred from the model.
2. **Exact Capacity Rule**: Selected count strictly equals `ceil(capacity_fraction * N)` across arbitrary N.
3. **Fairness Eligibility**: Groups require rows >= 10 and positives >= 3.
4. **Intra-Attribute Gap**: Observed Recall Gap evaluated solely within the same attribute.
5. **Score Bounds**: Continuous probabilities in [0, 1] and need scores in [0, 100].
6. **Non-Causal Language**: Factor explanations strictly avoid causal verbs.
7. **Strict 422 on Leakage**: Submitting `G3` or `support_needed` to `/predict` returns `422 Unprocessable Entity`.

---

## 📡 API Endpoints & Curl Examples

### 1. Health Status
```bash
curl -X GET http://127.0.0.1:8000/health
```

### 2. Fairness Audit & Observed Recall Gap
```bash
curl -X GET http://127.0.0.1:8000/fairness
```

### 3. Score Student Need (POST /predict)
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"student_id\": \"STU_001\", \"age\": 16, \"Medu\": 2, \"Fedu\": 2, \"traveltime\": 1, \"studytime\": 1, \"failures\": 2, \"famrel\": 4, \"freetime\": 3, \"goout\": 2, \"Dalc\": 1, \"Walc\": 2, \"health\": 3, \"absences\": 8, \"G1\": 7, \"G2\": 6, \"address\": \"U\", \"famsize\": \"GT3\", \"Pstatus\": \"T\", \"Mjob\": \"other\", \"Fjob\": \"other\", \"reason\": \"course\", \"guardian\": \"mother\", \"schoolsup\": \"no\", \"famsup\": \"yes\", \"paid\": \"no\", \"activities\": \"no\", \"nursery\": \"yes\", \"higher\": \"yes\", \"internet\": \"yes\", \"romantic\": \"no\"}"
```

### 4. Zero-Leakage Prevention Verification
Submitting `G3` or `support_needed` is rejected immediately:
```bash
curl -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"G3\": 7, \"age\": 16, \"address\": \"U\", ...}"
```
Returns: `HTTP 422 Unprocessable Entity` with message explaining the leakage violation.

---

## ⚖️ Ethical Governance & Limitations

Read [RESPONSIBLE_USE.md](file:///c:/Users/sahuh/OneDrive/Pictures/Desktop/FAIR%20STUDENT%20SYSTEM/RESPONSIBLE_USE.md) for complete details on:
1. **Decision Support vs. Automated Decision Making**: Why human counselors must review all allocations.
2. **Fairness-Through-Unawareness Limits**: Why demographic blindness prevents disparate treatment but does not eliminate all proxy bias.
3. **The Rejected Post-Hoc Fairness Experiment**: Full scientific documentation of why artificial log-odds shifting was rejected (failure to generalize out-of-sample and degradation of Brier calibration).
