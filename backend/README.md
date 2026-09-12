# Student Support AI — FastAPI Backend

Rock-solid, validated, and tested Python API serving the fairness-aware student prioritization model.

## Architecture
- **Framework**: FastAPI + Pydantic v2
- **ML Pipeline**: scikit-learn Logistic Regression inside a `ColumnTransformer` (StandardScaler for 15 numeric features, OneHotEncoder for 15 categorical features)
- **Fairness Rules**: Exact ceiling allocation `ceil(0.20 * N)`, group eligibility filter (n >= 10, positives >= 3), intra-attribute Observed Recall Gap
- **Explainability**: Linear coefficient-based attribution with non-causal language safeguards

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Training (Offline Artifact Generation)
```bash
python scripts/train.py
```
This trains the pipeline on `data/student-mat.csv`, runs fairness audits, executes 10-seed robustness cross-validation, and writes `artifacts/model.joblib` and `artifacts/metrics_summary.json`.

### 3. Run Backend Server
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Run Test Suite
```bash
pytest tests/ -v
```
