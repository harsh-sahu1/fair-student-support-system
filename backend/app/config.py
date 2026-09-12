"""
Application configuration for the Student Support AI backend.
"""

import os
from pathlib import Path
from typing import List

# Directory and artifact paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("DATA_PATH", str(BACKEND_DIR / "data" / "student-mat.csv")))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(BACKEND_DIR / "artifacts")))
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics_summary.json"
VALIDATION_PREDICTIONS_PATH = ARTIFACTS_DIR / "validation_predictions.json"

# Capacity defaults
DEFAULT_CAPACITY_FRACTION: float = float(os.getenv("DEFAULT_CAPACITY_FRACTION", "0.20"))

# CORS settings
CORS_ORIGINS_ENV = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,https://harsh-sahu1.github.io,http://localhost:3000,http://localhost:8000"
)
CORS_ORIGINS: List[str] = [origin.strip() for origin in CORS_ORIGINS_ENV.split(",") if origin.strip()]

# ML Feature schema
NUMERIC_FEATURES: List[str] = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2"
]

CATEGORICAL_FEATURES: List[str] = [
    "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
    "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

MODEL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PROTECTED_ATTRIBUTES: List[str] = ["sex", "school"]

LEAKAGE_ATTRIBUTES: List[str] = ["G3", "support_needed"]

TARGET_COLUMN: str = "support_needed"
