"""
Data processing module for UCI Student Performance dataset.
Enforces strict column allowlists and prevents leakage of target and sensitive features.
"""

from typing import Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.config import (
    DATA_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    PROTECTED_ATTRIBUTES,
    LEAKAGE_ATTRIBUTES,
    TARGET_COLUMN,
)


def load_raw_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the UCI Student Performance (mathematics) dataset from CSV.
    The file is semicolon-delimited.
    """
    path = file_path or str(DATA_PATH)
    df = pd.read_csv(path, sep=";")
    return df


def prepare_dataset(df: pd.DataFrame, id_prefix: str = "VAL") -> pd.DataFrame:
    """
    Prepares dataset with target variable and synthetic IDs for tracking.
    Target definition: support_needed = 1 if G3 < 10 else 0.
    """
    prepared = df.copy()
    if "G3" in prepared.columns:
        prepared[TARGET_COLUMN] = (prepared["G3"] < 10).astype(int)
    
    if "student_id" not in prepared.columns:
        prepared["student_id"] = [f"{id_prefix}{i:04d}" for i in range(len(prepared))]
        
    return prepared


def get_train_validation_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates reproducible train/validation split.
    Validation set represents the held-out benchmark for demo and audit metrics.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    return train_df.copy(), val_df.copy()


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts strictly allowed model features.
    Guarantees that sensitive attributes (sex, school) and target/leakage columns
    (G3, support_needed) are NEVER fed to the model pipeline.
    """
    # Defensive check
    for col in MODEL_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required model feature: '{col}'")
            
    # Explicit allowlist extraction
    X = df[MODEL_FEATURES].copy()
    
    # Assert zero leakage
    for forbidden in PROTECTED_ATTRIBUTES + LEAKAGE_ATTRIBUTES:
        assert forbidden not in X.columns, f"Leakage violation! '{forbidden}' found in feature matrix."
        
    return X
