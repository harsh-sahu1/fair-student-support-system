"""
Model training and inference module.
Implements the validated scikit-learn Logistic Regression pipeline with ColumnTransformer.
"""

from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from backend.app.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    MODEL_PATH,
)
from backend.app.ml.data_processing import extract_features


def build_pipeline(random_state: int = 42) -> Pipeline:
    """
    Builds the ColumnTransformer + LogisticRegression pipeline.
    StandardScaler for numeric, OneHotEncoder for categorical.
    Does NOT use class_weight='balanced' to preserve calibration.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    
    classifier = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        C=1.0,
        solver="lbfgs"
    )
    
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


class StudentSupportModel:
    """
    Wrapper for model lifecycle, predictions, and calibrated scoring.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline: Pipeline = build_pipeline(random_state=random_state)
        self.is_fitted: bool = False
        
    def fit(self, train_df: pd.DataFrame) -> "StudentSupportModel":
        """
        Fits the preprocessor and classifier strictly on the training split.
        """
        X_train = extract_features(train_df)
        y_train = train_df[TARGET_COLUMN].values
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        return self
        
    def predict_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns predicted probabilities P(support_needed = 1 | X).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load().")
        X = extract_features(df)
        probs = self.pipeline.predict_proba(X)[:, 1]
        return probs
        
    def predict_need_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Converts probability into an educator-friendly 0-100 Support Need Score.
        """
        probs = self.predict_probabilities(df)
        return np.round(probs * 100.0, 1)
        
    def attach_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of the dataframe with probability and need_score attached.
        """
        out = df.copy()
        probs = self.predict_probabilities(df)
        out["probability"] = np.round(probs, 4)
        out["need_score"] = np.round(probs * 100.0, 1)
        return out

    def get_transformed_feature_names(self) -> List[str]:
        """
        Retrieves feature names output by the ColumnTransformer.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        preprocessor = self.pipeline.named_steps["preprocessor"]
        return list(preprocessor.get_feature_names_out())

    def get_classifier_coefficients(self) -> np.ndarray:
        """
        Returns the logistic regression weights.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        clf = self.pipeline.named_steps["classifier"]
        return clf.coef_[0]

    def save(self, path: Optional[str] = None) -> None:
        """
        Persists the fitted model pipeline.
        """
        target_path = path or str(MODEL_PATH)
        joblib.dump(self.pipeline, target_path)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "StudentSupportModel":
        """
        Loads a persisted model pipeline.
        """
        target_path = path or str(MODEL_PATH)
        loaded_pipeline = joblib.load(target_path)
        instance = cls()
        instance.pipeline = loaded_pipeline
        instance.is_fitted = True
        return instance
