"""
Model training and inference module for Fair Student-Support Prioritization.
Provides defensible baseline models (Logistic Regression, Random Forest),
computes calibrated support-need scores (0-100), and outputs model artifacts.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import FEATURE_COLS, TARGET_COL, ID_COL, SENSITIVE_COL


class StudentSupportModel:
    """
    Predictive model for identifying students who may need additional academic support.
    
    IMPORTANT ETHICAL & ARCHITECTURAL NOTE:
    - Sensitive attributes (e.g. `student_group`) are strictly EXCLUDED from the feature matrix
      to prevent direct disparate treatment or proxy bias during model inference.
    - Group attributes are solely used downstream for fairness auditing and constrained re-ranking.
    """
    
    def __init__(self, model_type: str = "logistic_regression", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.feature_names = FEATURE_COLS
        self.pipeline = self._build_pipeline()
        self.is_fitted = False
        
    def _build_pipeline(self) -> Pipeline:
        if self.model_type == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=5,
                random_state=self.random_state,
                class_weight="balanced"
            )
        else:
            # Default: Logistic Regression - clean, calibrated, and highly explainable
            clf = LogisticRegression(
                C=1.0,
                random_state=self.random_state,
                class_weight="balanced",
                max_iter=1000
            )
            
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        
    def fit(self, train_df: pd.DataFrame) -> "StudentSupportModel":
        """
        Fits the model on the training dataframe using only valid educational features.
        """
        X_train = train_df[self.feature_names]
        y_train = train_df[TARGET_COL]
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        return self
        
    def predict_need_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """
        Outputs continuous probability of support need P(Y=1 | X).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating predictions.")
        X = df[self.feature_names]
        return self.pipeline.predict_proba(X)[:, 1]
        
    def predict_need_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Converts predicted probabilities into an educator-friendly 0-100 Support Need Score.
        Higher score = greater estimated need for intervention.
        """
        probs = self.predict_need_probabilities(df)
        scores = np.round(probs * 100, 1)
        return scores
        
    def attach_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attaches need_score and need_probability to a copy of the dataframe.
        """
        out_df = df.copy()
        probs = self.predict_need_probabilities(df)
        out_df["need_probability"] = np.round(probs, 4)
        out_df["need_score"] = np.round(probs * 100, 1)
        return out_df
        
    def get_feature_coefficients(self) -> Dict[str, float]:
        """
        Returns model coefficients or feature importances for explainability.
        For Logistic Regression, returns linear weights on scaled features.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
            
        clf = self.pipeline.named_steps["classifier"]
        if hasattr(clf, "coef_"):
            weights = clf.coef_[0]
            return dict(zip(self.feature_names, weights))
        elif hasattr(clf, "feature_importances_"):
            weights = clf.feature_importances_
            return dict(zip(self.feature_names, weights))
        return {}


def train_baseline_model(train_df: pd.DataFrame, model_type: str = "logistic_regression") -> StudentSupportModel:
    """
    Convenience function to instantiate and train a StudentSupportModel.
    """
    model = StudentSupportModel(model_type=model_type, random_state=42)
    model.fit(train_df)
    return model
