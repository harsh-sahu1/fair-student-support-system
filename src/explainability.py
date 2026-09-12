"""
Explainability module for Fair Student-Support Prioritization.
Provides mathematically grounded, feature-level attribution for individual student need scores.
Enforces non-causal educational language standards.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from src.data_processing import FEATURE_COLS

FEATURE_DESCRIPTIONS = {
    "attendance_rate": {
        "label": "Attendance Rate",
        "unit": "%",
        "low_text": "Low attendance relative to cohort",
        "high_text": "High consistent attendance",
        "direction": "inverse",  # Lower value -> higher risk
    },
    "assignment_completion_rate": {
        "label": "Assignment Completion Rate",
        "unit": "%",
        "low_text": "Low assignment completion rate",
        "high_text": "High assignment completion rate",
        "direction": "inverse",
    },
    "midterm_exam_score": {
        "label": "Midterm Exam Score",
        "unit": "pts",
        "low_text": "Below-average midterm examination score",
        "high_text": "Strong midterm examination performance",
        "direction": "inverse",
    },
    "previous_gpa": {
        "label": "Previous Cumulative CGPA",
        "unit": "/10",
        "low_text": "Low prior cumulative CGPA",
        "high_text": "Strong prior cumulative CGPA",
        "direction": "inverse",
    },
    "lms_engagement_hours": {
        "label": "Weekly LMS Engagement",
        "unit": "hrs/wk",
        "low_text": "Limited weekly digital platform engagement",
        "high_text": "Substantial weekly digital platform engagement",
        "direction": "inverse",
    },
    "late_submissions_count": {
        "label": "Late Submissions Count",
        "unit": "tasks",
        "low_text": "Minimal or zero late submissions",
        "high_text": "Elevated number of late assignment submissions",
        "direction": "direct",  # Higher value -> higher risk
    },
    "study_hours_weekly": {
        "label": "Weekly Study Hours",
        "unit": "hrs/wk",
        "low_text": "Low reported independent study hours",
        "high_text": "High reported independent study hours",
        "direction": "inverse",
    },
}


class StudentExplainer:
    """
    Computes transparent individual attributions based on standardized feature contributions
    to the linear predictor logit(p) = beta_0 + sum(beta_i * (x_i - mean_i) / std_i).
    """
    
    def __init__(self, model, reference_df: pd.DataFrame):
        self.model = model
        self.feature_names = FEATURE_COLS
        
        # Calculate cohort means and standard deviations from reference data
        self.means = reference_df[self.feature_names].mean().to_dict()
        self.stds = reference_df[self.feature_names].std().replace(0, 1.0).to_dict()
        self.coefs = model.get_feature_coefficients()
        
    def explain_student(self, student_series: pd.Series) -> Dict[str, Any]:
        """
        Generates individual factor attributions for a given student record.
        
        Strict Non-Causal Framing Rule:
        Returns clear statements regarding "factors associated with the model's prediction"
        and explicitly avoids causal claims such as "low attendance caused failure."
        """
        contributions = []
        
        for feat in self.feature_names:
            val = float(student_series[feat])
            mean_val = self.means[feat]
            std_val = self.stds[feat]
            z_score = (val - mean_val) / std_val
            
            weight = self.coefs.get(feat, 0.0)
            
            # Linear log-odds contribution: positive contribution increases predicted risk
            # For LogisticRegression, logit = intercept + sum(weight * z_score)
            risk_contrib = weight * z_score
            
            meta = FEATURE_DESCRIPTIONS.get(feat, {
                "label": feat,
                "unit": "",
                "low_text": f"Low {feat}",
                "high_text": f"High {feat}",
                "direction": "direct",
            })
            
            # Plain language descriptive factor summary
            diff_from_mean = val - mean_val
            if diff_from_mean < 0:
                qualifier = f"{abs(diff_from_mean):.1f} {meta['unit']} below cohort mean ({mean_val:.1f})"
                narrative = meta["low_text"]
            else:
                qualifier = f"{diff_from_mean:.1f} {meta['unit']} above cohort mean ({mean_val:.1f})"
                narrative = meta["high_text"]
                
            contributions.append({
                "feature": feat,
                "label": meta["label"],
                "value": val,
                "cohort_mean": round(mean_val, 1),
                "unit": meta["unit"],
                "risk_contribution": risk_contrib,
                "qualifier": qualifier,
                "narrative": narrative,
                "is_risk_factor": (risk_contrib > 0.1),
                "is_protective_factor": (risk_contrib < -0.1),
            })
            
        # Sort factors by strongest positive risk contributor to predicted need
        contributions.sort(key=lambda x: x["risk_contribution"], reverse=True)
        
        top_risk_factors = [c for c in contributions if c["is_risk_factor"]][:4]
        protective_factors = [c for c in reversed(contributions) if c["is_protective_factor"]][:2]
        
        # Build strict non-causal human-readable bullet points
        risk_bullets = [
            f"{c['narrative']} ({c['value']} {c['unit']}, {c['qualifier']})"
            for c in top_risk_factors
        ]
        
        return {
            "student_id": student_series.get("student_id", "Unknown"),
            "student_group": student_series.get("student_group", "Unknown"),
            "need_score": student_series.get("need_score", None),
            "top_contributing_factors": risk_bullets,
            "detailed_contributions": contributions,
            "protective_factors": [f"{c['narrative']} ({c['value']} {c['unit']})" for c in protective_factors],
            "language_disclaimer": "Factors listed represent statistical associations with the model's prioritization score and do not imply direct personal causation.",
        }
