"""
Explainability module providing coefficient-based feature attribution.
Translates statistical linear contributions into non-causal, plain-language factors for educators.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from backend.app.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
)
from backend.app.ml.model import StudentSupportModel
from backend.app.ml.data_processing import extract_features

EXPLANATION_DISCLAIMER = (
    "These indicators represent statistical associations identified by the model, "
    "not causal determinations. They do not imply that any single factor caused a student's "
    "academic status and should always be contextualized with human educator insight."
)


def _describe_factor(feature_name: str, value: Any) -> str:
    """
    Translates raw feature and observed student value into a clear, non-judgmental,
    non-causal phrase for educators.
    """
    if feature_name == "G2":
        try:
            val = float(value)
            if val < 10:
                return "Lower performance in the second assessment period (G2)"
            elif val < 12:
                return "Moderate performance in the second assessment period (G2)"
            else:
                return f"Second assessment score of {val}/20"
        except (ValueError, TypeError):
            return "Assessment score in the second period"

    if feature_name == "G1":
        try:
            val = float(value)
            if val < 10:
                return "Lower performance in the first assessment period (G1)"
            elif val < 12:
                return "Moderate performance in the first assessment period (G1)"
            else:
                return f"First assessment score of {val}/20"
        except (ValueError, TypeError):
            return "Assessment score in the first period"

    if feature_name == "failures":
        try:
            val = int(value)
            if val > 0:
                return f"A history of {val} past class failure(s)"
            return "No past class failures recorded"
        except (ValueError, TypeError):
            return "Past class performance record"

    if feature_name == "absences":
        try:
            val = int(value)
            if val >= 10:
                return f"Higher recorded school absences ({val} days)"
            elif val >= 5:
                return f"Recorded school absences ({val} days)"
            return f"Low absence record ({val} days)"
        except (ValueError, TypeError):
            return "Recorded school attendance pattern"

    if feature_name == "studytime":
        try:
            val = int(value)
            if val <= 1:
                return "Limited independent study time (<2 hours/week)"
            elif val == 2:
                return "Standard independent study time (2-5 hours/week)"
            return "Higher independent study time (>5 hours/week)"
        except (ValueError, TypeError):
            return "Independent weekly study time"

    if feature_name == "traveltime":
        try:
            val = int(value)
            if val >= 3:
                return "Longer daily commute to school (>30 minutes)"
            return "Local daily commute to school"
        except (ValueError, TypeError):
            return "Commute time to school"

    if feature_name == "schoolsup":
        return "Receiving extra educational school support" if str(value).lower() == "yes" else "No extra school support recorded"

    if feature_name == "famsup":
        return "Family educational support present" if str(value).lower() == "yes" else "Absence of family educational support"

    if feature_name == "paid":
        return "Not enrolled in extra paid math tutoring" if str(value).lower() == "no" else "Enrolled in extra paid tutoring"

    if feature_name == "higher":
        return "Does not plan to take higher education" if str(value).lower() == "no" else "Aspires to pursue higher education"

    if feature_name == "internet":
        return "No home internet access reported" if str(value).lower() == "no" else "Home internet access available"

    if feature_name == "Pstatus":
        return "Parents cohabiting (living together)" if str(value).upper() == "T" else "Parents living apart"

    if feature_name == "Medu":
        try:
            val = int(value)
            labels = {0: "none", 1: "primary", 2: "5th-9th grade", 3: "secondary", 4: "higher education"}
            return f"Mother's education level ({labels.get(val, 'level ' + str(val))})"
        except (ValueError, TypeError):
            return "Mother's education level"

    if feature_name == "Fedu":
        try:
            val = int(value)
            labels = {0: "none", 1: "primary", 2: "5th-9th grade", 3: "secondary", 4: "higher education"}
            return f"Father's education level ({labels.get(val, 'level ' + str(val))})"
        except (ValueError, TypeError):
            return "Father's education level"

    if feature_name == "famrel":
        return "Self-reported family relationships quality"

    if feature_name == "health":
        return "Current self-reported physical health indicator"

    if feature_name == "goout":
        return "Frequency of social activities with friends"

    if feature_name == "Dalc" or feature_name == "Walc":
        return "Reported alcohol consumption level"

    if feature_name == "freetime":
        return "Amount of free time after school"

    return f"Statistical indicator for '{feature_name}'"

TEACHER_FACING_PRIMARY_FEATURES = ["G1", "G2", "failures", "absences", "studytime"]
TEACHER_FACING_SECONDARY_FEATURES = ["schoolsup", "famsup", "activities", "higher"]
TEACHER_FACING_FEATURES = TEACHER_FACING_PRIMARY_FEATURES + TEACHER_FACING_SECONDARY_FEATURES

# Excluded from primary educator-facing explanations (retained in the 30-feature predictive model)
SUPPRESSED_LIFESTYLE_FEATURES = {
    "Dalc", "Walc", "guardian", "Mjob", "Fjob", "Medu", "Fedu", "famrel", "romantic", "age", "Pstatus", "famsize", "traveltime"
}

REVIEW_FOCUS_DISCLAIMER = (
    "Rule-based review focus for educator consideration; not a diagnosis or automated intervention decision."
)


def get_suggested_review_focus(student_row: Any) -> List[Dict[str, str]]:
    """
    Lightweight rule-based review focus indicator for educator consideration.
    Transparent heuristics based solely on existing educational and engagement indicators:
      - 📚 Academic Review
      - 🕐 Attendance Review
      - 👨‍🏫 Teacher Check-in
      - 🤝 Additional Support Review
    """
    focus_list = []
    
    # Extract values safely
    def get_val(key, default=0):
        if isinstance(student_row, dict):
            v = student_row.get(key, default)
        elif hasattr(student_row, "get"):
            v = student_row.get(key, default)
        else:
            v = getattr(student_row, key, default)
        return v

    try:
        g1 = float(get_val("G1", 12))
        g2 = float(get_val("G2", 12))
        failures = int(get_val("failures", 0))
        absences = int(get_val("absences", 0))
        studytime = int(get_val("studytime", 2))
        schoolsup = str(get_val("schoolsup", "no")).strip().lower()
        famsup = str(get_val("famsup", "yes")).strip().lower()
        higher = str(get_val("higher", "yes")).strip().lower()
    except (ValueError, TypeError):
        g1, g2, failures, absences, studytime = 12.0, 12.0, 0, 0, 2
        schoolsup, famsup, higher = "no", "yes", "yes"

    # 1. 📚 Academic Review: IF G1/G2 relatively low OR failures elevated
    if g1 < 10 or g2 < 10 or failures > 0:
        focus_list.append({
            "category": "📚 Academic Review",
            "reason": "Earlier assessment scores below passing threshold (G1/G2 < 10) or past class failures recorded."
        })

    # 2. 🕐 Attendance Review: IF absences are relatively high
    if absences >= 8:
        focus_list.append({
            "category": "🕐 Attendance Review",
            "reason": f"Elevated cumulative absence record ({absences} days) warranting attendance follow-up."
        })

    # 3. 👨‍🏫 Teacher Check-in: IF low academic performance AND low studytime OR multiple educational risk indicators
    if (g1 < 10 or g2 < 10) and studytime <= 1:
        focus_list.append({
            "category": "👨‍🏫 Teacher Check-in",
            "reason": "Lower assessment scores combined with limited self-directed study time (<2 hrs/week)."
        })
    elif failures >= 1 and absences >= 6:
        focus_list.append({
            "category": "👨‍🏫 Teacher Check-in",
            "reason": "Multiple concurrent educational indicators (past failure and attendance patterns) present."
        })

    # 4. 🤝 Additional Support Review: IF multiple educational indicators suggest human follow-up
    if schoolsup == "yes" or (failures >= 1 and famsup == "no") or higher == "no":
        focus_list.append({
            "category": "🤝 Additional Support Review",
            "reason": "Student support flags or educational pathway indicators suggest team consultation."
        })

    if not focus_list:
        focus_list.append({
            "category": "📚 Academic Review",
            "reason": "Routine educational progress check aligned with standard academic monitoring."
        })

    return focus_list


class FeatureExplainer:
    """
    Computes per-student linear feature attributions from the fitted pipeline.
    Prioritizes educator-actionable academic and engagement indicators, while
    suppressing personal, lifestyle, and family variables from primary explanations.
    """

    def __init__(self, model: StudentSupportModel):
        self.model = model
        self.pipeline = model.pipeline
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.coef = self.classifier.coef_[0]
        self.transformed_names = list(self.preprocessor.get_feature_names_out())

    def explain_student(self, student_row: pd.Series, top_k: int = 3) -> Dict[str, Any]:
        """
        Computes the top positive risk-increasing feature factors for a single student.
        Strictly prioritizes teacher-facing educational factors over lifestyle/family factors.
        """
        # Convert single student row to 1-row DataFrame
        df_single = pd.DataFrame([student_row.to_dict()])
        X_single = extract_features(df_single)
        
        # Transform via preprocessor
        X_trans = self.preprocessor.transform(X_single)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
            
        trans_values = X_trans[0]
        
        # Compute contribution = coef_j * value_j
        contributions = self.coef * trans_values
        
        # Group transformed features back to raw features
        raw_feature_contributions: Dict[str, float] = {feat: 0.0 for feat in MODEL_FEATURES}
        for name, cont in zip(self.transformed_names, contributions):
            parts = name.split("__", 1)
            raw_target = parts[1] if len(parts) > 1 else parts[0]
            for base_feat in MODEL_FEATURES:
                if raw_target == base_feat or raw_target.startswith(base_feat + "_"):
                    raw_feature_contributions[base_feat] += float(cont)
                    break
                    
        # Separate factors into teacher-facing educational factors vs other model features
        # Teacher-facing priority: Primary (G1, G2, failures, absences, studytime) then secondary
        primary_educational = []
        secondary_educational = []
        other_factors = []
        
        for feat, cont in raw_feature_contributions.items():
            if cont > 0:
                if feat in TEACHER_FACING_PRIMARY_FEATURES:
                    # Give primary educational indicators a strong priority weighting
                    primary_educational.append((feat, cont))
                elif feat in TEACHER_FACING_SECONDARY_FEATURES:
                    secondary_educational.append((feat, cont))
                elif feat not in SUPPRESSED_LIFESTYLE_FEATURES:
                    other_factors.append((feat, cont))

        primary_educational.sort(key=lambda x: x[1], reverse=True)
        secondary_educational.sort(key=lambda x: x[1], reverse=True)
        other_factors.sort(key=lambda x: x[1], reverse=True)
        
        # Prioritize teacher-facing factors strictly
        selected_factors = primary_educational + secondary_educational + other_factors
        top_factors_raw = selected_factors[:top_k]
        
        top_factor_descriptions = []
        detailed_list = []
        
        for feat, cont in top_factors_raw:
            val = student_row.get(feat, None)
            desc = _describe_factor(feat, val)
            top_factor_descriptions.append(desc)
            detailed_list.append({
                "feature": feat,
                "value": val,
                "contribution": round(cont, 4),
                "description": desc
            })
            
        # Fallback if fewer than top_k positive educational factors
        if not top_factor_descriptions:
            top_factor_descriptions = ["Recent assessment performance and academic history patterns"]
            
        review_focus = get_suggested_review_focus(student_row)
        
        return {
            "top_factors": top_factor_descriptions,
            "factor_details": detailed_list,
            "suggested_review_focus": review_focus,
            "review_focus_disclaimer": REVIEW_FOCUS_DISCLAIMER,
            "disclaimer": EXPLANATION_DISCLAIMER
        }

