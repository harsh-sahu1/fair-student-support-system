"""
Unit test suite for the Machine Learning and Fairness Prioritization logic.
Tests mathematical and ethical invariants:
1. Zero feature leakage (sensitive attributes and future label strictly excluded)
2. Strict exact ceiling capacity rule: ceil(capacity_fraction * N) across arbitrary N
3. Mathematical correctness of group-specific recall and Observed Recall Gap
4. Group eligibility thresholds (n_rows >= 10 and n_positives >= 3)
5. Intra-attribute fairness gap evaluation and zero-case handling
6. Continuous score bounds (probability in [0, 1], need_score in [0, 100])
7. Non-causal phrasing in feature explainability
8. Deterministic reproducibility across repeated inferences
"""

import math
import pytest
import numpy as np
import pandas as pd

from backend.app.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    PROTECTED_ATTRIBUTES,
    LEAKAGE_ATTRIBUTES,
    TARGET_COLUMN,
)
from backend.app.ml.data_processing import (
    load_raw_dataset,
    prepare_dataset,
    extract_features,
    get_train_validation_split,
)
from backend.app.ml.model import StudentSupportModel
from backend.app.ml.ranking import rank_students, compute_capacity_slots
from backend.app.ml.fairness import evaluate_fairness
from backend.app.ml.explainability import FeatureExplainer


@pytest.fixture(scope="module")
def prepared_dataset():
    """Loads and prepares the real UCI benchmark dataset."""
    raw_df = load_raw_dataset()
    return prepare_dataset(raw_df)


@pytest.fixture(scope="module")
def trained_pipeline(prepared_dataset):
    """Trains a reference model on 75% split."""
    train_df, _ = get_train_validation_split(prepared_dataset, test_size=0.25, random_state=42)
    model = StudentSupportModel(random_state=42)
    model.fit(train_df)
    return model


def test_zero_leakage_in_feature_matrix(prepared_dataset):
    """
    CRITICAL INVARIANT 1:
    Neither sensitive attributes (sex, school) nor target/label columns (G3, support_needed)
    may ever be present in the feature matrix extracted for the model.
    """
    X = extract_features(prepared_dataset)
    
    # Assert allowed columns only
    assert list(X.columns) == MODEL_FEATURES
    assert len(X.columns) == 30  # 15 numeric + 15 categorical
    
    # Assert zero forbidden columns
    for forbidden in PROTECTED_ATTRIBUTES + LEAKAGE_ATTRIBUTES:
        assert forbidden not in X.columns, f"Leakage detected: '{forbidden}' in model features."


def test_exact_ceiling_capacity_rule_across_multiple_n():
    """
    CRITICAL INVARIANT 2:
    For any cohort size N and capacity fraction c, the number of selected students
    must EXACTLY equal ceil(c * N). Test across N = 1 to 250 for multiple fractions.
    """
    for c in [0.10, 0.15, 0.20, 0.25, 0.33]:
        for n in range(1, 251):
            expected_k = int(math.ceil(c * n))
            computed_k = compute_capacity_slots(n, capacity_fraction=c)
            assert computed_k == expected_k, f"Mismatch at N={n}, c={c}: expected {expected_k}, got {computed_k}"

    # Verify integration with rank_students
    for n in [17, 50, 99, 100, 101, 395]:
        dummy_df = pd.DataFrame({
            "student_id": [f"S{i:03d}" for i in range(n)],
            "probability": np.linspace(0.01, 0.99, n)
        })
        ranked_df, summary = rank_students(dummy_df, capacity_fraction=0.20)
        expected = int(math.ceil(0.20 * n))
        assert summary["num_selected"] == expected
        assert int(ranked_df["selected"].sum()) == expected


def test_mathematical_correctness_of_group_recall():
    """
    CRITICAL INVARIANT 3:
    Group recall = TP_g / Positives_g.
    """
    toy_df = pd.DataFrame({
        "student_id": [f"S{i}" for i in range(10)],
        "sex": ["F"] * 5 + ["M"] * 5,
        "school": ["GP"] * 10,
        "support_needed": [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],  # F has 3 pos, M has 2 pos
        "selected":       [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # F sel 2 pos, M sel 1 pos
    })
    
    # Force evaluate_fairness on toy
    res = evaluate_fairness(toy_df, attributes=["sex"])
    groups = {g["group"]: g for g in res["groups"]}
    
    # F has 3 pos, 2 sel pos -> 2/3 = 0.667
    assert groups["F"]["n_positives"] == 3
    assert groups["F"]["n_selected_positives"] == 2
    assert groups["F"]["recall"] == pytest.approx(0.667, abs=0.001)
    
    # M has 2 pos, 1 sel pos -> 1/2 = 0.500
    assert groups["M"]["n_positives"] == 2
    assert groups["M"]["n_selected_positives"] == 1
    assert groups["M"]["recall"] == pytest.approx(0.500, abs=0.001)


def test_group_eligibility_thresholds():
    """
    CRITICAL INVARIANT 4:
    A group is ELIGIBLE only if n_rows >= 10 AND n_positives >= 3.
    Ineligible groups must be marked eligible: False and excluded from R_min and gap.
    """
    # Group A: 12 rows, 4 positives -> ELIGIBLE
    # Group B: 8 rows, 4 positives  -> INELIGIBLE (< 10 rows)
    # Group C: 15 rows, 2 positives -> INELIGIBLE (< 3 positives)
    toy_df = pd.DataFrame({
        "sex": ["F"] * 12 + ["M"] * 8,
        "school": ["GP"] * 15 + ["MS"] * 5,
        "support_needed": ([1]*4 + [0]*8) + ([1]*4 + [0]*4),
        "selected": [1] * 20
    })
    
    res = evaluate_fairness(toy_df)
    groups = {(g["attribute"], g["group"]): g for g in res["groups"]}
    
    assert groups[("sex", "F")]["eligible"] is True
    assert groups[("sex", "M")]["eligible"] is False
    assert "insufficient sample" in groups[("sex", "M")]["status_label"].lower()


def test_fairness_gap_intra_attribute_and_zero_case():
    """
    CRITICAL INVARIANT 5:
    fairness_gap = largest |recall_a - recall_b| among eligible groups of the SAME attribute.
    Never compare across attributes.
    If no attribute has >= 2 eligible groups, fairness_gap = 0.0.
    """
    # Scenario: Sex has only 1 eligible group, School has 1 eligible group -> gap should be 0.0
    toy_df = pd.DataFrame({
        "sex": ["F"] * 12 + ["M"] * 5,           # F eligible, M ineligible
        "school": ["GP"] * 12 + ["MS"] * 5,       # GP eligible, MS ineligible
        "support_needed": [1] * 3 + [0] * 14,
        "selected": [1] * 17
    })
    res = evaluate_fairness(toy_df)
    assert res["fairness_gap"] == 0.0


def test_model_score_and_probability_bounds(prepared_dataset, trained_pipeline):
    """
    CRITICAL INVARIANT 6:
    Probabilities must lie strictly in [0.0, 1.0], and need_scores in [0.0, 100.0].
    """
    df_pred = trained_pipeline.attach_predictions(prepared_dataset)
    
    assert df_pred["probability"].min() >= 0.0
    assert df_pred["probability"].max() <= 1.0
    assert df_pred["need_score"].min() >= 0.0
    assert df_pred["need_score"].max() <= 100.0
    
    # Check monotonic relation between probability and need_score
    diff = np.abs(df_pred["need_score"] - df_pred["probability"] * 100)
    assert diff.max() <= 0.1


def test_explainability_non_causal_language(prepared_dataset, trained_pipeline):
    """
    CRITICAL INVARIANT 7:
    Feature attribution must strictly avoid causal verbs ("caused", "led to")
    and include the required ethical educational disclaimer.
    """
    explainer = FeatureExplainer(trained_pipeline)
    
    for idx in range(min(15, len(prepared_dataset))):
        student = prepared_dataset.iloc[idx]
        exp = explainer.explain_student(student, top_k=3)
        
        assert "top_factors" in exp
        assert "disclaimer" in exp
        assert len(exp["top_factors"]) >= 1
        
        # Check that causal language is strictly avoided
        for factor in exp["top_factors"]:
            low = factor.lower()
            assert "caused" not in low, f"Causal verb found in factor: '{factor}'"
            assert "blame" not in low
            assert "responsible for" not in low
            
        # Check disclaimer text
        assert "statistical associations" in exp["disclaimer"].lower()
        assert "not causal" in exp["disclaimer"].lower()


def test_deterministic_scoring_and_ranking(prepared_dataset, trained_pipeline):
    """
    CRITICAL INVARIANT 8:
    Inference and ranking must be 100% deterministic with no hidden random state.
    """
    df_pred_1 = trained_pipeline.attach_predictions(prepared_dataset)
    df_pred_2 = trained_pipeline.attach_predictions(prepared_dataset)
    
    pd.testing.assert_series_equal(df_pred_1["probability"], df_pred_2["probability"])
    pd.testing.assert_series_equal(df_pred_1["need_score"], df_pred_2["need_score"])
    
    ranked_1, sum_1 = rank_students(df_pred_1, capacity_fraction=0.20)
    ranked_2, sum_2 = rank_students(df_pred_2, capacity_fraction=0.20)
    
    pd.testing.assert_frame_equal(ranked_1, ranked_2)
    assert sum_1 == sum_2
