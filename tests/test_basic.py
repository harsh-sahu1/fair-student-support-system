"""
Basic automated test suite for Fair Student-Support Prioritization ().
Tests core invariants:
1. Strict capacity constraint adherence (selected_students <= floor(N * capacity_pct / 100))
2. Mathematical correctness of group recall and observed recall gap
3. Model training and 0-100 score bounds
4. Explainability attribution and non-causal language guidelines
5. Fairness-aware re-ranking deterministic properties
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd

from src.data_processing import generate_synthetic_dataset, split_dataset, FEATURE_COLS
from src.model import StudentSupportModel
from src.fairness import compute_group_fairness_metrics
from src.ranking import rank_baseline, rank_fairness_aware
from src.explainability import StudentExplainer


@pytest.fixture
def sample_data():
    """Generates reproducible test dataset of 200 students."""
    return generate_synthetic_dataset(n_samples=200, seed=123)


@pytest.fixture
def trained_model(sample_data):
    """Trains a baseline model on sample data."""
    train_df, _ = split_dataset(sample_data, test_size=0.25, seed=123)
    model = StudentSupportModel(model_type="logistic_regression", random_state=123)
    model.fit(train_df)
    return model


def test_capacity_constraint_strictly_respected(sample_data, trained_model):
    """
    CRITICAL REQUIREMENT:
    Selected students must strictly equal floor(N * capacity_pct / 100).
    Default capacity is 20%.
    """
    df_pred = trained_model.attach_predictions(sample_data)
    n_total = len(df_pred)
    
    # Test default 20%
    capacity_pct = 20.0
    expected_k = int(np.floor(n_total * 0.20))
    
    ranked_df, summary = rank_fairness_aware(df_pred, capacity_pct=capacity_pct)
    
    assert summary["baseline_selected_count"] == expected_k, "Baseline selection violated capacity constraint."
    assert summary["fairness_selected_count"] == expected_k, "Fairness selection violated capacity constraint."
    assert int(ranked_df["fairness_selected"].sum()) == expected_k, "Total selected flag does not match K."
    
    # Test configurable capacity (e.g. 15% and 25%)
    for custom_pct in [15.0, 25.0]:
        custom_k = int(np.floor(n_total * (custom_pct / 100.0)))
        _, custom_summary = rank_fairness_aware(df_pred, capacity_pct=custom_pct)
        assert custom_summary["fairness_selected_count"] == custom_k, f"Failed at {custom_pct}% capacity."


def test_fairness_gap_calculation_mathematical_correctness():
    """
    Verifies that group recall = TP_g / TotalNeed_g and
    Observed Recall Gap = max(recall) - min(recall).
    """
    # Create deterministic toy dataframe
    # Group A: 10 students, 4 need support. We select 2 who need support -> recall = 2/4 = 50.0%
    # Group B: 10 students, 5 need support. We select 4 who need support -> recall = 4/5 = 80.0%
    # Expected gap = 80.0 - 50.0 = 30.0%
    toy_df = pd.DataFrame({
        "student_id": [f"T{i:02d}" for i in range(20)],
        "student_group": ["Group A"] * 10 + ["Group B"] * 10,
        "support_needed": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "is_selected":    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    })
    
    metrics = compute_group_fairness_metrics(toy_df, selection_col="is_selected")
    
    table = metrics["table"].set_index("group")
    assert table.loc["Group A", "recall_pct"] == 50.0
    assert table.loc["Group B", "recall_pct"] == 80.0
    assert metrics["observed_recall_gap"] == 30.0
    assert metrics["max_group"] == "Group B"
    assert metrics["min_group"] == "Group A"


def test_model_predictions_and_score_range(sample_data, trained_model):
    """
    Verifies model produces calibrated continuous probability and 0-100 need scores.
    """
    df_pred = trained_model.attach_predictions(sample_data)
    
    assert "need_score" in df_pred.columns
    assert "need_probability" in df_pred.columns
    
    assert df_pred["need_score"].min() >= 0.0
    assert df_pred["need_score"].max() <= 100.0
    assert df_pred["need_probability"].min() >= 0.0
    assert df_pred["need_probability"].max() <= 1.0


def test_explainability_attribution_and_non_causal_language(sample_data, trained_model):
    """
    Verifies explainability factors are computed and enforce strict non-causal phrasing.
    """
    df_pred = trained_model.attach_predictions(sample_data)
    explainer = StudentExplainer(trained_model, sample_data)
    
    student = df_pred.iloc[0]
    explanation = explainer.explain_student(student)
    
    assert "top_contributing_factors" in explanation
    assert "detailed_contributions" in explanation
    assert len(explanation["detailed_contributions"]) == len(FEATURE_COLS)
    
    # Check language disclaimer presence
    disclaimer = explanation["language_disclaimer"]
    assert "statistical associations" in disclaimer.lower()
    assert "causation" in disclaimer.lower()
    
    # Ensure no causal verbs like "caused" in factors
    for factor in explanation["top_contributing_factors"]:
        assert "caused" not in factor.lower()


def test_fairness_aware_ranking_deterministic_and_reduces_gap(sample_data, trained_model):
    """
    Verifies that the constrained marginal swap is deterministic and reduces or maintains the gap.
    """
    df_pred = trained_model.attach_predictions(sample_data)
    
    ranked_1, summary_1 = rank_fairness_aware(df_pred, capacity_pct=20.0)
    ranked_2, summary_2 = rank_fairness_aware(df_pred, capacity_pct=20.0)
    
    # Reproducibility check
    pd.testing.assert_frame_equal(ranked_1, ranked_2)
    assert summary_1["selection_changes"] == summary_2["selection_changes"]
    
    # Gap reduction check
    if summary_1["baseline_recall_gap"] is not None and summary_1["fairness_recall_gap"] is not None:
        assert summary_1["fairness_recall_gap"] <= summary_1["baseline_recall_gap"]
