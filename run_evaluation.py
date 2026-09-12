"""
CLI runner for complete evaluation of the Fair Student-Support Prioritization system.
Generates development and synthetic metrics, audits fairness improvements, and saves report to outputs/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from src.data_processing import load_or_create_dataset, split_dataset
from src.model import StudentSupportModel
from src.ranking import rank_fairness_aware
from src.fairness import compute_group_fairness_metrics, compare_fairness_policies
from src.evaluation import compute_classification_metrics, compute_ranking_metrics, evaluate_system_robustness


def main():
    print("=" * 70)
    print("  FAIR STUDENT-SUPPORT PRIORITIZATION EVALUATION")
    print("=" * 70)
    
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Dataset loading / generation
    df = load_or_create_dataset("data/students.csv", n_samples=1000, seed=42)
    train_df, test_df = split_dataset(df, test_size=0.25, seed=42)
    print(f"\n[1] Dataset Loaded: {len(df)} total students (Train: {len(train_df)}, Test: {len(test_df)})")
    print(f"    - Eligible Groups: {sorted(df['student_group'].unique())}")
    print(f"    - True Support Need Rate: {df['support_needed'].mean() * 100:.1f}%")
    
    # 2. Model training
    model = StudentSupportModel(model_type="logistic_regression", random_state=42)
    model.fit(train_df)
    full_pred = model.attach_predictions(df)
    test_pred = model.attach_predictions(test_df)
    print("\n[2] Model Trained: Calibrated Logistic Regression Pipeline")
    
    # Classification metrics
    clf_metrics = compute_classification_metrics(test_df["support_needed"].values, test_pred["need_probability"].values)
    print("    - Test Classification Metrics:")
    for k, v in clf_metrics.items():
        print(f"      * {k}: {v}")
        
    # 3. Prioritization & Capacity Constraint (20%)
    capacity_pct = 20.0
    ranked_df, ranking_summary = rank_fairness_aware(full_pred, capacity_pct=capacity_pct)
    rank_metrics = compute_ranking_metrics(full_pred, capacity_pct=capacity_pct)
    
    print(f"\n[3] 20% Capacity Constraint Verification:")
    print(f"    - Target Capacity: {capacity_pct}% of {len(df)} students = {ranking_summary['capacity_k']} slots")
    print(f"    - Baseline Selected Count: {ranking_summary['baseline_selected_count']}")
    print(f"    - Fairness Selected Count: {ranking_summary['fairness_selected_count']}")
    print(f"    - Constraint Adherence: EXACT ({ranking_summary['fairness_selected_count'] == ranking_summary['capacity_k']})")
    
    # 4. Fairness Audit & Improvement
    b_metrics = ranking_summary["baseline_metrics"]
    f_metrics = ranking_summary["fairness_metrics"]
    comp_df = compare_fairness_policies(b_metrics, f_metrics)
    
    print("\n[4] Fairness Audit & Group-Wise Recall:")
    print(comp_df.to_string(index=False))
    print(f"\n    - Baseline Observed Recall Gap: {ranking_summary['baseline_recall_gap']}%")
    print(f"    - Fairness-Aware Recall Gap:    {ranking_summary['fairness_recall_gap']}%")
    print(f"    - Total Recall Gap Reduction:   {ranking_summary['gap_reduction']}% pts")
    print(f"    - Total Marginal Swaps Made:    {ranking_summary['selection_changes']} students ({ranking_summary['selection_changes'] / len(df) * 100:.1f}% of cohort)")
    
    # 5. Robustness
    print("\n[5] Running Input Perturbation Robustness Test (+/- 3% noise)...")
    robustness = evaluate_system_robustness(df, model, capacity_pct=capacity_pct, noise_std=0.03, n_simulations=5)
    print(f"    - Mean Spearman Rank Correlation: {robustness['mean_spearman_correlation']}")
    print(f"    - Mean Jaccard Selection Overlap: {robustness['mean_jaccard_selection_overlap'] * 100:.1f}%")
    print(f"    - Perturbed Recall Gap:           {robustness['mean_perturbed_fairness_gap']}%")
    
    # 6. Save JSON Report
    report = {
        "dataset": {
            "total_students": len(df),
            "train_students": len(train_df),
            "test_students": len(test_df),
            "groups": sorted(df["student_group"].unique()),
            "true_need_rate_pct": round(df["support_needed"].mean() * 100, 1),
            "type": "Synthetic educational benchmark dataset (reproducible seed=42)"
        },
        "model": {
            "type": "Logistic Regression with StandardScaler",
            "test_classification_metrics": clf_metrics,
            "ranking_metrics": rank_metrics,
        },
        "capacity_and_fairness": {
            "capacity_pct": capacity_pct,
            "capacity_slots": ranking_summary["capacity_k"],
            "baseline_recall_gap_pct": ranking_summary["baseline_recall_gap"],
            "fairness_recall_gap_pct": ranking_summary["fairness_recall_gap"],
            "gap_reduction_pts": ranking_summary["gap_reduction"],
            "selection_changes_count": ranking_summary["selection_changes"],
            "group_breakdown": comp_df.to_dict(orient="records"),
        },
        "robustness": robustness,
    }
    
    report_path = os.path.join("outputs", "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[6] Evaluation report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
