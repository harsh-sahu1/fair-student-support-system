"""
Offline training and artifact generation script.
Produces:
  1. backend/artifacts/model.joblib
  2. backend/artifacts/validation_predictions.json
  3. backend/artifacts/metrics_summary.json
"""

import sys
import json
from pathlib import Path

# Ensure root backend path is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import pandas as pd

from backend.app.config import (
    DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    VALIDATION_PREDICTIONS_PATH,
    DEFAULT_CAPACITY_FRACTION,
    TARGET_COLUMN,
    ARTIFACTS_DIR
)
from backend.app.ml.data_processing import (
    load_raw_dataset,
    prepare_dataset,
    get_train_validation_split,
)
from backend.app.ml.model import StudentSupportModel
from backend.app.ml.ranking import rank_students
from backend.app.ml.fairness import evaluate_fairness
from backend.app.ml.explainability import FeatureExplainer
from backend.app.ml.evaluation import compute_metrics, run_10_seed_robustness


def run_training(random_state: int = 42) -> None:
    print("=" * 70)
    print("  STUDENT SUPPORT AI: MODEL TRAINING & ARTIFACT GENERATION")
    print("=" * 70)
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load and prepare dataset
    print(f"\n[1] Loading dataset from: {DATA_PATH}")
    raw_df = load_raw_dataset(str(DATA_PATH))
    df = prepare_dataset(raw_df)
    print(f"    - Total rows: {len(df)}")
    print(f"    - True Support Needed count: {int(df[TARGET_COLUMN].sum())} ({df[TARGET_COLUMN].mean()*100:.1f}%)")
    
    # 2. Train / Validation Split (25% held out)
    print(f"\n[2] Splitting into Train (75%) and Validation (25%) splits (random_state={random_state})...")
    train_df, val_df = get_train_validation_split(df, test_size=0.25, random_state=random_state)
    print(f"    - Train records: {len(train_df)}")
    print(f"    - Validation records: {len(val_df)}")
    
    # 3. Model Training
    print("\n[3] Training scikit-learn Logistic Regression pipeline with ColumnTransformer...")
    model = StudentSupportModel(random_state=random_state)
    model.fit(train_df)
    model.save(str(MODEL_PATH))
    print(f"    - Model persisted to: {MODEL_PATH}")
    
    # 4. Generate Validation Predictions & Ranking
    print(f"\n[4] Generating predictions and ranking at {DEFAULT_CAPACITY_FRACTION*100:.0f}% capacity...")
    val_with_preds = model.attach_predictions(val_df)
    ranked_val, rank_summary = rank_students(val_with_preds, capacity_fraction=DEFAULT_CAPACITY_FRACTION)
    print(f"    - Total validation cohort: {rank_summary['total_students']}")
    print(f"    - Selected for support: {rank_summary['num_selected']} (exact ceil rule)")
    
    # 5. Compute Feature Explanations
    print("\n[5] Generating top contributing factors for validation cohort...")
    explainer = FeatureExplainer(model)
    
    records = []
    for _, row in ranked_val.iterrows():
        explanation = explainer.explain_student(row, top_k=3)
        record = {
            "student_id": str(row["student_id"]),
            "rank": int(row["rank"]),
            "need_score": float(row["need_score"]),
            "probability": float(row["probability"]),
            "selected": bool(row["selected"]),
            "sex": str(row.get("sex", "Unknown")),
            "school": str(row.get("school", "Unknown")),
            "ground_truth_support_needed": int(row[TARGET_COLUMN]),
            "top_factors": explanation["top_factors"],
            "factor_details": explanation["factor_details"],
            "raw_features": {
                k: (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else str(v))
                for k, v in row.items()
                if k not in ["probability", "need_score", "rank", "selected"]
            }
        }
        records.append(record)
        
    with open(VALIDATION_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"    - Validation student records written to: {VALIDATION_PREDICTIONS_PATH}")
    
    # 6. Fairness Audit
    print("\n[6] Computing Group-Wise Fairness Audit...")
    fairness_results = evaluate_fairness(ranked_val)
    print(f"    - Worst Group Recall (R_min): {fairness_results['worst_group_recall']:.3f}")
    print(f"    - Observed Recall Gap: {fairness_results['fairness_gap']:.3f}")
    for g in fairness_results["groups"]:
        print(f"      * {g['attribute']}={g['group']}: rows={g['n_rows']}, pos={g['n_positives']}, sel_pos={g['n_selected_positives']}, recall={g['recall']:.3f}, eligible={g['eligible']}")

    # 7. Classification and Overview Metrics
    y_true = ranked_val[TARGET_COLUMN].values
    y_prob = ranked_val["probability"].values
    selected = ranked_val["selected"].values
    clf_metrics = compute_metrics(y_true, y_prob, selected)
    
    overview_metrics = {
        "total_students": len(ranked_val),
        "capacity_fraction": DEFAULT_CAPACITY_FRACTION,
        "num_selected": rank_summary["num_selected"],
        "average_need_score": float(np.round(ranked_val["need_score"].mean(), 1)),
        "worst_group_recall": fairness_results["worst_group_recall"],
        "fairness_gap": fairness_results["fairness_gap"],
        "overall_recall_at_capacity": clf_metrics["recall_at_capacity"],
        "brier_score": clf_metrics["brier_score"],
    }
    
    # 8. 10-Seed Robustness Evaluation
    print("\n[8] Running 10-seed robustness benchmark across random train/val splits...")
    robustness_metrics = run_10_seed_robustness(df, seeds=list(range(10)), test_size=0.25)
    print(f"    - Mean Recall@20%: {robustness_metrics['recall_at_capacity']['mean']} ± {robustness_metrics['recall_at_capacity']['std']}")
    print(f"    - Mean R_min:      {robustness_metrics['worst_group_recall']['mean']} ± {robustness_metrics['worst_group_recall']['std']}")
    print(f"    - Mean Gap:        {robustness_metrics['fairness_gap']['mean']} ± {robustness_metrics['fairness_gap']['std']}")
    print(f"    - Mean Brier:      {robustness_metrics['brier_score']['mean']} ± {robustness_metrics['brier_score']['std']}")
    
    # Save combined metrics summary
    metrics_summary = {
        "overview": overview_metrics,
        "fairness": fairness_results,
        "robustness": robustness_metrics,
        "meta": {
            "dataset": "UCI Student Performance (mathematics)",
            "total_rows": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "model_type": "Logistic Regression with StandardScaler and OneHotEncoder",
            "capacity_rule": "Exact ceil(0.20 * N)",
            "evaluation_note": "Demo/development metrics evaluated on held-out public validation split."
        }
    }
    
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n[9] Metrics summary written to: {METRICS_PATH}")
    print("=" * 70)
    print("  TRAINING COMPLETE! ALL ARTIFACTS GENERATED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    run_training()
