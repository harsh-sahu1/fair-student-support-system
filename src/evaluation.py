"""
Evaluation and robustness assessment module for Fair Student-Support Prioritization.
Computes classification metrics, ranking metrics (Recall@20%, Precision@20%, NDCG@20%),
fairness audits, and feature perturbation stability tests.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    ndcg_score,
)

from src.data_processing import FEATURE_COLS, TARGET_COL
from src.ranking import rank_baseline, rank_fairness_aware
from src.fairness import compute_group_fairness_metrics


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes standard classification performance metrics on held-out test data.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1_score": round(f1, 3),
        "roc_auc": round(auc, 3),
        "brier_score": round(brier, 4),
    }


def compute_ranking_metrics(
    df: pd.DataFrame,
    capacity_pct: float = 20.0,
    score_col: str = "need_score",
    target_col: str = TARGET_COL
) -> Dict[str, float]:
    """
    Computes top-K ranking quality metrics: Precision@K, Recall@K, and NDCG@K.
    """
    n_total = len(df)
    k = int(np.floor(n_total * (capacity_pct / 100.0)))
    
    sorted_df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    top_k = sorted_df.iloc[:k]
    
    total_positives = df[target_col].sum()
    tp_at_k = top_k[target_col].sum()
    
    precision_at_k = tp_at_k / k if k > 0 else 0.0
    recall_at_k = tp_at_k / total_positives if total_positives > 0 else 0.0
    
    # NDCG@K calculation
    y_true_binary = df[target_col].values.reshape(1, -1)
    scores = df[score_col].values.reshape(1, -1)
    ndcg_val = ndcg_score(y_true_binary, scores, k=k) if total_positives > 0 else 1.0
    
    return {
        f"precision_at_{int(capacity_pct)}pct": round(precision_at_k, 3),
        f"recall_at_{int(capacity_pct)}pct": round(recall_at_k, 3),
        f"ndcg_at_{int(capacity_pct)}pct": round(ndcg_val, 3),
        "capacity_k": k,
        "total_population": n_total,
    }


def evaluate_system_robustness(
    df: pd.DataFrame,
    model,
    capacity_pct: float = 20.0,
    noise_std: float = 0.03,
    n_simulations: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Tests ranking and selection stability under small input feature perturbations (+/- 3%).
    Simulates real-world variability in attendance reporting or assignment recording.
    
    Evaluates:
    - Spearman rank correlation of scores
    - Jaccard similarity of selected top-20% cohorts
    - Observed recall gap variance
    """
    np.random.seed(seed)
    
    # Baseline original run
    df_orig = model.attach_predictions(df)
    orig_ranked, orig_summary = rank_fairness_aware(df_orig, capacity_pct=capacity_pct)
    orig_selected_ids = set(orig_ranked[orig_ranked["fairness_selected"] == 1]["student_id"])
    orig_scores = orig_ranked.set_index("student_id")["need_score"]
    
    spearman_corrs = []
    jaccard_sims = []
    perturbed_gaps = []
    
    for i in range(n_simulations):
        noisy_df = df.copy()
        # Add relative Gaussian noise to numerical features
        for col in FEATURE_COLS:
            std_dev = noisy_df[col].std()
            noise = np.random.normal(0, noise_std * std_dev, size=len(noisy_df))
            noisy_df[col] = np.clip(noisy_df[col] + noise, 0, None)
            
        noisy_pred = model.attach_predictions(noisy_df)
        noisy_ranked, noisy_summary = rank_fairness_aware(noisy_pred, capacity_pct=capacity_pct)
        
        # 1. Spearman rank correlation on student need scores
        noisy_scores = noisy_ranked.set_index("student_id")["need_score"]
        common_ids = orig_scores.index.intersection(noisy_scores.index)
        corr, _ = spearmanr(orig_scores.loc[common_ids], noisy_scores.loc[common_ids])
        spearman_corrs.append(corr)
        
        # 2. Jaccard similarity of selected sets: |A intersect B| / |A union B|
        pert_selected_ids = set(noisy_ranked[noisy_ranked["fairness_selected"] == 1]["student_id"])
        intersection = len(orig_selected_ids.intersection(pert_selected_ids))
        union = len(orig_selected_ids.union(pert_selected_ids))
        jaccard = intersection / union if union > 0 else 1.0
        jaccard_sims.append(jaccard)
        
        # 3. Gap stability
        gap = noisy_summary["fairness_recall_gap"]
        if gap is not None:
            perturbed_gaps.append(gap)
            
    return {
        "mean_spearman_correlation": round(float(np.mean(spearman_corrs)), 3),
        "mean_jaccard_selection_overlap": round(float(np.mean(jaccard_sims)), 3),
        "mean_perturbed_fairness_gap": round(float(np.mean(perturbed_gaps)), 1) if perturbed_gaps else None,
        "baseline_fairness_gap": orig_summary["baseline_recall_gap"],
        "unperturbed_fairness_gap": orig_summary["fairness_recall_gap"],
        "n_simulations": n_simulations,
        "noise_scale_pct": round(noise_std * 100, 1),
    }
