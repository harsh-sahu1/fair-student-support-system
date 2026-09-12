"""
Fairness evaluation module for Fair Student-Support Prioritization.
Computes group-wise recall, selection rates, and the Observed Recall Gap.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def compute_group_fairness_metrics(
    df: pd.DataFrame,
    selection_col: str = "is_selected",
    target_col: str = "support_needed",
    group_col: str = "student_group"
) -> Dict[str, Any]:
    """
    Computes group-wise recall, selection counts, precision, and the Observed Recall Gap.
    
    Formula:
      Group Recall = (Selected students with support_needed = 1) / (All students with support_needed = 1 in that group)
      Observed Recall Gap = max(Group Recall) - min(Group Recall)
    
    Returns:
      Dictionary containing:
      - 'table': pd.DataFrame with group-level breakdown
      - 'observed_recall_gap': float (percentage difference between max and min recall)
      - 'max_group': group with highest recall
      - 'min_group': group with lowest recall
      - 'overall_recall': float (macro/total recall across all groups)
      - 'overall_precision': float
      - 'total_selected': int
      - 'capacity_pct': float (percentage of total students selected)
    """
    if selection_col not in df.columns:
        raise ValueError(f"Selection column '{selection_col}' not found in dataframe.")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe.")
        
    has_ground_truth = target_col in df.columns and df[target_col].notna().any()
    
    groups = sorted(df[group_col].unique())
    group_rows = []
    
    for g in groups:
        gdf = df[df[group_col] == g]
        total_students = len(gdf)
        selected_students = int(gdf[selection_col].sum())
        selection_rate = (selected_students / total_students) * 100 if total_students > 0 else 0.0
        
        if has_ground_truth:
            total_need = int(gdf[target_col].sum())
            true_positives = int(((gdf[selection_col] == 1) & (gdf[target_col] == 1)).sum())
            recall = (true_positives / total_need * 100) if total_need > 0 else 0.0
            precision = (true_positives / selected_students * 100) if selected_students > 0 else 0.0
        else:
            total_need = None
            true_positives = None
            recall = None
            precision = None
            
        group_rows.append({
            "group": g,
            "total_students": total_students,
            "total_support_needed": total_need,
            "selected_count": selected_students,
            "selection_rate_pct": np.round(selection_rate, 1),
            "true_positives": true_positives,
            "recall_pct": np.round(recall, 1) if recall is not None else None,
            "precision_pct": np.round(precision, 1) if precision is not None else None,
        })
        
    table_df = pd.DataFrame(group_rows)
    
    total_selected = int(df[selection_col].sum())
    capacity_pct = (total_selected / len(df)) * 100 if len(df) > 0 else 0.0
    
    if has_ground_truth:
        recalls = table_df["recall_pct"].dropna()
        max_recall = recalls.max()
        min_recall = recalls.min()
        observed_recall_gap = np.round(max_recall - min_recall, 1)
        
        max_group = table_df.loc[table_df["recall_pct"] == max_recall, "group"].iloc[0]
        min_group = table_df.loc[table_df["recall_pct"] == min_recall, "group"].iloc[0]
        
        total_need_overall = int(df[target_col].sum())
        total_tp_overall = int(((df[selection_col] == 1) & (df[target_col] == 1)).sum())
        overall_recall = np.round((total_tp_overall / total_need_overall) * 100, 1) if total_need_overall > 0 else 0.0
        overall_precision = np.round((total_tp_overall / total_selected) * 100, 1) if total_selected > 0 else 0.0
    else:
        observed_recall_gap = None
        max_group = None
        min_group = None
        overall_recall = None
        overall_precision = None
        
    return {
        "table": table_df,
        "observed_recall_gap": observed_recall_gap,
        "max_group": max_group,
        "min_group": min_group,
        "overall_recall": overall_recall,
        "overall_precision": overall_precision,
        "total_selected": total_selected,
        "capacity_pct": np.round(capacity_pct, 1),
    }


def compare_fairness_policies(
    baseline_metrics: Dict[str, Any],
    fair_metrics: Dict[str, Any]
) -> pd.DataFrame:
    """
    Creates a side-by-side comparative summary table between Baseline and Fairness-Aware selection.
    """
    b_table = baseline_metrics["table"].copy()
    f_table = fair_metrics["table"].copy()
    
    comparison = pd.DataFrame({
        "Eligible Group": b_table["group"],
        "Group Size": b_table["total_students"],
        "Baseline Selected": b_table["selected_count"],
        "Baseline Recall (%)": b_table["recall_pct"],
        "Fairness Selected": f_table["selected_count"],
        "Fairness Recall (%)": f_table["recall_pct"],
        "Recall Change (% pts)": np.round(f_table["recall_pct"] - b_table["recall_pct"], 1),
    })
    return comparison
