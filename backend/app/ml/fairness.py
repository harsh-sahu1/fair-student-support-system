"""
Fairness audit module for group-wise recall and gap computations.
Implements the exact benchmark specification for eligibility, R_min, and Observed Recall Gap.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from backend.app.config import PROTECTED_ATTRIBUTES, TARGET_COLUMN


def evaluate_fairness(
    df: pd.DataFrame,
    selection_col: str = "selected",
    target_col: str = TARGET_COLUMN,
    attributes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes group-level recall and intra-attribute fairness gaps.
    
    Eligibility rules:
      - n_rows >= 10
      - n_positives (target == 1) >= 3
      
    R_min:
      - Minimum recall across ALL eligible groups across all audited protected attributes.
      
    fairness_gap (Observed Recall Gap):
      - Largest |recall_a - recall_b| between eligible groups of the SAME protected attribute.
      - Never compare across different attributes (e.g. sex vs school).
      - Returns 0.0 if no attribute has >= 2 eligible groups.
    """
    attrs = attributes or PROTECTED_ATTRIBUTES
    group_results: List[Dict[str, Any]] = []
    eligible_recalls_by_attr: Dict[str, List[float]] = {attr: [] for attr in attrs}
    all_eligible_recalls: List[float] = []
    
    for attr in attrs:
        if attr not in df.columns:
            continue
            
        unique_groups = sorted(df[attr].dropna().unique().tolist())
        for grp in unique_groups:
            sub = df[df[attr] == grp]
            n_rows = int(len(sub))
            
            # Ground truth positives
            pos_mask = (sub[target_col] == 1)
            n_positives = int(pos_mask.sum())
            
            # Selected positives
            sel_pos_mask = (sub[selection_col] == True) & pos_mask
            n_selected_positives = int(sel_pos_mask.sum())
            
            # Group eligibility check
            eligible = bool((n_rows >= 10) and (n_positives >= 3))
            
            # Group recall
            if n_positives > 0:
                recall = float(np.round(n_selected_positives / n_positives, 3))
            else:
                recall = 0.0
                
            group_item = {
                "attribute": attr,
                "group": str(grp),
                "n_rows": n_rows,
                "n_positives": n_positives,
                "n_selected_positives": n_selected_positives,
                "eligible": eligible,
                "recall": recall,
                "status_label": "Eligible" if eligible else "not eligible (insufficient sample)"
            }
            group_results.append(group_item)
            
            if eligible:
                eligible_recalls_by_attr[attr].append(recall)
                all_eligible_recalls.append(recall)
                
    # Compute R_min: minimum recall across ALL eligible groups pooled
    if all_eligible_recalls:
        worst_group_recall = float(np.round(min(all_eligible_recalls), 3))
    else:
        worst_group_recall = 0.0
        
    # Compute fairness_gap: max |recall_a - recall_b| within the SAME attribute
    intra_attribute_gaps: List[float] = []
    for attr, recs in eligible_recalls_by_attr.items():
        if len(recs) >= 2:
            gap = max(recs) - min(recs)
            intra_attribute_gaps.append(gap)
            
    if intra_attribute_gaps:
        fairness_gap = float(np.round(max(intra_attribute_gaps), 3))
    else:
        fairness_gap = 0.0
        
    return {
        "groups": group_results,
        "worst_group_recall": worst_group_recall,
        "fairness_gap": fairness_gap,
        "metric_name": "Observed Recall Gap",
        "pooling_rule": "R_min pooled across all eligible groups; fairness_gap evaluated intra-attribute only."
    }
