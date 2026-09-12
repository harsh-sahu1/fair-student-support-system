"""
Prioritization and ranking module under strict capacity constraints.
Enforces the mathematical invariant: select exactly ceil(capacity_fraction * N) students.
"""

import math
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np


def compute_capacity_slots(total_students: int, capacity_fraction: float = 0.20) -> int:
    """
    Computes exact capacity allocation: ceil(capacity_fraction * N).
    Must not approximate or use floor/round.
    """
    if total_students <= 0:
        return 0
    return int(math.ceil(capacity_fraction * total_students))


def rank_students(
    df: pd.DataFrame,
    capacity_fraction: float = 0.20,
    prob_col: str = "probability",
    id_col: str = "student_id"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ranks students by predicted need probability descending with deterministic tie-breaking.
    Marks top K = ceil(capacity_fraction * N) as selected = True.
    """
    if df.empty:
        empty_df = df.copy()
        empty_df["rank"] = []
        empty_df["selected"] = []
        return empty_df, {
            "total_students": 0,
            "capacity_fraction": capacity_fraction,
            "num_selected": 0
        }

    ranked = df.copy()
    
    # Deterministic sorting: probability desc, then student_id asc
    sort_cols = [prob_col]
    sort_orders = [False]
    if id_col in ranked.columns:
        sort_cols.append(id_col)
        sort_orders.append(True)
        
    ranked = ranked.sort_values(by=sort_cols, ascending=sort_orders).reset_index(drop=True)
    
    # 1-indexed ranks
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    
    # Compute exact capacity count
    k = compute_capacity_slots(len(ranked), capacity_fraction=capacity_fraction)
    
    # Assign selection flag
    ranked["selected"] = ranked["rank"] <= k
    
    summary = {
        "total_students": len(ranked),
        "capacity_fraction": capacity_fraction,
        "num_selected": k,
        "selected_count_check": int(ranked["selected"].sum())
    }
    
    # Strict validation invariant
    assert summary["selected_count_check"] == k, (
        f"Capacity constraint violation! Expected {k}, got {summary['selected_count_check']}"
    )
    
    return ranked, summary


def rank_fairness_aware(
    df: pd.DataFrame,
    capacity_fraction: float = 0.20,
    prob_col: str = "probability",
    id_col: str = "student_id",
    group_col: str = "school"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Separate fairness-aware selection layer operating downstream of predicted probabilities.
    
    Architecture:
      Original ML model -> Predicted support probability -> Baseline ranking
      -> Separate fairness-aware selection/ranking layer -> Final Top-20% roster.
      
    Allocates capacity slots K = ceil(capacity_fraction * N) across demographic groups
    proportionally to their cohort representation (using largest remainder method), then selects
    the highest-probability candidates within each group.
    
    Preserves:
      - Exact capacity constraint: sum(selected) == ceil(capacity_fraction * N)
      - Unaltered, calibrated model probabilities
      - Zero label leakage (uses group membership and predicted score only)
    """
    if df.empty or group_col not in df.columns:
        return rank_students(df, capacity_fraction=capacity_fraction, prob_col=prob_col, id_col=id_col)

    k_total = compute_capacity_slots(len(df), capacity_fraction=capacity_fraction)
    if k_total <= 0:
        return rank_students(df, capacity_fraction=capacity_fraction, prob_col=prob_col, id_col=id_col)

    # Calculate group representation quotas via largest remainder method
    groups = df[group_col].unique()
    n_total = len(df)
    
    quotas: Dict[Any, int] = {}
    remainders: Dict[Any, float] = {}
    allocated = 0
    
    for grp in groups:
        grp_size = (df[group_col] == grp).sum()
        exact_share = k_total * (grp_size / n_total)
        base_quota = int(math.floor(exact_share))
        quotas[grp] = base_quota
        remainders[grp] = exact_share - base_quota
        allocated += base_quota
        
    # Distribute remaining slots to groups with largest remainder
    slots_left = k_total - allocated
    sorted_by_remainder = sorted(remainders.keys(), key=lambda g: remainders[g], reverse=True)
    for i in range(slots_left):
        quotas[sorted_by_remainder[i]] += 1
        
    # Within each group, select top candidates by predicted probability
    selected_ids = set()
    for grp in groups:
        grp_df = df[df[group_col] == grp].sort_values(
            by=[prob_col, id_col] if id_col in df.columns else [prob_col],
            ascending=[False, True] if id_col in df.columns else [False]
        )
        q = quotas[grp]
        sel_grp = grp_df.head(q)
        if id_col in df.columns:
            selected_ids.update(sel_grp[id_col].tolist())
        else:
            selected_ids.update(sel_grp.index.tolist())
            
    ranked = df.copy()
    if id_col in ranked.columns:
        ranked["selected"] = ranked[id_col].isin(selected_ids)
    else:
        ranked["selected"] = ranked.index.isin(selected_ids)
        
    # Order: selected students first sorted by prob desc, then unselected by prob desc
    ranked["_sort_key"] = ranked["selected"].apply(lambda x: 0 if x else 1)
    sort_cols = ["_sort_key", prob_col]
    sort_orders = [True, False]
    if id_col in ranked.columns:
        sort_cols.append(id_col)
        sort_orders.append(True)
        
    ranked = ranked.sort_values(by=sort_cols, ascending=sort_orders).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.drop(columns=["_sort_key"], inplace=True)
    
    summary = {
        "total_students": len(ranked),
        "capacity_fraction": capacity_fraction,
        "num_selected": k_total,
        "selected_count_check": int(ranked["selected"].sum()),
        "group_quotas": quotas,
        "fairness_layer": "group_proportional_capacity"
    }
    
    assert summary["selected_count_check"] == k_total, (
        f"Fairness-aware capacity violation! Expected {k_total}, got {summary['selected_count_check']}"
    )
    
    return ranked, summary

