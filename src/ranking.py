"""
Ranking and selection module for Fair Student-Support Prioritization.
Implements Baseline Top-K Ranking and Constrained Bounded Marginal Swap Re-Ranking.
Strictly respects the configurable capacity constraint (default: 20%).
"""

from typing import Tuple, Dict, List, Any
import numpy as np
import pandas as pd

from src.fairness import compute_group_fairness_metrics


def rank_baseline(
    df: pd.DataFrame,
    capacity_pct: float = 20.0,
    score_col: str = "need_score",
    id_col: str = "student_id"
) -> pd.DataFrame:
    """
    Ranks students purely by predicted support-need score descending.
    Selects exactly floor(N * capacity_pct / 100) students.
    """
    out_df = df.copy()
    n_total = len(out_df)
    capacity_k = int(np.floor(n_total * (capacity_pct / 100.0)))
    
    # Deterministic sorting: need_score desc, id_col asc for stable tie-breaking
    out_df = out_df.sort_values(by=[score_col, id_col], ascending=[False, True]).reset_index(drop=True)
    out_df["baseline_rank"] = np.arange(1, n_total + 1)
    out_df["baseline_selected"] = (out_df["baseline_rank"] <= capacity_k).astype(int)
    
    return out_df


def rank_fairness_aware(
    df: pd.DataFrame,
    capacity_pct: float = 20.0,
    score_col: str = "need_score",
    group_col: str = "student_group",
    target_col: str = "support_needed",
    id_col: str = "student_id",
    max_score_delta: float = 12.0,
    min_gap_target: float = 5.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies a constrained, deterministic marginal swap algorithm to improve group recall balance
    while strictly preserving the total capacity K and prioritizing high-need students.
    
    Methodology:
    1. Start from baseline top-K selection (capacity = floor(N * capacity_pct / 100)).
    2. Measure group-wise recall on eligible student groups.
    3. Identify the group with the lowest recall (g_under) and groups with highest recall (g_over).
    4. Consider the highest-need unselected student in g_under (student_in) and the lowest-need
       selected student in g_over (student_out).
    5. Swap them IF AND ONLY IF:
       - The score difference (score(student_out) - score(student_in)) <= max_score_delta.
       - The swap reduces the observed recall gap.
    6. Terminate when the observed recall gap is <= min_gap_target or no eligible swaps remain.
    
    Guarantees:
    - selected_students == capacity_k (Strict capacity adherence).
    - Fully deterministic and reproducible.
    - Preserves high-need students (bounded by max_score_delta).
    """
    # 1. Establish baseline ranking
    work_df = rank_baseline(df, capacity_pct=capacity_pct, score_col=score_col, id_col=id_col)
    n_total = len(work_df)
    capacity_k = int(np.floor(n_total * (capacity_pct / 100.0)))
    
    work_df["fairness_selected"] = work_df["baseline_selected"].copy()
    
    # Track swap audit log
    swaps_log = []
    
    # 2. Iterative bounded swap
    max_iterations = int(capacity_k * 0.4)  # Safety limit: at most 40% of selected pool can ever be adjusted
    
    for iteration in range(max_iterations):
        # Current fairness metrics
        metrics = compute_group_fairness_metrics(
            work_df,
            selection_col="fairness_selected",
            target_col=target_col,
            group_col=group_col
        )
        
        current_gap = metrics["observed_recall_gap"]
        if current_gap is None or current_gap <= min_gap_target:
            break
            
        g_min = metrics["min_group"]
        g_max = metrics["max_group"]
        
        # Candidate IN: unselected student in g_min with highest need_score (prefer true need if known)
        unselected_gmin = work_df[(work_df["fairness_selected"] == 0) & (work_df[group_col] == g_min)]
        if target_col in work_df.columns:
            # Look at unselected students who truly need support first, ordered by need_score
            in_candidates = unselected_gmin[unselected_gmin[target_col] == 1].sort_values(
                by=[score_col, id_col], ascending=[False, True]
            )
            if in_candidates.empty:
                in_candidates = unselected_gmin.sort_values(by=[score_col, id_col], ascending=[False, True])
        else:
            in_candidates = unselected_gmin.sort_values(by=[score_col, id_col], ascending=[False, True])
            
        if in_candidates.empty:
            break
            
        candidate_in = in_candidates.iloc[0]
        
        # Candidate OUT: selected student in g_max (or over-represented groups) with lowest need_score
        selected_gmax = work_df[(work_df["fairness_selected"] == 1) & (work_df[group_col] == g_max)]
        if target_col in work_df.columns:
            # Prefer swapping out non-needing students or lowest need score
            out_candidates = selected_gmax[selected_gmax[target_col] == 0].sort_values(
                by=[score_col, id_col], ascending=[True, True]
            )
            if out_candidates.empty:
                out_candidates = selected_gmax.sort_values(by=[score_col, id_col], ascending=[True, True])
        else:
            out_candidates = selected_gmax.sort_values(by=[score_col, id_col], ascending=[True, True])
            
        if out_candidates.empty:
            break
            
        candidate_out = out_candidates.iloc[0]
        
        score_diff = candidate_out[score_col] - candidate_in[score_col]
        if score_diff > max_score_delta:
            # Score sacrifice too high; abort to maintain merit integrity
            break
            
        # Simulate swap and check if gap strictly improves
        idx_in = candidate_in.name
        idx_out = candidate_out.name
        
        work_df.loc[idx_in, "fairness_selected"] = 1
        work_df.loc[idx_out, "fairness_selected"] = 0
        
        new_metrics = compute_group_fairness_metrics(
            work_df,
            selection_col="fairness_selected",
            target_col=target_col,
            group_col=group_col
        )
        new_gap = new_metrics["observed_recall_gap"]
        
        if new_gap is not None and new_gap < current_gap:
            # Commit swap
            swaps_log.append({
                "swap_id": iteration + 1,
                "student_in": candidate_in[id_col],
                "group_in": candidate_in[group_col],
                "score_in": candidate_in[score_col],
                "student_out": candidate_out[id_col],
                "group_out": candidate_out[group_col],
                "score_out": candidate_out[score_col],
                "score_delta": np.round(score_diff, 1),
                "previous_gap": current_gap,
                "new_gap": new_gap,
            })
        else:
            # Revert swap and exit
            work_df.loc[idx_in, "fairness_selected"] = 0
            work_df.loc[idx_out, "fairness_selected"] = 1
            break
            
    # Assign final fairness rank:
    # Selected students ranked 1..K by need_score desc, unselected ranked K+1..N by need_score desc
    selected_mask = (work_df["fairness_selected"] == 1)
    
    sel_df = work_df[selected_mask].sort_values(by=[score_col, id_col], ascending=[False, True])
    unsel_df = work_df[~selected_mask].sort_values(by=[score_col, id_col], ascending=[False, True])
    
    sel_df["fairness_rank"] = np.arange(1, len(sel_df) + 1)
    unsel_df["fairness_rank"] = np.arange(len(sel_df) + 1, n_total + 1)
    
    final_df = pd.concat([sel_df, unsel_df]).sort_values(by="fairness_rank").reset_index(drop=True)
    
    # Audit summary
    final_metrics = compute_group_fairness_metrics(
        final_df,
        selection_col="fairness_selected",
        target_col=target_col,
        group_col=group_col
    )
    
    baseline_metrics = compute_group_fairness_metrics(
        final_df,
        selection_col="baseline_selected",
        target_col=target_col,
        group_col=group_col
    )
    
    selection_changes = int((final_df["baseline_selected"] != final_df["fairness_selected"]).sum() // 2)
    
    summary = {
        "capacity_k": capacity_k,
        "capacity_pct": capacity_pct,
        "total_students": n_total,
        "baseline_selected_count": int(final_df["baseline_selected"].sum()),
        "fairness_selected_count": int(final_df["fairness_selected"].sum()),
        "baseline_recall_gap": baseline_metrics["observed_recall_gap"],
        "fairness_recall_gap": final_metrics["observed_recall_gap"],
        "gap_reduction": np.round(baseline_metrics["observed_recall_gap"] - final_metrics["observed_recall_gap"], 1) if baseline_metrics["observed_recall_gap"] is not None and final_metrics["observed_recall_gap"] is not None else 0.0,
        "selection_changes": selection_changes,
        "swaps_log": swaps_log,
        "baseline_metrics": baseline_metrics,
        "fairness_metrics": final_metrics,
    }
    
    return final_df, summary
