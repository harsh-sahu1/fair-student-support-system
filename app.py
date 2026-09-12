"""
Fair Student-Support Prioritization System
Educator-facing decision-support dashboard for allocating limited academic support capacity (strictly 20%)
with proactive auditing and mitigation of group recall disparities.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import load_or_create_dataset, split_dataset, FEATURE_COLS
from src.model import StudentSupportModel
from src.ranking import rank_baseline, rank_fairness_aware
from src.fairness import compute_group_fairness_metrics, compare_fairness_policies
from src.explainability import StudentExplainer
from src.evaluation import compute_classification_metrics, compute_ranking_metrics, evaluate_system_robustness

# ==============================================================================
# 1. PAGE SETUP & DESIGN SYSTEM CSS INJECTION
# ==============================================================================
st.set_page_config(
    page_title="Fair Student-Support Prioritization",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System CSS
# Follows strict guidelines: calm slate-blue primary, warm amber for support status,
# clean 1px borders, no generic AI-generated card gradients or rounded drop-shadows.
CUSTOM_CSS = """
<style>
/* Base Typography & Clean Palette */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    color: #0F172A;
}

/* Tighten default loose Streamlit margins */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Sidebar styling: calm operational rail */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #1E3A5F;
    font-weight: 600;
}

/* Operational Metric Cards (No drop-shadow, clean slate border) */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 4px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748B;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}
.metric-subtext {
    font-size: 12px;
    color: #475569;
    margin-top: 4px;
}

/* Status Badges */
.badge-selected {
    display: inline-block;
    background-color: #FEF3C7;
    color: #92400E;
    border: 1px solid #FCD34D;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 3px;
}
.badge-standard {
    display: inline-block;
    background-color: #F1F5F9;
    color: #475569;
    border: 1px solid #CBD5E1;
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 3px;
}
.badge-swap {
    display: inline-block;
    background-color: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 3px;
}

/* Note Banner: Educator Decision Support Context */
.educator-note {
    background: #F8FAFC;
    border-left: 3px solid #1E3A5F;
    padding: 10px 14px;
    font-size: 13px;
    color: #334155;
    margin-bottom: 16px;
    border-radius: 0 4px 4px 0;
}

/* Clean tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #E2E8F0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 500;
    color: #64748B;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    color: #1E3A5F !important;
    border-bottom: 2px solid #1E3A5F !important;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. DATA CACHING & PIPELINE EXECUTION
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_cached_pipeline_data(model_type: str = "logistic_regression"):
    """
    Loads dataset, trains model on train split, and generates predictions for the cohort.
    """
    df = load_or_create_dataset(filepath="data/students.csv", n_samples=1000, seed=42)
    train_df, test_df = split_dataset(df, test_size=0.25, seed=42)
    
    # Train support prediction model
    model = StudentSupportModel(model_type=model_type, random_state=42)
    model.fit(train_df)
    
    # Generate predictions
    full_pred = model.attach_predictions(df)
    test_pred = model.attach_predictions(test_df)
    
    # Build explainer
    explainer = StudentExplainer(model, reference_df=train_df)
    
    return {
        "raw_df": df,
        "train_df": train_df,
        "test_df": test_df,
        "full_pred": full_pred,
        "test_pred": test_pred,
        "model": model,
        "explainer": explainer,
    }


pipeline = get_cached_pipeline_data()
raw_df = pipeline["raw_df"]
full_pred = pipeline["full_pred"]
test_pred = pipeline["test_pred"]
model = pipeline["model"]
explainer = pipeline["explainer"]


# ==============================================================================
# 3. SIDEBAR CONTROLS (Persistent Operational Rail)
# ==============================================================================
with st.sidebar:
    st.markdown("### Support Allocation Controller")
    st.caption("Decision support instrument for academic counseling teams")
    
    st.markdown("---")
    st.markdown("**Intervention Capacity**")
    capacity_pct = st.slider(
        "School Support Capacity (%)",
        min_value=5,
        max_value=35,
        value=20,
        step=1,
        help="The fixed percentage of total students eligible for specialized intervention (Default: 20%)."
    )
    
    n_students = len(full_pred)
    exact_capacity = int(np.floor(n_students * (capacity_pct / 100.0)))
    st.caption(f"Allocated Slots: **{exact_capacity}** of **{n_students}** students")
    
    st.markdown("---")
    st.markdown("**Fairness Adjustment Parameters**")
    max_score_delta = st.slider(
        "Max Score Tolerance Delta",
        min_value=4.0,
        max_value=20.0,
        value=12.0,
        step=1.0,
        help="Maximum score difference permitted when swapping borderline candidates to balance group recall."
    )
    
    group_filter = st.selectbox(
        "Filter Cohort View",
        options=["All Groups", "Group A", "Group B", "Group C"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 11px; color: #64748B; line-height: 1.4;">
        <strong>Educational Decision Support Policy:</strong><br>
        Algorithmic outputs prioritize students for counselor review. Final intervention commitments remain with certified educators.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 4. RANKING & FAIRNESS EVALUATION EXECUTION
# ==============================================================================
ranked_df, ranking_summary = rank_fairness_aware(
    full_pred,
    capacity_pct=capacity_pct,
    score_col="need_score",
    max_score_delta=max_score_delta,
    min_gap_target=4.0
)

# Attach primary explanation snippet for each student
top_reasons = []
for _, row in ranked_df.iterrows():
    exp = explainer.explain_student(row)
    factors = exp["top_contributing_factors"]
    top_reasons.append(factors[0] if factors else "Balanced profile")
ranked_df["primary_risk_factor"] = top_reasons

# Filter view if requested
display_df = ranked_df if group_filter == "All Groups" else ranked_df[ranked_df["student_group"] == group_filter]

baseline_metrics = ranking_summary["baseline_metrics"]
fair_metrics = ranking_summary["fairness_metrics"]


# ==============================================================================
# 5. DASHBOARD HEADER & KPI BAR
# ==============================================================================
st.markdown("## Fair Student-Support Prioritization")
st.markdown(
    '<div class="educator-note">'
    '<strong>Operational Objective:</strong> Prioritize students with high predicted academic support need '
    'within a strict <strong>20% capacity limit</strong>, while minimizing severe disparities in support-needed recall '
    'across eligible student groups.'
    '</div>',
    unsafe_allow_html=True
)

# Top KPI Summary Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Student Cohort</div>
        <div class="metric-value">{n_students:,}</div>
        <div class="metric-subtext">Active student records</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Support Capacity</div>
        <div class="metric-value">{exact_capacity}</div>
        <div class="metric-subtext">Exactly {capacity_pct}% of population</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Selected Need Score</div>
        <div class="metric-value">{ranked_df[ranked_df['fairness_selected'] == 1]['need_score'].mean():.1f}</div>
        <div class="metric-subtext">Cohort mean: {ranked_df['need_score'].mean():.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Baseline Recall Gap</div>
        <div class="metric-value" style="color: #64748B;">{ranking_summary['baseline_recall_gap']}%</div>
        <div class="metric-subtext">Prior to fairness re-ranking</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    gap_color = "#0D9488" if ranking_summary['fairness_recall_gap'] <= 5.0 else "#D97706"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Observed Recall Gap</div>
        <div class="metric-value" style="color: {gap_color};">{ranking_summary['fairness_recall_gap']}%</div>
        <div class="metric-subtext">Reduced by {ranking_summary['gap_reduction']}% pts ({ranking_summary['selection_changes']} swaps)</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 6. MAIN APPLICATION TABS (Who? -> Why? -> Is This Fair? -> Model Quality)
# ==============================================================================
tab_who, tab_why, tab_fair, tab_model = st.tabs([
    "1. Who Needs Support?",
    "2. Why? (Attribution)",
    "3. Is This Fair? (Audit)",
    "4. Model Quality & Robustness"
])


# ------------------------------------------------------------------------------
# TAB 1: WHO NEEDS SUPPORT? (Prioritized Allocation Table)
# ------------------------------------------------------------------------------
with tab_who:
    st.markdown("### Prioritized Student Allocation")
    st.caption("Sorted by fairness-adjusted intervention priority. Top students strictly within capacity constraint are flagged for support.")
    
    col_search, col_stats = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("Search student by ID (e.g. S0042):", "").strip().upper()
    with col_stats:
        selected_in_view = int(display_df["fairness_selected"].sum())
        st.markdown(
            f"<div style='margin-top: 28px; font-size: 13px; color: #475569;'>"
            f"Showing <strong>{len(display_df)}</strong> students (<strong>{selected_in_view}</strong> prioritized for support)"
            f"</div>",
            unsafe_allow_html=True
        )

    # Filter search if provided
    table_view = display_df.copy()
    if search_query:
        table_view = table_view[table_view["student_id"].str.contains(search_query)]
        
    # Prepare clean presentation dataframe
    table_render = table_view[[
        "fairness_rank",
        "student_id",
        "student_group",
        "need_score",
        "previous_gpa",
        "attendance_rate",
        "midterm_exam_score",
        "late_submissions_count",
        "baseline_selected",
        "fairness_selected",
        "primary_risk_factor"
    ]].copy()
    
    # Format selection column
    def format_status(row):
        if row["fairness_selected"] == 1:
            if row["baseline_selected"] == 0:
                return "Selected (Fairness Adjusted)"
            return "Selected for Support"
        return "Standard Track"
        
    table_render["Status"] = table_render.apply(format_status, axis=1)
    
    table_render = table_render.rename(columns={
        "fairness_rank": "Rank",
        "student_id": "Student ID",
        "student_group": "Group",
        "need_score": "Need Score (0-100)",
        "previous_gpa": "Previous CGPA (/10)",
        "attendance_rate": "Attendance (%)",
        "midterm_exam_score": "Midterm Exam",
        "late_submissions_count": "Late Tasks",
        "primary_risk_factor": "Primary Associated Indicator"
    })
    
    table_render = table_render.drop(columns=["baseline_selected", "fairness_selected"])
    
    st.dataframe(
        table_render,
        use_container_width=True,
        hide_index=True,
        height=480
    )


# ------------------------------------------------------------------------------
# TAB 2: WHY? (Individual Attribution & Non-Causal Explanation)
# ------------------------------------------------------------------------------
with tab_why:
    st.markdown("### Individual Support-Need Attribution")
    st.caption("Inspect why the model assigned a specific support-need score. Strictly adheres to non-causal educator reporting.")
    
    # Student Selector
    all_student_ids = ranked_df["student_id"].tolist()
    default_id = search_query if search_query in all_student_ids else all_student_ids[0]
    
    sel_student_id = st.selectbox(
        "Select student to inspect:",
        options=all_student_ids,
        index=all_student_ids.index(default_id)
    )
    
    student_row = ranked_df[ranked_df["student_id"] == sel_student_id].iloc[0]
    explanation = explainer.explain_student(student_row)
    
    # Top Student Info Bar
    c_id, c_group, c_score, c_status = st.columns(4)
    with c_id:
        st.markdown(f"**Student ID:** `{student_row['student_id']}`")
    with c_group:
        st.markdown(f"**Eligible Group:** `{student_row['student_group']}`")
    with c_score:
        st.markdown(f"**Support Need Score:** `{student_row['need_score']}/100`")
    with c_status:
        if student_row["fairness_selected"] == 1:
            st.markdown('<span class="badge-selected">Selected for Intervention</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-standard">Standard Track</span>', unsafe_allow_html=True)
            
    st.markdown("---")
    
    col_bullets, col_chart = st.columns([1, 1])
    
    with col_bullets:
        st.markdown("#### Primary Associated Factors")
        st.markdown(
            '<div style="font-size: 13px; color: #475569; margin-bottom: 12px;">'
            'Factors associated with the model\'s support-need estimation (relative to cohort baseline):'
            '</div>',
            unsafe_allow_html=True
        )
        
        if explanation["top_contributing_factors"]:
            for factor in explanation["top_contributing_factors"]:
                st.markdown(f"- **{factor}**")
        else:
            st.markdown("- *No severe risk factors identified. Indicators align closely with cohort average.*")
            
        if explanation["protective_factors"]:
            st.markdown("#### Protective / Stabilizing Factors")
            for prot in explanation["protective_factors"]:
                st.markdown(f"- {prot}")
                
        st.markdown(
            f'<div style="font-size: 11px; color: #64748B; margin-top: 18px; line-height: 1.4; border-top: 1px solid #E2E8F0; padding-top: 8px;">'
            f'<strong>Educational Language Standard:</strong> {explanation["language_disclaimer"]}'
            f'</div>',
            unsafe_allow_html=True
        )
        
    with col_chart:
        st.markdown("#### Factor Attribution Waterfall")
        # Build clean horizontal contribution bar chart
        contribs = explanation["detailed_contributions"]
        labels = [c["label"] for c in contribs]
        values = [c["risk_contribution"] for c in contribs]
        colors = ["#D97706" if v > 0 else "#0D9488" for v in values]
        
        fig_attr = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.2f}" for v in values],
            textposition="outside"
        ))
        
        fig_attr.update_layout(
            margin=dict(l=10, r=30, t=20, b=20),
            height=340,
            xaxis_title="Log-Odds Contribution to Support Need",
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis=dict(zeroline=True, zerolinecolor="#CBD5E1", gridcolor="#F1F5F9"),
        )
        st.plotly_chart(fig_attr, use_container_width=True)
        
    # Full indicator scorecard
    st.markdown("#### Indicator Scorecard vs Cohort Baseline")
    ind_df = pd.DataFrame([
        {
            "Indicator": c["label"],
            "Student Value": f"{c['value']} {c['unit']}",
            "Cohort Average": f"{c['cohort_mean']} {c['unit']}",
            "Relative Assessment": c["qualifier"]
        }
        for c in contribs
    ])
    st.dataframe(ind_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 3: IS THIS FAIR? (Auditing Group Recall & Marginal Swaps)
# ------------------------------------------------------------------------------
with tab_fair:
    st.markdown("### Group Recall & Fairness Audit")
    st.caption("Evaluating support-needed recall across eligible student cohorts to eliminate unjustified systemic disparities.")
    
    # 1. Plotly Group-Wise Recall Chart with Direct Gap Annotation
    b_table = baseline_metrics["table"]
    f_table = fair_metrics["table"]
    
    groups = b_table["group"].tolist()
    b_recalls = b_table["recall_pct"].tolist()
    f_recalls = f_table["recall_pct"].tolist()
    
    fig_fairness = go.Figure()
    
    # Baseline Recall Bars
    fig_fairness.add_trace(go.Bar(
        name="Baseline Selection (Pure Score)",
        x=groups,
        y=b_recalls,
        marker_color="#94A3B8",
        text=[f"{r:.1f}%" for r in b_recalls],
        textposition="inside",
        textfont=dict(color="#FFFFFF", size=12)
    ))
    
    # Fairness-Aware Recall Bars
    fig_fairness.add_trace(go.Bar(
        name="Fairness-Aware Selection",
        x=groups,
        y=f_recalls,
        marker_color="#1E3A5F",
        text=[f"{r:.1f}%" for r in f_recalls],
        textposition="inside",
        textfont=dict(color="#FFFFFF", size=12)
    ))
    
    # Direct Callout Annotation for Observed Recall Gap
    fig_fairness.add_annotation(
        text=f"Observed Recall Gap: <b>{ranking_summary['fairness_recall_gap']}%</b><br>(reduced from {ranking_summary['baseline_recall_gap']}%)",
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        bgcolor="#FEF3C7",
        bordercolor="#FCD34D",
        borderwidth=1,
        borderpad=6,
        font=dict(size=12, color="#92400E"),
        align="right"
    )
    
    fig_fairness.update_layout(
        barmode="group",
        title=dict(text="Group-Wise Support-Needed Recall (%)", font=dict(size=15, color="#1E3A5F")),
        yaxis=dict(title="Recall (%)", range=[0, 100], gridcolor="#F1F5F9"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
    )
    st.plotly_chart(fig_fairness, use_container_width=True)
    
    # 2. Side-by-side comparative table
    st.markdown("#### Cohort Fairness Metrics Table")
    comp_df = compare_fairness_policies(baseline_metrics, fair_metrics)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # 3. Transparent Swap Audit Log
    st.markdown("#### Fairness Re-ranking Audit Log")
    if ranking_summary["swaps_log"]:
        st.markdown(
            f"<div style='font-size: 13px; color: #475569; margin-bottom: 8px;'>"
            f"Completed <strong>{len(ranking_summary['swaps_log'])}</strong> bounded marginal swaps "
            f"to reduce the observed recall gap by <strong>{ranking_summary['gap_reduction']}% pts</strong> "
            f"while preserving high-need candidates (max delta: {max_score_delta} pts)."
            f"</div>",
            unsafe_allow_html=True
        )
        swaps_df = pd.DataFrame(ranking_summary["swaps_log"])[[
            "swap_id",
            "student_in",
            "group_in",
            "score_in",
            "student_out",
            "group_out",
            "score_out",
            "score_delta",
            "previous_gap",
            "new_gap"
        ]].rename(columns={
            "swap_id": "Swap #",
            "student_in": "Student Prioritized (IN)",
            "group_in": "Group IN",
            "score_in": "Score IN",
            "student_out": "Student Deferred (OUT)",
            "group_out": "Group OUT",
            "score_out": "Score OUT",
            "score_delta": "Score Delta",
            "previous_gap": "Gap Before (%)",
            "new_gap": "Gap After (%)"
        })
        st.dataframe(swaps_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("*No swaps were required; baseline ranking already met the fairness parity target.*")


# ------------------------------------------------------------------------------
# TAB 4: MODEL PERFORMANCE & ROBUSTNESS
# ------------------------------------------------------------------------------
with tab_model:
    st.markdown("### Model Diagnostics & Robustness Evaluation")
    st.caption("Verification of statistical fidelity, top-20% ranking quality, and stability under feature perturbation.")
    
    # Classification Metrics on Test Split
    y_test_true = test_pred["support_needed"].values
    y_test_prob = test_pred["need_probability"].values
    clf_metrics = compute_classification_metrics(y_test_true, y_test_prob)
    rank_metrics = compute_ranking_metrics(full_pred, capacity_pct=capacity_pct)
    
    c_acc, c_prec, c_rec, c_auc, c_ndcg = st.columns(5)
    with c_acc:
        st.metric("Test Accuracy", f"{clf_metrics['accuracy'] * 100:.1f}%")
    with c_prec:
        st.metric("Test Precision", f"{clf_metrics['precision'] * 100:.1f}%")
    with c_rec:
        st.metric("Test Recall", f"{clf_metrics['recall'] * 100:.1f}%")
    with c_auc:
        st.metric("ROC-AUC Score", f"{clf_metrics['roc_auc']:.3f}")
    with c_ndcg:
        st.metric(f"NDCG@{int(capacity_pct)}%", f"{rank_metrics[f'ndcg_at_{int(capacity_pct)}pct']:.3f}")
        
    st.markdown("---")
    
    # Robustness Check
    st.markdown("#### Input Perturbation Robustness Check")
    st.markdown(
        '<div style="font-size: 13px; color: #475569; margin-bottom: 12px;">'
        'Assessing whether student rankings and fairness metrics remain stable under realistic '
        'input perturbations (simulating +/- 3% measurement noise in attendance or homework logs).'
        '</div>',
        unsafe_allow_html=True
    )
    
    with st.spinner("Running perturbation simulations..."):
        robustness = evaluate_system_robustness(
            raw_df,
            model,
            capacity_pct=capacity_pct,
            noise_std=0.03,
            n_simulations=5
        )
        
    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rank Correlation (Spearman)</div>
            <div class="metric-value">{robustness['mean_spearman_correlation']:.3f}</div>
            <div class="metric-subtext">High score order stability (> 0.95)</div>
        </div>
        """, unsafe_allow_html=True)
    with r_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Selection Overlap (Jaccard)</div>
            <div class="metric-value">{robustness['mean_jaccard_selection_overlap'] * 100:.1f}%</div>
            <div class="metric-subtext">Overlap in prioritized 20% pool</div>
        </div>
        """, unsafe_allow_html=True)
    with r_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Perturbed Fairness Gap</div>
            <div class="metric-value">{robustness['mean_perturbed_fairness_gap']}%</div>
            <div class="metric-subtext">Unperturbed gap: {robustness['unperturbed_fairness_gap']}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("#### Feature Coefficients (Standardized Scale)")
    coefs = model.get_feature_coefficients()
    coef_df = pd.DataFrame([
        {"Feature": k, "Weight / Relative Contribution": round(v, 3)}
        for k, v in coefs.items()
    ]).sort_values(by="Weight / Relative Contribution")
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
