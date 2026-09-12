"""
Data processing module for Fair Student-Support Prioritization.
Handles data loading, synthetic data generation, validation, and train/test splitting.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "attendance_rate",
    "assignment_completion_rate",
    "midterm_exam_score",
    "previous_gpa",
    "lms_engagement_hours",
    "late_submissions_count",
    "study_hours_weekly",
]

SENSITIVE_COL = "student_group"
TARGET_COL = "support_needed"
ID_COL = "student_id"


def generate_synthetic_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a reproducible, realistic synthetic educational dataset.
    
    Variables included:
    - student_id: Unique student identifier (e.g. S0001)
    - student_group: Eligible cohort group ('Group A', 'Group B', 'Group C')
    - attendance_rate: Attendance percentage (50% - 100%)
    - assignment_completion_rate: Percentage of assignments turned in (40% - 100%)
    - midterm_exam_score: Score on recent midterm examination (30 - 100)
    - previous_gpa: Cumulative Grade Point Average from previous term (1.5 - 4.0)
    - lms_engagement_hours: Weekly hours logged on digital learning management system (1 - 25)
    - late_submissions_count: Number of assignments submitted after deadline (0 - 12)
    - study_hours_weekly: Self-reported weekly independent study hours (2 - 30)
    
    Target Formulation (Defensible Ground Truth):
    A student is identified as genuinely needing support (support_needed = 1) if
    they exhibit composite academic distress (low exams/GPA) and disengagement 
    (low attendance/high late submissions).
    
    Systemic Pattern:
    To reflect real-world disparity where raw engagement features may have measurement
    discrepancies across cohorts (e.g. unequal digital access for Group B), a naive model
    evaluating raw features tends to have an unmitigated recall gap on Group B unless
    audited and mitigated.
    """
    np.random.seed(seed)
    
    # 1. Cohort demographic distribution: Group A (45%), Group B (35%), Group C (20%)
    groups = np.random.choice(["Group A", "Group B", "Group C"], size=n_samples, p=[0.45, 0.35, 0.20])
    student_ids = [f"S{i:04d}" for i in range(1, n_samples + 1)]
    
    # 2. Baseline academic metrics
    # Normal distributions with realistic educational bounds (Indian 10-point CGPA scale)
    attendance_rate = np.clip(np.random.normal(loc=82, scale=12, size=n_samples), 50.0, 100.0)
    assignment_completion_rate = np.clip(np.random.normal(loc=78, scale=14, size=n_samples), 40.0, 100.0)
    previous_gpa = np.clip(np.random.normal(loc=7.3, scale=1.35, size=n_samples), 3.5, 10.0)
    study_hours_weekly = np.clip(np.random.normal(loc=12, scale=5, size=n_samples), 2.0, 32.0)
    
    # Correlated academic performance: midterm score correlates with study hours, completion, and previous GPA (out of 10)
    midterm_noise = np.random.normal(loc=0, scale=8, size=n_samples)
    midterm_raw = (
        0.35 * (assignment_completion_rate) +
        0.30 * (previous_gpa / 10.0 * 100) +
        0.20 * (study_hours_weekly / 30.0 * 100) +
        0.15 * (attendance_rate) +
        midterm_noise
    )
    midterm_exam_score = np.clip(midterm_raw, 30.0, 100.0)
    
    # LMS engagement and late submissions
    lms_engagement_hours = np.clip(
        (assignment_completion_rate / 100.0) * 16 + np.random.normal(loc=4, scale=3, size=n_samples),
        1.0, 25.0
    )
    
    # Late submissions count inversely related to attendance and assignment rate
    late_submissions_lambda = np.clip(12 - (attendance_rate / 10.0), 0.5, 9.0)
    late_submissions_count = np.random.poisson(lam=late_submissions_lambda)
    late_submissions_count = np.clip(late_submissions_count, 0, 12)
    
    # 3. Defensible Ground Truth Definition: Multi-criteria Academic & Behavioral Need
    # Academic distress indicator: exam failing or CGPA critical (< 5.8 out of 10.0)
    academic_distress = (midterm_exam_score < 62.0) * 0.45 + (previous_gpa < 5.8) * 0.30 + (assignment_completion_rate < 65.0) * 0.25
    # Behavioral disengagement indicator: poor attendance and chronic late submissions
    behavioral_risk = (attendance_rate < 74.0) * 0.45 + (late_submissions_count >= 4) * 0.35 + (lms_engagement_hours < 5.5) * 0.20
    
    # Latent true support need score (0 to 1)
    latent_need = 0.55 * academic_distress + 0.45 * behavioral_risk + np.random.normal(loc=0, scale=0.08, size=n_samples)
    
    # Systemic digital access noise for Group B:
    # Some Group B students with high actual need show mixed digital telemetry due to off-campus bandwidth limits
    mask_grp_b = (groups == "Group B")
    latent_need[mask_grp_b] += np.where(midterm_exam_score[mask_grp_b] < 60, 0.12, 0.0)
    
    # Threshold for ground truth: ~25-28% of total student cohort truly need intervention
    need_threshold = np.percentile(latent_need, 73)
    support_needed = (latent_need >= need_threshold).astype(int)
    
    df = pd.DataFrame({
        ID_COL: student_ids,
        SENSITIVE_COL: groups,
        "attendance_rate": np.round(attendance_rate, 1),
        "assignment_completion_rate": np.round(assignment_completion_rate, 1),
        "midterm_exam_score": np.round(midterm_exam_score, 1),
        "previous_gpa": np.round(previous_gpa, 2),
        "lms_engagement_hours": np.round(lms_engagement_hours, 1),
        "late_submissions_count": late_submissions_count,
        "study_hours_weekly": np.round(study_hours_weekly, 1),
        TARGET_COL: support_needed,
    })
    
    return df


def load_or_create_dataset(filepath: str = "data/students.csv", n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Loads dataset from filepath if present; otherwise creates synthetic dataset and saves it.
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = generate_synthetic_dataset(n_samples=n_samples, seed=seed)
        df.to_csv(filepath, index=False)
    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    """
    Splits data into train and test sets, stratified by target and group to avoid leakage and sampling bias.
    """
    strat_key = df[TARGET_COL].astype(str) + "_" + df[SENSITIVE_COL].astype(str)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat_key)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
