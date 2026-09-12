import numpy as np
import pandas as pd
from backend.app.config import DATA_PATH, TARGET_COLUMN
from backend.app.ml.data_processing import load_raw_dataset, prepare_dataset, get_train_validation_split
from backend.app.ml.model import StudentSupportModel
from backend.app.ml.ranking import rank_students, compute_capacity_slots
from backend.app.ml.fairness import evaluate_fairness
from backend.app.ml.evaluation import compute_metrics

raw_df = load_raw_dataset(str(DATA_PATH))
df = prepare_dataset(raw_df)
train_df, val_df = get_train_validation_split(df, test_size=0.25, random_state=42)
model = StudentSupportModel(random_state=42)
model.fit(train_df)
val_pred = model.attach_predictions(val_df)

ranked_base, sum_b = rank_students(val_pred, capacity_fraction=0.20)
fb = evaluate_fairness(ranked_base)
cb = compute_metrics(ranked_base[TARGET_COLUMN].values, ranked_base['probability'].values, ranked_base['selected'].values)

print("BASELINE:")
print(f"Top-20% Recall: {cb['recall_at_capacity']*100:.1f}%")
print(f"Worst-Group Recall: {fb['worst_group_recall']*100:.1f}%")
print(f"Recall Gap: {fb['fairness_gap']*100:.1f}%")
print(f"Brier Score: {cb['brier_score']:.3f}")
for g in fb['groups']:
    print(f"  {g['attribute']} {g['group']}: sel_pos={g['n_selected_positives']}/{g['n_positives']}, recall={g['recall']:.3f}")

# Test candidate proportional capacity allocation:
# In validation, N=99, K=20.
# If we test group-proportional capacity allocation on school:
# GP fraction in cohort: 83/99 = 83.8% -> 17 slots
# MS fraction in cohort: 16/99 = 16.2% -> 3 slots
gp = ranked_base[ranked_base['school'] == 'GP'].sort_values('probability', ascending=False)
ms = ranked_base[ranked_base['school'] == 'MS'].sort_values('probability', ascending=False)
sel_ids = set(gp.iloc[:17]['student_id']).union(set(ms.iloc[:3]['student_id']))
df_fa1 = ranked_base.copy()
df_fa1['selected'] = df_fa1['student_id'].isin(sel_ids)
ffa1 = evaluate_fairness(df_fa1)
cfa1 = compute_metrics(df_fa1[TARGET_COLUMN].values, df_fa1['probability'].values, df_fa1['selected'].values)

print("\nFAIRNESS-AWARE (Proportional Capacity Allocation 17 GP / 3 MS):")
print(f"Top-20% Recall: {cfa1['recall_at_capacity']*100:.1f}%")
print(f"Worst-Group Recall: {ffa1['worst_group_recall']*100:.1f}%")
print(f"Recall Gap: {ffa1['fairness_gap']*100:.1f}%")
print(f"Brier Score: {cfa1['brier_score']:.3f}")
for g in ffa1['groups']:
    print(f"  {g['attribute']} {g['group']}: sel_pos={g['n_selected_positives']}/{g['n_positives']}, recall={g['recall']:.3f}")
