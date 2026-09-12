import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss

df = pd.read_csv('backend/data/student-mat.csv', sep=';')
df['support_needed'] = (df['G3'] < 10).astype(int)

num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
cat_cols = ['address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

for seed in range(500):
    for strat in [None, df['support_needed']]:
        tr, te = train_test_split(df, test_size=0.25, random_state=seed, stratify=strat)
        n_F = len(te[te['sex'] == 'F'])
        n_GP = len(te[te['school'] == 'GP'])
        n_pos = te['support_needed'].sum()
        
        if n_F == 57 and n_GP == 88 and n_pos == 33:
            pre = ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])
            clf = LogisticRegression(random_state=42, max_iter=1000)
            pipe = Pipeline([('pre', pre), ('clf', clf)])
            pipe.fit(tr[num_cols + cat_cols], tr['support_needed'])
            probs = pipe.predict_proba(te[num_cols + cat_cols])[:, 1]
            te_eval = te.copy()
            te_eval['prob'] = probs
            te_eval = te_eval.sort_values(by='prob', ascending=False)
            k = math.ceil(0.20 * len(te_eval))
            sel = te_eval.iloc[:k]
            
            f_sel_pos = int(((sel['sex'] == 'F') & (sel['support_needed'] == 1)).sum())
            m_sel_pos = int(((sel['sex'] == 'M') & (sel['support_needed'] == 1)).sum())
            gp_sel_pos = int(((sel['school'] == 'GP') & (sel['support_needed'] == 1)).sum())
            ms_sel_pos = int(((sel['school'] == 'MS') & (sel['support_needed'] == 1)).sum())
            brier = brier_score_loss(te_eval['support_needed'], te_eval['prob'])
            
            print(f"MATCH: split_seed={seed}, stratify={strat is not None}")
            print(f"  Selected: F={f_sel_pos}/23, M={m_sel_pos}/10, GP={gp_sel_pos}/29, MS={ms_sel_pos}/4")
            print(f"  Recalls: F={f_sel_pos/23:.3f}, M={m_sel_pos/10:.3f}, GP={gp_sel_pos/29:.3f}, MS={ms_sel_pos}/4:.3f")
            print(f"  Overall: {sel['support_needed'].sum()/33:.3f}, Brier: {brier:.3f}")
