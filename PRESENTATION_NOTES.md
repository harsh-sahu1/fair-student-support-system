# Hackathon Presentation Notes: Fair Student-Support Prioritization ()

---

## 1. 30-Second Elevator Pitch
"Schools face a critical bottleneck: limited support capacity allows them to help only 20% of their students. But when schools rank students purely on raw test scores or generic risk models, systemic disparities often cause students from under-resourced cohorts to be overlooked. We built a fair, transparent decision-support system that identifies students who need help while slashing the Observed Recall Gap between student groups from 15.7% down to 4.9%—strictly respecting the 20% capacity limit, with zero black-box magic."

---

## 2. 60-Second Solution Brief
"Our platform acts as a calm, trustworthy instrument for academic counselors:
1. **Prediction:** A transparent, calibrated Logistic Regression model evaluates educational indicators (attendance, coursework completion, exams) to produce an intuitive 0–100 Support-Need Score. Crucially, sensitive group attributes are excluded from inputs to prevent disparate treatment.
2. **Fairness-Aware Selection:** Instead of blindly selecting the top 20% by score, our system audits group recall. It performs a bounded, deterministic marginal swap: swapping borderline students between over- and under-served cohorts only if score sacrifice is minimal ($\le 12$ pts) and group recall balance strictly improves.
3. **Non-Causal Explainability:** For every student, counselors see the exact statistical factors elevating need (e.g. 'Attendance rate is 18% below cohort average'), framed without unfounded causal accusations.
4. **Operations Dashboard:** An educator-first Streamlit interface designed like a clinical instrument rather than a consumer SaaS app, showing 'Who needs support?', 'Why?', and 'Is it fair?' in seconds."

---

## 3. Architecture & Technical Decision Rationale
- **Model Choice:** Standardized Logistic Regression instead of deep neural networks.
  - *Why:* Highly calibrated probabilities, zero black-box opacity, convex convergence, and direct linear log-odds coefficients ($w_i \cdot z_i$) that enable mathematically exact feature attribution.
- **Fairness Policy:** Bounded Marginal Swap Algorithm rather than adversarial debiasing or unconstrained demographic quotas.
  - *Why:* Demographic parity forces equal selection regardless of need; marginal swapping directly targets **equal recall** (identifying support-needing students equally across groups) while strictly preserving high-need students and honoring the hard 20% capacity ceiling.
- **Explainability:** Feature contribution z-score analysis.
  - *Why:* SHAP is computationally expensive and overkill for linear pipelines; exact linear attribution is fast, deterministic, and easily validated.

---

## 4. Design Direction
> *"Designed specifically as a calm, high-legibility operations instrument for academic counselors making real allocation decisions between class periods—deliberately free from AI-generated tropes like terracotta accents, blurry drop-shadow cards, or noisy neon graphs."*

---

## 5. Live Demo Flow (Step-by-Step for Judges)

1. **Step 1 — Overview & The Core Constraint (30s):**
   - Point out the KPI header: Total 1,000 students, Capacity: exactly 200 students (20.0%).
   - Show that the Baseline Recall Gap was **15.7%** (Group B was being missed).
   - Point out that our algorithm brought the Observed Recall Gap down to **4.9%** by making just 8 targeted marginal swaps.

2. **Step 2 — Tab 1: 'Who Needs Support?' (30s):**
   - Show the prioritization table. Point out the rank, need score, and status badges (`Selected for Support` vs `Selected (Fairness Adjusted)`).
   - Search for a student (e.g. `S0042`) to demonstrate responsive lookup.

3. **Step 3 — Tab 2: 'Why?' Individual Attribution (45s):**
   - Select student `S0042` or an adjusted candidate.
   - Point out the clean Factor Attribution Waterfall chart and the indicator scorecard comparing the student's metrics to the cohort mean.
   - **Crucial Talking Point:** Highlight the non-causal language banner: *"Factors associated with the model's estimation"*—we never claim a factor 'caused' a student's distress.

4. **Step 4 — Tab 3: 'Is This Fair?' (45s):**
   - Show the Group Recall Bar Chart with the **direct delta callout annotation** on the chart.
   - Show the Fairness Re-ranking Audit Log table: demonstrate full transparency showing exactly which student was brought in, which was deferred, and the minimal score delta ($< 3.5$ pts).

5. **Step 5 — Tab 4: Model Quality & Robustness (30s):**
   - Highlight the 0.884 ROC-AUC and 84.8% test accuracy.
   - Highlight the **Input Perturbation Robustness Check**: under $\pm 3\%$ measurement noise, the Spearman rank correlation remains $> 0.98$ and selection overlap is $> 91\%$.

---

## 6. Anticipated Judge Questions & Concise Answers

### Q1: "Why didn't you include student group as an input feature in the model?"
> **Answer:** "Including demographic or group identifiers directly in the predictive model risks direct algorithmic disparate treatment and proxy stereotyping. Instead, we use educational performance indicators for prediction, and reserve group attributes strictly for downstream auditing and constrained fairness adjustments."

### Q2: "Isn't swapping students lowering the academic standard of who gets support?"
> **Answer:** "No. Our swap algorithm has a strict `max_score_delta` bounding condition (default 12 points). It only touches marginal students right at the 20% cutoff boundary where prediction uncertainty is highest. A student with a score of 95 is never displaced for a student with a score of 50. In our empirical run, the average need score of selected students remained virtually identical (81.4 vs 80.9)."

### Q3: "Why did you optimize for Recall parity rather than Demographic Parity?"
> **Answer:** "Demographic parity demands that 20% of every group be selected regardless of actual need, which is inefficient and unfair if one cohort has higher genuine distress. Equal Recall ensures that among students who *genuinely need help*, every group has an equal chance of being identified."

### Q4: "How do you know this model won't make hallucinations or fabricate explanations?"
> **Answer:** "We use zero LLM APIs or generative text models for explanations. Every explanation is computed deterministically from the standardized coefficients of our Logistic Regression pipeline ($w_i \cdot z_i$) and mapped to structured, educator-vetted educational indicators."

### Q5: "What happens if a school has 15% capacity instead of 20%?"
> **Answer:** "Our capacity controller is fully parametric. Slide the capacity slider to 15% in the sidebar, and the system dynamically recalibrates: exactly $\lfloor N \times 0.15 \rfloor$ slots are allocated, the marginal swaps re-run deterministically, and the audit updates instantly."

---

## 7. Five Critical Takeaways to Remember During Presentation
1. **The Core Metric:** We optimize for **Recall Parity** (reducing the Observed Recall Gap), not arbitrary demographic quotas.
2. **Strict Capacity Constraint:** Exactly 20% of eligible students are selected ($\lfloor N \times 0.20 \rfloor$).
3. **No Disparate Treatment:** Group attribute is never an input feature to the ML model.
4. **Bounded Swap:** We only swap borderline candidates within a strict score tolerance, preserving high-need students.
5. **Human in the Loop:** The system is an educator decision-support tool, not an automated punishment engine.
