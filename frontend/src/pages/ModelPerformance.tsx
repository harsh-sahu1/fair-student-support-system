import React, { useEffect, useState } from 'react';
import { fetchModelPerformance, ModelPerformanceResponse } from '../api/client';

export const ModelPerformance: React.FC = () => {
  const [data, setData] = useState<ModelPerformanceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadPerf = () => {
    setLoading(true);
    setError(null);
    fetchModelPerformance()
      .then((res) => setData(res))
      .catch((err) => setError(err.message || 'Failed to load model performance'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadPerf();
  }, []);

  if (loading) {
    return (
      <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading model calibration and robustness benchmark data...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="banner-offline">
        <div>
          <strong>Unable to load model performance metrics:</strong> {error || 'No response.'}
        </div>
        <button className="btn btn-outline" onClick={loadPerf}>Retry</button>
      </div>
    );
  }

  const { robustness, overview } = data;

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
          Model Calibration, Validation & Robustness Benchmark
        </h2>
        <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
          Detailed performance metrics across single-split holdout and 10-seed cross-validation on the UCI Student Performance benchmark.
        </p>
      </div>

      {/* Validation Holdout Metrics */}
      <div className="metrics-row">
        <div className="metric-box">
          <div className="metric-label">Holdout Brier Score</div>
          <div className="metric-value tabular-nums" style={{ color: 'var(--brand-primary)' }}>
            {overview.brier_score.toFixed(3)}
          </div>
          <div className="metric-desc">
            Calibration metric (lower is better; 0.0 = perfect probabilistic accuracy).
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Holdout Recall @ 20%</div>
          <div className="metric-value tabular-nums">
            {(overview.overall_recall_at_capacity * 100).toFixed(1)}%
          </div>
          <div className="metric-desc">
            Intervention capture rate on the held-out validation cohort of 99 students.
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Worst-Group Recall (R_min)</div>
          <div className="metric-value tabular-nums">
            {(overview.worst_group_recall * 100).toFixed(1)}%
          </div>
          <div className="metric-desc">
            Lowest positive recall rate among eligible demographic groups.
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Observed Recall Gap</div>
          <div className="metric-value tabular-nums">
            {(overview.fairness_gap * 100).toFixed(1)}%
          </div>
          <div className="metric-desc">
            Maximum disparity between eligible groups within the same demographic attribute.
          </div>
        </div>
      </div>

      {/* 10-Seed Robustness Cross-Validation Summary Table */}
      <div className="panel" style={{ padding: 0 }}>
        <div style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid var(--border-subtle)' }}>
          <h3 className="panel-title">10-Seed Cross-Validation Robustness Audit</h3>
          <p className="panel-subtitle">
            Mean and standard deviation computed across 10 independent random train/validation splits (seeds 0 through 9)
          </p>
        </div>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Evaluation Metric</th>
                <th>Validation Split (Seed=42)</th>
                <th>10-Seed Mean (± Std Dev)</th>
                <th>Educational Interpretation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Recall @ Top-20% Capacity</strong></td>
                <td className="tabular-nums">{(overview.overall_recall_at_capacity * 100).toFixed(1)}%</td>
                <td className="tabular-nums">
                  <strong>{(robustness.recall_at_capacity.mean * 100).toFixed(1)}%</strong>
                  <span style={{ color: 'var(--text-muted)' }}> (± {(robustness.recall_at_capacity.std * 100).toFixed(1)}%)</span>
                </td>
                <td style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                  Consistent prioritization efficiency capturing ~58-60% of students needing support within a 20% allocation.
                </td>
              </tr>
              <tr>
                <td><strong>Worst-Group Recall (R_min)</strong></td>
                <td className="tabular-nums">{(overview.worst_group_recall * 100).toFixed(1)}%</td>
                <td className="tabular-nums">
                  <strong>{(robustness.worst_group_recall.mean * 100).toFixed(1)}%</strong>
                  <span style={{ color: 'var(--text-muted)' }}> (± {(robustness.worst_group_recall.std * 100).toFixed(1)}%)</span>
                </td>
                <td style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                  Assures that even the least-advantaged eligible cohort retains over 50% support recall on average.
                </td>
              </tr>
              <tr>
                <td><strong>Observed Recall Gap</strong></td>
                <td className="tabular-nums">{(overview.fairness_gap * 100).toFixed(1)}%</td>
                <td className="tabular-nums">
                  <strong>{(robustness.fairness_gap.mean * 100).toFixed(1)}%</strong>
                  <span style={{ color: 'var(--text-muted)' }}> (± {(robustness.fairness_gap.std * 100).toFixed(1)}%)</span>
                </td>
                <td style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                  Reflects expected variance in small groups (e.g. MS has only 11-16 validation rows); shows why gaps must be evaluated over time.
                </td>
              </tr>
              <tr>
                <td><strong>Brier Score (Calibration)</strong></td>
                <td className="tabular-nums">{overview.brier_score.toFixed(3)}</td>
                <td className="tabular-nums">
                  <strong>{robustness.brier_score.mean.toFixed(3)}</strong>
                  <span style={{ color: 'var(--text-muted)' }}> (± {robustness.brier_score.std.toFixed(3)})</span>
                </td>
                <td style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                  Excellent probabilistic calibration, meaning predicted probabilities closely match true empirical frequency.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Educational Context & Model Specs */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
        <div className="panel">
          <h3 className="panel-title" style={{ marginBottom: '0.5rem' }}>Pipeline Architecture</h3>
          <ul style={{ paddingLeft: '1.25rem', fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <li><strong>Model:</strong> scikit-learn <code>LogisticRegression(solver='lbfgs', max_iter=1000)</code>.</li>
            <li><strong>Numeric Preprocessing:</strong> <code>StandardScaler()</code> applied to 15 numeric features.</li>
            <li><strong>Categorical Preprocessing:</strong> <code>OneHotEncoder(handle_unknown='ignore')</code> applied to 15 categorical features.</li>
            <li><strong>Unbalanced Class Weights:</strong> Class weights were deliberately NOT balanced to preserve accurate Brier score calibration.</li>
            <li><strong>Zero Leakage:</strong> Sensitive attributes (<code>sex</code>, <code>school</code>) and future label (<code>G3</code>) are strictly excluded from the pipeline.</li>
          </ul>
        </div>

        <div className="panel">
          <h3 className="panel-title" style={{ marginBottom: '0.5rem' }}>Small-Sample Guidance</h3>
          <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {robustness.interpretability_note}
          </p>
          <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            * Evaluated on UCI Student Performance dataset (395 total students, 130 support-needed cases).
          </div>
        </div>
      </div>
    </div>
  );
};
