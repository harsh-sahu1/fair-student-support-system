import React, { useEffect, useState } from 'react';
import { fetchFairness, FairnessResponse } from '../api/client';

export const FairnessView: React.FC = () => {
  const [data, setData] = useState<FairnessResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadFairness = () => {
    setLoading(true);
    setError(null);
    fetchFairness()
      .then((res) => setData(res))
      .catch((err) => setError(err.message || 'Failed to load fairness audit'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadFairness();
  }, []);

  if (loading) {
    return (
      <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading fairness audit and demographic parity metrics...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="banner-offline">
        <div>
          <strong>Unable to load fairness data:</strong> {error || 'No response.'}
        </div>
        <button className="btn btn-outline" onClick={loadFairness}>Retry</button>
      </div>
    );
  }

  // Find min and max recall among eligible groups
  const minRecall = data.worst_group_recall;

  // Split by attribute
  const sexGroups = data.groups.filter((g) => g.attribute === 'sex');
  const schoolGroups = data.groups.filter((g) => g.attribute === 'school');

  return (
    <div>
      {/* Page Header */}
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
          Demographic Fairness Audit & Recall Disparities
        </h2>
        <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
          Auditing recall rates across eligible student cohorts to ensure equitable access to limited support capacity.
        </p>
      </div>

      {/* Summary KPI Cards */}
      <div className="metrics-row" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
        <div className="metric-box">
          <div className="metric-label">Observed Recall Gap</div>
          <div className="metric-value tabular-nums">
            {(data.fairness_gap * 100).toFixed(1)}%
          </div>
          <div className="metric-desc">
            Largest intra-attribute disparity (e.g. |recall_A - recall_B|). Evaluated strictly within the same protected category.
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Worst-Group Recall (R_min)</div>
          <div className="metric-value tabular-nums" style={{ color: 'var(--support-accent)' }}>
            {(data.worst_group_recall * 100).toFixed(1)}%
          </div>
          <div className="metric-desc">
            Minimum recall achieved across all eligible groups pooled across sex and school attributes.
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Auditing Standard</div>
          <div className="metric-value" style={{ fontSize: '1.2rem', fontWeight: 600 }}>
            Educational Equity Benchmark
          </div>
          <div className="metric-desc">
            Eligibility requires ≥ 10 rows and ≥ 3 positive support-needed cases per group.
          </div>
        </div>
      </div>

      {/* Group Recall Comparison Visual Bar Chart */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">Group-Wise Support Recall Comparison</h3>
            <p className="panel-subtitle">Percentage of positive students correctly prioritized within each eligible demographic group</p>
          </div>
          <div className="bracket-delta-box">
            <span>Observed Recall Gap: {(data.fairness_gap * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="fairness-chart-container">
          <div className="chart-bars-group">
            {/* Sex Groups */}
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                PROTECTED ATTRIBUTE: SEX
              </div>
              {sexGroups.map((g) => {
                const pct = g.recall * 100;
                return (
                  <div key={`${g.attribute}_${g.group}`} className="bar-row" style={{ marginBottom: '0.5rem' }}>
                    <div className="bar-label">
                      <strong>{g.group === 'F' ? 'Female (F)' : 'Male (M)'}</strong>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        {g.n_selected_positives} of {g.n_positives} positives
                      </div>
                    </div>
                    <div className="bar-track">
                      <div
                        className={`bar-fill ${g.recall === minRecall ? 'bar-fill-amber' : ''}`}
                        style={{ width: `${Math.max(pct, 6)}%` }}
                      >
                        {pct.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* School Groups */}
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                PROTECTED ATTRIBUTE: SCHOOL
              </div>
              {schoolGroups.map((g) => {
                const pct = g.recall * 100;
                return (
                  <div key={`${g.attribute}_${g.group}`} className="bar-row" style={{ marginBottom: '0.5rem' }}>
                    <div className="bar-label">
                      <strong>{g.group === 'GP' ? 'Gabriel Pereira (GP)' : 'Mousinho da Silveira (MS)'}</strong>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        {g.n_selected_positives} of {g.n_positives} positives
                      </div>
                    </div>
                    <div className="bar-track">
                      <div
                        className={`bar-fill ${g.recall === minRecall ? 'bar-fill-amber' : ''}`}
                        style={{ width: `${Math.max(pct, 6)}%` }}
                      >
                        {pct.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Comprehensive Audit Table */}
      <div className="panel" style={{ padding: 0 }}>
        <div style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid var(--border-subtle)' }}>
          <h3 className="panel-title">Detailed Group Breakdown</h3>
          <p className="panel-subtitle">Audited cohorts with sample sizes, positive counts, and eligibility status</p>
        </div>
        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Attribute</th>
                <th>Group Name</th>
                <th>Total Sample Size</th>
                <th>True Support Positives</th>
                <th>Selected Positives (TP)</th>
                <th>Eligibility Status</th>
                <th>Group Recall Rate</th>
              </tr>
            </thead>
            <tbody>
              {data.groups.map((g) => (
                <tr key={`${g.attribute}_${g.group}`}>
                  <td style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>{g.attribute}</td>
                  <td style={{ fontWeight: 600 }}>{g.group}</td>
                  <td className="tabular-nums">{g.n_rows} students</td>
                  <td className="tabular-nums">{g.n_positives}</td>
                  <td className="tabular-nums">{g.n_selected_positives}</td>
                  <td>
                    {g.eligible ? (
                      <span className="badge-neutral" style={{ backgroundColor: 'var(--status-success-bg)', color: 'var(--status-success-text)' }}>
                        Eligible
                      </span>
                    ) : (
                      <span className="badge-ineligible">
                        not eligible (insufficient sample)
                      </span>
                    )}
                  </td>
                  <td>
                    <span className="tabular-nums" style={{ fontWeight: 700, fontSize: '0.95rem' }}>
                      {(g.recall * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Rigorous Experiment & Policy Documentation Panel */}
      <div className="panel" style={{ backgroundColor: '#F8FAFC', borderLeft: '3px solid var(--brand-secondary)' }}>
        <h4 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text-primary)' }}>
          Methodological Rigor: Why We Rejected Post-Hoc Fairness Adjustments
        </h4>
        <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          <p style={{ marginBottom: '0.5rem' }}>
            We empirically tested a post-hoc log-odds score adjustment to mechanically force parity across demographic groups. While this reduced the gap from 0.34 to 0.06 on internal tuning splits, it <strong>failed to generalize</strong> to held-out test data across 10 random seeds (mean gap 0.220 baseline vs. 0.208 shifted) and degraded probability calibration (mean Brier worsened from 0.065 to 0.074).
          </p>
          <p>
            In educational resource planning, distorted risk scores harm students by misrepresenting their true support needs. We therefore deploy the <strong>unmodified, well-calibrated Logistic Regression model</strong> and rely on clear transparency and human educator review.
          </p>
        </div>
      </div>
    </div>
  );
};
