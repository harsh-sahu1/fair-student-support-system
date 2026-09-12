import React, { useEffect, useState } from 'react';
import { fetchOverview, OverviewResponse } from '../api/client';

interface OverviewProps {
  onNavigateToPriorities?: () => void;
  onNavigateToFairness?: () => void;
}

export const Overview: React.FC<OverviewProps> = ({
  onNavigateToPriorities,
  onNavigateToFairness,
}) => {
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadOverview = () => {
    setLoading(true);
    setError(null);
    fetchOverview()
      .then((res) => setData(res))
      .catch((err) => setError(err.message || 'Failed to load cohort overview'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadOverview();
  }, []);

  if (loading) {
    return (
      <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading cohort allocation metrics...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="banner-offline">
        <div>
          <strong>Unable to load dashboard data:</strong> {error || 'No response from server.'}
        </div>
        <button className="btn btn-outline" onClick={loadOverview}>
          Retry Connection
        </button>
      </div>
    );
  }

  return (
    <div>
      {/* Ethical Framing Advisory Notice */}
      <div className="banner-ethical">
        <strong>Decision Support Advisory:</strong> This platform is an educator-facing decision instrument.
        Scores indicate statistical patterns from historical data under fixed intervention limits; they are
        neither psychological diagnoses nor automated allocation mandates. Human review by school teams is
        required prior to final intervention placement.
      </div>

      {/* Hero Metric Row */}
      <div className="metrics-row">
        {/* Capacity Allocation */}
        <div className="metric-box">
          <div className="metric-label">Allocated Support Slots</div>
          <div className="metric-value tabular-nums" style={{ color: 'var(--support-accent)' }}>
            {data.num_selected}
            <span style={{ fontSize: '1rem', fontWeight: 500, color: 'var(--text-muted)' }}>
              {' '}/ {data.total_students}
            </span>
          </div>
          <div className="metric-desc">
            Exact 20% capacity rule: <code>ceil(0.20 × N)</code> = {data.num_selected} students prioritized.
          </div>
        </div>

        {/* Observed Recall Gap */}
        <div className="metric-box">
          <div className="metric-label">Observed Recall Gap</div>
          <div className="metric-value tabular-nums">
            {(data.fairness_gap * 100).toFixed(1)}%
          </div>
          <div className="metric-desc">
            Maximum recall disparity between eligible groups within the same attribute (intra-attribute).
          </div>
        </div>

        {/* Worst-Group Recall */}
        <div className="metric-box">
          <div className="metric-label">Worst-Group Recall (R_min)</div>
          <div className="metric-value tabular-nums">
            {(data.worst_group_recall * 100).toFixed(1)}%
          </div>
          <div className="metric-desc">
            The lowest positive recall rate among any eligible demographic cohort in the validation pool.
          </div>
        </div>

        {/* Overall Recall */}
        <div className="metric-box">
          <div className="metric-label">Overall Recall @ Top-20%</div>
          <div className="metric-value tabular-nums">
            {(data.overall_recall_at_capacity * 100).toFixed(1)}%
          </div>
          <div className="metric-desc">
            Percentage of total benchmark students needing support captured within the top-20% capacity slots.
          </div>
        </div>
      </div>

      {/* Second Row: Calibration & Need Score */}
      <div className="metrics-row">
        <div className="metric-box">
          <div className="metric-label">Average Cohort Need Score</div>
          <div className="metric-value tabular-nums">
            {data.average_need_score.toFixed(1)}
            <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}> / 100</span>
          </div>
          <div className="metric-desc">
            Mean 0–100 need score across the full evaluation cohort. Higher scores indicate greater intervention priority.
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Calibration Utility (Brier Score)</div>
          <div className="metric-value tabular-nums">
            {data.brier_score.toFixed(3)}
          </div>
          <div className="metric-desc">
            Mean squared probability error (0.0 = perfect calibration). Unbalanced Logistic Regression preserves honest probabilities.
          </div>
        </div>
      </div>

      {/* Cohort Pulse & Workflow Panels */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
        {/* Prioritization Guidance Panel */}
        <div className="panel">
          <div className="panel-header">
            <div>
              <h3 className="panel-title">Operational Workflow</h3>
              <p className="panel-subtitle">How educators should review and action this roster</p>
            </div>
          </div>
          <ol style={{ paddingLeft: '1.25rem', fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <li style={{ marginBottom: '0.5rem' }}>
              <strong>Review the Prioritized List:</strong> Inspect students ranked in slots #1 through #{data.num_selected}.
            </li>
            <li style={{ marginBottom: '0.5rem' }}>
              <strong>Examine Non-Causal Contributing Factors:</strong> Click <em>Inspect Profile</em> to view specific historical indicators (G1/G2 trends, absence records, study time).
            </li>
            <li style={{ marginBottom: '0.5rem' }}>
              <strong>Apply Contextual Educator Discretion:</strong> Validate whether unmodeled circumstances (illness, bereavement, recent improvement) alter the recommendation.
            </li>
            <li>
              <strong>Audit Demographic Balance:</strong> Verify that group recall rates remain balanced across sex and school cohorts.
            </li>
          </ol>
          {onNavigateToPriorities && (
            <div style={{ marginTop: '1.25rem' }}>
              <button className="btn btn-primary" onClick={onNavigateToPriorities}>
                Open Prioritized List
              </button>
            </div>
          )}
        </div>

        {/* Fairness Standards Panel */}
        <div className="panel">
          <div className="panel-header">
            <div>
              <h3 className="panel-title">Fairness & Eligibility Audit</h3>
              <p className="panel-subtitle">Auditing criteria defined under educational equity benchmark</p>
            </div>
          </div>
          <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <p style={{ marginBottom: '0.5rem' }}>
              <strong>Group Eligibility Rule:</strong> A demographic group is eligible for fairness auditing only if it contains at least <strong>10 evaluation records</strong> and at least <strong>3 positive cases</strong> needing support.
            </p>
            <p style={{ marginBottom: '0.5rem' }}>
              <strong>Fairness-Through-Unawareness:</strong> Protected attributes (<code>sex</code>, <code>school</code>) are excluded from the model feature vector to eliminate explicit disparate treatment.
            </p>
            <p>
              <strong>Observed Recall Gap:</strong> Measures intra-attribute disparity (Female vs. Male, or GP vs. MS). The current validation gap is <strong>{(data.fairness_gap * 100).toFixed(1)}%</strong>.
            </p>
          </div>
          {onNavigateToFairness && (
            <div style={{ marginTop: '1.25rem' }}>
              <button className="btn btn-outline" onClick={onNavigateToFairness}>
                View Group Recall Breakdown
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
