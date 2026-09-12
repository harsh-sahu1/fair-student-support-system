import React, { useEffect, useState } from 'react';
import { fetchStudentDetail, StudentDetailResponse } from '../api/client';

interface StudentDrawerProps {
  studentId: string | null;
  onClose: () => void;
}

export const StudentDrawer: React.FC<StudentDrawerProps> = ({ studentId, onClose }) => {
  const [detail, setDetail] = useState<StudentDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!studentId) {
      setDetail(null);
      return;
    }

    let isMounted = true;
    setLoading(true);
    setError(null);

    fetchStudentDetail(studentId)
      .then((data) => {
        if (isMounted) setDetail(data);
      })
      .catch((err) => {
        if (isMounted) setError(err.message || 'Failed to load student details');
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [studentId]);

  if (!studentId) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="drawer-content" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Student Profile: {studentId}
            </h3>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Detailed assessment factors & model attribution
            </span>
          </div>
          <button className="btn btn-outline" onClick={onClose} style={{ padding: '0.3rem 0.6rem' }}>
            ✕ Close
          </button>
        </div>

        <div className="drawer-body">
          {loading && (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              Loading student data...
            </div>
          )}

          {error && (
            <div style={{ padding: '1.5rem', backgroundColor: '#FEF2F2', border: '1px solid #FECACA', color: '#991B1B', borderRadius: 4 }}>
              {error}
            </div>
          )}

          {detail && !loading && (
            <>
              {/* Primary Metric Card */}
              <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
                <div style={{ flex: 1, backgroundColor: 'var(--bg-subtle)', padding: '1rem', borderRadius: 6 }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Support Need Score</div>
                  <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }} className="tabular-nums">
                    {detail.need_score.toFixed(1)}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    Calibrated Prob: {(detail.probability * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={{ flex: 1, backgroundColor: 'var(--bg-subtle)', padding: '1rem', borderRadius: 6 }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Cohort Priority Rank</div>
                  <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    #{detail.rank}
                  </div>
                  <div>
                    {detail.selected ? (
                      <span className="badge-support">Selected for Support</span>
                    ) : (
                      <span className="badge-neutral">Regular Monitoring</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Demographic Group Info */}
              <div style={{ marginBottom: '1.5rem', padding: '0.75rem 1rem', backgroundColor: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: 4 }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                  Auditing Demographics (Never used as scoring features)
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Sex: <strong>{detail.sex}</strong> | School: <strong>{detail.school}</strong>
                  {detail.ground_truth_support_needed !== undefined && (
                    <span style={{ marginLeft: '1rem', color: 'var(--text-muted)' }}>
                      Benchmark Target: {detail.ground_truth_support_needed === 1 ? 'Needed Support (G3 < 10)' : 'Passing (G3 ≥ 10)'}
                    </span>
                  )}
                </div>
              </div>

              {/* Top Factors Section */}
              <div style={{ marginBottom: '1.5rem' }}>
                <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text-primary)' }}>
                  Key Contributing Factors (Statistical Associations)
                </h4>
                {detail.top_factors.map((factor, idx) => (
                  <div key={idx} className="factor-pill">
                    {factor}
                  </div>
                ))}
              </div>

              {/* Factor Contribution Details */}
              {detail.factor_details && detail.factor_details.length > 0 && (
                <div style={{ marginBottom: '1.5rem' }}>
                  <h4 style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                    Attribution Weights (Linear Contribution)
                  </h4>
                  <table style={{ width: '100%', fontSize: '0.75rem', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', textAlign: 'left' }}>
                        <th style={{ padding: '0.4rem 0' }}>Factor</th>
                        <th style={{ padding: '0.4rem 0' }}>Value</th>
                        <th style={{ padding: '0.4rem 0', textAlign: 'right' }}>Weight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.factor_details.map((fd, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid var(--bg-subtle)' }}>
                          <td style={{ padding: '0.4rem 0', color: 'var(--text-primary)' }}>{fd.feature}</td>
                          <td style={{ padding: '0.4rem 0', color: 'var(--text-secondary)' }}>{String(fd.value)}</td>
                          <td style={{ padding: '0.4rem 0', textAlign: 'right', fontWeight: 600, color: 'var(--brand-primary)' }}>
                            +{fd.contribution.toFixed(3)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Raw Features Inspect */}
              <div style={{ marginBottom: '1.5rem' }}>
                <details style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 500, color: 'var(--brand-secondary)' }}>
                    View All 30 Raw Educational Input Features
                  </summary>
                  <div style={{ marginTop: '0.75rem', maxHeight: '180px', overflowY: 'auto', background: 'var(--bg-subtle)', padding: '0.75rem', borderRadius: 4 }}>
                    <pre style={{ fontSize: '0.75rem', whiteSpace: 'pre-wrap' }}>
                      {JSON.stringify(detail.raw_features, null, 2)}
                    </pre>
                  </div>
                </details>
              </div>

              {/* Mandatory Language Disclaimer */}
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem', lineHeight: 1.4 }}>
                <strong>Ethical Use Note:</strong> {detail.disclaimer}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
