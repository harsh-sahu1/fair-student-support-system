import React, { useState } from 'react';
import { postPredict, StudentPrediction } from '../api/client';

export const PredictSandbox: React.FC = () => {
  const [formData, setFormData] = useState({
    student_id: 'STUDENT_PREVIEW_01',
    age: 16,
    Medu: 2,
    Fedu: 2,
    traveltime: 2,
    studytime: 1,
    failures: 1,
    famrel: 4,
    freetime: 3,
    goout: 3,
    Dalc: 1,
    Walc: 2,
    health: 3,
    absences: 8,
    G1: 8,
    G2: 7,
    address: 'U',
    famsize: 'GT3',
    Pstatus: 'T',
    Mjob: 'other',
    Fjob: 'other',
    reason: 'course',
    guardian: 'mother',
    schoolsup: 'no',
    famsup: 'yes',
    paid: 'no',
    activities: 'no',
    nursery: 'yes',
    higher: 'yes',
    internet: 'yes',
    romantic: 'no',
    sex: 'F',
    school: 'GP'
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<StudentPrediction | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'number' ? parseInt(value, 10) || 0 : value,
    }));
  };

  const handlePredict = async (injectG3: boolean = false) => {
    setLoading(true);
    setErrorMsg(null);
    setResult(null);

    const payload: any = { ...formData };
    if (injectG3) {
      payload['G3'] = 8; // Deliberate leakage test
    }

    try {
      const res = await postPredict(payload);
      if (res.predictions && res.predictions.length > 0) {
        setResult(res.predictions[0]);
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Error occurred during prediction.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
          Interactive Scoring Sandbox & API Tester
        </h2>
        <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
          Directly test the <code>POST /predict</code> API endpoint with custom student attributes or verify zero-leakage rejection.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '1.5rem' }}>
        {/* Form Panel */}
        <div className="panel">
          <h3 className="panel-title" style={{ marginBottom: '1rem' }}>Student Input Features</h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', fontSize: '0.8125rem' }}>
            <div>
              <label style={{ display: 'block', fontWeight: 500, marginBottom: '0.25rem' }}>First Period Grade (G1): {formData.G1}/20</label>
              <input
                type="number"
                name="G1"
                min={0}
                max={20}
                value={formData.G1}
                onChange={handleInputChange}
                style={{ width: '100%', padding: '0.4rem', border: '1px solid var(--border-subtle)', borderRadius: 4 }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontWeight: 500, marginBottom: '0.25rem' }}>Second Period Grade (G2): {formData.G2}/20</label>
              <input
                type="number"
                name="G2"
                min={0}
                max={20}
                value={formData.G2}
                onChange={handleInputChange}
                style={{ width: '100%', padding: '0.4rem', border: '1px solid var(--border-subtle)', borderRadius: 4 }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontWeight: 500, marginBottom: '0.25rem' }}>Past Class Failures</label>
              <input
                type="number"
                name="failures"
                min={0}
                max={4}
                value={formData.failures}
                onChange={handleInputChange}
                style={{ width: '100%', padding: '0.4rem', border: '1px solid var(--border-subtle)', borderRadius: 4 }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontWeight: 500, marginBottom: '0.25rem' }}>Recorded Absences</label>
              <input
                type="number"
                name="absences"
                min={0}
                max={93}
                value={formData.absences}
                onChange={handleInputChange}
                style={{ width: '100%', padding: '0.4rem', border: '1px solid var(--border-subtle)', borderRadius: 4 }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontWeight: 500, marginBottom: '0.25rem' }}>Weekly Study Time</label>
              <select
                name="studytime"
                value={formData.studytime}
                onChange={handleInputChange}
                style={{ width: '100%', padding: '0.4rem', border: '1px solid var(--border-subtle)', borderRadius: 4 }}
              >
                <option value={1}>&lt; 2 hours</option>
                <option value={2}>2 to 5 hours</option>
                <option value={3}>5 to 10 hours</option>
                <option value={4}>&gt; 10 hours</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontWeight: 500, marginBottom: '0.25rem' }}>Higher Education Plans</label>
              <select
                name="higher"
                value={formData.higher}
                onChange={handleInputChange}
                style={{ width: '100%', padding: '0.4rem', border: '1px solid var(--border-subtle)', borderRadius: 4 }}
              >
                <option value="yes">Yes</option>
                <option value="no">No</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontWeight: 500, marginBottom: '0.25rem' }}>Extra Educational Support</label>
              <select
                name="schoolsup"
                value={formData.schoolsup}
                onChange={handleInputChange}
                style={{ width: '100%', padding: '0.4rem', border: '1px solid var(--border-subtle)', borderRadius: 4 }}
              >
                <option value="no">No</option>
                <option value="yes">Yes</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontWeight: 500, marginBottom: '0.25rem' }}>Internet Access at Home</label>
              <select
                name="internet"
                value={formData.internet}
                onChange={handleInputChange}
                style={{ width: '100%', padding: '0.4rem', border: '1px solid var(--border-subtle)', borderRadius: 4 }}
              >
                <option value="yes">Yes</option>
                <option value="no">No</option>
              </select>
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <button
              className="btn btn-primary"
              disabled={loading}
              onClick={() => handlePredict(false)}
            >
              {loading ? 'Scoring...' : 'Score Student Need'}
            </button>

            <button
              className="btn btn-outline"
              disabled={loading}
              onClick={() => handlePredict(true)}
              style={{ borderColor: '#FCA5A5', color: '#991B1B' }}
            >
              Test Anti-Leakage (Inject G3)
            </button>
          </div>
        </div>

        {/* Output Panel */}
        <div className="panel">
          <h3 className="panel-title" style={{ marginBottom: '1rem' }}>Inference & Explainability Output</h3>

          {loading && (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              Calling <code>POST /predict</code> API...
            </div>
          )}

          {errorMsg && (
            <div style={{ padding: '1.25rem', backgroundColor: '#FEF2F2', border: '1px solid #FECACA', borderRadius: 4, color: '#991B1B', fontSize: '0.8125rem' }}>
              <strong style={{ display: 'block', marginBottom: '0.35rem' }}>API Error / Validation Response:</strong>
              {errorMsg}
            </div>
          )}

          {result && !loading && (
            <div>
              <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
                <div style={{ flex: 1, backgroundColor: 'var(--bg-subtle)', padding: '1rem', borderRadius: 6 }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Predicted Need Score</div>
                  <div style={{ fontSize: '2.25rem', fontWeight: 700, color: 'var(--text-primary)' }} className="tabular-nums">
                    {result.need_score.toFixed(1)}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    Probability: {(result.probability * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={{ flex: 1, backgroundColor: 'var(--bg-subtle)', padding: '1rem', borderRadius: 6 }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Priority Level</div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 600, marginTop: '0.5rem' }}>
                    {result.need_score >= 50.0 ? (
                      <span className="badge-support">High Intervention Need</span>
                    ) : (
                      <span className="badge-neutral">Moderate / Standard Monitoring</span>
                    )}
                  </div>
                </div>
              </div>

              <div style={{ marginBottom: '1.25rem' }}>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                  Model-Identified Statistical Associations
                </h4>
                {result.top_factors.map((tf, i) => (
                  <div key={i} className="factor-pill">
                    {tf}
                  </div>
                ))}
              </div>

              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' }}>
                {result.disclaimer}
              </div>
            </div>
          )}

          {!result && !errorMsg && !loading && (
            <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
              Modify student parameters on the left and click <strong>Score Student Need</strong> to see instant results.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
