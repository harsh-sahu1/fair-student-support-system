import React, { useState, useEffect } from 'react';
import { fetchStudents, StudentListItem } from '../api/client';
import { StudentDrawer } from '../components/StudentDrawer';

interface PrioritizedListProps {
  onSelectStudent?: (studentId: string) => void;
}

export const PrioritizedList: React.FC<PrioritizedListProps> = () => {
  const [students, setStudents] = useState<StudentListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [selectedOnly, setSelectedOnly] = useState(false);
  const [groupFilter, setGroupFilter] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  // Selected student for drawer
  const [activeStudentId, setActiveStudentId] = useState<string | null>(null);

  const loadData = () => {
    setLoading(true);
    setError(null);
    fetchStudents({
      selectedOnly: selectedOnly,
      group: groupFilter || undefined,
    })
      .then((data) => setStudents(data))
      .catch((err) => setError(err.message || 'Error fetching prioritized cohort'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, [selectedOnly, groupFilter]);

  const filteredStudents = students.filter((s) => {
    if (!searchQuery.trim()) return true;
    const query = searchQuery.toLowerCase();
    return (
      s.student_id.toLowerCase().includes(query) ||
      s.top_factors.some((f) => f.toLowerCase().includes(query))
    );
  });

  const selectedCount = students.filter((s) => s.selected).length;

  return (
    <div>
      {/* Top Section Header */}
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
          Prioritized Student Support Roster
        </h2>
        <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
          Ranked by continuous predictive need probability under fixed 20% allocation capacity.
        </p>
      </div>

      {/* Control Bar: Filters & Search */}
      <div className="panel" style={{ padding: '0.85rem 1.25rem', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            {/* Search Input */}
            <input
              type="text"
              placeholder="Search by ID or factor..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                padding: '0.45rem 0.75rem',
                fontSize: '0.8125rem',
                border: '1px solid var(--border-subtle)',
                borderRadius: '4px',
                width: '210px',
                outline: 'none',
              }}
            />

            {/* Group Filter */}
            <select
              value={groupFilter}
              onChange={(e) => setGroupFilter(e.target.value)}
              style={{
                padding: '0.45rem 0.75rem',
                fontSize: '0.8125rem',
                border: '1px solid var(--border-subtle)',
                borderRadius: '4px',
                backgroundColor: 'var(--bg-surface)',
                color: 'var(--text-primary)',
              }}
            >
              <option value="">All Demographic Groups</option>
              <option value="F">Sex: Female (F)</option>
              <option value="M">Sex: Male (M)</option>
              <option value="GP">School: Gabriel Pereira (GP)</option>
              <option value="MS">School: Mousinho da Silveira (MS)</option>
            </select>

            {/* Selected Only Toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8125rem', cursor: 'pointer', userSelect: 'none' }}>
              <input
                type="checkbox"
                checked={selectedOnly}
                onChange={(e) => setSelectedOnly(e.target.checked)}
              />
              <span>Selected for Support Only ({selectedCount})</span>
            </label>
          </div>

          <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
            Showing <strong>{filteredStudents.length}</strong> of <strong>{students.length}</strong> students
          </div>
        </div>
      </div>

      {/* Table Surface */}
      <div className="panel" style={{ padding: 0 }}>
        {loading && (
          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            Loading student prioritization records...
          </div>
        )}

        {error && (
          <div style={{ padding: '2rem', textAlign: 'center' }}>
            <div style={{ color: '#991B1B', marginBottom: '0.75rem' }}>{error}</div>
            <button className="btn btn-outline" onClick={loadData}>Retry</button>
          </div>
        )}

        {!loading && !error && (
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: '70px' }}>Rank</th>
                  <th style={{ width: '110px' }}>Student ID</th>
                  <th style={{ width: '110px' }}>Need Score</th>
                  <th style={{ width: '160px' }}>Allocation Status</th>
                  <th style={{ width: '130px' }}>Audit Cohort</th>
                  <th>Primary Contributing Factor</th>
                  <th style={{ width: '120px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredStudents.map((s) => (
                  <tr key={s.student_id} className={s.selected ? 'row-selected' : ''}>
                    <td>
                      <span style={{ fontWeight: 600, color: s.selected ? 'var(--support-text)' : 'var(--text-secondary)' }}>
                        #{s.rank}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontWeight: 600 }}>{s.student_id}</span>
                    </td>
                    <td>
                      <span
                        className="tabular-nums"
                        style={{
                          fontWeight: 700,
                          fontSize: '1rem',
                          color: s.need_score >= 80 ? 'var(--support-accent)' : 'var(--text-primary)',
                        }}
                      >
                        {s.need_score.toFixed(1)}
                      </span>
                    </td>
                    <td>
                      {s.selected ? (
                        <span className="badge-support">Selected for Support</span>
                      ) : (
                        <span className="badge-neutral">Regular Monitoring</span>
                      )}
                    </td>
                    <td style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                      Sex: {s.sex} | School: {s.school}
                    </td>
                    <td style={{ fontSize: '0.8125rem', color: 'var(--text-primary)' }}>
                      {s.top_factors[0] || 'General profile indicators'}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-outline"
                        style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
                        onClick={() => setActiveStudentId(s.student_id)}
                      >
                        Inspect Profile
                      </button>
                    </td>
                  </tr>
                ))}

                {filteredStudents.length === 0 && (
                  <tr>
                    <td colSpan={7} style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                      No students match the selected filter criteria.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Slide-over Profile Drawer */}
      <StudentDrawer
        studentId={activeStudentId}
        onClose={() => setActiveStudentId(null)}
      />
    </div>
  );
};
