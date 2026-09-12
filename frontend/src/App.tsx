import React, { useState, useEffect } from 'react';
import { fetchHealth, fetchOverview, OverviewResponse } from './api/client';
import { PrioritizedList } from './pages/PrioritizedList';
import { Overview } from './pages/Overview';
import { FairnessView } from './pages/FairnessView';
import { ModelPerformance } from './pages/ModelPerformance';
import { PredictSandbox } from './pages/PredictSandbox';

type NavSection = 'priorities' | 'overview' | 'fairness' | 'performance' | 'sandbox';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavSection>('priorities');
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [checkingHealth, setCheckingHealth] = useState(false);
  const [overviewData, setOverviewData] = useState<OverviewResponse | null>(null);

  const checkConnection = () => {
    setCheckingHealth(true);
    fetchHealth()
      .then((res) => {
        setBackendOnline(res.status === 'ok' && res.model_loaded);
        return fetchOverview();
      })
      .then((ov) => {
        setOverviewData(ov);
      })
      .catch(() => {
        setBackendOnline(false);
      })
      .finally(() => {
        setCheckingHealth(false);
      });
  };

  useEffect(() => {
    checkConnection();
    // Poll health status gently every 15 seconds
    const interval = setInterval(checkConnection, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-layout">
      {/* Persistent Left Navigation Rail */}
      <aside className="sidebar-rail">
        <div className="sidebar-header">
          <h1 className="sidebar-title">Student Support AI</h1>
          <div className="sidebar-subtitle">Fair Prioritization System</div>
        </div>

        <ul className="nav-list">
          <li>
            <button
              className={`nav-item-btn ${activeTab === 'priorities' ? 'active' : ''}`}
              onClick={() => setActiveTab('priorities')}
            >
              <span>📋</span>
              <span>Prioritized Roster</span>
            </button>
          </li>
          <li>
            <button
              className={`nav-item-btn ${activeTab === 'overview' ? 'active' : ''}`}
              onClick={() => setActiveTab('overview')}
            >
              <span>📊</span>
              <span>Cohort Overview</span>
            </button>
          </li>
          <li>
            <button
              className={`nav-item-btn ${activeTab === 'fairness' ? 'active' : ''}`}
              onClick={() => setActiveTab('fairness')}
            >
              <span>⚖️</span>
              <span>Fairness Audit</span>
            </button>
          </li>
          <li>
            <button
              className={`nav-item-btn ${activeTab === 'performance' ? 'active' : ''}`}
              onClick={() => setActiveTab('performance')}
            >
              <span>🎯</span>
              <span>Model Performance</span>
            </button>
          </li>
          <li>
            <button
              className={`nav-item-btn ${activeTab === 'sandbox' ? 'active' : ''}`}
              onClick={() => setActiveTab('sandbox')}
            >
              <span>🧪</span>
              <span>Scoring Sandbox</span>
            </button>
          </li>
        </ul>

        <div className="sidebar-footer">
          <div className="connection-badge">
            <span
              className={`dot-indicator ${
                backendOnline === true ? 'dot-online' : backendOnline === false ? 'dot-offline' : ''
              }`}
              style={{ backgroundColor: backendOnline === null ? '#94A3B8' : undefined }}
            />
            <span style={{ color: backendOnline ? '#F1F5F9' : '#FCA5A5' }}>
              {backendOnline === true
                ? 'Backend: Online'
                : backendOnline === false
                ? 'Backend: Offline'
                : 'Checking API...'}
            </span>
          </div>
          <div style={{ marginTop: '0.4rem', color: '#64748B', fontSize: '0.7rem' }}>
            FastAPI :8000 | React+Vite :5173
          </div>
        </div>
      </aside>

      {/* Main Operational Surface */}
      <div className="content-area">
        {/* Top Context Bar */}
        <header className="top-bar">
          <div className="top-context-title">
            Decision-Support Environment — Secondary Mathematics Benchmark
          </div>

          <div className="top-capacity-indicator">
            {overviewData && (
              <>
                <div className="top-capacity-item">
                  Cohort: <strong>{overviewData.total_students} Students</strong>
                </div>
                <div className="top-capacity-item">
                  Fixed Capacity: <strong>{overviewData.num_selected} Slots ({(overviewData.capacity_fraction * 100).toFixed(0)}%)</strong>
                </div>
                <div className="top-capacity-item">
                  Recall Gap: <strong>{(overviewData.fairness_gap * 100).toFixed(1)}%</strong>
                </div>
              </>
            )}
            <button
              className="btn btn-outline"
              style={{ padding: '0.25rem 0.55rem', fontSize: '0.75rem' }}
              onClick={checkConnection}
              disabled={checkingHealth}
            >
              {checkingHealth ? 'Syncing...' : 'Sync'}
            </button>
          </div>
        </header>

        {/* Scrollable Main Content */}
        <main className="main-scroll">
          {/* Friendly Offline Notice when Backend is Down */}
          {backendOnline === false && (
            <div className="banner-offline">
              <div>
                <strong style={{ display: 'block', marginBottom: '0.25rem' }}>
                  Backend API is unreachable at http://127.0.0.1:8000
                </strong>
                <span style={{ fontSize: '0.8125rem' }}>
                  Please verify that the FastAPI server is running. In your terminal, run:{' '}
                  <code>uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload</code>
                </span>
              </div>
              <button className="btn btn-primary" onClick={checkConnection}>
                Retry Connection
              </button>
            </div>
          )}

          {/* Active Section View */}
          {activeTab === 'priorities' && <PrioritizedList />}
          {activeTab === 'overview' && (
            <Overview
              onNavigateToPriorities={() => setActiveTab('priorities')}
              onNavigateToFairness={() => setActiveTab('fairness')}
            />
          )}
          {activeTab === 'fairness' && <FairnessView />}
          {activeTab === 'performance' && <ModelPerformance />}
          {activeTab === 'sandbox' && <PredictSandbox />}
        </main>
      </div>
    </div>
  );
};
