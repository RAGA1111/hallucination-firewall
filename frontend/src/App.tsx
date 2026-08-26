import React, { useCallback, useEffect, useRef, useState } from 'react';
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import ResultsPage from './pages/ResultsPage';
import PipelinePage from './pages/PipelinePage';
import AboutPage from './pages/AboutPage';
import type { HealthResponse, Page, VerifyResponse } from './lib/types';
import { fetchHealth } from './lib/api';

const POLL_INTERVAL = 12_000; // ms

const App: React.FC = () => {
  const [page, setPage] = useState<Page>('home');
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const pollHealth = useCallback(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    pollHealth();
    const id = setInterval(pollHealth, POLL_INTERVAL);
    return () => clearInterval(id);
  }, [pollHealth]);

  // When a result arrives → auto-navigate to results page
  const handleResult = useCallback((r: VerifyResponse) => {
    setResult(r);
    // Small delay so the user sees the pipeline "completed" state briefly
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setPage('results'), 350);
  }, []);

  const handleNav = useCallback((p: Page) => {
    setPage(p);
  }, []);

  const handleBack = useCallback(() => {
    setPage('home');
  }, []);

  return (
    <div className="app-layout">
      <Sidebar current={page} onNav={handleNav} health={health} />
      <main className="main-content" id="main" tabIndex={-1}>
        {page === 'home' && (
          <HomePage onResult={handleResult} />
        )}
        {page === 'results' && result ? (
          <ResultsPage result={result} onBack={handleBack} />
        ) : page === 'results' && !result ? (
          /* Edge case: user clicks Results before any run */
          <div>
            <div className="page-header">
              <h1 className="page-title">Results</h1>
              <p className="page-sub">No verification has been run yet.</p>
            </div>
            <div className="card">
              <div className="empty-state" style={{ padding: '32px 20px' }}>
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="20" x2="18" y2="10"/>
                  <line x1="12" y1="20" x2="12" y2="4"/>
                  <line x1="6" y1="20" x2="6" y2="14"/>
                </svg>
                <span>
                  No results yet.{' '}
                  <button
                    style={{ background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', fontSize: 'inherit', textDecoration: 'underline', padding: 0 }}
                    onClick={() => setPage('home')}
                  >
                    Run a verification
                  </button>{' '}
                  first.
                </span>
              </div>
            </div>
          </div>
        ) : null}
        {page === 'pipeline' && <PipelinePage />}
        {page === 'about' && <AboutPage />}
      </main>
    </div>
  );
};

export default App;
