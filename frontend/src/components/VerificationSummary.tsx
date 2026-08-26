import React from 'react';
import type { VerifyResponse } from '../lib/types';

interface Props {
  result: VerifyResponse;
  onDownload: () => void;
}

const VerificationSummary: React.FC<Props> = ({ result, onDownload }) => {
  const { summary, timing } = result;
  const trustScore = Math.round((1 - summary.hallucination_rate) * 100);
  const trustColor =
    trustScore >= 80 ? 'var(--green)' :
    trustScore >= 50 ? 'var(--amber)' : 'var(--red)';

  return (
    <div>
      {/* Top row */}
      <div className="topbar">
        <div>
          <h2 className="page-title" style={{ fontSize: '1.2rem' }}>Verification Results</h2>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-3)', marginTop: 2 }}>
            Completed in {timing.total_seconds.toFixed(2)}s
          </p>
        </div>
        <button className="download-btn" onClick={onDownload} aria-label="Download verification report">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          Download Report
        </button>
      </div>

      {/* Summary cards */}
      <div className="summary-grid">
        {/* Total claims */}
        <div className="summary-card">
          <div className="summary-card-label">Total Claims</div>
          <div className="summary-card-value" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {summary.total_claims}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text-3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>
          </div>
        </div>

        {/* Supported */}
        <div className="summary-card">
          <div className="summary-card-label">Supported</div>
          <div className="summary-card-value green" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {summary.supported}
            <span style={{ fontSize: '0.85rem', color: 'var(--text-2)' }}>
              {summary.total_claims > 0 ? `${((summary.supported / summary.total_claims) * 100).toFixed(1)}%` : '–'}
            </span>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
          </div>
          <div className="summary-card-bar">
            <div className="summary-card-bar-fill" style={{ width: `${summary.total_claims > 0 ? (summary.supported / summary.total_claims) * 100 : 0}%`, background: 'var(--green)' }} />
          </div>
        </div>

        {/* Hallucinated */}
        <div className="summary-card">
          <div className="summary-card-label">Hallucinated</div>
          <div className="summary-card-value red" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {summary.hallucinated}
            <span style={{ fontSize: '0.85rem', color: 'var(--text-2)' }}>
              {summary.total_claims > 0 ? `${(summary.hallucination_rate * 100).toFixed(1)}%` : '–'}
            </span>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--red)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
          </div>
          <div className="summary-card-bar">
            <div className="summary-card-bar-fill" style={{ width: `${summary.total_claims > 0 ? summary.hallucination_rate * 100 : 0}%`, background: 'var(--red)' }} />
          </div>
        </div>

        {/* Unverifiable */}
        <div className="summary-card">
          <div className="summary-card-label">Unverifiable</div>
          <div className="summary-card-value amber" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {summary.unverifiable}
            <span style={{ fontSize: '0.85rem', color: 'var(--text-2)' }}>
              {summary.total_claims > 0 ? `${((summary.unverifiable / summary.total_claims) * 100).toFixed(1)}%` : '–'}
            </span>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--amber)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
          </div>
          <div className="summary-card-bar">
            <div className="summary-card-bar-fill" style={{ width: `${summary.total_claims > 0 ? (summary.unverifiable / summary.total_claims) * 100 : 0}%`, background: 'var(--amber)' }} />
          </div>
        </div>

        {/* Overall trust score */}
        <div className="summary-card">
          <div className="summary-card-label">Overall Trust Score</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 4, marginTop: 4 }}>
            <span style={{ fontSize: '1.6rem', fontWeight: 700, color: trustColor }}>{trustScore}%</span>
          </div>
          <div style={{ marginTop: 6 }}>
            <div style={{ height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{ width: `${trustScore}%`, height: '100%', background: trustColor, borderRadius: 3, transition: 'width 0.6s ease' }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VerificationSummary;
