import React from 'react';
import type { HealthResponse, Page } from '../lib/types';

interface Props {
  current: Page;
  onNav: (p: Page) => void;
  health: HealthResponse | null;
}

const NAV: { id: Page; label: string; icon: React.ReactNode }[] = [
  {
    id: 'home',
    label: 'Verify',
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
        <polyline points="9 22 9 12 15 12 15 22"/>
      </svg>
    ),
  },
  {
    id: 'results',
    label: 'Results',
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
        <line x1="6" y1="20" x2="6" y2="14"/>
      </svg>
    ),
  },
  {
    id: 'pipeline',
    label: 'Pipeline',
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    ),
  },
  {
    id: 'about',
    label: 'About',
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="16" x2="12" y2="12"/>
        <line x1="12" y1="8" x2="12.01" y2="8"/>
      </svg>
    ),
  },
];

const ShieldIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    <polyline points="9 12 11 14 15 10"/>
  </svg>
);

const Sidebar: React.FC<Props> = ({ current, onNav, health }) => {
  const dotClass =
    health === null ? 'loading' :
    health.pipeline === 'ready' ? 'online' : 'offline';

  const statusText =
    health === null ? 'Connecting…' :
    health.pipeline === 'ready'
      ? `${health.kb_passages} passages loaded`
      : 'Pipeline not ready';

  return (
    <nav className="sidebar" aria-label="Main navigation">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon" aria-hidden="true">
          <ShieldIcon />
        </div>
        <div className="sidebar-brand-text">
          <div className="sidebar-brand-name">Hallucination</div>
          <div className="sidebar-brand-name" style={{ color: 'var(--accent)' }}>Firewall</div>
          <div className="sidebar-brand-sub">AI Response Verification</div>
        </div>
      </div>

      {/* Nav items */}
      <div className="sidebar-nav">
        {NAV.map(({ id, label, icon }) => (
          <button
            key={id}
            className={`sidebar-nav-item${current === id ? ' active' : ''}`}
            onClick={() => onNav(id)}
            aria-current={current === id ? 'page' : undefined}
          >
            {icon}
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* Footer status */}
      <div className="sidebar-footer">
        <div style={{ display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '6px' }}>
          <span className={`status-dot ${dotClass}`} aria-hidden="true" />
          <span style={{ fontWeight: 600, color: dotClass === 'online' ? 'var(--green)' : dotClass === 'offline' ? 'var(--red)' : 'var(--amber)' }}>
            {dotClass === 'online' ? 'Backend Online' : dotClass === 'offline' ? 'Backend Offline' : 'Connecting'}
          </span>
        </div>
        <div>{statusText}</div>
        {health?.mode && <div style={{ marginTop: 2 }}>Mode: {health.mode}</div>}
        <div style={{ marginTop: 8, color: 'var(--text-3)' }}>
          Built with ♥ for trustworthy AI<br />v1.0.0
        </div>
      </div>
    </nav>
  );
};

export default Sidebar;
