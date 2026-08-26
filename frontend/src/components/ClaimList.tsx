import React, { useState } from 'react';
import type { Claim } from '../lib/types';

interface Props {
  claims: Claim[];
  selectedIndex: number | null;
  onSelect: (i: number) => void;
}

type Filter = 'ALL' | 'SUPPORTED' | 'HALLUCINATED' | 'UNVERIFIABLE';

const labelClass = (l: string) => l.toLowerCase() as 'supported' | 'hallucinated' | 'unverifiable';

const Badge: React.FC<{ label: string }> = ({ label }) => (
  <span className={`badge badge-${labelClass(label)}`}>{label}</span>
);

const ChevronRight = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 18 15 12 9 6"/>
  </svg>
);

const ClaimList: React.FC<Props> = ({ claims, selectedIndex, onSelect }) => {
  const [filter, setFilter] = useState<Filter>('ALL');
  const [showAll, setShowAll] = useState(false);

  const filtered = filter === 'ALL' ? claims : claims.filter(c => c.final_label === filter);
  const LIMIT = 6;
  const visible = showAll ? filtered : filtered.slice(0, LIMIT);

  return (
    <>
      <div className="panel-header">
        <div>
          <div className="panel-title">Claims Extracted ({claims.length})</div>
        </div>
        <select
          className="filter-select"
          value={filter}
          onChange={e => { setFilter(e.target.value as Filter); setShowAll(false); }}
          aria-label="Filter claims by label"
        >
          <option value="ALL">All</option>
          <option value="SUPPORTED">Supported</option>
          <option value="HALLUCINATED">Hallucinated</option>
          <option value="UNVERIFIABLE">Unverifiable</option>
        </select>
      </div>

      <div className="panel-body">
        {visible.length === 0 ? (
          <div className="empty-state">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            No {filter !== 'ALL' ? filter.toLowerCase() : ''} claims found.
          </div>
        ) : (
          <>
            {visible.map((c) => {
              const realIdx = claims.indexOf(c);
              const lc = labelClass(c.final_label);
              const isSelected = selectedIndex === realIdx;
              return (
                <div
                  key={realIdx}
                  className={`claim-card${isSelected ? ` selected-${lc}` : ''}`}
                  onClick={() => onSelect(realIdx)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={e => e.key === 'Enter' && onSelect(realIdx)}
                  aria-pressed={isSelected}
                  aria-label={`Claim ${realIdx + 1}: ${c.claim}. ${c.final_label}`}
                >
                  <div className={`claim-card-num ${lc}`}>{realIdx + 1}</div>
                  <div className="claim-card-body">
                    <div className="claim-card-text">"{c.claim}"</div>
                    <div className="claim-card-meta">
                      <Badge label={c.final_label} />
                      <span className="claim-card-score">
                        {Math.round(c.claim_confidence * 100)}%
                      </span>
                    </div>
                  </div>
                  <div className="claim-card-chevron" aria-hidden="true">
                    <ChevronRight />
                  </div>
                </div>
              );
            })}

            {!showAll && filtered.length > LIMIT && (
              <button
                className="btn btn-ghost"
                style={{ width: '100%', marginTop: 4 }}
                onClick={() => setShowAll(true)}
              >
                View All {filtered.length} Claims
              </button>
            )}
          </>
        )}
      </div>
    </>
  );
};

export default ClaimList;
