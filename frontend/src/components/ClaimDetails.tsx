import React, { useState } from 'react';
import type { Claim } from '../lib/types';

interface Props {
  claim: Claim;
  claimIndex: number;
  onClose: () => void;
}

type Tab = 'evidence' | 'nli' | 'retriever' | 'selfcheck' | 'fusion';

const labelClass = (l: string) => l.toLowerCase() as 'supported' | 'hallucinated' | 'unverifiable';

const ScoreBar: React.FC<{ value: number; color: string }> = ({ value, color }) => (
  <div className="score-bar-wrap">
    <div className="score-bar-track">
      <div className="score-bar-fill" style={{ width: `${Math.round(value * 100)}%`, background: color }} />
    </div>
    <div className="score-bar-value">{Math.round(value * 100)}%</div>
  </div>
);

const scoreColor = (v: number) =>
  v >= 0.7 ? 'var(--green)' : v >= 0.45 ? 'var(--amber)' : 'var(--red)';

const TABS: { id: Tab; label: string }[] = [
  { id: 'evidence',  label: 'Evidence' },
  { id: 'nli',       label: 'NLI' },
  { id: 'retriever', label: 'Retriever' },
  { id: 'selfcheck', label: 'SelfCheck' },
  { id: 'fusion',    label: 'Fusion' },
];

const ClaimDetails: React.FC<Props> = ({ claim: c, claimIndex, onClose }) => {
  const [tab, setTab] = useState<Tab>('evidence');
  const lc = labelClass(c.final_label);

  const hasCorrected =
    c.correction_status === 'CORRECTED' &&
    c.corrected_claim &&
    c.corrected_claim !== c.claim;

  return (
    <>
      <div className="panel-header">
        <div>
          <div className="panel-title">Verification Details</div>
          <div className="panel-subtitle">Claim {claimIndex + 1}</div>
        </div>
        <button className="btn btn-icon" onClick={onClose} aria-label="Close details panel">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>

      <div className="panel-body">
        {/* Claim text + status */}
        <div className="detail-section">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10, marginBottom: 10 }}>
            <div style={{ fontSize: '0.97rem', fontWeight: 600, color: 'var(--text)', lineHeight: 1.4 }}>
              {c.claim}
            </div>
            <span className={`badge badge-${lc}`} style={{ flexShrink: 0 }}>{c.final_label}</span>
          </div>

          {/* Confidence bar */}
          <div style={{ marginBottom: 4 }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-3)', marginBottom: 4 }}>Confidence Score</div>
            <ScoreBar value={c.claim_confidence} color={scoreColor(c.claim_confidence)} />
          </div>
        </div>

        <div className="divider" />

        {/* Tabs */}
        <div className="tabs" role="tablist">
          {TABS.map(t => (
            <button
              key={t.id}
              className={`tab-btn${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
              role="tab"
              aria-selected={tab === t.id}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Evidence tab */}
        {tab === 'evidence' && (
          <div>
            {c.nli_evidence ? (
              <div className="detail-section">
                <div className="detail-section-title">Top Evidence</div>
                <div className="evidence-box">
                  <div className="evidence-source-row">
                    <span className="evidence-source-label">
                      {c.nli_evidence_meta?.[0]?.source
                        ? `Source: ${c.nli_evidence_meta[0].source}`
                        : 'Source: Knowledge Base'}
                    </span>
                    {c.nli_evidence_meta?.[0]?.retrieval_score != null && (
                      <span className="evidence-source-score">
                        Similarity {c.nli_evidence_meta[0].retrieval_score.toFixed(2)}
                      </span>
                    )}
                  </div>
                  <p style={{ margin: 0, lineHeight: 1.6 }}>
                    "{c.nli_evidence}"
                  </p>
                </div>
              </div>
            ) : (
              <div className="empty-state" style={{ padding: '20px 0' }}>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                </svg>
                No evidence retrieved for this claim.
              </div>
            )}

            {/* Correction */}
            {hasCorrected && (
              <div className="detail-section" style={{ marginTop: 14 }}>
                <div className="detail-section-title">Suggested Correction</div>
                <div className="correction-box">
                  <div className="correction-label">
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: 4 }}>
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                    Corrected Claim
                  </div>
                  <div className="correction-text">{c.corrected_claim}</div>
                </div>
                <div style={{ marginTop: 6, fontSize: '0.7rem', color: 'var(--text-3)' }}>
                  {c.correction_note}
                </div>
              </div>
            )}

            {/* Explanation */}
            <div className="detail-section" style={{ marginTop: 14 }}>
              <div className="detail-section-title">Explanation</div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-2)', lineHeight: 1.55, margin: 0 }}>
                {c.explanation}
              </p>
            </div>
          </div>
        )}

        {/* NLI tab */}
        {tab === 'nli' && (
          <div className="detail-section">
            <div className="detail-section-title">NLI (Natural Language Inference)</div>
            <div className="detail-row">
              <span className="detail-row-key">Label</span>
              <span className="detail-row-value">{c.nli_label}</span>
            </div>
            <div className="detail-row">
              <span className="detail-row-key">Confidence</span>
              <span className="detail-row-value" style={{ color: scoreColor(c.nli_confidence) }}>
                {Math.round(c.nli_confidence * 100)}%
              </span>
            </div>
            {c.nli_all_scores && Object.keys(c.nli_all_scores).length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-3)', marginBottom: 8 }}>All scores</div>
                {Object.entries(c.nli_all_scores).map(([k, v]) => (
                  <div key={k} style={{ marginBottom: 7 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem', marginBottom: 3 }}>
                      <span style={{ textTransform: 'capitalize', color: 'var(--text-2)' }}>{k}</span>
                      <span style={{ color: 'var(--text)', fontWeight: 600 }}>{(v * 100).toFixed(1)}%</span>
                    </div>
                    <ScoreBar value={v} color={
                      k === 'entailment' ? 'var(--green)' :
                      k === 'contradiction' ? 'var(--red)' : 'var(--amber)'} />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Retriever tab */}
        {tab === 'retriever' && (
          <div className="detail-section">
            <div className="detail-section-title">Evidence Retriever</div>
            <div className="detail-row">
              <span className="detail-row-key">Retrieval Score</span>
              <span className="detail-row-value" style={{ color: scoreColor(c.retrieval_score) }}>
                {c.retrieval_score.toFixed(4)}
              </span>
            </div>
            {c.nli_evidence_meta?.length > 0 && c.nli_evidence_meta.map((m, i) => (
              <div key={i} style={{ marginTop: 10, background: 'var(--input-bg)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '9px 11px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', marginBottom: 5 }}>
                  <span style={{ color: 'var(--text-3)' }}>{m.source}</span>
                  <span style={{ color: 'var(--green)', fontWeight: 700 }}>sim {m.retrieval_score.toFixed(3)}</span>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-2)', margin: 0, lineHeight: 1.5 }}>
                  {m.passage_preview}{m.passage_preview.length >= 240 ? '…' : ''}
                </p>
              </div>
            ))}
          </div>
        )}

        {/* SelfCheck tab */}
        {tab === 'selfcheck' && (
          <div className="detail-section">
            <div className="detail-section-title">SelfCheck Consistency</div>
            <div className="detail-row">
              <span className="detail-row-key">Label</span>
              <span className="detail-row-value">{c.selfcheck_label}</span>
            </div>
            <div className="detail-row">
              <span className="detail-row-key">Consistency Score</span>
              <span className="detail-row-value" style={{ color: scoreColor(c.selfcheck_score) }}>
                {Math.round(c.selfcheck_score * 100)}%
              </span>
            </div>
            {c.selfcheck_votes && c.selfcheck_votes.length > 0 && (
              <div style={{ marginTop: 10 }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-3)', marginBottom: 6 }}>Sample votes</div>
                <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                  {c.selfcheck_votes.map((v, i) => (
                    <span key={i} style={{
                      padding: '2px 8px', borderRadius: 3, fontSize: '0.72rem', fontWeight: 700,
                      background: v === 'yes' ? 'var(--green-bg)' : 'var(--red-bg)',
                      color: v === 'yes' ? 'var(--green)' : 'var(--red)',
                      border: `1px solid ${v === 'yes' ? 'var(--green-border)' : 'var(--red-border)'}`,
                    }}>{v}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Fusion tab */}
        {tab === 'fusion' && (
          <div className="detail-section">
            <div className="detail-section-title">Score Fusion</div>
            <div className="detail-row">
              <span className="detail-row-key">Fused Score</span>
              <span className="detail-row-value" style={{ color: scoreColor(c.fused_score) }}>
                {c.fused_score.toFixed(4)}
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-row-key">Final Decision</span>
              <span className={`badge badge-${lc}`}>{c.final_label}</span>
            </div>
            <div className="detail-row">
              <span className="detail-row-key">Hallucination Type</span>
              <span className="detail-row-value" style={{ textTransform: 'capitalize' }}>
                {c.hallucination_type ?? '—'}
              </span>
            </div>
            <div style={{ marginTop: 10 }}>
              <ScoreBar value={c.fused_score} color={scoreColor(c.fused_score)} />
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default ClaimDetails;
