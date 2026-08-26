import React from 'react';
import type { PipelineStage } from '../lib/types';

interface Props {
  stages: PipelineStage[];
}

const ICONS: Record<string, React.ReactNode> = {
  extract: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
    </svg>
  ),
  retrieval: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
  ),
  nli: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>
  ),
  selfcheck: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
    </svg>
  ),
  symbolic: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  ),
  fusion: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
      <line x1="6" y1="20" x2="6" y2="14"/>
    </svg>
  ),
  correction: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
    </svg>
  ),
};

const CheckIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);

const FailIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);

const VerificationPipeline: React.FC<Props> = ({ stages }) => (
  <div style={{ marginTop: '24px' }}>
    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '16px' }}>
      Verification Pipeline
    </div>
    <div className="pipeline-track" role="list" aria-label="Verification pipeline stages">
      {stages.map((stage) => {
        const icon = ICONS[stage.id] ?? ICONS['extract'];
        return (
          <div key={stage.id} className={`pipeline-stage ${stage.state}`} role="listitem">
            <div className={`stage-icon ${stage.state}`} aria-label={`${stage.label}: ${stage.state}`}>
              {stage.state === 'completed' ? <CheckIcon /> :
               stage.state === 'failed'    ? <FailIcon /> :
               icon}
            </div>
            <div className={`stage-label ${stage.state}`}>{stage.label}</div>
            <div className="stage-desc">{stage.description}</div>
            {/* invisible connector line drawn via CSS ::after */}
          </div>
        );
      })}
    </div>
  </div>
);

export const INITIAL_STAGES: PipelineStage[] = [
  { id: 'extract',    label: 'Claim Extraction', description: 'Extract statements',    icon: 'extract',    state: 'idle' },
  { id: 'retrieval',  label: 'Evidence Retrieval', description: 'Search knowledge base', icon: 'retrieval', state: 'idle' },
  { id: 'nli',        label: 'NLI Verification',  description: 'Entailment analysis',   icon: 'nli',       state: 'idle' },
  { id: 'selfcheck',  label: 'SelfCheck',          description: 'Consistency check',     icon: 'selfcheck', state: 'idle' },
  { id: 'symbolic',   label: 'Symbolic Check',     description: 'Rule-based verification', icon: 'symbolic', state: 'idle' },
  { id: 'fusion',     label: 'Score Fusion',       description: 'Aggregate results',     icon: 'fusion',    state: 'idle' },
  { id: 'correction', label: 'Correction',         description: 'Generate corrections',  icon: 'correction', state: 'idle' },
];

export default VerificationPipeline;
