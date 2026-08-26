import React, { useState, useCallback } from 'react';
import VerificationPipeline, { INITIAL_STAGES } from '../components/VerificationPipeline';
import type { PipelineStage, StageState, VerifyResponse } from '../lib/types';
import { verifyResponse } from '../lib/api';

interface Props {
  onResult: (r: VerifyResponse) => void;
}

const MAX_CHARS = 5000;

const STAGE_ORDER = ['extract', 'retrieval', 'nli', 'selfcheck', 'symbolic', 'fusion', 'correction'];

function advanceStages(stages: PipelineStage[], runningId: string): PipelineStage[] {
  return stages.map(s => {
    const idx = STAGE_ORDER.indexOf(s.id);
    const runIdx = STAGE_ORDER.indexOf(runningId);
    if (idx < runIdx) return { ...s, state: 'completed' as StageState };
    if (idx === runIdx) return { ...s, state: 'running' as StageState };
    return { ...s, state: 'idle' as StageState };
  });
}

function completeStages(stages: PipelineStage[]): PipelineStage[] {
  return stages.map(s => ({ ...s, state: 'completed' as StageState }));
}

function resetStages(stages: PipelineStage[]): PipelineStage[] {
  return stages.map(s => ({ ...s, state: 'idle' as StageState }));
}

const ShieldIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    <polyline points="9 12 11 14 15 10"/>
  </svg>
);

const HomePage: React.FC<Props> = ({ onResult }) => {
  const [text, setText] = useState('');
  const [question, setQuestion] = useState('');
  const [useSelfCheck, setUseSelfCheck] = useState(false);
  const [stages, setStages] = useState<PipelineStage[]>(INITIAL_STAGES);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Simulated stage progression tied to real timing
  const runPipelineAnimation = useCallback(async (payload: { response: string; question: string; use_selfcheck: boolean }) => {
    setError(null);
    setLoading(true);
    setStages(advanceStages(INITIAL_STAGES, 'extract'));

    // Stagger stage highlights — the real call is concurrent
    const delays: Record<string, number> = {
      extract: 0,
      retrieval: 800,
      nli: 2200,
      selfcheck: 4500,
      symbolic: 6000,
      fusion: 7000,
      correction: 8000,
    };

    const timers: ReturnType<typeof setTimeout>[] = [];
    STAGE_ORDER.forEach((id, i) => {
      if (i === 0) return; // already set
      timers.push(setTimeout(() => {
        setStages(prev => advanceStages(prev, id));
      }, delays[id]));
    });

    try {
      const result = await verifyResponse(payload);
      timers.forEach(clearTimeout);
      setStages(completeStages(INITIAL_STAGES));
      setTimeout(() => {
        setLoading(false);
        onResult(result);
      }, 400);
    } catch (e: unknown) {
      timers.forEach(clearTimeout);
      setStages(resetStages(INITIAL_STAGES));
      setLoading(false);
      setError(e instanceof Error ? e.message : 'Verification failed. Check backend is running.');
    }
  }, [onResult]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || loading) return;
    runPipelineAnimation({ response: text.trim(), question: question.trim(), use_selfcheck: useSelfCheck });
  };

  const charCount = text.length;
  const overLimit = charCount > MAX_CHARS;

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Verify AI Response</h1>
        <p className="page-sub">Paste an AI-generated response below to verify its factual accuracy.</p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit}>
          {/* Optional question field */}
          <div className="form-group" style={{ marginBottom: 14 }}>
            <label htmlFor="question-input">Question (optional)</label>
            <input
              id="question-input"
              type="text"
              placeholder="What question did you ask the AI? (improves results)"
              value={question}
              onChange={e => setQuestion(e.target.value)}
              disabled={loading}
            />
          </div>

          {/* Main textarea */}
          <div className="form-group">
            <label htmlFor="response-input">AI Response to Verify</label>
            <textarea
              id="response-input"
              placeholder="Paste AI response here…"
              value={text}
              onChange={e => setText(e.target.value)}
              disabled={loading}
              style={{
                minHeight: 160,
                borderColor: overLimit ? 'var(--red)' : undefined,
              }}
              aria-describedby="char-count"
              required
            />
            <div
              id="char-count"
              style={{
                textAlign: 'right',
                fontSize: '0.75rem',
                color: overLimit ? 'var(--red)' : 'var(--text-3)',
                marginTop: 4,
              }}
            >
              {charCount.toLocaleString()} / {MAX_CHARS.toLocaleString()}
            </div>
          </div>

          {/* SelfCheck toggle */}
          <div style={{ marginBottom: 18 }}>
            <label className="toggle-row" htmlFor="sc-toggle">
              <input
                id="sc-toggle"
                type="checkbox"
                checked={useSelfCheck}
                onChange={e => setUseSelfCheck(e.target.checked)}
                disabled={loading}
              />
              Enable SelfCheck sampling — slower but more accurate
            </label>
          </div>

          {/* Error */}
          {error && (
            <div
              role="alert"
              style={{
                background: 'var(--red-bg)',
                border: '1px solid var(--red-border)',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 13px',
                fontSize: '0.83rem',
                color: 'var(--red)',
                marginBottom: 14,
              }}
            >
              {error}
            </div>
          )}

          {/* Submit */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 12 }}>
            {loading && (
              <span style={{ fontSize: '0.78rem', color: 'var(--text-3)' }}>
                Running verification pipeline…
              </span>
            )}
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !text.trim() || overLimit}
              aria-busy={loading}
            >
              {loading ? (
                <><span className="spinner" aria-hidden="true" /> Verifying…</>
              ) : (
                <><ShieldIcon /> Verify Response</>
              )}
            </button>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-3)', textAlign: 'right', marginTop: 5 }}>
            Runs full verification pipeline
          </div>
        </form>

        {/* Pipeline stages */}
        <VerificationPipeline stages={stages} />
      </div>
    </div>
  );
};

export default HomePage;
