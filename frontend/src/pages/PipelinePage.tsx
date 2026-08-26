import React from 'react';

interface FlowStep {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  purpose: string;
  input: string;
  output: string;
  tech: string;
}

const steps: FlowStep[] = [
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
      </svg>
    ),
    title: 'Claim Decomposition',
    subtitle: 'Break the response into atomic, independently verifiable statements',
    purpose: 'LLM responses often contain multiple claims per sentence. Each must be verified individually.',
    input: 'Raw LLM response string',
    output: 'List of atomic claim strings',
    tech: 'Ollama (llama3.2:1b) with JSON extraction + heuristic sentence fallback',
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
    ),
    title: 'Evidence Retrieval',
    subtitle: 'Find the most semantically similar passages from the knowledge base',
    purpose: 'Provides grounding evidence for NLI scoring. High retrieval similarity means the KB can speak to the claim.',
    input: 'Atomic claim string',
    output: 'Top-k passages with cosine similarity scores',
    tech: 'FAISS IndexFlatIP · all-MiniLM-L6-v2 embeddings · Wikipedia REST API fallback',
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 0 2 2z"/>
      </svg>
    ),
    title: 'NLI Verification',
    subtitle: 'Determine whether evidence entails, contradicts, or is neutral to the claim',
    purpose: 'Cross-encoder NLI is the primary signal — more accurate than cosine similarity alone.',
    input: '"evidence [SEP] claim" text pair',
    output: 'Label (SUPPORTED / CONTRADICTED / NEUTRAL) + confidence + per-class probabilities',
    tech: 'cross-encoder/nli-deberta-v3-small (HuggingFace Transformers)',
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
      </svg>
    ),
    title: 'SelfCheck',
    subtitle: 'Ask the LLM the same question N times and measure answer consistency',
    purpose: 'Hallucinated facts produce inconsistent answers across samples. Consistency is a reliability signal.',
    input: 'Claim string + optional context',
    output: 'Consistency score (0–1) · yes/no vote distribution · CONSISTENT / INCONSISTENT / UNCERTAIN',
    tech: 'Ollama (llama3.2:1b) · N=3 samples at temperature 0.7 · async parallel calls',
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      </svg>
    ),
    title: 'Symbolic Verification',
    subtitle: 'Evaluate any numeric or date constraints embedded in the claim',
    purpose: 'Catches arithmetic errors like "born in 1879, died 76 years later in 1935" — no evidence needed.',
    input: 'Claim string',
    output: 'has_logic (bool) · passed (bool) · note string',
    tech: 'Ollama → Python AST · restricted node allowlist (no eval / exec)',
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
      </svg>
    ),
    title: 'Score Fusion',
    subtitle: 'Combine NLI, SelfCheck, and symbolic signals into a single weighted verdict',
    purpose: 'No single signal is sufficient. Fusion weights NLI at 60% and SelfCheck at 40%.',
    input: 'NLI result + SelfCheck result + fuse_high / fuse_low thresholds',
    output: 'final_label (SUPPORTED / HALLUCINATED / UNVERIFIABLE) · fused_score (0–1)',
    tech: 'Weighted fusion · contradiction priority rule · configurable thresholds',
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3"/><path d="M19.07 4.93A10 10 0 0 0 4.93 19.07"/><path d="M4.93 4.93a10 10 0 0 0 0 14.14"/>
      </svg>
    ),
    title: 'Hallucination Classification',
    subtitle: 'Assign a semantic type to each hallucinated claim',
    purpose: 'Different hallucination types require different handling — factual errors vs. unverifiable claims.',
    input: 'Verification result dict',
    output: 'hallucination_type string (e.g. "factual_error", "unknown")',
    tech: 'Rule-based classifier in core/hallucination_type.py',
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
      </svg>
    ),
    title: 'Regeneration',
    subtitle: 'Rewrite hallucinated claims using Chain-of-Verification (CoVe)',
    purpose: 'Rather than just flagging errors, the system attempts to produce a corrected, evidence-grounded rewrite.',
    input: 'Hallucinated claim + retrieved evidence',
    output: 'corrected_claim · correction_status (CORRECTED / UNCHANGED / FAILED)',
    tech: 'Ollama 3-step CoVe chain · NLI validation of output · max 2 attempts',
  },
];

const PipelinePage: React.FC = () => (
  <div>
    <div className="page-header">
      <h1 className="page-title">How It Works</h1>
      <p className="page-sub">
        Hallucination Firewall runs a multi-stage verification pipeline on every AI response.
        Each stage adds an independent signal before the final verdict is produced.
      </p>
    </div>

    <div className="card" style={{ marginBottom: 20, padding: '14px 18px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap', justifyContent: 'center' }}>
        {['Input', 'Decompose', 'Retrieve', 'NLI', 'SelfCheck', 'Symbolic', 'Fusion', 'Classify', 'Correct', 'Output'].map((s, i, arr) => (
          <React.Fragment key={s}>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: i === 0 || i === arr.length - 1 ? 'var(--accent)' : 'var(--text-2)' }}>{s}</span>
            {i < arr.length - 1 && (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--border-strong)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="9 18 15 12 9 6"/>
              </svg>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>

    <div className="pipeline-flow">
      {steps.map((step, idx) => (
        <div key={idx} className="pipeline-flow-item">
          <div className="flow-line" aria-hidden="true" />
          <div className="flow-dot" aria-hidden="true">{step.icon}</div>
          <div className="flow-content">
            <div className="flow-title">{idx + 1}. {step.title}</div>
            <div className="flow-subtitle">{step.subtitle}</div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-2)', margin: 0, lineHeight: 1.55 }}>
              {step.purpose}
            </p>
            <div className="flow-detail-grid">
              <div className="flow-detail-item">
                <div className="flow-detail-key">Input</div>
                <div className="flow-detail-value">{step.input}</div>
              </div>
              <div className="flow-detail-item">
                <div className="flow-detail-key">Output</div>
                <div className="flow-detail-value">{step.output}</div>
              </div>
              <div className="flow-detail-item" style={{ gridColumn: 'span 2' }}>
                <div className="flow-detail-key">Technology</div>
                <div className="flow-detail-value">{step.tech}</div>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  </div>
);

export default PipelinePage;
