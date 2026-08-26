import React from 'react';

interface TechItem {
  name: string;
  role: string;
  icon: React.ReactNode;
}

const accent = (children: React.ReactNode) => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">{children}</svg>
);

const TECH: TechItem[] = [
  {
    name: 'FAISS',
    role: 'Vector similarity search',
    icon: accent(<><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></>),
  },
  {
    name: 'DeBERTa-v3-small',
    role: 'NLI cross-encoder',
    icon: accent(<><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 0 2 2z"/></>),
  },
  {
    name: 'all-MiniLM-L6-v2',
    role: 'Sentence embeddings',
    icon: accent(<><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></>),
  },
  {
    name: 'Ollama (llama3.2:1b)',
    role: 'Local LLM inference',
    icon: accent(<><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></>),
  },
  {
    name: 'FastAPI',
    role: 'REST API backend',
    icon: accent(<><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></>),
  },
  {
    name: 'React + TypeScript',
    role: 'Frontend framework',
    icon: accent(<><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></>),
  },
  {
    name: 'HuggingFace Transformers',
    role: 'NLI model loading',
    icon: accent(<><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></>),
  },
  {
    name: 'Wikipedia REST API',
    role: 'Dynamic KB augmentation',
    icon: accent(<><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></>),
  },
];

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div style={{ marginBottom: 28 }}>
    <h2 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text)', marginBottom: 10, paddingBottom: 8, borderBottom: '1px solid var(--border)' }}>
      {title}
    </h2>
    {children}
  </div>
);

const AboutPage: React.FC = () => (
  <div style={{ maxWidth: 780 }}>
    <div className="page-header">
      <h1 className="page-title">About Hallucination Firewall</h1>
      <p className="page-sub">Final Year Project — AI Response Verification System</p>
    </div>

    {/* Project card */}
    <div className="card" style={{ marginBottom: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 16 }}>
        <div style={{ width: 44, height: 44, borderRadius: 10, background: 'var(--accent-glow)', border: '1px solid rgba(59,130,246,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <polyline points="9 12 11 14 15 10"/>
          </svg>
        </div>
        <div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text)' }}>Hallucination Firewall</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-2)' }}>Evidence-based AI response verification · v1.0.0</div>
        </div>
      </div>
      <p style={{ fontSize: '0.9rem', color: 'var(--text-2)', lineHeight: 1.65, margin: 0 }}>
        Hallucination Firewall is an inference-time verification system that detects and corrects
        factual hallucinations in AI-generated responses. It decomposes each response into atomic
        claims, verifies each independently using retrieval-augmented NLI, SelfCheck consistency
        sampling, and symbolic reasoning, then fuses all signals into a calibrated verdict.
        Hallucinated claims are automatically rewritten using retrieved evidence via
        Chain-of-Verification (CoVe).
      </p>
    </div>

    <Section title="Purpose">
      <p style={{ fontSize: '0.875rem', color: 'var(--text-2)', lineHeight: 1.65 }}>
        Large language models are prone to generating confident but factually incorrect statements —
        a phenomenon known as hallucination. This system provides a transparent, explainable
        verification layer that can be placed in front of any LLM response before it reaches the user.
        Every decision is backed by retrievable evidence, not opaque model confidence alone.
      </p>
    </Section>

    <Section title="Verification Pipeline">
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 10 }}>
        {[
          ['Claim Decomposition', 'LLM-based atomic claim extraction'],
          ['Evidence Retrieval', 'FAISS vector search + Wikipedia fallback'],
          ['NLI Verification', 'DeBERTa-v3-small cross-encoder'],
          ['SelfCheck', 'LLM consistency sampling (N=3)'],
          ['Symbolic Verification', 'Numeric/date constraint checking via AST'],
          ['Score Fusion', '60% NLI + 40% SelfCheck weighted fusion'],
          ['Hallucination Classification', 'Semantic error type assignment'],
          ['Regeneration', 'Chain-of-Verification rewriting'],
        ].map(([name, desc]) => (
          <div key={name} style={{ background: 'var(--input-bg)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '9px 12px' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text)', marginBottom: 3 }}>{name}</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-3)' }}>{desc}</div>
          </div>
        ))}
      </div>
    </Section>

    <Section title="Technologies Used">
      <div className="about-tech-grid">
        {TECH.map(t => (
          <div key={t.name} className="tech-pill">
            <div className="tech-pill-icon" aria-hidden="true">{t.icon}</div>
            <div>
              <div className="tech-pill-name">{t.name}</div>
              <div className="tech-pill-role">{t.role}</div>
            </div>
          </div>
        ))}
      </div>
    </Section>

    <Section title="Architecture Decisions">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {[
          ['No fine-tuning required', 'All models used zero-shot — no labelled hallucination data needed.'],
          ['Local-first', 'LLM inference runs on Ollama locally. No external AI API calls, no data leaves the machine.'],
          ['Graceful degradation', 'If Ollama is offline, the system falls back to NLI-only verification with Python sentence splitting.'],
          ['Transparent verdicts', 'Every claim verdict exposes the evidence, NLI probabilities, and SelfCheck votes that produced it.'],
        ].map(([k, v]) => (
          <div key={k} style={{ background: 'var(--input-bg)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '10px 13px', display: 'flex', gap: 10 }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginTop: 2 }}>
              <polyline points="20 6 9 17 4 12"/>
            </svg>
            <div>
              <span style={{ fontSize: '0.83rem', fontWeight: 600, color: 'var(--text)' }}>{k} — </span>
              <span style={{ fontSize: '0.83rem', color: 'var(--text-2)' }}>{v}</span>
            </div>
          </div>
        ))}
      </div>
    </Section>

    <div style={{ fontSize: '0.75rem', color: 'var(--text-3)', paddingTop: 8, borderTop: '1px solid var(--border)' }}>
      Built with ♥ for trustworthy AI · Hallucination Firewall v1.0.0
    </div>
  </div>
);

export default AboutPage;
