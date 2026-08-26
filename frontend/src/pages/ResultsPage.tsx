import React, { useState, useCallback } from 'react';
import type { Claim, VerifyResponse } from '../lib/types';
import VerificationSummary from '../components/VerificationSummary';
import ClaimList from '../components/ClaimList';
import ClaimDetails from '../components/ClaimDetails';

interface Props {
  result: VerifyResponse;
  onBack: () => void;
}

const CopyIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
  </svg>
);

const BackIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="15 18 9 12 15 6"/>
  </svg>
);

const ResultsPage: React.FC<Props> = ({ result, onBack }) => {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(0);
  const [copied, setCopied] = useState(false);

  const selectedClaim: Claim | null = selectedIndex !== null ? result.claims[selectedIndex] : null;

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(result.original_response).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [result.original_response]);

  const handleDownload = useCallback(() => {
    const data = {
      original_response: result.original_response,
      question: result.question,
      summary: result.summary,
      timing: result.timing,
      claims: result.claims.map(c => ({
        claim: c.claim,
        final_label: c.final_label,
        claim_confidence: c.claim_confidence,
        corrected_claim: c.corrected_claim,
        correction_status: c.correction_status,
        explanation: c.explanation,
      })),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hallucination-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  return (
    <div>
      {/* Back button */}
      <div style={{ marginBottom: 18 }}>
        <button className="btn btn-ghost" onClick={onBack} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <BackIcon />
          New Verification
        </button>
      </div>

      {/* Summary */}
      <VerificationSummary result={result} onDownload={handleDownload} />

      {/* Three-panel layout */}
      <div className="results-panels">

        {/* Panel 1 — Original response */}
        <div className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-title">Original AI Response</div>
              <div className="panel-subtitle">{result.question || 'No question provided'}</div>
            </div>
            <button className="btn btn-icon" onClick={handleCopy} title="Copy response" aria-label="Copy original response">
              {copied ? (
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
              ) : (
                <CopyIcon />
              )}
            </button>
          </div>
          <div className="panel-body">
            <p className="original-response-text">{result.original_response}</p>
          </div>
        </div>

        {/* Panel 2 — Claim list */}
        <div className="panel">
          <ClaimList
            claims={result.claims}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
          />
        </div>

        {/* Panel 3 — Claim details */}
        <div className="panel">
          {selectedClaim !== null && selectedIndex !== null ? (
            <ClaimDetails
              claim={selectedClaim}
              claimIndex={selectedIndex}
              onClose={() => setSelectedIndex(null)}
            />
          ) : (
            <>
              <div className="panel-header">
                <div className="panel-title">Verification Details</div>
              </div>
              <div className="panel-body">
                <div className="no-selection-placeholder">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                  </svg>
                  <span>Select a claim to see<br />verification details</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;
