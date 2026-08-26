export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export interface Claim {
  claim: string;
  final_label: 'SUPPORTED' | 'HALLUCINATED' | 'UNVERIFIABLE' | string;
  fused_score: number;
  claim_confidence?: number;
  retrieval_score?: number;
  nli_label: string;
  nli_confidence: number;
  nli_evidence?: string;
  nli_all_scores?: Record<string, number>;
  hallucination_type?: string;
  selfcheck_label: string;
  selfcheck_score: number;
  corrected_claim: string;
  correction_status: string;
  explanation: string;
}

export interface VerifyResponseData {
  original_response: string;
  question: string;
  final_response: string;
  summary: {
    total_claims: number;
    supported: number;
    hallucinated: number;
    unverifiable: number;
    corrected: number;
    hallucination_rate: number;
  };
  claims: Claim[];
  timing: {
    total_seconds: number;
    decomposition_seconds?: number;
    retrieval_seconds?: number;
    verification_seconds?: number;
    regeneration_seconds?: number;
  };
  status: string;
  annotated_sentences?: Array<{
    sentence: string;
    final_label: string;
    claim_confidence: number;
  }>;
  final_response_html?: string;
  error?: string;
}

export interface HealthResponse {
  status: string;
  mode?: string;
  ollama: string;
  pipeline: string;
  kb_passages: number;
  startup_time_seconds: number | null;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  return res.json();
}

export async function verifyResponse(
  response: string,
  question: string = '',
  useSelfCheck: boolean = false
): Promise<VerifyResponseData> {
  const res = await fetch(`${API_BASE_URL}/verify`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      response,
      question,
      use_selfcheck: useSelfCheck,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `Server returned ${res.status}`);
  }

  return res.json();
}

export async function addKbPassages(passages: string[]): Promise<{ status: string; added: number; total_passages: number }> {
  const res = await fetch(`${API_BASE_URL}/kb/add`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ passages }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to add passages' }));
    throw new Error(err.detail || `Server error ${res.status}`);
  }

  return res.json();
}
