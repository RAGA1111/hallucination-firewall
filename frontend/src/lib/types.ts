/**
 * Types derived directly from the FastAPI response models in api/main.py.
 * Do NOT invent fields — every field here exists in the backend response.
 */

// ── NLI evidence metadata ─────────────────────────────────────────────────────
export interface NliEvidenceMeta {
  retrieval_score: number;
  source: string;
  nli_label: string;
  nli_confidence: number;
  all_scores: Record<string, number>;
  passage_preview: string;
}

// ── Single verified + corrected claim ────────────────────────────────────────
export interface Claim {
  claim: string;
  final_label: 'SUPPORTED' | 'HALLUCINATED' | 'UNVERIFIABLE';
  status: string; // "VERIFIED" | "HALLUCINATED" | "UNVERIFIABLE" | "ERROR" | "TIMEOUT"
  fused_score: number;
  claim_confidence: number;
  nli_label: string;
  nli_confidence: number;
  nli_all_scores: Record<string, number>;
  nli_evidence: string;
  nli_evidence_meta: NliEvidenceMeta[];
  retrieval_score: number;
  selfcheck_label: string;
  selfcheck_score: number;
  selfcheck_votes: string[];
  explanation: string;
  hallucination_type: string;
  // Correction fields added by regenerator
  corrected_claim: string;
  correction_status: 'CORRECTED' | 'UNCHANGED' | 'UNVERIFIABLE' | 'FAILED';
  correction_attempt: number;
  correction_note: string;
}

// ── Pipeline summary block ───────────────────────────────────────────────────
export interface PipelineSummary {
  total_claims: number;
  supported: number;
  hallucinated: number;
  unverifiable: number;
  corrected: number;
  correction_failed: number;
  hallucination_rate: number;
  correction_rate: number;
}

// ── Timing block ─────────────────────────────────────────────────────────────
export interface PipelineTiming {
  total_seconds: number;
  decompose_seconds?: number;
  verify_seconds?: number;
  correction_seconds?: number;
  rebuild_seconds?: number;
}

// ── Annotated sentence (for HTML overlay) ───────────────────────────────────
export interface AnnotatedSentence {
  sentence: string;
  final_label: 'SUPPORTED' | 'HALLUCINATED' | 'UNVERIFIABLE';
  claim_confidence: number;
  css_class: string;
}

// ── Full /verify and /ask response ───────────────────────────────────────────
export interface VerifyResponse {
  original_response: string;
  question: string;
  final_response: string;
  summary: PipelineSummary;
  claims: Claim[];
  timing: PipelineTiming;
  status: string;
  annotated_sentences: AnnotatedSentence[] | null;
  final_response_html: string | null;
  error: string | null;
}

// ── GET /health ───────────────────────────────────────────────────────────────
export interface HealthResponse {
  status: string;
  mode: string;
  ollama: string;
  pipeline: string;
  kb_passages: number;
  startup_time_seconds: number | null;
}

// ── POST /verify request ─────────────────────────────────────────────────────
export interface VerifyRequest {
  response: string;
  question?: string;
  use_selfcheck?: boolean;
}

// ── Pipeline stage (frontend-only, not from API) ─────────────────────────────
export type StageState = 'idle' | 'running' | 'completed' | 'failed';

export interface PipelineStage {
  id: string;
  label: string;
  description: string;
  icon: string;
  state: StageState;
}

// ── App pages (frontend routing) ─────────────────────────────────────────────
export type Page = 'home' | 'results' | 'pipeline' | 'about';
