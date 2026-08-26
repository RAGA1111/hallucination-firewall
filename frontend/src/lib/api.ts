/**
 * API layer — all fetch calls live here.
 * Components never call fetch directly.
 */

import type { HealthResponse, VerifyRequest, VerifyResponse } from './types';

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

export function verifyResponse(payload: VerifyRequest): Promise<VerifyResponse> {
  return request<VerifyResponse>('/verify', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function addKbPassages(
  passages: string[]
): Promise<{ status: string; added: number; total_passages: number }> {
  return request('/kb/add', {
    method: 'POST',
    body: JSON.stringify({ passages }),
  });
}

export function fetchKbInfo(): Promise<Record<string, unknown>> {
  return request('/kb/info');
}
