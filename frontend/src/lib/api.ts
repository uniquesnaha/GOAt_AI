export type Language = "ta" | "hi";

export interface StageLatencies {
  embed_ms?: number | null;
  dense_ms?: number | null;
  bm25_ms?: number | null;
  retrieval_ms?: number | null;
  context_ms?: number | null;
  prompt_prep_ms?: number | null;
  model_first_token_ms?: number | null;
  generation_ttft_ms?: number | null;
  generation_complete_ms?: number | null;
  full_rag_ttft_ms?: number | null;
  full_rag_complete_ms?: number | null;
  request_overhead_ms?: number | null;
}

export interface GuardrailInfo {
  blocked: boolean;
  stage?: string | null;
  code?: string | null;
  reason?: string | null;
}

export interface SourceEvidence {
  parent_id: string;
  chunk_id?: string | null;
  lane?: string | null;
  score?: number | null;
  text: string;
}

export interface QueryResponse {
  answer: string;
  language: Language;
  sources: SourceEvidence[];
  stage_latencies: StageLatencies;
  guardrail: GuardrailInfo;
  retried: boolean;
}

export async function postQuery(query: string, language: Language): Promise<QueryResponse> {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, language }),
  });

  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail ?? `Request failed (${res.status})`);
  }

  return res.json();
}

export interface TranscriptEvent {
  event: string;
  text?: string | null;
  is_final: boolean;
  stt_latency_ms?: number;
}

export function openSttSocket(
  language: Language,
  onEvent: (evt: TranscriptEvent) => void
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/api/stt/stream?language=${language}`);
  ws.binaryType = "arraybuffer";

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onEvent(data);
    } catch {
      // ignore malformed frames
    }
  };

  return ws;
}
