import { useState } from "react";
import type { GuardrailInfo, SourceEvidence, StageLatencies } from "../lib/api";
import { LatencyPanel } from "./LatencyPanel";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  sources?: SourceEvidence[];
  stageLatencies?: StageLatencies;
  guardrail?: GuardrailInfo;
  retried?: boolean;
  pending?: boolean;
}

const GUARDRAIL_LABEL: Record<string, string> = {
  empty_query: "Empty query",
  query_too_long: "Query too long",
  unsupported_language: "Unsupported language",
  unsafe_content: "Blocked: unsafe content",
  no_grounded_context: "Not found in knowledge base",
  ungrounded_answer: "Not found in knowledge base",
};

const LANE_COLORS: Record<string, string> = {
  dense: "text-sky-400",
  bm25: "text-emerald-400",
  fallback: "text-slate-500",
};

function EvidenceCard({
  evidence,
  index,
}: {
  evidence: SourceEvidence;
  index: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const lane = (evidence.lane ?? "unknown").toUpperCase();
  const laneColor =
    LANE_COLORS[(evidence.lane ?? "").toLowerCase()] ?? "text-slate-400";

  return (
    <div className="mt-2 rounded-lg border border-goat-border bg-black/20 p-3 text-xs">
      {/* Header */}
      <div className={`mb-1.5 flex items-center gap-2 font-semibold uppercase tracking-widest ${laneColor}`}>
        <span className="text-slate-500">SOURCE {index + 1}</span>
        <span className="text-slate-600">·</span>
        <span>{lane}</span>
      </div>

      {/* Evidence snippet */}
      <blockquote className="border-l-2 border-goat-accent/40 pl-2.5 text-slate-300 leading-relaxed">
        "{evidence.text}"
      </blockquote>

      {/* Expandable parent ID */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="mt-1.5 text-[10px] text-slate-600 hover:text-slate-400 transition-colors"
      >
        {expanded ? "▲ hide" : "▼ details"}
      </button>

      {expanded && (
        <div className="mt-1.5 space-y-0.5 text-[10px] text-slate-500 font-mono">
          <div>Parent: {evidence.parent_id}</div>
          {evidence.chunk_id && (
            <div>Chunk: {evidence.chunk_id}</div>
          )}
          {typeof evidence.score === "number" && (
            <div>Score: {evidence.score.toFixed(4)}</div>
          )}
        </div>
      )}
    </div>
  );
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const blocked = message.guardrail?.blocked;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-goat-accent2/20 border border-goat-accent2/30 text-slate-100"
            : blocked
            ? "bg-amber-400/10 border border-amber-400/30 text-amber-100"
            : "bg-goat-panel border border-goat-border text-slate-100"
        }`}
      >
        {message.pending ? (
          <div className="flex items-center gap-1.5 py-1">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-goat-accent" />
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-goat-accent [animation-delay:0.15s]" />
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-goat-accent [animation-delay:0.3s]" />
          </div>
        ) : (
          <>
            {blocked && message.guardrail?.code && (
              <div className="mb-1 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-amber-300">
                <span>⚠</span>
                {GUARDRAIL_LABEL[message.guardrail.code] ?? message.guardrail.code}
              </div>
            )}

            <p className="whitespace-pre-wrap leading-relaxed">{message.text}</p>

            {message.sources && message.sources.length > 0 && (
              <div className="mt-3">
                <div className="text-[10px] font-semibold uppercase tracking-widest text-slate-500 mb-1">
                  Grounded Evidence
                </div>
                {message.sources.map((src, i) => (
                  <EvidenceCard key={src.parent_id + i} evidence={src} index={i} />
                ))}
              </div>
            )}

            {message.retried && (
              <div className="mt-1 text-[11px] text-slate-500">retried once (transient Qdrant error)</div>
            )}

            {!isUser && message.stageLatencies && (
              <LatencyPanel latencies={message.stageLatencies} />
            )}
          </>
        )}
      </div>
    </div>
  );
}
