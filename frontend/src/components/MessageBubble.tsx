import type { GuardrailInfo, StageLatencies } from "../lib/api";
import { LatencyPanel } from "./LatencyPanel";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  sources?: string[];
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
              <div className="mt-2 flex flex-wrap gap-1">
                {message.sources.map((source) => (
                  <span
                    key={source}
                    className="rounded-full bg-black/30 px-2 py-0.5 text-[11px] text-slate-400"
                  >
                    #{source}
                  </span>
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
