import { useState } from "react";
import type { StageLatencies } from "../lib/api";

const STAGE_LABELS: [keyof StageLatencies, string][] = [
  ["embed_ms", "Embed"],
  ["dense_ms", "Dense (Qdrant)"],
  ["bm25_ms", "Sparse (BM25)"],
  ["retrieval_ms", "Retrieval total"],
  ["context_ms", "Context build"],
  ["prompt_prep_ms", "Prompt prep"],
  ["model_first_token_ms", "Model first token"],
  ["generation_ttft_ms", "Generation TTFT"],
  ["full_rag_ttft_ms", "Core RAG TTFT"],
  ["request_overhead_ms", "Request overhead"],
];

export function LatencyPanel({ latencies }: { latencies: StageLatencies }) {
  const [open, setOpen] = useState(false);

  const fullTtft = latencies.full_rag_ttft_ms;
  const underTarget = typeof fullTtft === "number" && fullTtft < 200;

  return (
    <div className="mt-2 text-xs">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-slate-400 hover:text-slate-200 transition-colors"
      >
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            underTarget ? "bg-goat-accent" : "bg-amber-400"
          }`}
        />
        {typeof fullTtft === "number" ? `${fullTtft.toFixed(1)}ms TTFT` : "latency"}
        <span className="text-slate-600">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 rounded-lg border border-goat-border bg-black/20 p-3">
          {STAGE_LABELS.map(([key, label]) => {
            const value = latencies[key];
            if (typeof value !== "number") return null;
            return (
              <div key={key} className="flex justify-between gap-3 text-slate-400">
                <span>{label}</span>
                <span className="font-mono text-slate-300">{value.toFixed(1)}ms</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
