# GOAt AI

A Tamil/Hindi grounded RAG assistant — Qwen embedder + Qwen generator, dense
(Qdrant) + sparse (BM25) retrieval fused via weighted RRF, packaged as a
deployable backend + frontend with Sarvam streaming speech-to-text and
guardrails on top.

**The RAG logic itself is untouched.** `scripts/benchmark_full_rag_t4_latency_winner.py`
is the golden reference that produced the winning ~143ms P50 / 20-of-20
under-200ms result — it stays byte-identical forever. Everything under
`backend/`, `frontend/`, and `deploy/` packages that logic for real
deployment; it never redesigns it. See `backend/parity/test_parity.py` for
the mechanical proof the two stay behaviorally identical.

## Architecture

```
                 ┌─────────────┐        ┌───────────────────────────┐
 mic / text  →   │  frontend   │  →     │          backend          │
 (GOAt AI UI)    │  (nginx)    │  /api  │        (FastAPI)          │
                 └─────────────┘        │                            │
                        │               │  guardrails (in)           │
                        │  WS audio     │       ↓                    │
                        └─────────────→ │  Sarvam streaming STT      │
                                        │       ↓ transcript          │
                                        │  app.rag.engine.FullRAG     │
                                        │   embed → Qdrant + BM25     │
                                        │   → weighted RRF → context  │
                                        │       ↓                     │
                                        │  guardrails (grounding)     │
                                        │       ↓                     │
                                        │  Qwen generator              │
                                        │       ↓                     │
                                        │  guardrails (out: NOT_FOUND)│
                                        └──────────┬──────────────────┘
                                                   │
                                          ┌────────┴────────┐
                                          │  Qdrant (local)  │
                                          └──────────────────┘
```

## Repo layout

- `scripts/` — golden reference, never edited.
- `backend/app/rag/engine.py` — the same RAG logic, made importable; only
  data-root and Qdrant-URL resolution were parametrized for deployment.
- `backend/app/guardrails/` — input / grounding / output guardrails, wrap
  the engine, never edit it.
- `backend/app/stt/` — Sarvam realtime streaming STT client.
- `backend/app/api/` — the FastAPI harness (`/api/query`, `/api/stt/stream`,
  health/readiness/metrics).
- `backend/benchmarks/benchmark_full_rag.py` — the offline P50/P70/P90/P95/P100
  + recall/hit measurement, same math as the golden script.
- `backend/parity/test_parity.py` — deployment gate comparing golden vs.
  modular output.
- `frontend/` — React + Vite + TypeScript + Tailwind chat UI ("GOAt AI"),
  language switch, mic streaming, per-stage latency + guardrail badges.
- `deploy/` — docker-compose (Qdrant + backend + frontend) and a GCP GPU-VM
  deployment guide.
- `data/` — empty on purpose. See **DATA_SETUP.md**.

## Quickstart (once data is in place — see DATA_SETUP.md)

```bash
# 1. Index Qdrant (once)
cd backend
pip install -r requirements.txt
export GOAT_DATA_ROOT=$(pwd)/../data
python -m app.indexing.index_qdrant

# 2. (recommended) prove the modular engine matches the golden script
python -m parity.test_parity

# 3. Run the backend
export SARVAM_API_KEY=...        # for streaming STT
uvicorn app.main:app --reload

# 4. Run the frontend
cd ../frontend
npm install
npm run dev
```

Or with Docker Compose end-to-end (GPU host required):

```bash
cd deploy
cp .env.example .env   # fill in SARVAM_API_KEY
docker compose --env-file .env up -d --build
```

For a full GCP GPU VM deployment, see `deploy/gcloud/README.md`.

## Hackathon requirements → where they're met

1. **Speech-to-text** — Sarvam realtime streaming STT (`backend/app/stt/sarvam_client.py`,
   `backend/app/api/routes_stt.py`), browser mic captured as raw PCM16 and
   streamed over WebSocket end-to-end.
2. **Chunking** — fixed-size 384-char chunks with 96-char overlap
   (`chunks_25k/fixed_384_96/`), parent/child hierarchy (child-level dense
   search rolled up to parent-level fusion), metadata-aware (per-language
   BM25 k1/b tuning, per-collection dense weighting) — unchanged from the
   golden reference. `DATA_SETUP.md` documents how to add further chunking
   strategies as additional `CHUNKS`/`CFG` entries without touching the
   retrieval math.
3. **Latency target (<200ms)** — the frozen `CONTEXT_CHAR_BUDGET=350`,
   `MAX_NEW_TOKENS=16`, and retrieval depths are exactly what produced
   20-of-20 queries under 200ms in the golden run.
4. **Latency analytics** — `backend/benchmarks/benchmark_full_rag.py` reports
   P50/P70/P90/P95/P100 across 100+ deterministic queries; also exposed live
   via `GET /api/metrics`.
5. **Harness** — `backend/app/api/routes_query.py`: structured
   request/response schemas, discrete timed stages, a single whole-call
   retry on transient Qdrant errors (flagged, excluded from latency stats),
   structured error responses.
6. **Guardrails** — `backend/app/guardrails/guardrails.py`: input screening,
   a structural grounding check (no retrieved context → never call the
   generator), and graceful surfacing of the model's own `NOT_FOUND`
   sentinel instead of leaking it raw.
