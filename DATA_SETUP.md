# Data setup

This repo ships with no data — the chunked corpora, precomputed embeddings, and
evaluation query files are produced offline and are expected to already exist
(you generated them alongside `scripts/benchmark_full_rag_t4_latency_winner.py`).
Drop them into `data/` at the **exact** paths below. These paths mirror the
golden script's own path expectations (`ROOT / "chunks_25k" / ...`,
`ROOT / "embeddings_256" / ...`, and `ROOT / "data" / "eval" / ...` — note the
extra `data/` nesting under eval, inherited unchanged from the original script)
with `GOAT_DATA_ROOT` pointed at this `data/` folder.

```
data/
├── chunks_25k/
│   └── fixed_384_96/
│       ├── tamil.parquet          # parent_id + text/chunk_text/content column
│       └── hindi.parquet
│
├── embeddings_256/
│   └── fixed_384_96/
│       ├── ta/
│       │   ├── embeddings.npy     # float32, shape (N, 256)
│       │   └── metadata.parquet   # parent_id (+ optional chunk_id, vector_row)
│       └── hi/
│           ├── embeddings.npy
│           └── metadata.parquet
│
├── data/
│   └── eval/
│       ├── tamil_validation_queries.parquet   # query_id/id + query/translated_query/target_query/Query
│       └── hindi_validation_queries.parquet
│
└── final_retrieval_tuning/
    └── candidate_ceiling_per_query.parquet     # query_id, language, relevant_parent_ids
                                                  # (analysis_type == "candidate_depth_100" rows used)
```

## After the files are in place

1. **Set the data root** (defaults to `./data` relative to the repo root if unset):
   ```
   export GOAT_DATA_ROOT=/path/to/GOAt_AI/data
   ```

2. **Index Qdrant** (loads `embeddings_256/...` into the local vector DB, builds HNSW):
   ```
   cd backend
   python -m app.indexing.index_qdrant
   ```

3. **Run the parity gate** to confirm the modularized backend behaves identically
   to the golden reference script before trusting it in production:
   ```
   python -m parity.test_parity
   ```

4. **Run the offline latency/quality benchmark** (P50/P70/P90/P95/P100 + recall/hit),
   the same measurement the golden script produces:
   ```
   python -m benchmarks.benchmark_full_rag --per-language 50
   ```

5. Start the backend (`uvicorn app.main:app`) — `/readyz` will report `not_ready`
   with a reason until the data files above are present and Qdrant is populated.

If you use a different chunking strategy or add more languages later, keep the
same directory shape (`chunks_25k/<strategy_name>/...`) and add a matching entry
to `CHUNKS`/`CFG`/`COLLECTIONS` in `backend/app/rag/engine.py` and
`backend/app/indexing/index_qdrant.py` — do not change the retrieval math itself,
only the data pointers.
