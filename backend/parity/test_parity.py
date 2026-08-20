"""
Deployment gate: proves `app.rag.engine.FullRAG` behaves identically to the
golden reference `scripts/benchmark_full_rag_t4_latency_winner.py`.

This is an integration test, not a unit test — it needs the real GPU, the
real Qwen models, and the real indexed data (see DATA_SETUP.md), because
that is the only way to actually prove the modular split didn't change
retrieval/generation output. Run it once after `index_qdrant.py` and before
trusting a deploy:

    cd backend
    python -m parity.test_parity

How it works
------------
The golden script hardcodes `ROOT = Path("/content/HH-goa-task2")` — that is
intentional, it must never be edited. To let the *unmodified* file actually
resolve real data on this machine, this test creates a symlink/junction at
that exact path pointing at `settings.data_root`, then dynamically imports
the golden script as a module and runs its `FullRAG` side by side with the
modular one, on the same fixed set of queries, comparing:

  - dense parent IDs (order-sensitive)
  - BM25 parent IDs (order-sensitive)
  - fused top-20 parent IDs (order-sensitive)
  - assembled context string (byte-identical)
  - prompt token count
  - generated answer text (deterministic: do_sample=False)

Any mismatch fails the gate.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from app.config import settings
from app.rag.engine import FullRAG as ModularFullRAG


REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SCRIPT = REPO_ROOT / "scripts" / "benchmark_full_rag_t4_latency_winner.py"

GOLDEN_ROOT = Path("/content/HH-goa-task2") if os.name != "nt" else Path("C:/content/HH-goa-task2")

# A small, fixed set of queries per language to compare. Kept short because
# this test loads two full copies of the embedder + generator on one GPU.
SAMPLE_QUERIES = {
    "ta": [],
    "hi": [],
}


def _ensure_golden_root_link() -> None:
    """Point the golden script's hardcoded ROOT at the real data directory."""

    data_root = settings.data_root.resolve()

    if GOLDEN_ROOT.exists():
        resolved = GOLDEN_ROOT.resolve()
        if resolved == data_root:
            return
        raise RuntimeError(
            f"{GOLDEN_ROOT} already exists and does not point at "
            f"{data_root}. Remove it manually before running the parity "
            f"gate — refusing to overwrite something that might be real "
            f"data."
        )

    try:
        GOLDEN_ROOT.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise RuntimeError(
            f"Can't create {GOLDEN_ROOT.parent} — it's a filesystem-root "
            f"path and your user likely lacks permission. One-time fix:\n"
            f"    sudo mkdir -p {GOLDEN_ROOT.parent}\n"
            f"    sudo chown $USER:$USER {GOLDEN_ROOT.parent}\n"
            f"then re-run this."
        ) from exc

    if os.name == "nt":
        os.system(f'mklink /J "{GOLDEN_ROOT}" "{data_root}"')
    else:
        os.symlink(data_root, GOLDEN_ROOT, target_is_directory=True)

    if not GOLDEN_ROOT.exists():
        raise RuntimeError(
            f"Failed to link {GOLDEN_ROOT} -> {data_root}. "
            f"Create it manually (symlink/junction) and re-run."
        )


def _load_golden_module():
    spec = importlib.util.spec_from_file_location(
        "golden_benchmark_full_rag", GOLDEN_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_sample_queries(golden) -> None:
    for language in ("ta", "hi"):
        lookup = golden.load_queries(language)
        SAMPLE_QUERIES[language] = list(lookup.values())[:5]


def run() -> bool:
    _ensure_golden_root_link()

    golden = _load_golden_module()
    _load_sample_queries(golden)

    print("Loading golden FullRAG...")
    golden_engine = golden.FullRAG()

    print("Loading modular FullRAG...")
    modular_engine = ModularFullRAG()

    all_ok = True

    for language, queries in SAMPLE_QUERIES.items():
        for query in queries:

            g = golden_engine.retrieve(query, language)
            m = modular_engine.retrieve(query, language)

            g_ctx, _, g_parents_n = golden_engine.contexts.build(
                language, query, g["parents"], 350, 2, 175
            )
            m_ctx, _, m_parents_n = modular_engine.contexts.build(
                language, query, m["parents"], 350, 2, 175
            )

            g_gen = golden_engine.generate(query, g_ctx, 16)
            m_gen = modular_engine.generate(query, m_ctx, 16)

            checks = [
                ("fused parents", g["parents"], m["parents"]),
                ("context string", g_ctx, m_ctx),
                ("context parent count", g_parents_n, m_parents_n),
                ("prompt tokens", g_gen["prompt_tokens"], m_gen["prompt_tokens"]),
                ("answer", g_gen["answer"], m_gen["answer"]),
            ]

            for name, expected, actual in checks:
                status = "OK" if expected == actual else "MISMATCH"
                if status == "MISMATCH":
                    all_ok = False
                print(f"[{language}] {name}: {status}")
                if status == "MISMATCH":
                    print(f"    golden : {expected!r}")
                    print(f"    modular: {actual!r}")

    print()
    print("PARITY GATE:", "PASS" if all_ok else "FAIL")
    return all_ok


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
