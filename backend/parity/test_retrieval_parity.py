"""
Retrieval parity gate.

Correction 1 (quality patch): This gate now checks RETRIEVAL PARITY ONLY.
The full-pipeline parity gate (context + answer + prompt tokens) has been
intentionally retired because the quality patch deliberately changes:

  - ContextStore.build()  (175→168 chars, prefix-cut→query-centered window,
                           guessed child→actual retrieved child)
  - MAX_NEW_TOKENS        (16→24)
  - sources/provenance    (structured SourceEvidence)

WHAT IS CHECKED (must be 100% identical):
  - fused top-20 parent IDs (order-sensitive)

WHAT IS NOT CHECKED (intentionally different after quality patch):
  - context string
  - assembled answer
  - prompt token count

Run from backend/ after `index_qdrant.py` and before trusting a deploy:

    cd backend
    python -m parity.test_retrieval_parity

How it works
------------
The golden script hardcodes `ROOT = Path("/content/HH-goa-task2")` — that is
intentional, it must never be edited. To let the *unmodified* file resolve
real data on this machine, this test creates a symlink/junction at that
exact path pointing at `settings.data_root`, then dynamically imports
the golden script as a module and runs its `FullRAG.retrieve()` side by side
with the modular one, on the same fixed set of queries, comparing only the
fused Top-20 parent list.
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

# A small, fixed set of queries per language to compare.
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

            # ------------------------------------------------------------------
            # Compare all three lists.
            # The modular engine exposes "dense_parents" and "sparse_parents"
            # as extra return-dict keys. The golden script cannot be edited so
            # it may not have those keys — fall back gracefully to fused-only
            # for the golden side when they're absent.
            # ------------------------------------------------------------------

            g_dense  = g.get("dense_parents")
            g_sparse = g.get("sparse_parents")
            m_dense  = m.get("dense_parents")
            m_sparse = m.get("sparse_parents")

            checks = [
                ("fused top-20", g["parents"], m["parents"]),
            ]

            if g_dense is not None and m_dense is not None:
                checks.insert(0, ("dense parents", g_dense, m_dense))

            if g_sparse is not None and m_sparse is not None:
                checks.insert(1, ("sparse parents", g_sparse, m_sparse))

            for name, expected, actual in checks:
                status = "OK" if expected == actual else "MISMATCH"
                if status == "MISMATCH":
                    all_ok = False
                print(f"[{language}] {name}: {status}")
                if status == "MISMATCH":
                    print(f"    golden : {expected!r}")
                    print(f"    modular: {actual!r}")

    print()
    print("RETRIEVAL PARITY GATE:", "PASS" if all_ok else "FAIL")
    return all_ok


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
