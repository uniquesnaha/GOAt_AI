"""
Retrieval parity gate.

This gate checks RETRIEVAL PARITY ONLY.

The grounded-quality patch intentionally changes post-retrieval behavior:

- ContextStore.build()
    175 -> 168 chars
    prefix-cut -> query-centered evidence window
    guessed child -> actual retrieved child where available

- MAX_NEW_TOKENS
    16 -> 24

- sources / provenance
    raw parent IDs -> structured SourceEvidence

Therefore context strings, prompt token counts, and generated answers are
EXPECTED to differ from the frozen golden benchmark.

WHAT IS CHECKED
---------------
Must remain identical:

- fused Top-20 parent IDs, order-sensitive

If dense/sparse parent lists are exposed by both implementations, they are
also compared automatically.

WHAT IS NOT CHECKED
-------------------
Intentionally different after the quality patch:

- context string
- context parent count
- prompt token count
- answer text
- source/provenance representation


DOCKER / QDRANT NOTE
--------------------
The frozen golden script hardcodes:

    http://127.0.0.1:6333

That was correct in the original Colab/local-host environment where Qdrant
shared the same host namespace.

Inside Docker Compose, however, this parity gate runs in a temporary backend
container while Qdrant runs in a separate service container.

Therefore:

    127.0.0.1:6333

would incorrectly point back to the parity-test container itself.

This test DOES NOT edit the golden script.

Instead, after instantiating golden.FullRAG(), it replaces only the golden
engine's Qdrant client with one pointing at settings.qdrant_url, normally:

    http://qdrant:6333

This is infrastructure adaptation only. It does not alter:

- query embedding
- dense retrieval parameters
- HNSW ef
- BM25
- parent collapse
- RRF configuration/math
- final Top-20 ranking


Run from backend/:

    python -m parity.test_retrieval_parity
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from app.config import settings
from app.rag.engine import FullRAG as ModularFullRAG


# =============================================================================
# PATHS
# =============================================================================

REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

GOLDEN_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "benchmark_full_rag_t4_latency_winner.py"
)

GOLDEN_ROOT = (
    Path("/content/HH-goa-task2")
    if os.name != "nt"
    else Path("C:/content/HH-goa-task2")
)


# =============================================================================
# FIXED PARITY QUERY SET
# =============================================================================

SAMPLE_QUERIES = {
    "ta": [],
    "hi": [],
}


# =============================================================================
# GOLDEN DATA ROOT ADAPTER
# =============================================================================

def _ensure_golden_root_link() -> None:
    """
    Point the golden script's hardcoded ROOT at the deployed data directory.

    The golden script itself is never edited.
    """

    data_root = (
        settings.data_root
        .resolve()
    )

    if GOLDEN_ROOT.exists():

        resolved = (
            GOLDEN_ROOT
            .resolve()
        )

        if resolved == data_root:
            return

        raise RuntimeError(
            f"{GOLDEN_ROOT} already exists and does not point at "
            f"{data_root}. Refusing to overwrite it because it may "
            f"contain real data."
        )

    try:

        GOLDEN_ROOT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    except PermissionError as exc:

        raise RuntimeError(
            f"Cannot create {GOLDEN_ROOT.parent}. "
            "The parity container/user does not have permission."
        ) from exc


    if os.name == "nt":

        exit_code = os.system(
            f'mklink /J "{GOLDEN_ROOT}" "{data_root}"'
        )

        if exit_code != 0:

            raise RuntimeError(
                f"Failed to create junction:\n"
                f"{GOLDEN_ROOT} -> {data_root}"
            )

    else:

        os.symlink(
            data_root,
            GOLDEN_ROOT,
            target_is_directory=True,
        )


    if not GOLDEN_ROOT.exists():

        raise RuntimeError(
            f"Failed to link "
            f"{GOLDEN_ROOT} -> {data_root}"
        )


# =============================================================================
# GOLDEN MODULE LOADER
# =============================================================================

def _load_golden_module():

    if not GOLDEN_SCRIPT.exists():

        raise RuntimeError(
            "Golden benchmark script not found at:\n"
            f"{GOLDEN_SCRIPT}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "golden_benchmark_full_rag",
            GOLDEN_SCRIPT,
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):

        raise RuntimeError(
            "Could not create import specification "
            "for golden benchmark."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


# =============================================================================
# SAMPLE QUERY LOADER
# =============================================================================

def _load_sample_queries(
    golden,
) -> None:

    for language in (
        "ta",
        "hi",
    ):

        lookup = (
            golden.load_queries(
                language
            )
        )

        SAMPLE_QUERIES[
            language
        ] = list(
            lookup.values()
        )[:5]


# =============================================================================
# QDRANT FACTORY
# =============================================================================

def _make_deployed_qdrant_client() -> QdrantClient:
    """
    Create a Qdrant client using deployment infrastructure settings.

    This changes ONLY the endpoint/transport used by the parity harness.
    Retrieval behavior/configuration is untouched.
    """

    return QdrantClient(
        url=
            settings.qdrant_url,

        prefer_grpc=
            settings.qdrant_prefer_grpc,

        grpc_port=
            settings.qdrant_grpc_port,

        timeout=
            30,

        check_compatibility=
            False,
    )


# =============================================================================
# QDRANT CONNECTIVITY PRE-FLIGHT
# =============================================================================

def _verify_qdrant_connectivity() -> None:
    """
    Fail before loading two GPU models if Qdrant is unreachable.
    """

    print(
        "Checking deployed Qdrant endpoint:",
        settings.qdrant_url,
    )

    client = (
        _make_deployed_qdrant_client()
    )

    try:

        collections = (
            client
            .get_collections()
            .collections
        )

    except Exception as exc:

        raise RuntimeError(
            "Parity gate cannot reach Qdrant at "
            f"{settings.qdrant_url}. "
            "Inside Docker Compose this should normally be "
            "'http://qdrant:6333'."
        ) from exc


    names = {
        collection.name
        for collection
        in collections
    }


    expected = {
        "hhgoa_fixed384_ta",
        "hhgoa_fixed384_hi",
    }


    missing = (
        expected
        -
        names
    )


    if missing:

        raise RuntimeError(
            "Qdrant is reachable, but required collections "
            f"are missing: {sorted(missing)}"
        )


    print(
        "Qdrant connectivity: OK"
    )

    print(
        "Required collections: OK"
    )


# =============================================================================
# PARITY GATE
# =============================================================================

def run() -> bool:

    # ---------------------------------------------------------------------
    # 1. Infrastructure/data preparation
    # ---------------------------------------------------------------------

    _ensure_golden_root_link()

    _verify_qdrant_connectivity()


    # ---------------------------------------------------------------------
    # 2. Load untouched golden module
    # ---------------------------------------------------------------------

    golden = (
        _load_golden_module()
    )

    _load_sample_queries(
        golden
    )


    # ---------------------------------------------------------------------
    # 3. Golden engine
    # ---------------------------------------------------------------------

    print()
    print(
        "Loading golden FullRAG..."
    )

    golden_engine = (
        golden.FullRAG()
    )


    # IMPORTANT:
    #
    # Golden FullRAG hardcodes:
    #
    #     http://127.0.0.1:6333
    #
    # That endpoint is invalid from a temporary Docker backend container,
    # because localhost refers to that backend container itself.
    #
    # Replace ONLY the infrastructure endpoint.
    #
    # The untouched golden retrieve() body still executes the same:
    #
    # embed -> Qdrant -> parent collapse -> BM25 -> weighted RRF -> Top-20
    #

    golden_engine.qdrant = (
        _make_deployed_qdrant_client()
    )


    print(
        "Golden Qdrant endpoint adapted to:",
        settings.qdrant_url,
    )


    # ---------------------------------------------------------------------
    # 4. Modular/patched engine
    # ---------------------------------------------------------------------

    print()
    print(
        "Loading modular FullRAG..."
    )

    modular_engine = (
        ModularFullRAG()
    )


    # ---------------------------------------------------------------------
    # 5. Retrieval parity
    # ---------------------------------------------------------------------

    all_ok = True

    total_queries = 0

    fused_matches = 0


    print()
    print(
        "=" * 80
    )

    print(
        "RETRIEVAL PARITY"
    )

    print(
        "=" * 80
    )


    for language, queries in (
        SAMPLE_QUERIES.items()
    ):

        for query_index, query in enumerate(
            queries,
            start=1,
        ):

            total_queries += 1


            # -------------------------------------------------------------
            # Run the exact frozen golden retrieval.
            # -------------------------------------------------------------

            golden_result = (
                golden_engine.retrieve(
                    query,
                    language,
                )
            )


            # -------------------------------------------------------------
            # Run the modular retrieval.
            # -------------------------------------------------------------

            modular_result = (
                modular_engine.retrieve(
                    query,
                    language,
                )
            )


            # -------------------------------------------------------------
            # Fused Top-20 is the mandatory parity contract.
            # -------------------------------------------------------------

            golden_fused = (
                golden_result[
                    "parents"
                ]
            )

            modular_fused = (
                modular_result[
                    "parents"
                ]
            )


            fused_ok = (
                golden_fused
                ==
                modular_fused
            )


            if fused_ok:

                fused_matches += 1

            else:

                all_ok = False


            print(
                f"[{language}] "
                f"query {query_index}: "
                f"fused top-20 = "
                f"{'OK' if fused_ok else 'MISMATCH'}"
            )


            if not fused_ok:

                print(
                    "    query:"
                )

                print(
                    f"    {query}"
                )

                print(
                    "    golden :",
                    golden_fused,
                )

                print(
                    "    modular:",
                    modular_fused,
                )


            # -------------------------------------------------------------
            # Optional lane-level checks.
            #
            # These are only performed if BOTH implementations expose the
            # corresponding lists. The frozen golden script is not modified
            # merely to make debugging metadata available.
            # -------------------------------------------------------------

            golden_dense = (
                golden_result.get(
                    "dense_parents"
                )
            )

            modular_dense = (
                modular_result.get(
                    "dense_parents"
                )
            )


            if (
                golden_dense is not None
                and
                modular_dense is not None
            ):

                dense_ok = (
                    golden_dense
                    ==
                    modular_dense
                )

                print(
                    f"    dense parents = "
                    f"{'OK' if dense_ok else 'MISMATCH'}"
                )

                if not dense_ok:

                    all_ok = False

                    print(
                        "        golden :",
                        golden_dense,
                    )

                    print(
                        "        modular:",
                        modular_dense,
                    )


            golden_sparse = (
                golden_result.get(
                    "sparse_parents"
                )
            )

            modular_sparse = (
                modular_result.get(
                    "sparse_parents"
                )
            )


            if (
                golden_sparse is not None
                and
                modular_sparse is not None
            ):

                sparse_ok = (
                    golden_sparse
                    ==
                    modular_sparse
                )

                print(
                    f"    sparse parents = "
                    f"{'OK' if sparse_ok else 'MISMATCH'}"
                )

                if not sparse_ok:

                    all_ok = False

                    print(
                        "        golden :",
                        golden_sparse,
                    )

                    print(
                        "        modular:",
                        modular_sparse,
                    )


    # ---------------------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "RETRIEVAL PARITY SUMMARY"
    )

    print(
        "=" * 80
    )


    print(
        "Queries checked:",
        total_queries,
    )

    print(
        "Fused Top-20 exact matches:",
        f"{fused_matches}/{total_queries}",
    )


    if total_queries:

        parity_percent = (
            100.0
            *
            fused_matches
            /
            total_queries
        )

    else:

        parity_percent = 0.0


    print(
        "Fused Top-20 parity:",
        f"{parity_percent:.1f}%",
    )


    print()

    print(
        "RETRIEVAL PARITY GATE:",
        "PASS"
        if all_ok
        else "FAIL",
    )


    return all_ok


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    ok = run()

    sys.exit(
        0
        if ok
        else 1
    )