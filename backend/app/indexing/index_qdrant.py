"""
Qdrant indexer.

Verbatim extraction of `scripts/index_local_qdrant.py` (kept untouched at
that path forever). The only changes relative to the golden script:

  1. `ROOT` is read from `app.config.settings.data_root` instead of being
     derived from this file's own location.
  2. The Qdrant client points at `app.config.settings.qdrant_url` /
     `qdrant_grpc_port` instead of the hardcoded `localhost`.

All indexing logic (collection creation, HNSW config, batch upload, the
"only upload if point count mismatches" guard, enabling HNSW after upload)
is unchanged.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from qdrant_client import QdrantClient, models

from app.config import settings


ROOT = settings.data_root

EMBED_ROOT = (
    ROOT
    / "embeddings_256"
    / "fixed_384_96"
)

COLLECTIONS = {
    "ta": "hhgoa_fixed384_ta",
    "hi": "hhgoa_fixed384_hi",
}

VECTOR_DIM = 256


client = QdrantClient(
    url=settings.qdrant_url,
    prefer_grpc=settings.qdrant_prefer_grpc,
    grpc_port=settings.qdrant_grpc_port,
    timeout=60,
    check_compatibility=False,
)


def wait_for_hnsw(
    collection: str,
    expected_points: int,
):

    print("Waiting for HNSW build...")

    deadline = time.time() + 180

    while time.time() < deadline:

        info = client.get_collection(
            collection
        )

        points = int(
            info.points_count or 0
        )

        indexed = int(
            info.indexed_vectors_count or 0
        )

        status = str(
            info.status
        )

        pct = (
            100.0 * indexed / points
            if points
            else 0.0
        )

        print(
            f"\rpoints={points:,} "
            f"indexed={indexed:,} "
            f"({pct:.1f}%) "
            f"status={status}",
            end="",
            flush=True,
        )

        if (
            points == expected_points
            and
            indexed >= int(
                expected_points * 0.95
            )
        ):
            print()
            print("HNSW READY ✅")
            return

        time.sleep(2)

    print()

    raise RuntimeError(
        f"HNSW did not finish for {collection}. "
        f"Check Docker/Qdrant logs."
    )


def index_language(
    language: str,
):

    collection = (
        COLLECTIONS[
            language
        ]
    )

    folder = (
        EMBED_ROOT
        / language
    )

    vector_path = (
        folder
        / "embeddings.npy"
    )

    metadata_path = (
        folder
        / "metadata.parquet"
    )

    vectors = np.load(
        vector_path,
        mmap_mode="r",
    )

    metadata = pd.read_parquet(
        metadata_path
    )

    if "vector_row" in metadata.columns:

        metadata = (
            metadata
            .sort_values(
                "vector_row"
            )
            .reset_index(drop=True)
        )

    if len(vectors) != len(metadata):

        raise RuntimeError(
            "Vector/metadata mismatch"
        )

    if vectors.shape[1] != VECTOR_DIM:

        raise RuntimeError(
            f"Expected 256 dims, got "
            f"{vectors.shape[1]}"
        )

    metadata["parent_id"] = (
        metadata["parent_id"]
        .astype(str)
    )

    if "chunk_id" not in metadata.columns:

        metadata["chunk_id"] = [
            f"{language}_{i}"
            for i in range(
                len(metadata)
            )
        ]

    metadata["chunk_id"] = (
        metadata["chunk_id"]
        .astype(str)
    )

    expected = len(vectors)

    print()
    print("=" * 80)
    print(
        language.upper(),
        collection,
    )
    print("=" * 80)

    print(
        f"Expected vectors: "
        f"{expected:,}"
    )

    # ------------------------------------------------------------------
    # If collection already contains all vectors,
    # DO NOT upload again.
    # ------------------------------------------------------------------

    collection_ready = False

    if client.collection_exists(
        collection
    ):

        info = client.get_collection(
            collection
        )

        current_points = int(
            info.points_count or 0
        )

        if current_points == expected:

            print(
                "Vectors already uploaded ✅"
            )

            collection_ready = True

        else:

            print(
                "Existing collection has "
                "wrong point count."
            )

            print(
                "Recreating collection..."
            )

            client.delete_collection(
                collection
            )

    # ------------------------------------------------------------------
    # Create + upload only when needed
    # ------------------------------------------------------------------

    if not collection_ready:

        print(
            "Creating collection..."
        )

        # Disable HNSW while bulk uploading.
        client.create_collection(
            collection_name=
                collection,

            vectors_config=
                models.VectorParams(
                    size=
                        VECTOR_DIM,

                    distance=
                        models.Distance.COSINE,
                ),

            hnsw_config=
                models.HnswConfigDiff(
                    m=16,
                    ef_construct=128,
                ),

            optimizers_config=
                models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
        )

        client.create_payload_index(
            collection_name=
                collection,

            field_name=
                "parent_id",

            field_schema=
                models.PayloadSchemaType.KEYWORD,

            wait=True,
        )

        batch_size = 512

        start_time = (
            time.perf_counter()
        )

        for start in range(
            0,
            expected,
            batch_size,
        ):

            end = min(
                start + batch_size,
                expected,
            )

            points = []

            for i in range(
                start,
                end,
            ):

                points.append(
                    models.PointStruct(
                        id=i,

                        vector=
                            np.asarray(
                                vectors[i],
                                dtype=np.float32,
                            ).tolist(),

                        payload={
                            "language":
                                language,

                            "parent_id":
                                metadata.iloc[i][
                                    "parent_id"
                                ],

                            "chunk_id":
                                metadata.iloc[i][
                                    "chunk_id"
                                ],
                        },
                    )
                )

            client.upsert(
                collection_name=
                    collection,

                points=
                    points,

                wait=True,
            )

            print(
                f"\rUploaded "
                f"{end:,}/"
                f"{expected:,}",
                end="",
                flush=True,
            )

        print()

        print(
            f"Upload completed in "
            f"{time.perf_counter() - start_time:.1f}s"
        )

    # ------------------------------------------------------------------
    # IMPORTANT:
    # turn HNSW indexing ON AFTER upload.
    #
    # Low threshold intentionally forces HNSW for this ~40k collection.
    # ------------------------------------------------------------------

    print(
        "Enabling HNSW indexing..."
    )

    client.update_collection(
        collection_name=
            collection,

        optimizers_config=
            models.OptimizersConfigDiff(
                indexing_threshold=1000,
            ),
    )

    wait_for_hnsw(
        collection,
        expected,
    )


def main():

    index_language("ta")
    index_language("hi")

    print()
    print("=" * 80)
    print(
        "LOCAL QDRANT READY ✅"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
