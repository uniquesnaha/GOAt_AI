from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qdrant_client import (
    QdrantClient,
    models,
)

from app.config import settings


VECTOR_DIM = 256

COLLECTIONS = {
    "ta":
        "hhgoa_350k_fixed384_ta",

    "hi":
        "hhgoa_350k_fixed384_hi",
}


def wait_for_hnsw(
    client: QdrantClient,
    collection: str,
    expected_points: int,
    timeout_seconds: int,
):

    print(
        "Waiting for HNSW..."
    )

    deadline = (
        time.time()
        +
        timeout_seconds
    )

    while time.time() < deadline:

        info = (
            client.get_collection(
                collection
            )
        )

        points = int(
            info.points_count
            or 0
        )

        indexed = int(
            info.indexed_vectors_count
            or 0
        )

        pct = (
            100.0
            *
            indexed
            /
            points
            if points
            else 0.0
        )

        print(
            f"\rpoints={points:,} "
            f"indexed={indexed:,} "
            f"({pct:.1f}%) "
            f"status={info.status}",
            end="",
            flush=True,
        )

        if (
            points
            ==
            expected_points
            and
            indexed
            >=
            int(
                expected_points
                *
                0.95
            )
        ):

            print()

            print(
                "HNSW READY"
            )

            return

        time.sleep(5)

    print()

    raise RuntimeError(
        "HNSW build timeout"
    )


def index_language(
    client: QdrantClient,
    language: str,
    data_root: Path,
    batch_size: int,
    hnsw_timeout: int,
):

    collection = (
        COLLECTIONS[
            language
        ]
    )

    folder = (
        data_root
        /
        "embeddings_350k_256"
        /
        "fixed_384_96"
        /
        language
    )

    vector_path = (
        folder
        /
        "embeddings.npy"
    )

    metadata_path = (
        folder
        /
        "metadata.parquet"
    )

    print()
    print("=" * 80)
    print(language.upper())
    print(collection)
    print("=" * 80)

    vectors = np.load(
        vector_path,
        mmap_mode="r",
    )

    metadata = pd.read_parquet(
        metadata_path,
    )

    if "vector_row" in metadata.columns:

        metadata = (
            metadata
            .sort_values(
                "vector_row"
            )
            .reset_index(
                drop=True
            )
        )

    if (
        len(vectors)
        !=
        len(metadata)
    ):

        raise RuntimeError(
            "Vector/metadata row mismatch"
        )

    if (
        vectors.shape[1]
        !=
        VECTOR_DIM
    ):

        raise RuntimeError(
            f"Expected {VECTOR_DIM} dimensions, "
            f"got {vectors.shape[1]}"
        )

    parent_ids = (
        metadata[
            "parent_id"
        ]
        .astype(str)
        .to_numpy()
    )

    chunk_ids = (
        metadata[
            "chunk_id"
        ]
        .astype(str)
        .to_numpy()
    )

    expected = (
        len(vectors)
    )

    print(
        "Expected vectors:",
        f"{expected:,}",
    )

    collection_ready = False

    if client.collection_exists(
        collection
    ):

        info = (
            client.get_collection(
                collection
            )
        )

        current = int(
            info.points_count
            or 0
        )

        if current == expected:

            print(
                "New 350k collection already "
                "contains expected vectors."
            )

            collection_ready = True

        else:

            print(
                "350k collection exists with wrong "
                f"count ({current:,})."
            )

            print(
                "Deleting ONLY the new 350k collection."
            )

            client.delete_collection(
                collection
            )

    if not collection_ready:

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

        started = (
            time.perf_counter()
        )

        for start in range(
            0,
            expected,
            batch_size,
        ):

            end = min(
                start
                +
                batch_size,

                expected,
            )

            points = [
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
                            str(
                                parent_ids[i]
                            ),

                        "chunk_id":
                            str(
                                chunk_ids[i]
                            ),
                    },
                )

                for i in range(
                    start,
                    end,
                )
            ]

            client.upsert(
                collection_name=
                    collection,

                points=
                    points,

                wait=True,
            )

            elapsed = (
                time.perf_counter()
                -
                started
            )

            rate = (
                end
                /
                max(
                    elapsed,
                    0.001,
                )
            )

            eta = (
                (
                    expected
                    -
                    end
                )
                /
                max(
                    rate,
                    0.001,
                )
            )

            print(
                f"\rUploaded "
                f"{end:,}/{expected:,} "
                f"| {rate:.0f} vectors/s "
                f"| ETA {eta/60:.1f} min",
                end="",
                flush=True,
            )

        print()

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
        client=
            client,

        collection=
            collection,

        expected_points=
            expected,

        timeout_seconds=
            hnsw_timeout,
    )


def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--language",
        choices=[
            "ta",
            "hi",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--hnsw-timeout",
        type=int,
        default=1800,
    )

    args = parser.parse_args()

    client = (
        QdrantClient(
            url=
                settings.qdrant_url,

            prefer_grpc=
                settings.qdrant_prefer_grpc,

            grpc_port=
                settings.qdrant_grpc_port,

            timeout=
                60,

            check_compatibility=
                False,
        )
    )

    languages = (
        [
            "ta",
            "hi",
        ]
        if args.language == "both"
        else [
            args.language
        ]
    )

    for language in languages:

        index_language(
            client=
                client,

            language=
                language,

            data_root=
                args.data_root,

            batch_size=
                args.batch_size,

            hnsw_timeout=
                args.hnsw_timeout,
        )

    print()
    print("=" * 80)
    print("350K QDRANT COLLECTIONS READY")
    print("=" * 80)


if __name__ == "__main__":
    main()
