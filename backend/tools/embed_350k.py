from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from sentence_transformers import (
    SentenceTransformer,
)


MODEL_NAME = (
    "Qwen/Qwen3-Embedding-0.6B"
)

EMBED_DIM = 256


LANGUAGE_FILES = {
    "ta": "tamil.parquet",
    "hi": "hindi.parquet",
}


def embed_language(
    model,
    language: str,
    data_root: Path,
    output_root: Path,
    encode_batch_size: int,
    parquet_batch_rows: int,
    limit: int | None,
):

    input_path = (
        data_root
        / "chunks_350k"
        / "fixed_384_96"
        / LANGUAGE_FILES[
            language
        ]
    )

    output_dir = (
        output_root
        / "fixed_384_96"
        / language
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_path = (
        output_dir
        / "embeddings.npy"
    )

    metadata_path = (
        output_dir
        / "metadata.parquet"
    )

    parquet = (
        pq.ParquetFile(
            input_path
        )
    )

    total_rows = int(
        parquet.metadata.num_rows
    )

    if limit is not None:
        total_rows = min(
            total_rows,
            limit,
        )

    print()
    print("=" * 80)
    print(language.upper())
    print("=" * 80)

    print(
        "Input:",
        input_path,
    )

    print(
        "Rows:",
        total_rows,
    )

    vectors = (
        np.lib.format.open_memmap(
            vector_path,
            mode="w+",
            dtype=np.float32,
            shape=(
                total_rows,
                EMBED_DIM,
            ),
        )
    )

    metadata_schema = (
        pa.schema([
            (
                "vector_row",
                pa.int64(),
            ),
            (
                "parent_id",
                pa.string(),
            ),
            (
                "chunk_id",
                pa.string(),
            ),
        ])
    )

    metadata_writer = (
        pq.ParquetWriter(
            metadata_path,
            metadata_schema,
            compression="zstd",
        )
    )

    offset = 0
    start_time = (
        time.perf_counter()
    )

    try:

        for batch in parquet.iter_batches(
            batch_size=
                parquet_batch_rows,

            columns=[
                "parent_id",
                "chunk_id",
                "text",
            ],
        ):

            if offset >= total_rows:
                break

            table = (
                pa.Table
                .from_batches([
                    batch
                ])
            )

            df = (
                table
                .to_pandas()
            )

            remaining = (
                total_rows
                -
                offset
            )

            if len(df) > remaining:
                df = df.iloc[
                    :remaining
                ]

            texts = (
                df["text"]
                .fillna("")
                .astype(str)
                .tolist()
            )

            if not texts:
                continue

            torch.cuda.synchronize()

            batch_start = (
                time.perf_counter()
            )

            encoded = (
                model.encode(
                    texts,

                    batch_size=
                        encode_batch_size,

                    truncate_dim=
                        EMBED_DIM,

                    normalize_embeddings=
                        True,

                    convert_to_numpy=
                        True,

                    show_progress_bar=
                        False,
                )
            )

            torch.cuda.synchronize()

            encoded = np.asarray(
                encoded,
                dtype=np.float32,
            )

            end = (
                offset
                +
                len(encoded)
            )

            vectors[
                offset:end
            ] = encoded

            metadata_batch = (
                pa.table({
                    "vector_row":
                        np.arange(
                            offset,
                            end,
                            dtype=np.int64,
                        ),

                    "parent_id":
                        df["parent_id"]
                        .astype(str)
                        .tolist(),

                    "chunk_id":
                        df["chunk_id"]
                        .astype(str)
                        .tolist(),
                })
            )

            metadata_writer.write_table(
                metadata_batch
            )

            elapsed = (
                time.perf_counter()
                -
                start_time
            )

            speed = (
                end
                /
                max(
                    elapsed,
                    0.001,
                )
            )

            eta = (
                (
                    total_rows
                    -
                    end
                )
                /
                max(
                    speed,
                    0.001,
                )
            )

            print(
                f"\r{language.upper()} "
                f"{end:,}/{total_rows:,} "
                f"| {speed:.1f} chunks/s "
                f"| ETA {eta/60:.1f} min",
                end="",
                flush=True,
            )

            offset = end

    finally:

        metadata_writer.close()

        vectors.flush()

    print()

    if offset != total_rows:
        raise RuntimeError(
            f"Expected {total_rows} embeddings, "
            f"wrote {offset}"
        )

    sample_size = min(
        1000,
        total_rows,
    )

    norms = np.linalg.norm(
        np.asarray(
            vectors[
                :sample_size
            ],
            dtype=np.float32,
        ),
        axis=1,
    )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    info = {
        "language":
            language,

        "model":
            MODEL_NAME,

        "embedding_dim":
            EMBED_DIM,

        "normalize_embeddings":
            True,

        "query_prompt_used":
            False,

        "rows":
            total_rows,

        "elapsed_seconds":
            elapsed,

        "chunks_per_second":
            total_rows
            /
            max(
                elapsed,
                0.001,
            ),

        "mean_sample_norm":
            float(
                norms.mean()
            ),
    }

    (
        output_dir
        /
        "embedding_info.json"
    ).write_text(
        json.dumps(
            info,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Embeddings:",
        vector_path,
    )

    print(
        "Metadata:",
        metadata_path,
    )

    print(
        "Mean norm:",
        round(
            float(
                norms.mean()
            ),
            6,
        ),
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
        "--output-root",
        type=Path,
        default=(
            Path("/data")
            /
            "embeddings_350k_256"
        ),
    )

    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--parquet-batch-rows",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Loading:",
        MODEL_NAME,
    )

    model = (
        SentenceTransformer(
            MODEL_NAME,

            device="cuda",

            model_kwargs={
                "torch_dtype":
                    torch.float16,

                "attn_implementation":
                    "sdpa",
            },
        )
    )

    model.eval()

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

        embed_language(
            model=
                model,

            language=
                language,

            data_root=
                args.data_root,

            output_root=
                args.output_root,

            encode_batch_size=
                args.encode_batch_size,

            parquet_batch_rows=
                args.parquet_batch_rows,

            limit=
                args.limit,
        )


if __name__ == "__main__":
    main()
