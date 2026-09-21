#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

# Required by GenoGramma deterministic CUDA execution.
# Must be defined before the first CuBLAS operation.
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import torch


BASE = Path(__file__).resolve().parents[2]

STAGE55 = (
    BASE
    / "pipeline"
    / "55_extract_genogramma_pairaware_representations.py"
)


def load_stage55():
    spec = importlib.util.spec_from_file_location(
        "genogramma_stage55_inference",
        str(STAGE55),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {STAGE55}"
        )

    mod = importlib.util.module_from_spec(spec)

    sys.modules[
        "genogramma_stage55_inference"
    ] = mod

    spec.loader.exec_module(mod)

    return mod


def resolve_device(value):
    if value == "auto":
        value = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )

    return torch.device(value)


def validate_input(path):
    if not path.is_file():
        raise RuntimeError(
            f"Input NPZ not found: {path}"
        )

    z = np.load(
        path,
        allow_pickle=False,
    )

    required = {
        "family_rows",
        "strand",
    }

    missing = required - set(z.files)

    if missing:
        raise RuntimeError(
            f"Input NPZ missing keys: {sorted(missing)}"
        )

    family_rows = np.asarray(
        z["family_rows"],
        dtype=np.int64,
    )

    strand = np.asarray(
        z["strand"],
        dtype=np.int64,
    )

    if family_rows.ndim != 2:
        raise RuntimeError(
            f"family_rows must be 2D; got {family_rows.shape}"
        )

    if strand.ndim != 2:
        raise RuntimeError(
            f"strand must be 2D; got {strand.shape}"
        )

    if family_rows.shape != strand.shape:
        raise RuntimeError(
            "family_rows and strand shape mismatch: "
            f"{family_rows.shape} vs {strand.shape}"
        )

    if family_rows.shape[1] != 64:
        raise RuntimeError(
            "GenoGramma compatible windows must contain exactly "
            f"64 genes; got {family_rows.shape[1]}"
        )

    strand_values = set(
        np.unique(strand).tolist()
    )

    if not strand_values.issubset({0, 1}):
        raise RuntimeError(
            "strand must contain only encoded values 0/1; "
            f"observed {sorted(strand_values)}"
        )

    return family_rows, strand


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Extract fold-specific 768D GenoGramma representations "
            "from compatible 64-gene family-index windows."
        )
    )

    ap.add_argument(
        "--input",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    family_rows, strand = validate_input(
        input_path
    )

    stage55 = load_stage55()

    family_embeddings = np.load(
        stage55.FAMILY_EMBEDDINGS,
        mmap_mode="r",
    )

    if family_embeddings.shape != (276517, 480):
        raise RuntimeError(
            "Unexpected family embedding matrix: "
            f"{family_embeddings.shape}"
        )

    if family_rows.size:
        lo = int(family_rows.min())
        hi = int(family_rows.max())

        if lo < 0 or hi >= family_embeddings.shape[0]:
            raise RuntimeError(
                "family_rows contains index outside the formal "
                f"vocabulary: min={lo}, max={hi}, "
                f"valid=0..{family_embeddings.shape[0]-1}"
            )

    device = resolve_device(
        args.device
    )

    (
        model,
        ckpt,
        ckpt_path,
        checkpoint_sha,
    ) = stage55.build_model(
        args.fold,
        device,
    )

    model.eval()

    n = family_rows.shape[0]

    tmp_path = output_path.with_name(
        output_path.name + ".tmp.npy"
    )

    if tmp_path.exists():
        tmp_path.unlink()

    out = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, 768),
    )

    with torch.inference_mode():

        for start in range(
            0,
            n,
            args.batch_size,
        ):
            end = min(
                start + args.batch_size,
                n,
            )

            rows_np = np.asarray(
                family_rows[start:end],
                dtype=np.int64,
            )

            strand_np = np.asarray(
                strand[start:end],
                dtype=np.int64,
            )

            family_np = np.asarray(
                family_embeddings[rows_np],
                dtype=np.float32,
            )

            family_t = (
                torch.from_numpy(family_np)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            strand_t = (
                torch.from_numpy(strand_np)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            token = model.prepare_tokens(
                family_t,
                strand_t,
            )

            if tuple(token.shape) != (
                end - start,
                64,
                128,
            ):
                raise RuntimeError(
                    "Unexpected token shape: "
                    f"{tuple(token.shape)}"
                )

            pair_rep = stage55.gg.encode_pairaware(
                model,
                token,
                stage55.LEFT,
                stage55.RIGHT,
            )

            if tuple(pair_rep.shape) != (
                end - start,
                768,
            ):
                raise RuntimeError(
                    "Unexpected GenoGramma representation shape: "
                    f"{tuple(pair_rep.shape)}"
                )

            out[start:end] = (
                pair_rep
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

    out.flush()
    del out

    tmp_path.replace(
        output_path
    )

    metadata = {
        "model": "GenoGramma",
        "mode": "compatible_npz_embedding",
        "fold": int(args.fold),
        "device": str(device),
        "input": str(input_path),
        "output": str(output_path),
        "n_windows": int(n),
        "window_length": 64,
        "target_pair_indices_0based": [31, 32],
        "representation_dimension": 768,
        "representation_dtype": "float32",
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": checkpoint_sha,
        "formal_outer_fold": int(
            ckpt.get(
                "formal_outer_fold",
                -1,
            )
        ),
        "family_embedding_matrix": str(
            stage55.FAMILY_EMBEDDINGS
        ),
        "implementation": (
            "model.prepare_tokens + "
            "canonical gg.encode_pairaware"
        ),
    }

    meta_path = output_path.with_suffix(
        output_path.suffix + ".json"
    )

    meta_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[PASS] GenoGramma compatible-NPZ embedding"
    )
    print(
        "       input   =",
        input_path,
    )
    print(
        "       output  =",
        output_path,
    )
    print(
        "       shape   =",
        (n, 768),
    )
    print(
        "       dtype   = float32"
    )
    print(
        "       fold    =",
        args.fold,
    )
    print(
        "       device  =",
        device,
    )
    print(
        "       metadata=",
        meta_path,
    )


if __name__ == "__main__":
    main()
