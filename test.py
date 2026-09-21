#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
MODEL = BASE / "pipeline" / "revision_modules" / "genogramma.py"
FAMILY_EMB = BASE / "model_assets" / "family_embeddings" / "00_family_embeddings_276517x480.npy"
EXPECTED_MODEL_SHA = "7579820f2324f101fa7412285233f9ab1d80548f234269a5ad7ef47a76be2e01"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def locate_checkpoint() -> Path:
    roots = [
        BASE / "checkpoints" / "genogramma_formal",
        BASE / "checkpoints" / "genogramma_formal_pretrained",
    ]
    for root in roots:
        p = root / "fold_1" / "best_checkpoint.pt"
        if p.is_file():
            return p
    raise SystemExit(
        "[FAIL] no packaged formal fold_1 checkpoint found under "
        "checkpoints/genogramma_formal or checkpoints/genogramma_formal_pretrained"
    )

def audit() -> None:
    for p in (MODEL, FAMILY_EMB):
        if not p.is_file():
            raise SystemExit(f"[FAIL] required asset missing: {p}")

    model_sha = sha256(MODEL)
    if model_sha != EXPECTED_MODEL_SHA:
        raise SystemExit(
            f"[FAIL] canonical GenoGramma SHA mismatch: {model_sha}"
        )

    x = np.load(FAMILY_EMB, mmap_mode="r")
    if x.shape != (276517, 480):
        raise SystemExit(f"[FAIL] family embedding shape: {x.shape}; expected (276517, 480)")
    if x.dtype != np.float16:
        raise SystemExit(f"[FAIL] family embedding dtype: {x.dtype}; expected float16")

    ckpt = locate_checkpoint()
    obj = torch.load(
        ckpt,
        map_location="cpu",
        weights_only=False,
    )

    if "model" not in obj:
        raise SystemExit(f"[FAIL] checkpoint[model] missing: {ckpt}")

    print("[PASS] canonical GenoGramma module")
    print("       SHA256 =", model_sha)
    print("[PASS] family embeddings")
    print("       shape  =", x.shape)
    print("       dtype  =", x.dtype)
    print("[PASS] packaged checkpoint")
    print("       path   =", ckpt)
    print("       config  =", obj.get("model_config"))

def main() -> None:
    ap = argparse.ArgumentParser(
        description="GenoGramma public smoke/audit entry point."
    )
    ap.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit packaged assets only.",
    )
    ap.add_argument(
        "--mode",
        choices=["smoke", "embed", "predict"],
        default="smoke",
    )
    ap.add_argument("--input", default=None)
    ap.add_argument("--head", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    audit()

    if args.audit_only or args.mode == "smoke":
        print("[FINAL PASS] bundled GenoGramma smoke/audit test complete")
        return

    if not args.input:
        raise SystemExit(
            "[GUARD] --input is required for embed/predict modes."
        )

    if args.mode == "embed":
        if not args.input:
            raise SystemExit(
                "[FAIL] --input is required for --mode embed"
            )

        if not args.output:
            raise SystemExit(
                "[FAIL] --output is required for --mode embed"
            )

        engine = (
            BASE
            / "pipeline"
            / "inference"
            / "embed_compatible_npz.py"
        )

        cmd = [
            sys.executable,
            str(engine),
            "--input",
            str(args.input),
            "--output",
            str(args.output),
            "--fold",
            str(args.fold),
            "--batch-size",
            str(args.batch_size),
            "--device",
            str(args.device),
        ]

        print(
            "[RUN]",
            " ".join(cmd),
            flush=True,
        )

        subprocess.run(
            cmd,
            check=True,
        )

        print(
            "[FINAL PASS] GenoGramma embed mode complete"
        )

        return

    if args.mode == "predict":
        if not args.input:
            raise SystemExit(
                "[FAIL] --input is required for --mode predict"
            )

        if not args.output:
            raise SystemExit(
                "[FAIL] --output is required for --mode predict"
            )

        engine = (
            BASE
            / "pipeline"
            / "inference"
            / "predict_operon.py"
        )

        cmd = [
            sys.executable,
            str(engine),
            "--input",
            str(args.input),
            "--output",
            str(args.output),
            "--fold",
            str(args.fold),
            "--batch-size",
            str(args.batch_size),
            "--device",
            str(args.device),
        ]

        if args.head:
            cmd.extend([
                "--head",
                str(args.head),
            ])

        print(
            "[RUN]",
            " ".join(cmd),
            flush=True,
        )

        subprocess.run(
            cmd,
            check=True,
        )

        print(
            "[FINAL PASS] GenoGramma predict mode complete"
        )

        return
if __name__ == "__main__":
    main()
