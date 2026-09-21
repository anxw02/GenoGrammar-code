#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import torch
import torch.nn as nn
import torch.nn.functional as F


BASE = Path(__file__).resolve().parents[2]

EMBED_ENGINE = (
    BASE
    / "pipeline"
    / "inference"
    / "embed_compatible_npz.py"
)

DEFAULT_HEAD_ROOT = (
    BASE
    / "checkpoints"
    / "stage56_operon_heads"
)


class GenoGrammaPairHead(nn.Module):

    def __init__(
        self,
        representation_dim=768,
        hidden_dim=128,
        dropout=0.2,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            representation_dim,
            hidden_dim,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            2,
        )

    def forward(self, x):

        x = self.fc1(x)

        x = F.gelu(x)

        x = self.dropout(x)

        return self.fc2(x)


def resolve_device(value):

    if value == "auto":
        value = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        value == "cuda"
        and
        not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    return torch.device(value)


def resolve_head(
    supplied,
    fold,
):

    if supplied is not None:
        p = Path(supplied).resolve()
    else:
        p = (
            DEFAULT_HEAD_ROOT
            / f"fold_{fold}"
            / "best_head.pt"
        )

    if not p.is_file():
        raise RuntimeError(
            f"Head checkpoint not found: {p}"
        )

    return p


def load_head(
    path,
    fold,
    device,
):

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "head_state_dict",
        "zscore_mean",
        "zscore_sd",
        "protocol",
    }

    missing = required - set(obj)

    if missing:
        raise RuntimeError(
            "Incompatible downstream head; "
            f"missing keys: {sorted(missing)}"
        )

    if (
        "model_name" in obj
        and
        obj["model_name"] != "GenoGramma"
    ):
        raise RuntimeError(
            "Head model_name is not GenoGramma"
        )

    if (
        "outer_fold" in obj
        and
        int(obj["outer_fold"]) != int(fold)
    ):
        raise RuntimeError(
            "Requested encoder fold and head "
            f"outer_fold disagree: {fold} vs "
            f"{obj[outer_fold]}"
        )

    protocol = obj["protocol"]

    rep_dim = int(
        protocol.get(
            "representation_dim",
            768,
        )
    )

    hidden_dim = int(
        protocol.get(
            "hidden_dim",
            128,
        )
    )

    dropout = float(
        protocol.get(
            "dropout",
            0.2,
        )
    )

    if rep_dim != 768:
        raise RuntimeError(
            f"Unsupported representation_dim={rep_dim}"
        )

    if hidden_dim != 128:
        raise RuntimeError(
            f"Unsupported hidden_dim={hidden_dim}"
        )

    model = GenoGrammaPairHead(
        representation_dim=rep_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    model.load_state_dict(
        obj["head_state_dict"],
        strict=True,
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    if n_params != 98690:
        raise RuntimeError(
            f"Unexpected head parameters: {n_params}"
        )

    model.to(device)

    model.eval()

    mean = np.asarray(
        obj["zscore_mean"],
        dtype=np.float32,
    )

    sd = np.asarray(
        obj["zscore_sd"],
        dtype=np.float32,
    )

    if mean.shape != (768,):
        raise RuntimeError(
            f"zscore_mean shape={mean.shape}"
        )

    if sd.shape != (768,):
        raise RuntimeError(
            f"zscore_sd shape={sd.shape}"
        )

    if (
        not np.all(np.isfinite(mean))
        or
        not np.all(np.isfinite(sd))
    ):
        raise RuntimeError(
            "Non-finite z-score parameters"
        )

    if np.any(sd <= 0):
        raise RuntimeError(
            "zscore_sd contains non-positive values"
        )

    return model, mean, sd, obj


def predict(
    model,
    rep,
    mean,
    sd,
    device,
    batch_size,
):

    n = len(rep)

    p0 = np.empty(
        n,
        dtype=np.float32,
    )

    p1 = np.empty(
        n,
        dtype=np.float32,
    )

    with torch.inference_mode():

        for start in range(
            0,
            n,
            batch_size,
        ):

            end = min(
                start + batch_size,
                n,
            )

            x = np.asarray(
                rep[start:end],
                dtype=np.float32,
            )

            x = (
                (x - mean)
                /
                sd
            ).astype(
                np.float32,
                copy=False,
            )

            xt = (
                torch.from_numpy(x)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            prob = torch.softmax(
                model(xt),
                dim=1,
            )

            prob = (
                prob
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            p0[start:end] = prob[:, 0]
            p1[start:end] = prob[:, 1]

    return p0, p1


def main():

    ap = argparse.ArgumentParser(
        description=(
            "Predict the bundled DOOR2-derived "
            "adjacent-gene operon-status task "
            "from GenoGramma-compatible 64-gene windows."
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
        "--head",
        default=None,
        help=(
            "Optional compatible Stage56-format "
            "downstream head. If omitted, the "
            "bundled formal operon head for --fold "
            "is used."
        ),
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

    input_path = Path(
        args.input
    ).resolve()

    output_path = Path(
        args.output
    ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(
        input_path,
        allow_pickle=False,
    ) as z:

        if "family_rows" not in z.files:
            raise RuntimeError(
                "Input NPZ missing family_rows"
            )

        n = int(
            z["family_rows"].shape[0]
        )

        if "row_index" in z.files:
            row_index = np.asarray(
                z["row_index"]
            )
        else:
            row_index = np.arange(
                n,
                dtype=np.int64,
            )

    if row_index.shape != (n,):
        raise RuntimeError(
            f"row_index shape={row_index.shape}"
        )

    device = resolve_device(
        args.device
    )

    head_path = resolve_head(
        args.head,
        args.fold,
    )

    (
        head,
        mean,
        sd,
        head_obj,
    ) = load_head(
        head_path,
        args.fold,
        device,
    )

    with tempfile.TemporaryDirectory(
        prefix="genogramma_predict_"
    ) as td:

        rep_path = (
            Path(td)
            / "representation_768D.npy"
        )

        cmd = [
            sys.executable,
            str(EMBED_ENGINE),
            "--input",
            str(input_path),
            "--output",
            str(rep_path),
            "--fold",
            str(args.fold),
            "--batch-size",
            str(args.batch_size),
            "--device",
            str(args.device),
        ]

        subprocess.run(
            cmd,
            check=True,
        )

        rep = np.load(
            rep_path,
            mmap_mode="r",
        )

        if rep.shape != (n, 768):
            raise RuntimeError(
                f"Representation shape={rep.shape}"
            )

        p0, p1 = predict(
            head,
            rep,
            mean,
            sd,
            device,
            args.batch_size,
        )

    pred = (
        p1 >= 0.5
    ).astype(
        np.int64
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "row_index",
            "prob_class_0",
            "prob_class_1",
            "predicted_class",
            "fold",
            "head_checkpoint",
        ])

        for i in range(n):

            w.writerow([
                row_index[i],
                float(p0[i]),
                float(p1[i]),
                int(pred[i]),
                int(args.fold),
                str(head_path),
            ])

    meta = {
        "model": "GenoGramma",
        "task": (
            "DOOR2-derived adjacent-gene "
            "operon-status prediction"
        ),
        "label_note": (
            "DOOR2 labels are external "
            "computational annotations, not "
            "experimental ground truth."
        ),
        "input": str(input_path),
        "output": str(output_path),
        "n_windows": int(n),
        "window_length": 64,
        "target_pair_indices_0based": [31, 32],
        "representation_dimension": 768,
        "fold": int(args.fold),
        "head_checkpoint": str(head_path),
        "head_best_epoch": head_obj.get(
            "best_epoch"
        ),
        "head_parameters": 98690,
        "head_architecture": (
            "768->128->GELU->Dropout->2"
        ),
        "zscore_from_checkpoint": True,
        "probability_semantics": {
            "prob_class_0": (
                "predicted probability of "
                "DOOR2-derived negative class"
            ),
            "prob_class_1": (
                "predicted probability of "
                "DOOR2-derived positive class "
                "(same-operon annotation)"
            ),
        },
        "decision_threshold": 0.5,
        "deployment_note": (
            "This bundled head is specific to "
            "the formal DOOR2-derived operon task. "
            "It is not a universal zero-shot "
            "phenotype predictor."
        ),
    }

    meta_path = Path(
        str(output_path) + ".json"
    )

    meta_path.write_text(
        json.dumps(
            meta,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[PASS] GenoGramma formal operon prediction"
    )

    print(
        "       windows =",
        n,
    )

    print(
        "       fold    =",
        args.fold,
    )

    print(
        "       head    =",
        head_path,
    )

    print(
        "       output  =",
        output_path,
    )

    print(
        "       metadata=",
        meta_path,
    )


if __name__ == "__main__":
    main()
