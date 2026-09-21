#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

END_TO_END_TRAINING_INTERFACE = True

BASE = Path(__file__).resolve().parent

PREP = BASE / "pipeline" / "preprocessing" / "prepare_formal_workspace.py"
RAW_TO_FAMILY = BASE / "pipeline" / "preprocessing" / "run_raw_to_family.py"
SSL_SCRIPT = BASE / "pipeline" / "training" / "formal_lineage_clean_pretrain.py"
STAGE55 = BASE / "pipeline" / "55_extract_genogramma_pairaware_representations.py"
STAGE56 = BASE / "pipeline" / "56_genogramma_pairaware_probe.py"

PACKAGED_FAMILY = (
    BASE / "model_assets" / "family_embeddings"
    / "00_family_embeddings_276517x480.npy"
)
SSL_INDEX = BASE / "data" / "07_1_ssl_encoder"
UPSTREAM = BASE / "model_assets" / "formal_upstream"
LINEAGE_ASSETS = UPSTREAM / "lineage_pretraining"
ELIGIBLE_SPLIT = (
    UPSTREAM / "r4_eligible" / "fold_1_split_manifest.csv"
)

EXPECTED_FAMILY_SHAPE = (276517, 480)
N_FOLDS = 5


def run(cmd, env=None):
    cmd = [str(x) for x in cmd]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def require_file(path: Path):
    if not path.is_file():
        raise SystemExit(f"[FAIL] missing file: {path}")


def require_dir(path: Path):
    if not path.is_dir():
        raise SystemExit(f"[FAIL] missing directory: {path}")


def safe_link(src: Path, dst: Path):
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_symlink():
        if dst.resolve() == src:
            return
        dst.unlink()
    elif dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    dst.symlink_to(src, target_is_directory=src.is_dir())


def copy_file(src: Path, dst: Path):
    require_file(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def validate_family(path: Path):
    require_file(path)
    x = np.load(path, mmap_mode="r")
    if x.shape != EXPECTED_FAMILY_SHAPE:
        raise SystemExit(
            f"[FAIL] family embedding shape {x.shape} "
            f"!= {EXPECTED_FAMILY_SHAPE}"
        )
    if x.dtype != np.float16:
        raise SystemExit(
            f"[FAIL] family embedding dtype {x.dtype} != float16"
        )
    print(
        f"[PASS] family embedding {path} | "
        f"shape={x.shape} dtype={x.dtype}"
    )


def prepare_ssl_bridge(work: Path, family: Path):
    require_dir(SSL_INDEX)
    require_dir(LINEAGE_ASSETS)
    require_file(ELIGIBLE_SPLIT)

    safe_link(SSL_INDEX, work / "07_1_ssl_encoder")
    safe_link(
        family,
        work / "08_explicit_order_ssl"
        / "00_family_embeddings_276517x480.npy",
    )

    lineage_dst = (
        work / "new_paper_data" / "05_downstream_function"
        / "lineage_pretraining"
    )
    lineage_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in sorted(LINEAGE_ASSETS.iterdir()):
        if src.is_file():
            copy_file(src, lineage_dst / src.name)
            copied += 1

    if copied == 0:
        raise SystemExit("[FAIL] no frozen lineage assets copied")

    copy_file(
        ELIGIBLE_SPLIT,
        work / "new_paper_data" / "03_ablation_falsification"
        / "R4_context_ssl_samplewise" / "fold_1"
        / "split_manifest.csv",
    )

    print("[PASS] frozen SSL protocol bridge installed")


def validate_ssl_checkpoints(root: Path):
    for fold in range(1, N_FOLDS + 1):
        p = root / f"fold_{fold}" / "best_checkpoint.pt"
        require_file(p)

        obj = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        if int(obj.get("formal_outer_fold", -1)) != fold:
            raise SystemExit(
                f"[FAIL] outer-fold metadata mismatch: {p}"
            )

        if bool(obj.get("phenotype_labels_used", True)):
            raise SystemExit(
                f"[FAIL] phenotype labels used during SSL: {p}"
            )

        for key in (
            "outer_holdout_used_for_training",
            "outer_holdout_used_for_early_stopping",
            "outer_holdout_used_for_model_selection",
        ):
            if bool(obj.get(key, True)):
                raise SystemExit(f"[FAIL] {key}=True: {p}")

        print(
            f"[PASS] new SSL fold {fold} | "
            f"best_epoch={obj.get('best_epoch')} | {p}"
        )


def print_plan(route: str, work: Path, resume: bool, downstream: bool):
    print("=" * 96)
    print("GENOGRAMMA — END-TO-END TRAINING PLAN")
    print("=" * 96)
    print("package root :", BASE)
    print("work root    :", work)
    print("route        :", route)
    print("resume       :", resume)
    print("downstream   :", downstream)
    print()
    print("Execution plan:")
    print("  1. Install frozen formal 994-genome / ANI95 workspace")
    if route == "raw-genomes":
        print("  2. Prodigal gene calling on the frozen formal genome universe")
        print("  3. MMseqs2 easy-linclust: identity=0.50, coverage=0.80, cov-mode=0")
        print("  4. Deterministic family representatives")
        print("  5. ESM-2 t12 35M residue-mean -> 480D float16 family embeddings")
    else:
        print("  2-5. Reuse packaged byte-verified 276517 x 480 float16 family embeddings")
    print("  6. Install frozen ANI95 lineage / outer-fold protocol metadata")
    print("  7. Train 5 lineage-clean GenoGramma SSL folds from random initialization")
    print("  8. Validate leakage metadata in all NEW checkpoints")
    if downstream:
        print("  9. Stage55 using ONLY the NEW SSL checkpoints")
        print(" 10. Stage56 formal DOOR2-derived operon probe/evaluation")
    else:
        print("  9-10. Downstream evaluation skipped by request")
    print()
    print("Important:")
    print("  - run.py remains the manuscript checkpoint reproduction route.")
    print("  - train.py does not replace or overwrite packaged formal checkpoints.")
    print("  - ESM-2 is pretrained; GenoGramma SSL and downstream heads are retrained.")
    if route == "raw-genomes":
        deps = {
            "prodigal": shutil.which("prodigal"),
            "mmseqs": shutil.which("mmseqs"),
        }
        try:
            import esm  # noqa: F401
            esm_ok = True
        except Exception:
            esm_ok = False
        print()
        print("Current raw-route dependency visibility:")
        for k, v in deps.items():
            print(f"  {k:8s}: {v or 'NOT ON PATH'}")
        print(f"  fair-esm : {'available' if esm_ok else 'NOT IMPORTABLE'}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Train GenoGramma end to end. With no route flag, the default "
            "is the frozen formal raw-genome -> ESM -> GenoGramma -> "
            "downstream workflow."
        )
    )

    route = ap.add_mutually_exclusive_group()
    route.add_argument(
        "--from-family-embeddings",
        action="store_true",
        help=(
            "Skip raw genome / Prodigal / MMseqs2 / ESM regeneration and "
            "start from the packaged byte-verified family embedding matrix."
        ),
    )
    route.add_argument(
        "--from-raw",
        action="store_true",
        help=(
            "Explicit alias for the default complete raw-genome route."
        ),
    )

    ap.add_argument(
        "--plan",
        "--plan-only",
        dest="plan_only",
        action="store_true",
        help="Print the resolved workflow and do not train.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse supported completed preprocessing/training outputs.",
    )
    ap.add_argument(
        "--skip-downstream",
        action="store_true",
        help="Stop after the five newly trained GenoGramma SSL checkpoints.",
    )
    ap.add_argument(
        "--work-root",
        "--output",
        dest="work_root",
        default=None,
        help="Training workspace/output root.",
    )
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=60.0,
        help="Raw-route disk safety threshold (default: 60 GB).",
    )

    args = ap.parse_args()

    if args.work_root:
        work = Path(args.work_root).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = (
            BASE / "results" / "training_runs"
            / f"genogramma_train_{stamp}"
        ).resolve()

    route_name = (
        "family-embeddings"
        if args.from_family_embeddings
        else "raw-genomes"
    )

    sources = (PREP, RAW_TO_FAMILY, SSL_SCRIPT, STAGE55, STAGE56)
    missing = [p for p in sources if not p.is_file()]
    if missing:
        raise SystemExit(
            "[FAIL] missing training source(s):\n  "
            + "\n  ".join(map(str, missing))
        )

    require_file(PACKAGED_FAMILY)

    print_plan(
        route_name,
        work,
        args.resume,
        not args.skip_downstream,
    )

    if args.plan_only:
        print("[FINAL PASS] plan only; no training executed")
        return

    work.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable,
        PREP,
        "--work-root",
        work,
    ])

    if route_name == "raw-genomes":
        available = free_gb(BASE)
        print(
            f"[INFO] free disk before raw rebuild = "
            f"{available:.1f} GB"
        )
        if available < args.min_free_gb:
            raise SystemExit(
                f"[FAIL] raw route safety threshold requires >= "
                f"{args.min_free_gb:.1f} GB free; only "
                f"{available:.1f} GB available. "
                f"Use a larger disk or --from-family-embeddings."
            )

        cmd = [
            sys.executable,
            RAW_TO_FAMILY,
            "--work-root",
            work,
        ]
        if args.resume:
            cmd.append("--resume")
        run(cmd)

        family = (
            work / "06_family_esm2"
            / "00_family_embeddings_276517x480.npy"
        )
    else:
        family = PACKAGED_FAMILY

    validate_family(family)
    prepare_ssl_bridge(work, family)

    ssl_out = (
        work / "new_paper_data" / "09_genogramma"
        / "02_lineage_clean_pretraining"
    )
    ssl_out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["GENOGRAMMAR_DATA_ROOT"] = str(work)
    env["GENOGRAMMAR_REVISION_OUTPUT"] = str(ssl_out)
    env["PYTHONPATH"] = (
        str(BASE / "pipeline" / "training")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("PYTHONUNBUFFERED", "1")

    run([sys.executable, "-u", SSL_SCRIPT], env=env)
    validate_ssl_checkpoints(ssl_out)

    if args.skip_downstream:
        print("=" * 96)
        print("[FINAL PASS] GenoGramma SSL retraining complete")
        print("new checkpoint root =", ssl_out)
        return

    downstream_root = (
        work / "new_paper_data" / "10_genogramma_pairaware"
    )
    downstream_root.mkdir(parents=True, exist_ok=True)

    denv = os.environ.copy()
    denv["GENOGRAMMA_RESULT_ROOT"] = str(downstream_root)
    denv["GENOGRAMMA_CHECKPOINT_ROOT"] = str(ssl_out)
    denv["GENOGRAMMA_FAMILY_EMBEDDINGS"] = str(family)
    denv.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    denv.setdefault("PYTHONUNBUFFERED", "1")

    run([sys.executable, "-u", STAGE55], env=denv)
    run([sys.executable, "-u", STAGE56], env=denv)

    summary = (
        downstream_root / "04_stage56_pairaware_probe"
        / "04_model_summary.csv"
    )
    freeze = (
        downstream_root / "04_stage56_pairaware_probe"
        / "06_stage56_result_freeze.json"
    )

    require_file(summary)
    require_file(freeze)

    print("=" * 96)
    print("[FINAL PASS] GenoGramma end-to-end training/evaluation complete")
    print("=" * 96)
    print("route                =", route_name)
    print("family embeddings    =", family)
    print("new checkpoint root  =", ssl_out)
    print("downstream root      =", downstream_root)
    print("downstream summary   =", summary)
    print("downstream freeze    =", freeze)


if __name__ == "__main__":
    main()
