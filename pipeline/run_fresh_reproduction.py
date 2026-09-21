from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

# Required for deterministic CUDA/cuBLAS operations when
# torch.use_deterministic_algorithms(True) is enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve()

BASE = Path(__file__).resolve().parents[1]

PAPER = (
    BASE
)

CANONICAL_ROOT = (
    BASE / "runtime_assets" / "source_tree" / "new_paper_data"
    / "10_genogramma_pairaware"
)

REPRODUCTION_PARENT = (
    BASE / "results" / "reproduction_runs"
)


STAGES = [
    (
        "55",
        PAPER
        / "pipeline"
        / "55_extract_genogramma_pairaware_representations.py",
    ),

    (
        "56",
        PAPER
        / "pipeline"
        / "56_genogramma_pairaware_probe.py",
    ),

    (
        "57B2",
        PAPER
        / "pipeline"
        / "57B2_extract_native_matched_pair_features.py",
    ),

    (
        "57C",
        PAPER
        / "pipeline"
        / "57C_matched_pairaware_baseline_comparison.py",
    ),

    (
        "58A",
        PAPER
        / "pipeline"
        / "58A_strand_shortcut_same_strand_audit.py",
    ),

    (
        "58B",
        PAPER
        / "pipeline"
        / "58B_lineage_level_bootstrap.py",
    ),

    (
        "59A",
        PAPER
        / "pipeline"
        / "59A_genogramma_768d_component_ablation.py",
    ),

    (
        "59B",
        PAPER
        / "pipeline"
        / "59B_ablation_lineage_bootstrap.py",
    ),

    (
        "60",
        PAPER
        / "pipeline"
        / "60_final_operon_evidence_freeze.py",
    ),
]


GENERATED_OUTPUTS = {
    "03_stage55_pairaware_representations",
    "04_stage56_pairaware_probe",
    "07_stage57B2_native_matched_features",
    "08_stage57C_matched_comparison",
    "09_stage58A_strand_shortcut_audit",
    "10_stage58B_lineage_bootstrap",
    "11_stage59A_768d_ablation",
    "12_stage59B_ablation_lineage_bootstrap",
    "13_stage60_final_operon_evidence",
}


def numeric_prefix(
    name: str,
):

    token = name.split(
        "_",
        1,
    )[
        0
    ]


    try:

        return int(
            token
        )

    except Exception:

        return None


def seed_static_prerequisites(
    fresh_root: Path,
):

    copied = []


    fresh_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    for src in CANONICAL_ROOT.iterdir():

        if src.name in GENERATED_OUTPUTS:

            continue


        prefix = numeric_prefix(
            src.name
        )


        # Fresh formal downstream requires only the preregistration /
        # pre-Stage55 static files under this result root.
        if (
            prefix is None
            or
            prefix > 2
        ):

            continue


        dst = (
            fresh_root
            /
            src.name
        )


        if dst.exists():

            continue


        if src.is_file():

            shutil.copy2(
                src,
                dst,
            )


        elif src.is_dir():

            shutil.copytree(
                src,
                dst,
                symlinks=False,
            )


        copied.append(
            str(
                dst
            )
        )


    return copied


def validate_assets():

    required = [
        BASE / "runtime_assets" / "source_tree" / "new_paper_data"
        / "08_stage50_order_dependent_downstream"
        / "07_pair_to_genogrammar_alignment"
        / "13_STAGE50_operon_formal_model_inputs.npz",

        BASE / "runtime_assets" / "source_tree" / "new_paper_data"
        / "08_stage50_order_dependent_downstream"
        / "07_pair_to_genogrammar_alignment"
        / "14B_STAGE50_operon_model_input_metadata_with_row_index.csv.gz",

        PAPER
        / "data"
        / "08_explicit_order_ssl"
        / "00_family_embeddings_276517x480.npy",

        PAPER
        / "requirements_lock.txt",

        PAPER
        / "model_assets"
        / "esm2"
        / "esm2_t12_35M_UR50D.pt",
    ]


    for p in required:

        if not p.exists():

            raise RuntimeError(
                f"Missing packaged reproduction asset: {p}"
            )


    checkpoints = list(
        (
            PAPER
            / "checkpoints"
        ).rglob(
            "*.pt"
        )
    )


    if not checkpoints:

        raise RuntimeError(
            "No packaged model checkpoints found"
        )


    return checkpoints


def main(
    argv=None,
):

    parser = argparse.ArgumentParser(
        description=(
            "Run the formal GenoGramma Stage55-60 downstream "
            "reproduction in an isolated output root."
        )
    )


    parser.add_argument(
        "--fresh",
        action="store_true",
        help=argparse.SUPPRESS,
    )


    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Validate fresh-run assets and stage graph "
            "without executing any model computation."
        ),
    )


    parser.add_argument(
        "--fresh-id",
        default=None,
        help=(
            "Optional isolated reproduction run ID."
        ),
    )


    args = parser.parse_args(
        argv
    )


    checkpoints = validate_assets()


    for stage, script in STAGES:

        if not script.exists():

            raise RuntimeError(
                f"Missing formal Stage {stage}: {script}"
            )


    run_id = (
        args.fresh_id
        or
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )


    fresh_root = (
        REPRODUCTION_PARENT
        /
        run_id
    )


    print("=" * 100)

    print(
        "GENOGRAMMA — ISOLATED FRESH DOWNSTREAM REPRODUCTION"
    )

    print("=" * 100)

    print(
        "[INFO] package root       =",
        BASE,
    )

    print(
        "[INFO] canonical root     =",
        CANONICAL_ROOT,
    )

    print(
        "[INFO] fresh result root  =",
        fresh_root,
    )

    print(
        "[INFO] checkpoints        =",
        len(
            checkpoints
        ),
    )

    print(
        "[INFO] plan only          =",
        args.plan_only,
    )


    print()
    print(
        "===== FRESH STAGE PLAN ====="
    )


    for stage, script in STAGES:

        print(
            f"{stage:>4}  {script.name}"
        )


    if args.plan_only:

        print()
        print(
            "[PASS] all packaged assets available"
        )

        print(
            "[PASS] all formal Stage55-60 scripts available"
        )

        print(
            "[PASS] fresh result root override supported"
        )

        print(
            "[PASS] no training executed"
        )

        return 0


    if fresh_root.exists():

        raise RuntimeError(
            f"Fresh output root already exists: {fresh_root}"
        )


    REPRODUCTION_PARENT.mkdir(
        parents=True,
        exist_ok=True,
    )


    copied = seed_static_prerequisites(
        fresh_root
    )


    print(
        f"[PASS] seeded static prerequisites = {len(copied)}"
    )


    env = os.environ.copy()


    env[
        "GENOGRAMMA_RESULT_ROOT"
    ] = str(
        fresh_root
    )


    records = []


    for stage, script in STAGES:

        print()
        print(
            "-" * 100
        )

        print(
            f"FRESH STAGE {stage}: {script.name}"
        )

        print(
            "-" * 100
        )


        log = (
            fresh_root
            /
            f"fresh_stage_{stage}.log"
        )


        with open(
            log,
            "w",
        ) as fh:

            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(
                        script
                    ),
                ],
                cwd=PAPER,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
            )


            rc = proc.wait()


        record = {
            "stage":
                stage,

            "script":
                str(
                    script
                ),

            "returncode":
                rc,

            "log":
                str(
                    log
                ),
        }


        records.append(
            record
        )


        print(
            f"[INFO] RC={rc}"
        )

        print(
            f"[INFO] log={log}"
        )


        if rc != 0:

            (
                fresh_root
                / "FRESH_RUN_FAILED.json"
            ).write_text(
                json.dumps(
                    {
                        "status":
                            "FAILED",

                        "failed_stage":
                            stage,

                        "records":
                            records,
                    },
                    indent=2,
                )
            )


            raise RuntimeError(
                f"Fresh reproduction failed at Stage {stage}"
            )


        print(
            f"[PASS] Stage {stage}"
        )


    (
        fresh_root
        / "FRESH_RUN_COMPLETE.json"
    ).write_text(
        json.dumps(
            {
                "status":
                    "GENOGRAMMA_FRESH_REPRODUCTION_COMPLETE",

                "run_id":
                    run_id,

                "fresh_root":
                    str(
                        fresh_root
                    ),

                "records":
                    records,
            },
            indent=2,
        )
    )


    print()
    print("=" * 100)

    print(
        "[FINAL PASS] isolated fresh downstream reproduction complete"
    )

    print(
        "[OUTPUT]",
        fresh_root,
    )

    print("=" * 100)

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
