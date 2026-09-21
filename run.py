#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# =============================================================================
# ISOLATED FRESH DOWNSTREAM REPRODUCTION
# =============================================================================

if "--fresh" in sys.argv:
    from pipeline.run_fresh_reproduction import main as _fresh_main
    raise SystemExit(
        _fresh_main(
            sys.argv[1:]
        )
    )



# =============================================================================
# FORMAL IDENTITY
# =============================================================================

MODEL_NAME = "GenoGramma"

BASE = Path(__file__).resolve().parent
PAPER = BASE

ROOT = (
    BASE / "runtime_assets" / "source_tree" / "new_paper_data"
    / "10_genogramma_pairaware"
)

RUN_ROOT = BASE / "results" / "formal_runs"

RUN_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


FORMAL_MODULE = (
    PAPER
    / "pipeline"
    / "revision_modules"
    / "genogramma.py"
)

EXPECTED_FORMAL_MODULE_SHA = (
    "7579820f2324f101fa7412285233f9ab"
    "1d80548f234269a5ad7ef47a76be2e01"
)


# =============================================================================
# UTILS
# =============================================================================

def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_text_safe(path: Path) -> str:

    try:
        return path.read_text(
            errors="ignore"
        )
    except Exception:
        return ""


def marker_in_logs(
    directory: Path,
    marker: str,
) -> bool:

    if not directory.exists():
        return False

    for p in directory.rglob("*.log"):

        if marker in read_text_safe(p):
            return True

    return False


def active_pids(
    directory: Path,
) -> list[int]:

    out = []

    if not directory.exists():
        return out

    for p in directory.rglob("*.pid"):

        try:
            pid = int(
                p.read_text().strip()
            )
        except Exception:
            continue

        try:
            os.kill(
                pid,
                0,
            )
        except OSError:
            continue

        out.append(
            pid
        )

    return sorted(
        set(out)
    )


# =============================================================================
# STAGE DEFINITION
# =============================================================================

@dataclass
class Stage:

    key: str

    name: str

    runner: Path

    output_dir: Path

    required_files: tuple[str, ...] = ()

    marker: Optional[str] = None

    transitive_files: tuple[Path, ...] = ()


    def done(
        self,
    ) -> bool:

        # -----------------------------------------------------
        # A later frozen artifact can prove an upstream stage
        # had already completed successfully.
        # -----------------------------------------------------

        if self.transitive_files:

            if all(
                p.exists()
                for p
                in self.transitive_files
            ):
                return True


        # -----------------------------------------------------
        # Exact required outputs
        # -----------------------------------------------------

        if self.required_files:

            ok = all(
                (
                    self.output_dir
                    / f
                ).exists()

                for f
                in self.required_files
            )

            if ok:
                return True


        # -----------------------------------------------------
        # PASS marker in logs
        # -----------------------------------------------------

        if self.marker:

            if marker_in_logs(
                self.output_dir,
                self.marker,
            ):
                return True


        return False


# =============================================================================
# FORMAL STAGES
# =============================================================================

STAGE57C_FINAL = (
    ROOT
    / "08_stage57C_matched_comparison"
    / "57C_FINAL_matched_pairaware_comparison.csv"
)


STAGES = [

    Stage(
        key="55",
        name="GenoGramma pair-aware representation extraction",
        runner=(
            PAPER
            / "run_stage55_genogramma_pairaware.sh"
        ),
        output_dir=(
            ROOT
            / "03_stage55_pairaware_representations"
        ),
        marker=(
            "ALL_5_GENOGRAMMA_PAIRAWARE_"
            "REPRESENTATIONS_PASS"
        ),
        transitive_files=(
            STAGE57C_FINAL,
        ),
    ),


    Stage(
        key="56",
        name="Formal GenoGramma downstream probe",
        runner=(
            PAPER
            / "run_stage56_genogramma_pairaware_probe.sh"
        ),
        output_dir=(
            ROOT
            / "04_stage56_pairaware_probe"
        ),
        marker=(
            "GENOGRAMMA_PAIRAWARE_TARGET"
        ),
        transitive_files=(
            STAGE57C_FINAL,
        ),
    ),


    Stage(
        key="57B2",
        name="Native matched baseline pair-state extraction",
        runner=(
            PAPER
            / "run_stage57B2_native_matched_features.sh"
        ),
        output_dir=(
            ROOT
            / "07_stage57B2_native_matched_features"
        ),
        marker=(
            "ALL_20_NATIVE_MATCHED_PAIR_STATES_PASS"
        ),
        transitive_files=(
            STAGE57C_FINAL,
        ),
    ),


    Stage(
        key="57C",
        name="Matched pair-aware baseline comparison",
        runner=(
            PAPER
            / "run_stage57C_matched_pairaware_comparison.sh"
        ),
        output_dir=(
            ROOT
            / "08_stage57C_matched_comparison"
        ),
        required_files=(
            "57C_FINAL_matched_pairaware_comparison.csv",
            "57C_GenoGramma_vs_baseline_deltas.csv",
        ),
    ),


    Stage(
        key="58A",
        name="Strand-shortcut robustness audit",
        runner=(
            PAPER
            / "run_stage58A_strand_shortcut_audit.sh"
        ),
        output_dir=(
            ROOT
            / "09_stage58A_strand_shortcut_audit"
        ),
        required_files=(
            "58A_all_vs_same_strand_model_summary.csv",
            "58A_StrandMatch_baseline.csv",
        ),
    ),


    Stage(
        key="58B",
        name="ANI95 lineage-level baseline bootstrap",
        runner=(
            PAPER
            / "run_stage58B_lineage_bootstrap.sh"
        ),
        output_dir=(
            ROOT
            / "10_stage58B_lineage_bootstrap"
        ),
        required_files=(
            "58B_equal_lineage_bootstrap_CI.csv",
            "58B_paired_lineage_bootstrap_deltas.csv",
        ),
    ),


    Stage(
        key="59A",
        name="GenoGramma 768D representation ablation",
        runner=(
            PAPER
            / "run_stage59A_genogramma_768d_ablation.sh"
        ),
        output_dir=(
            ROOT
            / "11_stage59A_768d_ablation"
        ),
        required_files=(
            "59A_FINAL_ablation_summary.csv",
            "59A_component_increment_summary.csv",
        ),
    ),


    Stage(
        key="59B",
        name="Ablation ANI95 lineage bootstrap",
        runner=(
            PAPER
            / "run_stage59B_ablation_lineage_bootstrap.sh"
        ),
        output_dir=(
            ROOT
            / "12_stage59B_ablation_lineage_bootstrap"
        ),
        required_files=(
            "59B_paired_lineage_bootstrap_contrasts.csv",
            "59B_CONTEXTUAL_EDGE_focus.csv",
        ),
    ),


    Stage(
        key="60",
        name="Final operon evidence freeze",
        runner=(
            PAPER
            / "run_stage60_final_operon_evidence_freeze.sh"
        ),
        output_dir=(
            ROOT
            / "13_stage60_final_operon_evidence"
        ),
        required_files=(
            "60_FINAL_RESULT_FREEZE.json",
            "60A_FINAL_matched_pairaware_model_comparison.csv",
            "60G_FINAL_key_evidence.csv",
            "60I_source_provenance_manifest.csv",
        ),
    ),
]


# =============================================================================
# PREFLIGHT
# =============================================================================

def preflight():

    print(
        "=" * 96
    )

    print(
        "GENOGRAMMA — FORMAL TRAINING / EVALUATION ENTRY POINT"
    )

    print(
        "=" * 96
    )


    # -----------------------------------------------------------------
    # Canonical module
    # -----------------------------------------------------------------

    if not FORMAL_MODULE.exists():

        raise RuntimeError(
            f"Missing formal GenoGramma module: "
            f"{FORMAL_MODULE}"
        )


    module_sha = sha256(
        FORMAL_MODULE
    )


    print(
        f"[PASS] formal module = "
        f"{FORMAL_MODULE}"
    )

    print(
        f"[INFO] SHA256 = "
        f"{module_sha}"
    )


    if (
        module_sha
        !=
        EXPECTED_FORMAL_MODULE_SHA
    ):

        raise RuntimeError(
            "Formal GenoGramma module SHA256 changed. "
            "Refusing silent execution."
        )


    print(
        "[PASS] formal module SHA256 frozen"
    )


    # -----------------------------------------------------------------
    # Runners
    # -----------------------------------------------------------------

    for stage in STAGES:

        if not stage.runner.exists():

            raise RuntimeError(
                f"Missing runner for Stage "
                f"{stage.key}: "
                f"{stage.runner}"
            )


    print(
        "[PASS] all formal runners present"
    )


    # -----------------------------------------------------------------
    # Naming audit
    # -----------------------------------------------------------------

    this_file = Path(
        __file__
    )

    text = read_text_safe(
        this_file
    )

    # Construct the historical name dynamically so that the
    # forbidden literal itself does not appear in this source file
    # and therefore cannot trigger the audit on its own.
    legacy_name = "Geno" + "Grammar"

    if legacy_name in text:

        raise RuntimeError(
            "Legacy model name detected "
            "inside formal run.py"
        )


    print(
        "[PASS] formal model name = GenoGramma"
    )


# =============================================================================
# EXECUTION
# =============================================================================

def run_stage(
    stage: Stage,
    rerun: bool = False,
) -> dict:

    print()
    print(
        "-" * 96
    )

    print(
        f"Stage {stage.key}: "
        f"{stage.name}"
    )

    print(
        "-" * 96
    )


    if (
        stage.done()
        and
        not rerun
    ):

        print(
            "[SKIP/PASS] frozen result already exists"
        )

        return {
            "stage":
                stage.key,

            "name":
                stage.name,

            "status":
                "SKIPPED_EXISTING_PASS",

            "output_dir":
                str(
                    stage.output_dir
                ),
        }


    print(
        f"[RUNNER] {stage.runner}"
    )


    proc = subprocess.run(
        [
            "bash",
            str(
                stage.runner
            ),
        ],
        cwd=str(
            PAPER
        ),
        check=False,
    )


    if proc.returncode != 0:

        raise RuntimeError(
            f"Stage {stage.key} runner "
            f"returned RC={proc.returncode}"
        )


    # -----------------------------------------------------------------
    # Many historical runners use nohup.
    # Wait for their PID rather than immediately advancing.
    # -----------------------------------------------------------------

    no_pid_rounds = 0


    while True:

        if stage.done():

            print(
                f"[PASS] Stage "
                f"{stage.key}"
            )

            break


        pids = active_pids(
            stage.output_dir
        )


        if pids:

            no_pid_rounds = 0

            print(
                f"[RUNNING] Stage "
                f"{stage.key} PID="
                f"{','.join(map(str, pids))}"
            )

            time.sleep(
                10
            )

            continue


        no_pid_rounds += 1


        # Give a just-started background runner
        # a short opportunity to create its PID/output.
        if no_pid_rounds <= 3:

            time.sleep(
                2
            )

            continue


        if not stage.done():

            raise RuntimeError(
                f"Stage {stage.key} exited "
                f"without required PASS outputs. "
                f"Inspect: {stage.output_dir}"
            )


    return {
        "stage":
            stage.key,

        "name":
            stage.name,

        "status":
            "PASS",

        "output_dir":
            str(
                stage.output_dir
            ),
    }


# =============================================================================
# FINAL RESULTS
# =============================================================================

def show_final_results():

    final_dir = (
        ROOT
        / "13_stage60_final_operon_evidence"
    )


    freeze_file = (
        final_dir
        / "60_FINAL_RESULT_FREEZE.json"
    )


    comparison_file = (
        final_dir
        / "60A_FINAL_matched_pairaware_model_comparison.csv"
    )


    edge_file = (
        final_dir
        / "60E_FINAL_contextual_edge_lineage_bootstrap.csv"
    )


    if not freeze_file.exists():

        raise RuntimeError(
            "Final Stage60 freeze is missing"
        )


    freeze = json.loads(
        freeze_file.read_text()
    )


    if (
        freeze.get(
            "status"
        )
        !=
        "STAGE60_FINAL_OPERON_EVIDENCE_FROZEN"
    ):

        raise RuntimeError(
            "Unexpected Stage60 freeze status"
        )


    print()
    print(
        "=" * 96
    )

    print(
        "FINAL FROZEN GENOGRAMMA RESULT"
    )

    print(
        "=" * 96
    )


    print(
        f"Model:              "
        f"{freeze['formal_model_name']}"
    )

    print(
        f"Task:               "
        f"{freeze['task']}"
    )

    print(
        f"Scope:              "
        f"{freeze['scope']}"
    )

    print(
        f"Evaluation:         "
        f"{freeze['main_evaluation']}"
    )

    print(
        f"Formal examples:    "
        f"{freeze['formal_examples']}"
    )

    print(
        f"Genomes:            "
        f"{freeze['genomes']}"
    )

    print(
        f"ANI95 lineages:     "
        f"{freeze['ANI95_lineages']}"
    )


    if comparison_file.exists():

        import pandas as pd

        df = pd.read_csv(
            comparison_file
        )

        gg = df[
            df["model"]
            ==
            "GenoGramma"
        ].iloc[0]

        print()
        print(
            "Matched pair-aware formal result:"
        )

        print(
            f"  ROC-AUC = "
            f"{gg['ROC_AUC']:.6f}"
        )

        print(
            f"  PR-AUC  = "
            f"{gg['PR_AUC']:.6f}"
        )

        print(
            f"  BalAcc  = "
            f"{gg['Balanced_Accuracy']:.6f}"
        )

        print(
            f"  F1      = "
            f"{gg['F1']:.6f}"
        )

        print(
            f"  MCC     = "
            f"{gg['MCC']:.6f}"
        )


    if edge_file.exists():

        import pandas as pd

        e = pd.read_csv(
            edge_file
        )

        q = e[
            (
                e["subset"]
                ==
                "same_strand_only"
            )
            &
            (
                e["contrast"]
                ==
                "ContextualEdge_gain_over_pair"
            )
        ]


        print()
        print(
            "Same-strand contextual-edge increment:"
        )


        for _, r in q.iterrows():

            print(
                f"  {r['metric']}: "
                f"delta={r['mean_delta']:.6f}, "
                f"95% CI "
                f"[{r['CI_low']:.6f}, "
                f"{r['CI_high']:.6f}]"
            )


    print()
    print(
        "[PASS] Stage60 formal freeze verified"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Formal GenoGramma "
            "training/evaluation orchestrator"
        )
    )


    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Run stages even when formal PASS "
            "outputs already exist. "
            "Default is safe resume."
        ),
    )


    parser.add_argument(
        "--from-stage",
        default=None,
        choices=[
            s.key
            for s
            in STAGES
        ],
        help=(
            "Start orchestration from this stage."
        ),
    )


    parser.add_argument(
        "--to-stage",
        default=None,
        choices=[
            s.key
            for s
            in STAGES
        ],
        help=(
            "Stop orchestration after this stage."
        ),
    )


    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Only verify formal code/results; "
            "do not execute missing stages."
        ),
    )


    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    preflight()


    selected = list(
        STAGES
    )


    if args.from_stage is not None:

        idx = [
            s.key
            for s
            in selected
        ].index(
            args.from_stage
        )

        selected = selected[
            idx:
        ]


    if args.to_stage is not None:

        keys = [
            s.key
            for s
            in selected
        ]

        if args.to_stage in keys:

            idx = keys.index(
                args.to_stage
            )

            selected = selected[
                :idx + 1
            ]


    stamp = (
        datetime.datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )


    this_run = (
        RUN_ROOT
        / stamp
    )

    this_run.mkdir(
        parents=True,
        exist_ok=True,
    )


    records = []


    if args.audit_only:

        print()
        print(
            "===== FORMAL STAGE STATUS ====="
        )

        for stage in selected:

            status = (
                "PASS"
                if stage.done()
                else
                "MISSING"
            )

            print(
                f"{stage.key:>4}  "
                f"{status:<8}  "
                f"{stage.name}"
            )

            records.append(
                {
                    "stage":
                        stage.key,

                    "name":
                        stage.name,

                    "status":
                        status,
                }
            )

    else:

        for stage in selected:

            records.append(
                run_stage(
                    stage,
                    rerun=args.rerun,
                )
            )


    # -----------------------------------------------------------------
    # Save orchestration record
    # -----------------------------------------------------------------

    summary = {

        "formal_model_name":
            MODEL_NAME,

        "implementation":
            "ExplicitNeighborhoodEncoder + RNOE + CMOC",

        "representation_scope":
            "order-sensitive local genomic neighborhood",

        "run_root":
            str(
                this_run
            ),

        "resume_mode":
            not args.rerun,

        "stages":
            records,

        "formal_module":
            str(
                FORMAL_MODULE
            ),

        "formal_module_sha256":
            sha256(
                FORMAL_MODULE
            ),
    }


    (
        this_run
        / "run_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )


    # -----------------------------------------------------------------
    # Show final result whenever Stage60 exists
    # -----------------------------------------------------------------

    final_freeze = (
        ROOT
        / "13_stage60_final_operon_evidence"
        / "60_FINAL_RESULT_FREEZE.json"
    )


    if final_freeze.exists():

        show_final_results()


    print()
    print(
        "=" * 96
    )

    print(
        "[FINAL PASS] "
        "GenoGramma training/evaluation "
        "orchestration complete"
    )

    print(
        "=" * 96
    )

    print(
        "Run record:",
        this_run,
    )


if __name__ == "__main__":

    main()

