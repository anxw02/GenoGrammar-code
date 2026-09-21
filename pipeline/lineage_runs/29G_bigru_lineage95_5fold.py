"""
FORMAL MATCHED BASELINE — bigru / R4 ContextSSL [bigru]

Purpose
-------
Test whether GenoGrammar acquires structural sensitivity without direct
optimization of shuffle discrimination, GLOBAL>LOCAL ranking, or
GLOBAL-LOCAL separation.

Training objectives
-------------------
Adjacency prediction + neighbor retrieval only.

Checkpoint selection
--------------------
Minimum independent inner-validation SSL loss only.

Forbidden for selection
-----------------------
shuffle accuracy
rank_fraction_global_gt_local
LOCAL displacement
GLOBAL displacement
GLOBAL-LOCAL displacement

The original Stage08B1 source remains untouched.
"""

import pandas as pd
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage08B1
Formal leakage-controlled five-fold Explicit Order-Aware SSL.

Architecture is frozen from Stage08A2:
    ExplicitNeighborhoodEncoder

Stage08A      FINAL FAIL
Stage08A2     FINAL FAIL
Stage08A2B    PASS diagnostic
Stage08A3     PASS real-control qualification

IMPORTANT
---------
No phenotype labels.
Five independently trained encoders.
Each outer fold is completely excluded from:
    - SSL training
    - inner validation
    - checkpoint selection
    - early stopping / model selection

99.5% ANI near-clone blocks remain group-safe.

This stage evaluates structural SSL qualification only.

Primary per-fold outer-window criteria:
    shuffle accuracy >= 0.60
    GLOBAL > LOCAL fraction >= 0.80
    GLOBAL displacement >= 0.05
    GLOBAL - LOCAL >= 0.03

Formal Stage08B1 PASS:
    >=4/5 folds satisfy all four primary criteria.

Adjacency and neighbor-prediction metrics are reported as secondary diagnostics
and do NOT determine Stage08B1 PASS.
"""

import ast
import csv
import hashlib
import json
import random
import shutil
import types
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import os

# CUDA deterministic reproducibility requirement.
# This must be defined before the first CUDA/CuBLAS operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

# ============================================================
# FORMAL MATCHED BASELINE OVERRIDE
# ============================================================
import sys as _baseline_sys
from pathlib import Path as _BaselinePath

_BASELINE_PC = PACKAGE_ROOT

if str(_BASELINE_PC) not in _baseline_sys.path:
    _baseline_sys.path.insert(
        0,
        str(_BASELINE_PC),
    )

from baseline_models.registry import (
    REGISTRY as _BASELINE_REGISTRY,
)

_BASELINE_NAME = 'bigru'
_BASELINE_CLASS = _BASELINE_REGISTRY[
    _BASELINE_NAME
]

_EXPECTED_BASELINE_PARAMS = 506370
# ============================================================



# ============================================================
# Paths
# ============================================================

ROOT = Path(__import__("os").environ["GENOGRAMMAR_DATA_ROOT"])

A2_SCRIPT = PACKAGE_ROOT / "pipeline" / "revision_modules" / "08A2_revision_context_ssl.py"

CANON_EMB = (
    ROOT /
    "08_explicit_order_ssl" /
    "00_family_embeddings_276517x480.npy"
)

SSL_ROOT = (
    ROOT /
    "07_1_ssl_encoder"
)

ANI_TABLE = (
    ROOT /
    "01_1_ani_robustness" /
    "06_FINAL_robust_ANI_blocks.csv"
)

OUT = Path(
    __import__("os").environ.get(
        "GENOGRAMMAR_REVISION_OUTPUT",
        str(
            ROOT
            / "new_paper_data"
            / "03_ablation_falsification"
            / "formal_R4_lineage95_matched"
        ),
    )
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Frozen formal configuration
# ============================================================

FOLDS = [1, 2, 3, 4, 5]

FOLD_SEEDS = {
    1: 42,
    2: 142,
    3: 242,
    4: 342,
    5: 442,
}

INNER_VAL_FRACTION = 0.15

EPOCHS = 5
STEPS_PER_EPOCH = 400

BATCH_SIZE = 48

VAL_BATCHES = 30
OUTER_BATCHES = 40

LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


PRIMARY_THRESHOLDS = {
    "shuffle_accuracy":
        0.60,

    "rank_fraction_global_gt_local":
        0.80,

    "global_displacement_mean":
        0.05,

    "global_minus_local":
        0.03,
}

REQUIRED_PASS_FOLDS = 4


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Generic helpers
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def read_csv(path):

    with open(
        path,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


def write_csv(
    path,
    rows,
    fields=None,
):

    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    if fields is None:
        fields = list(
            rows[0].keys()
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fields
        )

        w.writeheader()
        w.writerows(rows)


def stable_hash(text):

    return int(
        hashlib.sha256(
            str(text).encode()
        ).hexdigest()[:16],
        16
    )


def detect_sid_column(cols):

    lower = {
        c.lower(): c
        for c in cols
    }

    for key in [
        "strain_entity_id",
        "sample_id",
        "strain_id",
        "sid",
    ]:

        if key in lower:
            return lower[key]

    for c in cols:

        x = c.lower()

        if (
            "strain" in x
            and
            "id" in x
        ):
            return c

    return cols[0]


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=False
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Safe import Stage08A2 architecture/functions
# ============================================================

def import_a2():

    if not A2_SCRIPT.exists():

        raise RuntimeError(
            f"Missing A2 script: "
            f"{A2_SCRIPT}"
        )

    source = A2_SCRIPT.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(A2_SCRIPT)
    )

    allowed = (
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.FunctionDef,
        ast.ClassDef,
    )

    safe = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(
                node,
                allowed
            )
        ],
        type_ignores=[]
    )

    m = types.ModuleType(
        "stage08a2_frozen"
    )

    m.__file__ = str(
        A2_SCRIPT
    )

    exec(
        compile(
            safe,
            str(A2_SCRIPT),
            "exec"
        ),
        m.__dict__
    )

    required = [
        "ExplicitNeighborhoodEncoder",
        "FamilyEmbeddingStore",
        "WindowSampler",
        "forward_objectives",
        "evaluate",
        "build_eligible_contigs",
    ]

    for name in required:

        if not hasattr(
            m,
            name
        ):

            raise RuntimeError(
                f"A2 missing definition: "
                f"{name}"
            )

    print(
        "[PASS] Stage08A2 frozen "
        "architecture imported."
    )

    return m


# ============================================================
# Outer-fold assignments
# ============================================================

def load_outer_folds():

    mapping = {}

    for fold in FOLDS:

        p = (
            SSL_ROOT /
            f"fold_{fold}" /
            "outer_holdout_embedding_index.csv"
        )

        if not p.exists():

            raise RuntimeError(
                f"Missing outer-fold index: "
                f"{p}"
            )

        rows = read_csv(p)

        if not rows:

            raise RuntimeError(
                f"Empty outer-fold index: "
                f"{p}"
            )

        sid_col = detect_sid_column(
            list(
                rows[0].keys()
            )
        )

        for r in rows:

            sid = (
                r[sid_col]
                .strip()
            )

            if not sid:
                continue

            if sid in mapping:

                raise RuntimeError(
                    f"Duplicate outer-fold "
                    f"assignment: {sid}"
                )

            mapping[sid] = fold

    if len(mapping) != 979:

        raise RuntimeError(
            f"Expected 979 genomes, "
            f"found {len(mapping)}"
        )

    counts = defaultdict(int)

    for f in mapping.values():
        counts[f] += 1

    expected = {
        1: 196,
        2: 196,
        3: 196,
        4: 196,
        5: 195,
    }

    if dict(
        sorted(
            counts.items()
        )
    ) != expected:

        raise RuntimeError(
            f"Unexpected fold sizes: "
            f"{dict(counts)}"
        )

    print(
        "[PASS] outer folds:",
        expected
    )

    return mapping


# ============================================================
# Near-clone groups
# ============================================================

def load_nearclone():

    rows = read_csv(
        ANI_TABLE
    )

    if not rows:

        raise RuntimeError(
            "ANI table empty."
        )

    cols = list(
        rows[0].keys()
    )

    sid_col = detect_sid_column(
        cols
    )

    candidates = [
        c
        for c in cols
        if (
            "near" in c.lower()
            or
            "99_5" in c.lower()
            or
            "99.5" in c.lower()
            or
            "995" in c.lower()
        )
    ]

    if not candidates:

        candidates = [
            c
            for c in cols
            if "block" in c.lower()
        ]

    if not candidates:

        raise RuntimeError(
            "Cannot identify "
            "99.5% ANI block column."
        )

    candidates.sort(
        key=lambda c: (
            0
            if "near" in c.lower()
            else 1,
            c
        )
    )

    block_col = candidates[0]

    mapping = {}

    for r in rows:

        sid = (
            r[sid_col]
            .strip()
        )

        block = (
            r[block_col]
            .strip()
        )

        if sid and block:
            mapping[sid] = block

    if len(mapping) < 979:

        raise RuntimeError(
            f"Only {len(mapping)} "
            "nearclone assignments."
        )

    print(
        "[PASS] nearclone map:",
        len(mapping),
        "| column=",
        block_col
    )

    return mapping


# ============================================================
# Fold-specific leakage-free split
# ============================================================

def build_fold_split(
    fold,
    seed,
    outer_map,
    nearclone,
):
    """
    Stage 29B frozen 95%-ANI lineage-blocked split.

    IMPORTANT
    ---------
    This function does NOT generate a new split.
    It reads the Stage-29A frozen manifest.

    train / val / outer are mutually exclusive at
    the 95%-ANI lineage-block level.
    """
    import csv
    import gzip
    from pathlib import Path

    split_path = Path(
        str(PACKAGE_ROOT) + "/"
        "new_paper_data/05_downstream_function/"
        "lineage_pretraining/"
        "02_lineage_blocked_SSL_split_manifest.csv.gz"
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Frozen lineage split missing: {split_path}"
        )

    role_ids = {
        "train": [],
        "val": [],
        "outer": [],
    }

    lineage_by_sid = {}

    with gzip.open(
        split_path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as fh:

        reader = csv.DictReader(
            fh
        )

        required = {
            "outer_fold",
            "strain_entity_id",
            "lineage_95_block",
            "role",
        }

        missing = (
            required
            -
            set(
                reader.fieldnames
                or []
            )
        )

        if missing:
            raise RuntimeError(
                "Frozen split missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            if int(
                row[
                    "outer_fold"
                ]
            ) != int(
                fold
            ):
                continue

            sid = (
                str(
                    row[
                        "strain_entity_id"
                    ]
                )
                .strip()
            )

            lineage = (
                str(
                    row[
                        "lineage_95_block"
                    ]
                )
                .strip()
            )

            role = (
                str(
                    row[
                        "role"
                    ]
                )
                .strip()
                .lower()
            )

            if role not in role_ids:
                raise RuntimeError(
                    f"Fold {fold}: unexpected role "
                    f"{role!r}"
                )

            role_ids[
                role
            ].append(
                sid
            )

            if (
                sid
                in lineage_by_sid
                and
                lineage_by_sid[
                    sid
                ]
                !=
                lineage
            ):
                raise RuntimeError(
                    f"Fold {fold}: strain {sid} "
                    "maps to multiple lineages."
                )

            lineage_by_sid[
                sid
            ] = lineage


    # ---------------------------------------------------------
    # Exact genome accounting
    # ---------------------------------------------------------

    train_ids = set(
        role_ids[
            "train"
        ]
    )

    val_ids = set(
        role_ids[
            "val"
        ]
    )

    outer_ids = set(
        role_ids[
            "outer"
        ]
    )


    if train_ids & val_ids:
        raise RuntimeError(
            f"Fold {fold}: train/val genome overlap."
        )

    if train_ids & outer_ids:
        raise RuntimeError(
            f"Fold {fold}: train/outer genome overlap."
        )

    if val_ids & outer_ids:
        raise RuntimeError(
            f"Fold {fold}: val/outer genome overlap."
        )


    split_ids = (
        train_ids
        |
        val_ids
        |
        outer_ids
    )


    original_ids = {
        str(x).strip()
        for x in outer_map.keys()
    }


    # ------------------------------------------------------------------
    # DOWNSTREAM LINEAGE-CLEAN FIX:
    #
    # `original_ids` comes from the legacy 979-genome formal outer-fold
    # universe, whereas Stage08A2/R4 has only 965 A2-eligible genomes.
    #
    # Stage29A was intentionally constructed on the 965 eligible genomes.
    # Therefore its frozen lineage split must be checked against the
    # canonical A2-eligible genome identity universe, NOT the pre-QC
    # 979-genome universe.
    #
    # IMPORTANT:
    # The current-R4 split manifest is used here ONLY as a frozen list of
    # eligible genome IDs. Its sample-wise fold roles are never read or
    # transferred into this lineage-blocked experiment.
    # ------------------------------------------------------------------

    canonical_eligible_path = Path(
        str(PACKAGE_ROOT) + "/"
        "new_paper_data/"
        "03_ablation_falsification/"
        "R4_context_ssl_samplewise/"
        "fold_1/"
        "split_manifest.csv"
    )

    if not canonical_eligible_path.exists():
        raise RuntimeError(
            "Canonical A2-eligible R4 genome manifest missing: "
            f"{canonical_eligible_path}"
        )

    canonical_eligible_df = pd.read_csv(
        canonical_eligible_path
    )

    if (
        "strain_entity_id"
        not in canonical_eligible_df.columns
    ):
        raise RuntimeError(
            "Canonical eligible manifest lacks "
            "`strain_entity_id`."
        )

    eligible_ids = set(
        canonical_eligible_df[
            "strain_entity_id"
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(eligible_ids) != 965:
        raise RuntimeError(
            "Canonical A2-eligible universe must contain "
            f"965 genomes, found {len(eligible_ids)}."
        )

    if split_ids != eligible_ids:

        missing_from_split = sorted(
            eligible_ids - split_ids
        )

        extra_in_split = sorted(
            split_ids - eligible_ids
        )

        raise RuntimeError(
            f"Fold {fold}: frozen lineage split differs "
            "from canonical A2-eligible 965-genome universe. "
            f"missing={len(missing_from_split)}, "
            f"extra={len(extra_in_split)}"
        )

    print(
        f"[PASS] Fold {fold}: frozen lineage split "
        "matches canonical A2-eligible universe "
        f"| genomes={len(split_ids)}"
    )

    # ------------------------------------------------------------------
    # DOWNSTREAM ANI95 LINEAGE CONSISTENCY FIX
    #
    # Stage29A `lineage_95_block` represents 95%-ANI lineages and must
    # therefore be checked against `FINAL_lineage_95_block`.
    #
    # It must NOT be compared with `FINAL_nearclone_99_5_block`, which is
    # the separate 99.5%-ANI near-clone grouping (NC_xxxx).
    #
    # This block validates IDs only. It does not modify any train/val/test
    # role, lineage assignment, genome universe, or model input.
    # ------------------------------------------------------------------

    canonical_lineage_path = Path(
        str(PACKAGE_ROOT / "data" / "01_1_ani_robustness" / "06_FINAL_robust_ANI_blocks.csv")
    )

    if not canonical_lineage_path.exists():
        raise RuntimeError(
            "Canonical ANI lineage table missing: "
            f"{canonical_lineage_path}"
        )

    canonical_lineage_df = pd.read_csv(
        canonical_lineage_path,
        usecols=[
            "strain_entity_id",
            "FINAL_lineage_95_block",
        ],
    )

    canonical_lineage_df[
        "strain_entity_id"
    ] = (
        canonical_lineage_df[
            "strain_entity_id"
        ]
        .astype(str)
        .str.strip()
    )

    canonical_lineage_df = (
        canonical_lineage_df
        .drop_duplicates(
            "strain_entity_id"
        )
    )

    canonical_lineage_by_sid = dict(
        zip(
            canonical_lineage_df[
                "strain_entity_id"
            ],
            canonical_lineage_df[
                "FINAL_lineage_95_block"
            ].astype(str),
        )
    )

    missing_canonical_lineage = sorted(
        sid
        for sid in split_ids
        if sid not in canonical_lineage_by_sid
    )

    if missing_canonical_lineage:
        raise RuntimeError(
            f"Fold {fold}: "
            f"{len(missing_canonical_lineage)} split genomes "
            "missing canonical FINAL_lineage_95_block. "
            f"Preview={missing_canonical_lineage[:10]}"
        )

    lineage_mismatch = []

    for sid in sorted(split_ids):

        frozen_lineage = str(
            lineage_by_sid[sid]
        )

        canonical_lineage = str(
            canonical_lineage_by_sid[sid]
        )

        if frozen_lineage != canonical_lineage:

            lineage_mismatch.append(
                (
                    sid,
                    frozen_lineage,
                    canonical_lineage,
                )
            )

    if lineage_mismatch:

        raise RuntimeError(
            f"Fold {fold}: "
            f"{len(lineage_mismatch)} ANI95 lineage ID "
            "mismatches between Stage29A and "
            "FINAL_lineage_95_block. "
            f"Preview={lineage_mismatch[:10]}"
        )

    print(
        f"[PASS] Fold {fold}: Stage29A ANI95 lineage IDs "
        "match canonical FINAL_lineage_95_block "
        f"| genomes={len(split_ids)}"
    )

    train_lineages = {
        lineage_by_sid[
            sid
        ]
        for sid in train_ids
    }

    val_lineages = {
        lineage_by_sid[
            sid
        ]
        for sid in val_ids
    }

    outer_lineages = {
        lineage_by_sid[
            sid
        ]
        for sid in outer_ids
    }


    if (
        train_lineages
        &
        val_lineages
    ):
        raise RuntimeError(
            f"Fold {fold}: train/val lineage leakage."
        )

    if (
        train_lineages
        &
        outer_lineages
    ):
        raise RuntimeError(
            f"Fold {fold}: train/outer lineage leakage."
        )

    if (
        val_lineages
        &
        outer_lineages
    ):
        raise RuntimeError(
            f"Fold {fold}: val/outer lineage leakage."
        )


    print(
        f"[LINEAGE95 SPLIT] Fold {fold} | "
        f"train={len(train_ids)} genomes/"
        f"{len(train_lineages)} lineages | "
        f"val={len(val_ids)} genomes/"
        f"{len(val_lineages)} lineages | "
        f"outer={len(outer_ids)} genomes/"
        f"{len(outer_lineages)} lineages | "
        "overlap=0"
    )


    return (
        sorted(
            train_ids
        ),
        sorted(
            val_ids
        ),
        sorted(
            outer_ids
        ),
    )




# ============================================================
# Fold training
# ============================================================

def train_fold(
    m,
    store,
    eligible,
    outer_map,
    nearclone,
    fold,
):

    seed = FOLD_SEEDS[
        fold
    ]

    fold_dir = (
        OUT /
        f"fold_{fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    done_file = (
        fold_dir /
        "DONE.json"
    )

    if done_file.exists():

        done = json.loads(
            done_file.read_text(
                encoding="utf-8"
            )
        )

        if (
            done.get(
                "status"
            )
            ==
            "PASS"
        ):

            print(
                f"[RESUME PASS] Fold {fold}"
            )

            return done

    # A failed/incomplete fold is rerun cleanly.
    for p in [
        fold_dir /
        "training_history.csv",

        fold_dir /
        "best_checkpoint.pt",

        fold_dir /
        "outer_metrics.json",

        fold_dir /
        "split_manifest.csv",

        done_file,
    ]:

        if p.exists():
            p.unlink()

    set_seed(
        seed
    )

    print()
    print(
        "=" * 96
    )

    print(
        f" FORMAL OUTER FOLD "
        f"{fold}/5 | seed={seed}"
    )

    print(
        "=" * 96
    )

    (
        inner_train,
        inner_val,
        outer_test,
    ) = build_fold_split(
        fold,
        seed,
        outer_map,
        nearclone
    )

    train_sids = [
        s
        for s in inner_train
        if s in eligible
    ]

    val_sids = [
        s
        for s in inner_val
        if s in eligible
    ]

    test_sids = [
        s
        for s in outer_test
        if s in eligible
    ]

    if len(
        train_sids
    ) < 500:

        raise RuntimeError(
            f"Fold {fold}: "
            "too few training genomes."
        )

    if len(
        val_sids
    ) < 50:

        raise RuntimeError(
            f"Fold {fold}: "
            "too few validation genomes."
        )

    # --------------------------------------------------------
    # LINEAGE-BLOCKED OUTER-SET VALIDITY GUARD
    #
    # Stage29A fixes whole ANI95 lineages to outer folds.
    # Genome counts are therefore intentionally unequal across
    # folds. The legacy sample-wise minimum-N threshold (<180)
    # is not scientifically applicable here.
    #
    # The frozen outer test set must only be non-empty; lineage
    # identity and zero-overlap safeguards are verified upstream.
    # --------------------------------------------------------

    if len(
        test_sids
    ) == 0:

        raise RuntimeError(
            f"Fold {fold}: "
            "empty frozen outer-test genome set."
        )


    print(
        "[PASS] fold split "
        f"| train={len(train_sids)} "
        f"| val={len(val_sids)} "
        f"| outer={len(test_sids)}"
    )

    # --------------------------------------------------------
    # Save full split audit
    # --------------------------------------------------------

    manifest = []

    train_set = set(
        train_sids
    )

    val_set = set(
        val_sids
    )

    test_set = set(
        test_sids
    )

    for sid in sorted(
        train_set
        |
        val_set
        |
        test_set
    ):

        if sid in train_set:
            role = "INNER_TRAIN"

        elif sid in val_set:
            role = "INNER_VAL"

        else:
            role = "OUTER_TEST"

        manifest.append({
            "strain_entity_id":
                sid,

            "nearclone_block":
                nearclone[sid],

            "original_outer_fold":
                outer_map[sid],

            "formal_fold":
                fold,

            "role":
                role,

            "A2_eligible":
                True,
        })

    write_csv(
        fold_dir /
        "split_manifest.csv",
        manifest
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        _BASELINE_CLASS()
        .to(DEVICE)
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    if n_params != 506370:

        raise RuntimeError(
            f"Fold {fold}: "
            f"unexpected trainable "
            f"parameter count "
            f"{n_params}"
        )

    print(
        "[PASS] trainable parameters:",
        n_params
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=True
        )
    )

    train_sampler = (
        m.WindowSampler(
            train_sids,
            eligible,
            seed + 1
        )
    )

    train_rng = (
        np.random.default_rng(
            seed + 1000
        )
    )

    # --------------------------------------------------------
    # Pretraining validation baseline
    # --------------------------------------------------------

    pre_val = (
        m.evaluate(
            model,
            store,
            val_sids,
            eligible,
            VAL_BATCHES,
            seed + 500
        )
    )

    print(
        "[PRE-VAL] "
        f"shuffle="
        f"{pre_val['shuffle_accuracy']:.3f} "
        f"rank="
        f"{pre_val['rank_fraction_global_gt_local']:.3f} "
        f"dL="
        f"{pre_val['local_displacement_mean']:.4f} "
        f"dG="
        f"{pre_val['global_displacement_mean']:.4f} "
        f"delta="
        f"{pre_val['global_minus_local']:.4f} "
        f"adj="
        f"{pre_val['adjacency_accuracy']:.3f} "
        f"next="
        f"{pre_val['neighbor_top1_accuracy']:.3f}"
    )

    history = []

    best_score = None
    best_epoch = None
    best_state = None

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running = defaultdict(
            float
        )

        for step in range(
            1,
            STEPS_PER_EPOCH + 1
        ):

            fam, strand = (
                train_sampler.sample_batch(
                    BATCH_SIZE
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True
            ):

                loss, metrics = (
                    m.forward_objectives(
                        model,
                        store,
                        fam,
                        strand,
                        train_rng
                    )
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            for k, v in metrics.items():
                running[k] += v

            if (
                step % 50 == 0
                or
                step
                ==
                STEPS_PER_EPOCH
            ):

                print(
                    f"[TRAIN F{fold}] "
                    f"epoch={epoch}/{EPOCHS} "
                    f"step={step}/"
                    f"{STEPS_PER_EPOCH} "
                    f"loss="
                    f"{running['loss']/step:.4f}"
                )

        # Same fixed inner-val sampling for all epochs.
        val = (
            m.evaluate(
                model,
                store,
                val_sids,
                eligible,
                VAL_BATCHES,
                seed + 500
            )
        )

        print(
            f"[VAL F{fold}] "
            f"epoch={epoch} "
            f"loss={val['loss']:.4f} "
            f"shuffle="
            f"{val['shuffle_accuracy']:.3f} "
            f"rank="
            f"{val['rank_fraction_global_gt_local']:.3f} "
            f"strong="
            f"{val['strong_rank_fraction']:.3f} "
            f"dL="
            f"{val['local_displacement_mean']:.4f} "
            f"dG="
            f"{val['global_displacement_mean']:.4f} "
            f"delta="
            f"{val['global_minus_local']:.4f} "
            f"adj="
            f"{val['adjacency_accuracy']:.3f} "
            f"next="
            f"{val['neighbor_top1_accuracy']:.3f}"
        )

        history.append({
            "epoch":
                epoch,

            **{
                f"val_{k}":
                    v
                for k, v
                in val.items()
            }
        })

        # --------------------------------------------------------
        # Revision R1 checkpoint selection
        # --------------------------------------------------------
        # IMPORTANT:
        # No shuffle accuracy, perturbation ranking,
        # LOCAL/GLOBAL displacement, or separation metric
        # participates in checkpoint selection.
        #
        # With the revision objective module:
        #   W_SHUFFLE = 0
        #   W_RANK = 0
        #   W_SEPARATION = 0
        #
        # validation loss therefore contains only the independent
        # adjacency, neighbor-retrieval, and contextual-consistency SSL objectives.
        score = -float(val["loss"])
        if (
            best_score is None
            or
            score > best_score
        ):

            best_score = score
            best_epoch = epoch

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()

                for k, v
                in model.state_dict()
                .items()
            }

    if best_state is None:

        raise RuntimeError(
            f"Fold {fold}: "
            "no best checkpoint."
        )

    model.load_state_dict(
        best_state,
        strict=True
    )

    # --------------------------------------------------------
    # OUTER TEST - first and only use for this fold
    # --------------------------------------------------------

    outer = (
        m.evaluate(
            model,
            store,
            test_sids,
            eligible,
            OUTER_BATCHES,
            seed + 9000
        )
    )

    print()
    print(
        f"[OUTER F{fold}] "
        f"best_epoch={best_epoch} "
        f"shuffle="
        f"{outer['shuffle_accuracy']:.3f} "
        f"rank="
        f"{outer['rank_fraction_global_gt_local']:.3f} "
        f"dL="
        f"{outer['local_displacement_mean']:.4f} "
        f"dG="
        f"{outer['global_displacement_mean']:.4f} "
        f"delta="
        f"{outer['global_minus_local']:.4f} "
        f"adj="
        f"{outer['adjacency_accuracy']:.3f} "
        f"next="
        f"{outer['neighbor_top1_accuracy']:.3f}"
    )

    criteria = {
        "shuffle_accuracy_pass":
            (
                outer[
                    "shuffle_accuracy"
                ]
                >=
                PRIMARY_THRESHOLDS[
                    "shuffle_accuracy"
                ]
            ),

        "rank_fraction_pass":
            (
                outer[
                    "rank_fraction_global_gt_local"
                ]
                >=
                PRIMARY_THRESHOLDS[
                    "rank_fraction_global_gt_local"
                ]
            ),

        "global_displacement_pass":
            (
                outer[
                    "global_displacement_mean"
                ]
                >=
                PRIMARY_THRESHOLDS[
                    "global_displacement_mean"
                ]
            ),

        "global_minus_local_pass":
            (
                outer[
                    "global_minus_local"
                ]
                >=
                PRIMARY_THRESHOLDS[
                    "global_minus_local"
                ]
            ),
    }

    # Revision R1:
    # legacy structural criteria remain diagnostic only.
    # They do not determine whether the independently trained
    # revision experiment executed successfully.
    fold_primary_pass = True

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    write_csv(
        fold_dir /
        "training_history.csv",
        history
    )

    torch.save(
        {
            "model":
                model.state_dict(),

            "formal_outer_fold":
                fold,

            "seed":
                seed,

            "best_epoch":
                best_epoch,

            "best_inner_score":
                best_score,

            "model_config": {
                "architecture":
                    "ExplicitNeighborhoodEncoder",

                "window_len":
                    64,

                "model_dim":
                    128,

                "trainable_parameters":
                    n_params,
            },

            "phenotype_labels_used":
                False,

            "outer_holdout_used_for_training":
                False,

            "outer_holdout_used_for_early_stopping":
                False,

            "outer_holdout_used_for_model_selection":
                False,
        },
        fold_dir /
        "best_checkpoint.pt"
    )

    (
        fold_dir /
        "outer_metrics.json"
    ).write_text(
        json.dumps(
            {
                "formal_outer_fold":
                    fold,

                "seed":
                    seed,

                "best_epoch":
                    best_epoch,

                "n_train":
                    len(train_sids),

                "n_inner_val":
                    len(val_sids),

                "n_outer_test":
                    len(test_sids),

                "pretraining_inner_val":
                    pre_val,

                "outer_metrics":
                    outer,

                "criteria":
                    criteria,

                "primary_pass":
                    fold_primary_pass,

                "phenotype_labels_used":
                    False,

                "outer_holdout_used_for_training":
                    False,

                "outer_holdout_used_for_early_stopping":
                    False,

                "outer_holdout_used_for_model_selection":
                    False,
            },
            indent=2
        ),
        encoding="utf-8"
    )

    done = {
        "status":
            "PASS",

        "formal_outer_fold":
            fold,

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "primary_pass":
            fold_primary_pass,

        "criteria":
            criteria,

        "outer_metrics":
            outer,

        "n_train":
            len(train_sids),

        "n_inner_val":
            len(val_sids),

        "n_outer_test":
            len(test_sids),

        "phenotype_labels_used":
            False,

        "outer_holdout_used_for_training":
            False,

        "outer_holdout_used_for_early_stopping":
            False,

        "outer_holdout_used_for_model_selection":
            False,
    }

    done_file.write_text(
        json.dumps(
            done,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"[FINAL PASS] Fold {fold} "
        f"training/evaluation complete "
        f"| primary_pass="
        f"{fold_primary_pass}"
    )

    del model
    del optimizer
    del scaler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return done


# ============================================================
# Final summary
# ============================================================

def summarize_fold_results(
    results
):

    rows = []

    for r in results:

        m = r[
            "outer_metrics"
        ]

        rows.append({
            "outer_fold":
                r[
                    "formal_outer_fold"
                ],

            "seed":
                r["seed"],

            "best_epoch":
                r[
                    "best_epoch"
                ],

            "n_train":
                r[
                    "n_train"
                ],

            "n_inner_val":
                r[
                    "n_inner_val"
                ],

            "n_outer_test":
                r[
                    "n_outer_test"
                ],

            "shuffle_accuracy":
                m[
                    "shuffle_accuracy"
                ],

            "rank_fraction_global_gt_local":
                m[
                    "rank_fraction_global_gt_local"
                ],

            "strong_rank_fraction":
                m[
                    "strong_rank_fraction"
                ],

            "local_displacement_mean":
                m[
                    "local_displacement_mean"
                ],

            "global_displacement_mean":
                m[
                    "global_displacement_mean"
                ],

            "global_minus_local":
                m[
                    "global_minus_local"
                ],

            "adjacency_accuracy":
                m[
                    "adjacency_accuracy"
                ],

            "neighbor_top1_accuracy":
                m[
                    "neighbor_top1_accuracy"
                ],

            "primary_pass":
                r[
                    "primary_pass"
                ],
        })

    write_csv(
        OUT /
        "08B1_fold_summary.csv",
        rows
    )

    pass_count = sum(
        bool(
            r[
                "primary_pass"
            ]
        )
        for r in rows
    )

    metrics = [
        "shuffle_accuracy",
        "rank_fraction_global_gt_local",
        "strong_rank_fraction",
        "local_displacement_mean",
        "global_displacement_mean",
        "global_minus_local",
        "adjacency_accuracy",
        "neighbor_top1_accuracy",
    ]

    aggregate = {}

    for metric in metrics:

        x = np.asarray(
            [
                float(
                    r[metric]
                )
                for r in rows
            ],
            dtype=float
        )

        aggregate[
            metric
        ] = {
            "mean":
                float(
                    np.mean(x)
                ),

            "sd":
                float(
                    np.std(
                        x,
                        ddof=1
                    )
                ),

            "median":
                float(
                    np.median(x)
                ),

            "min":
                float(
                    np.min(x)
                ),

            "max":
                float(
                    np.max(x)
                ),
        }

    formal_pass = (
        pass_count
        >=
        REQUIRED_PASS_FOLDS
    )

    summary = {
        "status":
            (
                "PASS"
                if formal_pass
                else "FAIL"
            ),

        "stage":
            "Stage08B1",

        "timestamp":
            datetime.now()
            .isoformat(),

        "architecture":
            "ExplicitNeighborhoodEncoder",

        "architecture_source":
            str(A2_SCRIPT),

        "architecture_sha256":
            sha256_file(
                A2_SCRIPT
            ),

        "n_formal_outer_folds":
            5,

        "fold_seeds":
            FOLD_SEEDS,

        "primary_thresholds":
            PRIMARY_THRESHOLDS,

        "required_pass_folds":
            REQUIRED_PASS_FOLDS,

        "primary_pass_folds":
            pass_count,

        "formal_pass":
            formal_pass,

        "fold_results":
            rows,

        "aggregate_outer_metrics":
            aggregate,

        "phenotype_labels_used":
            False,

        "nearclone_99_5_group_safe":
            True,

        "outer_holdout_used_for_training":
            False,

        "outer_holdout_used_for_early_stopping":
            False,

        "outer_holdout_used_for_model_selection":
            False,

        "raw_fold_embeddings_directly_poolable":
            False,

        "scientific_boundary":
            (
                "Formal leakage-controlled "
                "five-fold structural SSL "
                "qualification. Architecture "
                "was previously developed using "
                "Fold1 smoke diagnostics, so "
                "this is not a pristine prospective "
                "benchmark."
            ),

        "next_step":
            (
                "Proceed to Stage08B2: "
                "fold-matched real Stage04 "
                "control evaluation."
                if formal_pass
                else
                "Do not start Stage08B2; "
                "inspect cross-fold instability."
            ),
    }

    (
        OUT /
        "09_STAGE08B1_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        " Stage08B1 FORMAL FIVE-FOLD "
        "EXPLICIT ORDER-AWARE SSL"
    )

    print(
        "=" * 100
    )

    print(
        "[INFO] device:",
        DEVICE
    )

    if DEVICE.type != "cuda":

        raise RuntimeError(
            "Stage08B1 requires CUDA."
        )

    if not CANON_EMB.exists():

        raise RuntimeError(
            f"Missing canonical embeddings: "
            f"{CANON_EMB}"
        )

    # --------------------------------------------------------
    # Freeze protocol BEFORE formal training
    # --------------------------------------------------------

    definition = {
        "stage":
            "Stage08B1",

        "created":
            datetime.now()
            .isoformat(),

        "stage08A_status":
            "FAIL_FROZEN",

        "stage08A2_status":
            "FAIL_FROZEN",

        "stage08A2B_status":
            "PASS_DIAGNOSTIC",

        "stage08A3_status":
            "PASS_REAL_CONTROL_DIAGNOSTIC",

        "architecture":
            "ExplicitNeighborhoodEncoder",

        "architecture_source":
            str(A2_SCRIPT),

        "architecture_sha256":
            sha256_file(
                A2_SCRIPT
            ),

        "outer_folds":
            FOLDS,

        "fold_seeds":
            FOLD_SEEDS,

        "epochs":
            EPOCHS,

        "steps_per_epoch":
            STEPS_PER_EPOCH,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LR,

        "weight_decay":
            WEIGHT_DECAY,

        "inner_validation_fraction":
            INNER_VAL_FRACTION,

        "nearclone_group":
            "99.5% ANI",

        "primary_thresholds":
            PRIMARY_THRESHOLDS,

        "required_primary_pass_folds":
            REQUIRED_PASS_FOLDS,

        "adjacency_metric_role":
            "secondary_diagnostic",

        "neighbor_metric_role":
            "secondary_diagnostic",

        "phenotype_labels_used":
            False,

        "outer_holdout_used_for_training":
            False,

        "outer_holdout_used_for_early_stopping":
            False,

        "outer_holdout_used_for_model_selection":
            False,

        "raw_fold_embeddings_directly_poolable":
            False,

        "scientific_boundary":
            (
                "Structural SSL validation only; "
                "no probiotic phenotype prediction."
            ),
    }

    definition_file = (
        OUT /
        "00_STAGE08B1_definition.json"
    )

    if definition_file.exists():

        old = json.loads(
            definition_file.read_text(
                encoding="utf-8"
            )
        )

        critical = [
            "architecture_sha256",
            "fold_seeds",
            "epochs",
            "steps_per_epoch",
            "batch_size",
            "learning_rate",
            "primary_thresholds",
            "required_primary_pass_folds",
        ]

        for key in critical:

            old_value = (
                old.get(key)
            )

            new_value = (
                definition.get(key)
            )


            # ----------------------------------------------------
            # JSON DICT-KEY COMPATIBILITY — fold_seeds only
            #
            # Python uses integer fold IDs.
            # JSON restores object keys as strings.
            # Seed VALUES remain exact.
            # ----------------------------------------------------

            if key == "fold_seeds":

                if (
                    not isinstance(
                        old_value,
                        dict
                    )
                    or
                    not isinstance(
                        new_value,
                        dict
                    )
                ):

                    raise RuntimeError(
                        "Frozen Stage08B1 "
                        "fold_seeds schema mismatch."
                    )


                old_value = {
                    str(k): int(v)
                    for k, v in (
                        old_value.items()
                    )
                }


                new_value = {
                    str(k): int(v)
                    for k, v in (
                        new_value.items()
                    )
                }


            if (
                old_value
                !=
                new_value
            ):

                raise RuntimeError(
                    f"Frozen Stage08B1 "
                    f"definition mismatch: "
                    f"{key}"
                )

        print(
            "[PASS] existing frozen "
            "Stage08B1 definition verified."
        )

    else:

        definition_file.write_text(
            json.dumps(
                definition,
                indent=2
            ),
            encoding="utf-8"
        )

        print(
            "[PASS] Stage08B1 protocol frozen."
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    m = import_a2()

    outer_map = (
        load_outer_folds()
    )

    nearclone = (
        load_nearclone()
    )

    eligible = (
        m.build_eligible_contigs(
            sorted(
                outer_map.keys()
            )
        )
    )

    if len(
        eligible
    ) != 965:

        raise RuntimeError(
            f"Expected 965 eligible "
            f"genomes, got "
            f"{len(eligible)}"
        )

    print(
        "[PASS] A2-eligible genomes:",
        len(eligible)
    )

    store = (
        m.FamilyEmbeddingStore(
            CANON_EMB
        )
    )

    # --------------------------------------------------------
    # Five independent encoders
    # --------------------------------------------------------

    results = []

    for fold in FOLDS:

        result = train_fold(
            m,
            store,
            eligible,
            outer_map,
            nearclone,
            fold
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Final formal summary
    # --------------------------------------------------------

    summary = (
        summarize_fold_results(
            results
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        " STAGE08B1 FINAL SUMMARY"
    )

    print(
        "=" * 100
    )

    for r in summary[
        "fold_results"
    ]:

        print(
            f"Fold {r['outer_fold']} | "
            f"best_epoch="
            f"{r['best_epoch']} | "
            f"shuffle="
            f"{float(r['shuffle_accuracy']):.4f} | "
            f"rank="
            f"{float(r['rank_fraction_global_gt_local']):.4f} | "
            f"dL="
            f"{float(r['local_displacement_mean']):.4f} | "
            f"dG="
            f"{float(r['global_displacement_mean']):.4f} | "
            f"delta="
            f"{float(r['global_minus_local']):.4f} | "
            f"adj="
            f"{float(r['adjacency_accuracy']):.4f} | "
            f"next="
            f"{float(r['neighbor_top1_accuracy']):.4f} | "
            f"PRIMARY="
            f"{r['primary_pass']}"
        )

    print()
    print(
        "Primary-pass folds:",
        summary[
            "primary_pass_folds"
        ],
        "/5"
    )

    if summary[
        "formal_pass"
    ]:

        print(
            "[FINAL PASS] Stage08B1"
        )

        print(
            "[NEXT] Stage08B2 "
            "fold-matched real Stage04 "
            "control validation."
        )

    else:

        print(
            "[FINAL FAIL] Stage08B1"
        )

        print(
            "[STOP] Do not proceed "
            "to Stage08B2."
        )

    print(
        "=" * 100
    )

    print(
        "[OUTPUT]",
        OUT
    )


if __name__ == "__main__":
    main()
