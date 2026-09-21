#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage10A
95% ANI LINEAGE-BLOCKED five-fold Explicit Order-Aware SSL robustness test.

PURPOSE
-------
Test whether explicit genome-order learning generalizes beyond closely related
genomes when entire 95% ANI lineages are excluded from training.

IMPORTANT
---------
- Same frozen architecture as Stage08B1.
- Same SSL objectives.
- Same training hyperparameters.
- Same pre-registered primary thresholds.
- NO phenotype labels.
- Models are trained FROM SCRATCH.
- Entire 95% ANI lineage blocks remain inside one outer fold.
- Inner train/validation split is ALSO lineage-blocked.
- Outer holdout is never used for training, early stopping, or model selection.

This is a harder phylogenetic robustness test than the 99.5% near-clone-safe B1.

Primary per-fold criteria inherited unchanged from Stage08B1:
    shuffle accuracy >= 0.60
    GLOBAL > LOCAL rank fraction >= 0.80
    GLOBAL displacement >= 0.05
    GLOBAL - LOCAL >= 0.03

Formal Stage10A PASS:
    >=4/5 outer lineage-blocked folds satisfy all criteria.

Adjacency and neighbor metrics remain secondary diagnostics.
"""

import ast
import csv
import hashlib
import json
import random
import types
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Paths
# ============================================================

ROOT = Path(__import__("os").environ["GENOGRAMMAR_DATA_ROOT"])

A2_SCRIPT = (
    ROOT /
    "08A2_explicit_neighborhood_ssl_smoke.py"
)

CANON_EMB = (
    ROOT /
    "08_explicit_order_ssl" /
    "00_family_embeddings_276517x480.npy"
)

ANI_TABLE = (
    ROOT /
    "01_1_ani_robustness" /
    "06_FINAL_robust_ANI_blocks.csv"
)

B1_ROOT = (
    ROOT /
    "08B_formal_5fold_order_ssl"
)

STAGE09_SUMMARY = (
    ROOT /
    "09_content_only_baseline" /
    "06_STAGE09_summary.json"
)

OUT = (
    ROOT /
    "10A_lineage95_order_ssl"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Frozen configuration
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

        low = c.lower()

        if (
            "strain" in low
            and
            "id" in low
        ):
            return c

    return cols[0]


def detect_lineage95_column(cols):

    exact = [
        "FINAL_lineage_95_block",
        "FINAL_lineage95_block",
        "lineage_95_block",
        "lineage95_block",
        "ANI95_cluster",
        "ani95_cluster",
    ]

    for x in exact:
        if x in cols:
            return x

    candidates = []

    for c in cols:

        low = c.lower()

        if (
            "95" in low
            and
            "99" not in low
            and
            (
                "lineage" in low
                or
                "cluster" in low
                or
                "block" in low
            )
        ):

            candidates.append(c)

    if not candidates:

        raise RuntimeError(
            "Cannot detect the 95% ANI "
            "lineage/block column.\n"
            f"Available columns: {cols}"
        )

    candidates.sort(
        key=lambda c: (
            0
            if "lineage" in c.lower()
            else 1,
            c
        )
    )

    return candidates[0]


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
# Safety gate: Stage09
# ============================================================

def verify_stage09():

    if not STAGE09_SUMMARY.exists():

        raise RuntimeError(
            "Stage09 summary missing."
        )

    x = json.loads(
        STAGE09_SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    if not x.get(
        "all_acceptance_criteria_pass",
        False
    ):

        raise RuntimeError(
            "Stage09 must formally PASS."
        )

    print(
        "[PASS] Stage09 content-only "
        "baseline formally passed."
    )


# ============================================================
# Safe import A2
# ============================================================

def import_a2():

    if not A2_SCRIPT.exists():

        raise RuntimeError(
            f"Missing A2 script: {A2_SCRIPT}"
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
                f"A2 missing definition: {name}"
            )

    print(
        "[PASS] frozen explicit-order "
        "architecture imported."
    )

    return m


# ============================================================
# Canonical 965-genome cohort
# ============================================================

def load_b1_cohort():

    sids = set()

    for fold in FOLDS:

        p = (
            B1_ROOT /
            f"fold_{fold}" /
            "split_manifest.csv"
        )

        if not p.exists():

            raise RuntimeError(
                f"Missing B1 split manifest: {p}"
            )

        rows = read_csv(p)

        for r in rows:

            if (
                r["role"].strip()
                ==
                "OUTER_TEST"
            ):

                sids.add(
                    r[
                        "strain_entity_id"
                    ].strip()
                )

    if len(sids) != 965:

        raise RuntimeError(
            f"Expected canonical cohort "
            f"N=965, got N={len(sids)}"
        )

    print(
        "[PASS] canonical Stage08/09 cohort:",
        len(sids)
    )

    return sorted(sids)


# ============================================================
# 95% ANI lineage map
# ============================================================

def load_lineage95(
    canonical_sids
):

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

    lineage_col = detect_lineage95_column(
        cols
    )

    wanted = set(
        canonical_sids
    )

    mapping = {}

    for r in rows:

        sid = (
            r[
                sid_col
            ].strip()
        )

        if sid not in wanted:
            continue

        lineage = (
            r[
                lineage_col
            ].strip()
        )

        if not lineage:

            raise RuntimeError(
                f"Missing 95% lineage: {sid}"
            )

        mapping[sid] = lineage

    missing = (
        wanted
        -
        set(mapping)
    )

    if missing:

        raise RuntimeError(
            f"Missing lineage assignments "
            f"for {len(missing)} genomes."
        )

    groups = defaultdict(list)

    for sid, lineage in mapping.items():

        groups[
            lineage
        ].append(sid)

    sizes = sorted(
        [
            len(v)
            for v in groups.values()
        ],
        reverse=True
    )

    print(
        "[PASS] 95% ANI lineage map:"
    )

    print(
        "       column       =",
        lineage_col
    )

    print(
        "       genomes      =",
        len(mapping)
    )

    print(
        "       lineages     =",
        len(groups)
    )

    print(
        "       largest      =",
        sizes[0]
    )

    print(
        "       median size  =",
        float(
            np.median(sizes)
        )
    )

    return (
        mapping,
        groups,
        lineage_col,
    )


# ============================================================
# Deterministic lineage-blocked outer folds
# ============================================================

def build_outer_folds(
    groups
):

    items = list(
        groups.items()
    )

    # Largest blocks first, stable hash for tie breaking.
    items.sort(
        key=lambda kv: (
            -len(
                kv[1]
            ),
            stable_hash(
                "OUTER95::"
                +
                str(
                    kv[0]
                )
            ),
        )
    )

    fold_groups = {
        f: []
        for f in FOLDS
    }

    fold_n = {
        f: 0
        for f in FOLDS
    }

    for lineage, members in items:

        target = min(
            FOLDS,
            key=lambda f: (
                fold_n[f],
                len(
                    fold_groups[f]
                ),
                f,
            )
        )

        fold_groups[
            target
        ].append(
            lineage
        )

        fold_n[
            target
        ] += len(
            members
        )

    outer_map = {}

    audit_rows = []

    for fold in FOLDS:

        for lineage in fold_groups[
            fold
        ]:

            members = sorted(
                groups[
                    lineage
                ]
            )

            audit_rows.append({
                "outer_fold":
                    fold,

                "lineage_95_block":
                    lineage,

                "n_genomes":
                    len(
                        members
                    ),
            })

            for sid in members:

                if sid in outer_map:

                    raise RuntimeError(
                        f"Duplicate genome: {sid}"
                    )

                outer_map[
                    sid
                ] = fold

    if len(
        outer_map
    ) != 965:

        raise RuntimeError(
            f"Expected 965 assignments, "
            f"got {len(outer_map)}"
        )

    write_csv(
        OUT /
        "01_lineage95_outer_fold_blocks.csv",
        audit_rows
    )

    print(
        "[PASS] deterministic "
        "lineage-blocked outer folds:"
    )

    for fold in FOLDS:

        n = sum(
            1
            for f in outer_map.values()
            if f == fold
        )

        print(
            f"       Fold {fold}: "
            f"N={n}, "
            f"lineages="
            f"{len(fold_groups[fold])}"
        )

    return (
        outer_map,
        fold_groups,
    )


# ============================================================
# Lineage-blocked inner split
# ============================================================

def inner_split(
    fold,
    seed,
    outer_map,
    lineage_map,
):

    outer_test = sorted(
        sid
        for sid, f
        in outer_map.items()
        if f == fold
    )

    remaining = sorted(
        sid
        for sid, f
        in outer_map.items()
        if f != fold
    )

    test_lineages = {
        lineage_map[s]
        for s in outer_test
    }

    rem_lineages = defaultdict(
        list
    )

    for sid in remaining:

        lineage = lineage_map[
            sid
        ]

        if lineage in test_lineages:

            raise RuntimeError(
                f"Outer lineage leakage: "
                f"{lineage}"
            )

        rem_lineages[
            lineage
        ].append(
            sid
        )

    # Deterministic lineage ordering.
    lineage_order = sorted(
        rem_lineages,
        key=lambda x:
            stable_hash(
                f"INNER95::{fold}::{seed}::{x}"
            )
    )

    target_val = max(
        1,
        int(
            round(
                len(remaining)
                *
                INNER_VAL_FRACTION
            )
        )
    )

    val_lineages = set()
    val_n = 0

    for lineage in lineage_order:

        if (
            val_n
            >=
            target_val
            and
            len(
                val_lineages
            ) >= 2
        ):
            break

        val_lineages.add(
            lineage
        )

        val_n += len(
            rem_lineages[
                lineage
            ]
        )

    inner_val = sorted(
        sid
        for lineage in val_lineages
        for sid in rem_lineages[
            lineage
        ]
    )

    inner_train = sorted(
        sid
        for lineage, members
        in rem_lineages.items()
        if lineage not in val_lineages
        for sid in members
    )

    tr_lin = {
        lineage_map[s]
        for s in inner_train
    }

    va_lin = {
        lineage_map[s]
        for s in inner_val
    }

    te_lin = {
        lineage_map[s]
        for s in outer_test
    }

    if tr_lin & va_lin:
        raise RuntimeError(
            "Inner train/val lineage leakage."
        )

    if tr_lin & te_lin:
        raise RuntimeError(
            "Train/test lineage leakage."
        )

    if va_lin & te_lin:
        raise RuntimeError(
            "Val/test lineage leakage."
        )

    return (
        inner_train,
        inner_val,
        outer_test,
    )


# ============================================================
# Train one 95%-blocked fold
# ============================================================

def train_fold(
    m,
    store,
    eligible,
    outer_map,
    lineage_map,
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

        old = json.loads(
            done_file.read_text(
                encoding="utf-8"
            )
        )

        if old.get(
            "status"
        ) == "PASS":

            print(
                f"[RESUME PASS] Fold {fold}"
            )

            return old

    set_seed(
        seed
    )

    print()
    print(
        "=" * 100
    )

    print(
        f" LINEAGE95 OUTER FOLD "
        f"{fold}/5 | seed={seed}"
    )

    print(
        "=" * 100
    )

    (
        inner_train,
        inner_val,
        outer_test,
    ) = inner_split(
        fold,
        seed,
        outer_map,
        lineage_map,
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

    if not train_sids:
        raise RuntimeError(
            f"Fold {fold}: empty train."
        )

    if not val_sids:
        raise RuntimeError(
            f"Fold {fold}: empty val."
        )

    if not test_sids:
        raise RuntimeError(
            f"Fold {fold}: empty test."
        )

    print(
        "[PASS] lineage-safe split "
        f"| train={len(train_sids)} "
        f"| val={len(val_sids)} "
        f"| outer={len(test_sids)}"
    )

    print(
        "[PASS] lineage counts "
        f"| train="
        f"{len({lineage_map[s] for s in train_sids})} "
        f"| val="
        f"{len({lineage_map[s] for s in val_sids})} "
        f"| outer="
        f"{len({lineage_map[s] for s in test_sids})}"
    )

    # --------------------------------------------------------
    # Split audit
    # --------------------------------------------------------

    manifest = []

    for role, sids in [
        (
            "INNER_TRAIN",
            train_sids
        ),
        (
            "INNER_VAL",
            val_sids
        ),
        (
            "OUTER_TEST",
            test_sids
        ),
    ]:

        for sid in sids:

            manifest.append({
                "strain_entity_id":
                    sid,

                "lineage_95_block":
                    lineage_map[
                        sid
                    ],

                "formal_fold":
                    fold,

                "role":
                    role,
            })

    write_csv(
        fold_dir /
        "split_manifest.csv",
        manifest
    )

    # --------------------------------------------------------
    # Model from scratch
    # --------------------------------------------------------

    model = (
        m.ExplicitNeighborhoodEncoder()
        .to(DEVICE)
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    if n_params != 508182:

        raise RuntimeError(
            f"Unexpected parameter count: "
            f"{n_params}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True
    )

    train_sampler = m.WindowSampler(
        train_sids,
        eligible,
        seed + 1,
    )

    train_rng = np.random.default_rng(
        seed + 1000
    )

    pre_val = m.evaluate(
        model,
        store,
        val_sids,
        eligible,
        VAL_BATCHES,
        seed + 500,
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
                        train_rng,
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
                GRAD_CLIP,
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

        val = m.evaluate(
            model,
            store,
            val_sids,
            eligible,
            VAL_BATCHES,
            seed + 500,
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
            },
        })

        # EXACT same selection rule as Stage08B1.
        score = (
            2.0
            *
            val[
                "shuffle_accuracy"
            ]

            +
            2.0
            *
            val[
                "rank_fraction_global_gt_local"
            ]

            +
            5.0
            *
            max(
                0.0,
                val[
                    "global_minus_local"
                ]
            )

            +
            val[
                "adjacency_accuracy"
            ]

            +
            0.25
            *
            val[
                "neighbor_top1_accuracy"
            ]
        )

        if (
            best_score is None
            or
            score > best_score
        ):

            best_score = float(
                score
            )

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
            "No best checkpoint."
        )

    model.load_state_dict(
        best_state,
        strict=True
    )

    # ========================================================
    # OUTER LINEAGE TEST
    # ========================================================

    outer = m.evaluate(
        model,
        store,
        test_sids,
        eligible,
        OUTER_BATCHES,
        seed + 9000,
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

    primary_pass = all(
        criteria.values()
    )

    print()
    print(
        f"[OUTER95 F{fold}] "
        f"best_epoch={best_epoch} "
        f"shuffle="
        f"{outer['shuffle_accuracy']:.4f} "
        f"rank="
        f"{outer['rank_fraction_global_gt_local']:.4f} "
        f"dL="
        f"{outer['local_displacement_mean']:.4f} "
        f"dG="
        f"{outer['global_displacement_mean']:.4f} "
        f"delta="
        f"{outer['global_minus_local']:.4f} "
        f"adj="
        f"{outer['adjacency_accuracy']:.4f} "
        f"next="
        f"{outer['neighbor_top1_accuracy']:.4f} "
        f"PRIMARY={primary_pass}"
    )

    write_csv(
        fold_dir /
        "training_history.csv",
        history,
    )

    torch.save(
        {
            "model":
                model.state_dict(),

            "stage":
                "Stage10A",

            "outer_fold":
                fold,

            "split_unit":
                "95% ANI lineage",

            "seed":
                seed,

            "best_epoch":
                best_epoch,

            "best_inner_score":
                best_score,

            "trainable_parameters":
                n_params,

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
        "best_checkpoint.pt",
    )

    done = {
        "status":
            "PASS",

        "stage":
            "Stage10A",

        "outer_fold":
            fold,

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "n_train":
            len(
                train_sids
            ),

        "n_inner_val":
            len(
                val_sids
            ),

        "n_outer_test":
            len(
                test_sids
            ),

        "n_train_lineages":
            len({
                lineage_map[s]
                for s in train_sids
            }),

        "n_inner_val_lineages":
            len({
                lineage_map[s]
                for s in val_sids
            }),

        "n_outer_test_lineages":
            len({
                lineage_map[s]
                for s in test_sids
            }),

        "outer_metrics":
            outer,

        "criteria":
            criteria,

        "primary_pass":
            primary_pass,

        "lineage_95_group_safe":
            True,

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
        encoding="utf-8",
    )

    print(
        f"[FINAL PASS] "
        f"Fold {fold} computation "
        f"complete | "
        f"primary_pass={primary_pass}"
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

def summarize(
    results,
    lineage_col,
):

    rows = []

    for r in results:

        m = r[
            "outer_metrics"
        ]

        rows.append({
            "outer_fold":
                r[
                    "outer_fold"
                ],

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

            "n_train_lineages":
                r[
                    "n_train_lineages"
                ],

            "n_inner_val_lineages":
                r[
                    "n_inner_val_lineages"
                ],

            "n_outer_test_lineages":
                r[
                    "n_outer_test_lineages"
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
        "08_fold_summary.csv",
        rows,
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
                    r[
                        metric
                    ]
                )

                for r in rows
            ],
            dtype=float,
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

    pass_count = sum(
        bool(
            r[
                "primary_pass"
            ]
        )

        for r in rows
    )

    formal_pass = (
        pass_count
        >=
        REQUIRED_PASS_FOLDS
    )

    result = {
        "status":
            (
                "PASS"
                if formal_pass
                else "FAIL"
            ),

        "stage":
            "Stage10A",

        "timestamp":
            datetime.now()
            .isoformat(),

        "evaluation":
            (
                "95% ANI lineage-blocked "
                "five-fold explicit-order SSL"
            ),

        "lineage_column":
            lineage_col,

        "n_genomes":
            965,

        "n_folds":
            5,

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

        "lineage_95_group_safe":
            True,

        "inner_split_lineage_95_group_safe":
            True,

        "models_trained_from_scratch":
            True,

        "phenotype_labels_used":
            False,

        "raw_fold_embeddings_directly_poolable":
            False,

        "scientific_boundary":
            (
                "Lineage-level structural "
                "representation robustness; "
                "not phenotype prediction."
            ),

        "next_step":
            (
                "Proceed to Stage10B: "
                "95%-lineage fold-matched "
                "real Stage04 controls."
                if formal_pass
                else
                "Do not claim lineage-level "
                "generalization; inspect failed folds."
            ),
    }

    (
        OUT /
        "09_STAGE10A_summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8",
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        " Stage10A 95% ANI LINEAGE-BLOCKED "
        "EXPLICIT ORDER SSL"
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
            "CUDA required."
        )

    verify_stage09()

    m = import_a2()

    canonical = load_b1_cohort()

    (
        lineage_map,
        lineage_groups,
        lineage_col,
    ) = load_lineage95(
        canonical
    )

    (
        outer_map,
        outer_groups,
    ) = build_outer_folds(
        lineage_groups
    )

    # --------------------------------------------------------
    # Verify A2 eligibility
    # --------------------------------------------------------

    eligible = (
        m.build_eligible_contigs(
            canonical
        )
    )

    missing_eligible = (
        set(canonical)
        -
        set(eligible)
    )

    if missing_eligible:

        raise RuntimeError(
            f"Canonical 965 cohort should "
            f"all be A2-eligible, but "
            f"{len(missing_eligible)} are missing."
        )

    if len(
        eligible
    ) != 965:

        raise RuntimeError(
            f"Expected 965 eligible genomes, "
            f"got {len(eligible)}"
        )

    print(
        "[PASS] A2-eligible genomes:",
        len(eligible)
    )

    # --------------------------------------------------------
    # Freeze protocol before results
    # --------------------------------------------------------

    definition = {
        "stage":
            "Stage10A",

        "created":
            datetime.now()
            .isoformat(),

        "purpose":
            (
                "Hard phylogenetic robustness "
                "test of explicit genome-order "
                "representation."
            ),

        "architecture":
            "ExplicitNeighborhoodEncoder",

        "architecture_source":
            str(
                A2_SCRIPT
            ),

        "architecture_sha256":
            sha256_file(
                A2_SCRIPT
            ),

        "n_genomes":
            965,

        "outer_group":
            "95% ANI lineage",

        "lineage_column":
            lineage_col,

        "outer_fold_assignment":
            (
                "Deterministic greedy "
                "lineage-block balancing."
            ),

        "inner_validation_group":
            "95% ANI lineage",

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

        "primary_thresholds":
            PRIMARY_THRESHOLDS,

        "required_pass_folds":
            REQUIRED_PASS_FOLDS,

        "phenotype_labels_used":
            False,

        "architecture_retuned":
            False,

        "thresholds_retuned":
            False,

        "models_trained_from_scratch":
            True,
    }

    def_path = (
        OUT /
        "00_STAGE10A_definition.json"
    )

    if def_path.exists():

        old = json.loads(
            def_path.read_text(
                encoding="utf-8"
            )
        )

        critical = [
            "architecture_sha256",
            "outer_group",
            "fold_seeds",
            "epochs",
            "steps_per_epoch",
            "batch_size",
            "learning_rate",
            "primary_thresholds",
            "required_pass_folds",
        ]

        for key in critical:

            if (
                old.get(
                    key
                )
                !=
                definition.get(
                    key
                )
            ):

                raise RuntimeError(
                    f"Frozen Stage10A "
                    f"definition mismatch: "
                    f"{key}"
                )

        print(
            "[PASS] existing Stage10A "
            "protocol verified."
        )

    else:

        def_path.write_text(
            json.dumps(
                definition,
                indent=2
            ),
            encoding="utf-8",
        )

        print(
            "[PASS] Stage10A protocol frozen."
        )

    store = m.FamilyEmbeddingStore(
        CANON_EMB
    )

    results = []

    for fold in FOLDS:

        result = train_fold(
            m,
            store,
            eligible,
            outer_map,
            lineage_map,
            fold,
        )

        results.append(
            result
        )

    summary = summarize(
        results,
        lineage_col,
    )

    print()
    print(
        "=" * 100
    )

    print(
        " STAGE10A FINAL SUMMARY"
    )

    print(
        "=" * 100
    )

    for r in summary[
        "fold_results"
    ]:

        print(
            f"Fold {r['outer_fold']} | "
            f"N={r['n_outer_test']} | "
            f"lineages="
            f"{r['n_outer_test_lineages']} | "
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
            f"PASS="
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

    print()

    if summary[
        "formal_pass"
    ]:

        print(
            "[FINAL PASS] Stage10A"
        )

        print(
            "[NEXT] Stage10B "
            "95%-lineage fold-matched "
            "real Stage04 controls."
        )

    else:

        print(
            "[FINAL FAIL] Stage10A"
        )

        print(
            "[STOP] Do not claim "
            "95% lineage-level generalization."
        )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
