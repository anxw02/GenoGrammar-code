from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import re

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE = Path("/root/autodl-tmp/axw")
PC = BASE / "paper_code"
NEW = BASE / "new_paper_data"

STRICT_ROOT = (
    NEW
    / "05_downstream_function"
    / "masked_center_cog"
)

STRICT_MANIFEST = (
    STRICT_ROOT
    / "09_masked_center_COG_manifest_STRICT.csv.gz"
)

STRICT_FREEZE = (
    STRICT_ROOT
    / "12_STRICT_dataset_freeze.json"
)

STRAIN_LINEAGE = (
    STRICT_ROOT
    / "02_strain_lineage95_map.csv"
)

OUT = (
    NEW
    / "05_downstream_function"
    / "lineage_pretraining"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

R4_PIPELINE = (
    PC
    / "pipeline"
    / "21_revision_R4_context_ssl.py"
)

R4_ROOT = (
    NEW
    / "03_ablation_falsification"
    / "R4_context_ssl_samplewise"
)

BASELINE_ROOT = (
    NEW
    / "02_strong_baselines"
    / "formal_R4_matched"
)

BASELINE_PIPE_ROOT = (
    PC
    / "pipeline"
    / "baseline_runs"
)


SEED = 42
N_FOLDS = 5

# Match the approximate inner-validation fraction used previously.
TARGET_VAL_FRACTION = 0.125


# =============================================================================
# HELPERS
# =============================================================================

def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def stable_hash(text):

    s = (
        f"{SEED}|{text}"
        .encode("utf-8")
    )

    return hashlib.sha256(
        s
    ).hexdigest()


def find_column(columns, candidates):

    low = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:

        if candidate.lower() in low:
            return low[
                candidate.lower()
            ]

    return None


print("=" * 120)
print("STAGE 29A — STRICT 95%-ANI PRETRAINING SPLIT FREEZE")
print("=" * 120)


# =============================================================================
# 0. INPUT GATES
# =============================================================================

for path in [
    STRICT_MANIFEST,
    STRICT_FREEZE,
    STRAIN_LINEAGE,
    R4_PIPELINE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input: {path}"
        )

    print(
        "[PASS] input:",
        path,
    )


freeze = json.loads(
    STRICT_FREEZE.read_text()
)

if freeze.get(
    "status"
) != "FROZEN":

    raise RuntimeError(
        "Strict downstream dataset is not frozen."
    )


print(
    "[PASS] strict dataset status = FROZEN"
)


# =============================================================================
# 1. LOAD STRICT DOWNSTREAM FOLDS
# =============================================================================

print()
print("=" * 120)
print("1. DOWNSTREAM LINEAGE FOLD MAP")
print("=" * 120)

strict = pd.read_csv(
    STRICT_MANIFEST,
    compression="gzip",
    low_memory=False,
)

strict[
    "lineage_95_block"
] = (
    strict[
        "lineage_95_block"
    ]
    .astype(str)
    .str.strip()
)

strict[
    "strain_entity_id"
] = (
    strict[
        "strain_entity_id"
    ]
    .astype(str)
    .str.strip()
)


lineage_fold_nunique = (
    strict.groupby(
        "lineage_95_block"
    )["outer_fold"]
    .nunique()
)


bad = lineage_fold_nunique[
    lineage_fold_nunique != 1
]

if len(bad):

    raise RuntimeError(
        f"{len(bad)} lineage blocks occur "
        "in multiple outer folds."
    )


lineage_fold = (
    strict[
        [
            "lineage_95_block",
            "outer_fold",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "outer_fold",
            "lineage_95_block",
        ]
    )
    .reset_index(
        drop=True
    )
)


if lineage_fold[
    "lineage_95_block"
].nunique() != 83:

    raise RuntimeError(
        "Expected 83 downstream lineage blocks."
    )


print(
    "[PASS] frozen downstream lineages =",
    lineage_fold[
        "lineage_95_block"
    ].nunique(),
)

print()

print(
    lineage_fold.groupby(
        "outer_fold"
    )[
        "lineage_95_block"
    ]
    .nunique()
    .rename(
        "n_test_lineages"
    )
    .to_string()
)


LINEAGE_FOLD_OUT = (
    OUT
    / "01_lineage95_outer_fold_map.csv"
)

lineage_fold.to_csv(
    LINEAGE_FOLD_OUT,
    index=False,
)


lineage_to_fold = dict(
    zip(
        lineage_fold[
            "lineage_95_block"
        ],
        lineage_fold[
            "outer_fold"
        ].astype(int),
    )
)


# =============================================================================
# 2. LOAD ALL 965 LINEAGE-ELIGIBLE GENOMES
# =============================================================================

print()
print("=" * 120)
print("2. PRETRAINING GENOME UNIVERSE")
print("=" * 120)

strain_lineage = pd.read_csv(
    STRAIN_LINEAGE,
    low_memory=False,
)

strain_lineage[
    "strain_entity_id"
] = (
    strain_lineage[
        "strain_entity_id"
    ]
    .astype(str)
    .str.strip()
)

strain_lineage[
    "lineage_95_block"
] = (
    strain_lineage[
        "lineage_95_block"
    ]
    .astype(str)
    .str.strip()
)


if strain_lineage[
    "strain_entity_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicated strain IDs in strain-lineage map."
    )


n_genomes = len(
    strain_lineage
)

n_lineages = strain_lineage[
    "lineage_95_block"
].nunique()


print(
    "[PASS] lineage-eligible genomes =",
    n_genomes,
)

print(
    "[PASS] lineage blocks =",
    n_lineages,
)


if n_genomes != 965:

    print(
        "[WARN] expected historical value 965, "
        f"observed {n_genomes}"
    )


missing_fold_lineages = sorted(
    set(
        strain_lineage[
            "lineage_95_block"
        ]
    )
    -
    set(
        lineage_to_fold
    )
)


if missing_fold_lineages:

    raise RuntimeError(
        "Some eligible pretraining lineages are absent "
        "from frozen downstream folds: "
        f"{missing_fold_lineages[:20]}"
    )


# =============================================================================
# 3. BUILD STRICT PRETRAINING SPLITS
#
# Outer test:
#   ALL genomes belonging to downstream fold's held-out lineages.
#
# Inner validation:
#   whole lineages only; deterministic hash order.
#   We accumulate lineages until ~12.5% of non-test genomes.
#
# Train:
#   all remaining non-test, non-val lineages.
# =============================================================================

print()
print("=" * 120)
print("3. BUILD TRAIN / VAL / OUTER SPLITS")
print("=" * 120)


long_rows = []
fold_rows = []


for fold in range(
    1,
    N_FOLDS + 1,
):

    outer_lineages = {
        lin
        for lin, f in lineage_to_fold.items()
        if int(f) == fold
    }


    outer = strain_lineage[
        strain_lineage[
            "lineage_95_block"
        ].isin(
            outer_lineages
        )
    ].copy()


    pool = strain_lineage[
        ~strain_lineage[
            "lineage_95_block"
        ].isin(
            outer_lineages
        )
    ].copy()


    lineage_sizes = (
        pool.groupby(
            "lineage_95_block"
        )[
            "strain_entity_id"
        ]
        .nunique()
        .to_dict()
    )


    candidate_lineages = sorted(
        lineage_sizes,
        key=lambda x: stable_hash(
            f"fold={fold}|{x}"
        ),
    )


    target_val_n = max(
        1,
        int(
            round(
                len(pool)
                *
                TARGET_VAL_FRACTION
            )
        ),
    )


    val_lineages = set()
    val_n = 0


    for lin in candidate_lineages:

        val_lineages.add(
            lin
        )

        val_n += int(
            lineage_sizes[
                lin
            ]
        )

        if val_n >= target_val_n:
            break


    val = pool[
        pool[
            "lineage_95_block"
        ].isin(
            val_lineages
        )
    ].copy()


    train = pool[
        ~pool[
            "lineage_95_block"
        ].isin(
            val_lineages
        )
    ].copy()


    train_l = set(
        train[
            "lineage_95_block"
        ]
    )

    val_l = set(
        val[
            "lineage_95_block"
        ]
    )

    outer_l = set(
        outer[
            "lineage_95_block"
        ]
    )


    if train_l & val_l:

        raise RuntimeError(
            f"Fold {fold}: train/val lineage overlap."
        )

    if train_l & outer_l:

        raise RuntimeError(
            f"Fold {fold}: train/outer lineage overlap."
        )

    if val_l & outer_l:

        raise RuntimeError(
            f"Fold {fold}: val/outer lineage overlap."
        )


    if (
        len(train)
        +
        len(val)
        +
        len(outer)
        !=
        len(strain_lineage)
    ):

        raise RuntimeError(
            f"Fold {fold}: genome accounting failure."
        )


    for role, df in [
        ("train", train),
        ("val", val),
        ("outer", outer),
    ]:

        for row in df.itertuples(
            index=False
        ):

            long_rows.append(
                {
                    "outer_fold":
                        fold,

                    "strain_entity_id":
                        row.strain_entity_id,

                    "lineage_95_block":
                        row.lineage_95_block,

                    "role":
                        role,
                }
            )


    fold_rows.append(
        {
            "outer_fold":
                fold,

            "train_genomes":
                len(train),

            "val_genomes":
                len(val),

            "outer_genomes":
                len(outer),

            "train_lineages":
                len(train_l),

            "val_lineages":
                len(val_l),

            "outer_lineages":
                len(outer_l),

            "train_val_lineage_overlap":
                len(
                    train_l
                    &
                    val_l
                ),

            "train_outer_lineage_overlap":
                len(
                    train_l
                    &
                    outer_l
                ),

            "val_outer_lineage_overlap":
                len(
                    val_l
                    &
                    outer_l
                ),
        }
    )


split_manifest = pd.DataFrame(
    long_rows
)

split_summary = pd.DataFrame(
    fold_rows
)


print(
    split_summary.to_string(
        index=False
    )
)


if (
    split_summary[
        [
            "train_val_lineage_overlap",
            "train_outer_lineage_overlap",
            "val_outer_lineage_overlap",
        ]
    ]
    !=
    0
).any().any():

    raise RuntimeError(
        "Lineage overlap detected."
    )


print()
print(
    "[STRICT PASS] train/val/outer "
    "lineage overlap = 0 in all 5 folds"
)


SPLIT_OUT = (
    OUT
    / "02_lineage_blocked_SSL_split_manifest.csv.gz"
)

SUMMARY_OUT = (
    OUT
    / "03_lineage_blocked_SSL_split_summary.csv"
)

split_manifest.to_csv(
    SPLIT_OUT,
    index=False,
    compression="gzip",
)

split_summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


# =============================================================================
# 4. AUDIT EXISTING SAMPLE-WISE R4 CHECKPOINT EXPOSURE
# =============================================================================

print()
print("=" * 120)
print("4. EXISTING R4 CHECKPOINT LINEAGE-EXPOSURE AUDIT")
print("=" * 120)


existing_split_files = sorted(
    R4_ROOT.glob(
        "fold_*/split_manifest.csv"
    )
)


exposure_rows = []


if not existing_split_files:

    print(
        "[WARN] no existing R4 split_manifest.csv files found"
    )

else:

    for path in existing_split_files:

        fold_match = re.search(
            r"fold_(\d+)",
            str(path),
        )

        r4_fold = (
            int(
                fold_match.group(1)
            )
            if fold_match
            else -1
        )


        df = pd.read_csv(
            path,
            low_memory=False,
        )


        id_col = find_column(
            df.columns,
            [
                "strain_entity_id",
                "sample_id",
                "genome_id",
                "strain_id",
            ],
        )


        role_col = find_column(
            df.columns,
            [
                "role",
                "split",
                "subset",
                "partition",
            ],
        )


        if id_col is None:

            print(
                "[WARN] cannot identify strain column:",
                path,
                list(
                    df.columns
                ),
            )

            continue


        df[
            id_col
        ] = (
            df[
                id_col
            ]
            .astype(str)
            .str.strip()
        )


        tmp = df.merge(
            strain_lineage[
                [
                    "strain_entity_id",
                    "lineage_95_block",
                ]
            ],
            left_on=id_col,
            right_on="strain_entity_id",
            how="left",
        )


        if role_col is not None:

            role = (
                tmp[
                    role_col
                ]
                .astype(str)
                .str.lower()
            )

            pretrain_mask = (
                role.str.contains(
                    "train"
                )
                |
                role.str.contains(
                    "val"
                )
            )

        else:

            # Without role information, all genomes in the manifest
            # are conservatively considered "seen".
            pretrain_mask = np.ones(
                len(tmp),
                dtype=bool,
            )


        seen_lineages = set(
            tmp.loc[
                pretrain_mask,
                "lineage_95_block",
            ]
            .dropna()
            .astype(str)
        )


        for downstream_fold in range(
            1,
            6,
        ):

            heldout = {
                lin
                for lin, f in lineage_to_fold.items()
                if int(f) == downstream_fold
            }

            overlap = (
                seen_lineages
                &
                heldout
            )

            exposure_rows.append(
                {
                    "existing_R4_fold":
                        r4_fold,

                    "downstream_outer_fold":
                        downstream_fold,

                    "n_seen_lineages":
                        len(
                            seen_lineages
                        ),

                    "n_downstream_test_lineages":
                        len(
                            heldout
                        ),

                    "n_test_lineages_seen_during_SSL":
                        len(
                            overlap
                        ),

                    "strict_lineage_clean":
                        len(
                            overlap
                        )
                        ==
                        0,
                }
            )


if exposure_rows:

    exposure = pd.DataFrame(
        exposure_rows
    )

    print(
        exposure.to_string(
            index=False
        )
    )

    exposure.to_csv(
        OUT
        / "04_existing_R4_lineage_exposure_audit.csv",
        index=False,
    )

    if (
        ~exposure[
            "strict_lineage_clean"
        ]
    ).any():

        print()
        print(
            "[EXPECTED FINDING] existing sample-wise R4 "
            "checkpoints are NOT valid for strict "
            "95%-ANI-held-out downstream claims."
        )

else:

    print(
        "[INFO] existing R4 exposure could not "
        "be fully parsed; new lineage-blocked "
        "pretraining is still required."
    )


# =============================================================================
# 5. CHECKPOINT INVENTORY
# =============================================================================

print()
print("=" * 120)
print("5. EXISTING CHECKPOINT INVENTORY")
print("=" * 120)


checkpoint_roots = {
    "GenoGrammar_R4":
        R4_ROOT,

    "mean_pool":
        BASELINE_ROOT
        / "mean_pool",

    "cnn1d":
        BASELINE_ROOT
        / "cnn1d",

    "bigru":
        BASELINE_ROOT
        / "bigru",

    "transformer_small":
        BASELINE_ROOT
        / "transformer_small",
}


checkpoint_rows = []


for model_name, root in checkpoint_roots.items():

    if not root.exists():
        continue

    files = []

    for pattern in [
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.bin",
    ]:

        files.extend(
            root.rglob(
                pattern
            )
        )


    for path in sorted(
        set(
            files
        )
    ):

        fold_match = re.search(
            r"fold_(\d+)",
            str(path),
        )

        checkpoint_rows.append(
            {
                "model":
                    model_name,

                "fold":
                    (
                        int(
                            fold_match.group(1)
                        )
                        if fold_match
                        else None
                    ),

                "path":
                    str(path),

                "size_MB":
                    round(
                        path.stat().st_size
                        /
                        1024
                        /
                        1024,
                        3,
                    ),
            }
        )


checkpoint_inventory = pd.DataFrame(
    checkpoint_rows
)


if len(
    checkpoint_inventory
):

    print(
        checkpoint_inventory.to_string(
            index=False
        )
    )

    checkpoint_inventory.to_csv(
        OUT
        / "05_existing_checkpoint_inventory.csv",
        index=False,
    )

else:

    print(
        "[WARN] no checkpoint files discovered"
    )


# =============================================================================
# 6. PIPELINE SOURCE INTERFACE AUDIT
# =============================================================================

print()
print("=" * 120)
print("6. PIPELINE SOURCE INTERFACE AUDIT")
print("=" * 120)


source_paths = [
    R4_PIPELINE,
]


for name in [
    "mean_pool",
    "cnn1d",
    "bigru",
    "transformer_small",
]:

    p = (
        BASELINE_PIPE_ROOT
        / f"25_R4_{name}_5fold.py"
    )

    if p.exists():
        source_paths.append(
            p
        )


patterns = re.compile(
    r"(split|outer_fold|train_ids|val_ids|"
    r"outer_ids|eligible|checkpoint|best_|"
    r"torch\.save|load_state_dict|"
    r"split_manifest)",
    re.I,
)


interface_lines = []


for path in source_paths:

    text = path.read_text(
        errors="replace"
    ).splitlines()


    interface_lines.append(
        "=" * 120
    )

    interface_lines.append(
        str(path)
    )

    interface_lines.append(
        "=" * 120
    )


    for i, line in enumerate(
        text,
        1,
    ):

        if patterns.search(
            line
        ):

            lo = max(
                1,
                i - 2
            )

            hi = min(
                len(text),
                i + 2
            )

            for j in range(
                lo,
                hi + 1,
            ):

                interface_lines.append(
                    f"{j:05d}: "
                    f"{text[j-1]}"
                )

            interface_lines.append(
                ""
            )


interface_out = (
    OUT
    / "06_pipeline_split_checkpoint_interface_audit.txt"
)

interface_out.write_text(
    "\n".join(
        interface_lines
    ),
    encoding="utf-8",
)


print(
    "[PASS] source audit written:"
)

print(
    interface_out
)


# =============================================================================
# 7. FREEZE DEFINITION
# =============================================================================

definition = {
    "stage":
        "29A",

    "purpose":
        (
            "freeze lineage-blocked SSL pretraining folds "
            "before independent masked-center COG evaluation"
        ),

    "strict_downstream_manifest":
        str(
            STRICT_MANIFEST
        ),

    "strict_downstream_manifest_sha256":
        sha256(
            STRICT_MANIFEST
        ),

    "strict_dataset_freeze":
        str(
            STRICT_FREEZE
        ),

    "strict_dataset_freeze_sha256":
        sha256(
            STRICT_FREEZE
        ),

    "pretraining_genome_universe":
        str(
            STRAIN_LINEAGE
        ),

    "n_genomes":
        int(
            n_genomes
        ),

    "n_lineages":
        int(
            n_lineages
        ),

    "n_outer_folds":
        N_FOLDS,

    "outer_group":
        "lineage_95_block",

    "inner_validation_group":
        "lineage_95_block",

    "target_inner_val_fraction":
        TARGET_VAL_FRACTION,

    "inner_val_assignment":
        (
            "deterministic SHA256 ordering of training lineages; "
            "whole lineages accumulated to approximately "
            "12.5% of non-test genomes"
        ),

    "lineage_overlap_train_val_outer":
        0,

    "status":
        "FROZEN",
}


freeze_out = (
    OUT
    / "00_lineage_pretraining_freeze.json"
)

freeze_out.write_text(
    json.dumps(
        definition,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 120)
print("STAGE 29A FINAL REPORT")
print("=" * 120)

print(
    "Pretraining genomes:",
    n_genomes,
)

print(
    "95%-ANI lineages:",
    n_lineages,
)

print()

print(
    split_summary.to_string(
        index=False
    )
)

print()
print(
    "[SPLIT MANIFEST]",
    SPLIT_OUT,
)

print(
    "[FREEZE]",
    freeze_out,
)

print(
    "[CHECKPOINT INVENTORY]",
    OUT
    / "05_existing_checkpoint_inventory.csv",
)

print(
    "[PIPELINE INTERFACE]",
    interface_out,
)

print()

print(
    "[FINAL PASS] strict lineage-blocked "
    "SSL pretraining folds frozen"
)

print(
    "[NEXT] build formal R4 ContextSSL lineage-blocked "
    "retraining runner for all five encoders"
)
