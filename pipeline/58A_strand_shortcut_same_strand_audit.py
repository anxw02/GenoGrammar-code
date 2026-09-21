from __future__ import annotations

import os

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


BASE = Path(__file__).resolve().parents[1]

ROOT=(
    Path(os.environ.get("GENOGRAMMA_RESULT_ROOT", str(BASE / "runtime_assets" / "source_tree" / "new_paper_data" / "10_genogramma_pairaware")))
)

INPUT_NPZ=(
    BASE / "runtime_assets" / "source_tree" / "new_paper_data"
    / "08_stage50_order_dependent_downstream"
    / "07_pair_to_genogrammar_alignment"
    / "13_STAGE50_operon_formal_model_inputs.npz"
)

GG_PRED=(
    ROOT
    / "04_stage56_pairaware_probe"
    / "03_all_oof_predictions.csv.gz"
)

BL_PRED=(
    ROOT
    / "08_stage57C_matched_comparison"
    / "57C_baseline_oof_predictions.csv.gz"
)

OUT=(
    ROOT
    / "09_stage58A_strand_shortcut_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_SHA=(
    "978e233098e4ccda12460e821d01fb076"
    "26f909d8afd4dafa2737c867081ab3e"
)

LEFT=31
RIGHT=32

MODELS=[
    "GenoGramma",
    "CNN1D",
    "BiGRU",
    "TransformerSmall",
    "MeanPool",
]


def sha256(path):

    h=hashlib.sha256()

    with open(path,"rb") as f:
        for b in iter(
            lambda:f.read(1024*1024),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def metrics(
    y,
    p,
):

    y=np.asarray(
        y,
        dtype=int,
    )

    p=np.asarray(
        p,
        dtype=float,
    )

    if len(
        np.unique(y)
    ) < 2:

        return None

    pred=(
        p >= 0.5
    ).astype(int)

    tn,fp,fn,tp=confusion_matrix(
        y,
        pred,
        labels=[0,1],
    ).ravel()

    specificity=(
        tn/(tn+fp)
        if (tn+fp)
        else np.nan
    )

    return {
        "N":
            int(len(y)),

        "Positive":
            int(y.sum()),

        "Negative":
            int(
                len(y)-y.sum()
            ),

        "Positive_rate":
            float(y.mean()),

        "ROC_AUC":
            float(
                roc_auc_score(
                    y,p
                )
            ),

        "PR_AUC":
            float(
                average_precision_score(
                    y,p
                )
            ),

        "Accuracy":
            float(
                accuracy_score(
                    y,pred
                )
            ),

        "Balanced_Accuracy":
            float(
                balanced_accuracy_score(
                    y,pred
                )
            ),

        "Precision":
            float(
                precision_score(
                    y,pred,
                    zero_division=0,
                )
            ),

        "Recall":
            float(
                recall_score(
                    y,pred,
                    zero_division=0,
                )
            ),

        "Specificity":
            float(specificity),

        "F1":
            float(
                f1_score(
                    y,pred,
                    zero_division=0,
                )
            ),

        "MCC":
            float(
                matthews_corrcoef(
                    y,pred
                )
            ),
    }


# ----------------------------------------------------------------------
# Formal input audit
# ----------------------------------------------------------------------

actual_sha=sha256(
    INPUT_NPZ
)

if actual_sha != EXPECTED_SHA:
    raise RuntimeError(
        "Formal input SHA mismatch"
    )


z=np.load(
    INPUT_NPZ
)

strand=np.asarray(
    z["strand"],
    dtype=np.int64,
)

y=np.asarray(
    z["label_index"],
    dtype=np.int64,
)

row_index=np.asarray(
    z["row_index"],
    dtype=np.int64,
)

outer_fold=np.asarray(
    z["outer_fold"],
    dtype=np.int64,
)


if strand.shape != (
    164302,
    64,
):
    raise RuntimeError(
        f"Unexpected strand shape "
        f"{strand.shape}"
    )


left_strand=strand[
    :,
    LEFT
]

right_strand=strand[
    :,
    RIGHT
]

same_strand=(
    left_strand
    ==
    right_strand
)


base=pd.DataFrame(
    {
        "row_index":
            row_index,

        "y_true":
            y,

        "outer_fold":
            outer_fold,

        "left_strand":
            left_strand,

        "right_strand":
            right_strand,

        "same_strand":
            same_strand.astype(int),
    }
)


# ----------------------------------------------------------------------
# Load frozen OOF predictions
# ----------------------------------------------------------------------

gg=pd.read_csv(
    GG_PRED
)

bl=pd.read_csv(
    BL_PRED
)

pred=pd.concat(
    [
        gg[
            [
                "row_index",
                "model",
                "outer_fold",
                "y_true",
                "prob_positive",
                "strain_entity_id",
                "lineage_95_block",
            ]
        ],
        bl[
            [
                "row_index",
                "model",
                "outer_fold",
                "y_true",
                "prob_positive",
                "strain_entity_id",
                "lineage_95_block",
            ]
        ],
    ],
    ignore_index=True,
)


if set(
    pred["model"].unique()
) != set(MODELS):

    raise RuntimeError(
        f"Unexpected model set: "
        f"{sorted(pred['model'].unique())}"
    )


pred=pred.merge(
    base[
        [
            "row_index",
            "left_strand",
            "right_strand",
            "same_strand",
        ]
    ],
    on="row_index",
    how="left",
    validate="m:1",
)


if pred[
    "same_strand"
].isna().any():

    raise RuntimeError(
        "Strand merge failure"
    )


# ----------------------------------------------------------------------
# Dataset-level strand composition
# ----------------------------------------------------------------------

composition=[]

for flag,name in [
    (1,"same_strand"),
    (0,"opposite_strand"),
]:

    q=base[
        base[
            "same_strand"
        ]
        ==
        flag
    ]

    composition.append(
        {
            "subset":
                name,

            "N":
                len(q),

            "Positive":
                int(
                    q[
                        "y_true"
                    ].sum()
                ),

            "Negative":
                int(
                    len(q)
                    -
                    q[
                        "y_true"
                    ].sum()
                ),

            "Positive_rate":
                float(
                    q[
                        "y_true"
                    ].mean()
                ),
        }
    )


composition_df=pd.DataFrame(
    composition
)

composition_df.to_csv(
    OUT
    /
    "58A_strand_label_composition.csv",
    index=False,
)


# ----------------------------------------------------------------------
# StrandMatch-only baseline
# ----------------------------------------------------------------------

strand_score=base[
    "same_strand"
].to_numpy(
    dtype=float
)

strand_metrics=metrics(
    y,
    strand_score,
)


strand_row={
    "model":
        "StrandMatch",

    "subset":
        "all_pairs",

    **strand_metrics,
}


pd.DataFrame(
    [
        strand_row
    ]
).to_csv(
    OUT
    /
    "58A_StrandMatch_baseline.csv",
    index=False,
)


# ----------------------------------------------------------------------
# All-pairs vs same-strand OOF performance
# ----------------------------------------------------------------------

summary=[]


for model_name in MODELS:

    q=pred[
        pred[
            "model"
        ]
        ==
        model_name
    ].copy()


    if len(q) != 164302:
        raise RuntimeError(
            f"{model_name}: "
            f"OOF N={len(q)}"
        )


    m_all=metrics(
        q[
            "y_true"
        ],
        q[
            "prob_positive"
        ],
    )


    q_same=q[
        q[
            "same_strand"
        ]
        ==
        1
    ]


    m_same=metrics(
        q_same[
            "y_true"
        ],
        q_same[
            "prob_positive"
        ],
    )


    if m_same is None:
        raise RuntimeError(
            f"{model_name}: "
            "same-strand subset has one class"
        )


    summary.append(
        {
            "model":
                model_name,

            "subset":
                "all_pairs",

            **m_all,
        }
    )


    summary.append(
        {
            "model":
                model_name,

            "subset":
                "same_strand_only",

            **m_same,
        }
    )


summary_df=pd.DataFrame(
    summary
)


summary_df.to_csv(
    OUT
    /
    "58A_all_vs_same_strand_model_summary.csv",
    index=False,
)


# ----------------------------------------------------------------------
# Same-strand GenoGramma deltas
# ----------------------------------------------------------------------

same_table=summary_df[
    summary_df[
        "subset"
    ]
    ==
    "same_strand_only"
].copy()


gg_same=same_table[
    same_table[
        "model"
    ]
    ==
    "GenoGramma"
].iloc[0]


delta=[]


for _,r in same_table.iterrows():

    if r[
        "model"
    ] == "GenoGramma":
        continue

    delta.append(
        {
            "baseline":
                r[
                    "model"
                ],

            "GenoGramma_same_ROC":
                gg_same[
                    "ROC_AUC"
                ],

            "baseline_same_ROC":
                r[
                    "ROC_AUC"
                ],

            "Delta_ROC":
                gg_same[
                    "ROC_AUC"
                ]
                -
                r[
                    "ROC_AUC"
                ],

            "GenoGramma_same_PR":
                gg_same[
                    "PR_AUC"
                ],

            "baseline_same_PR":
                r[
                    "PR_AUC"
                ],

            "Delta_PR":
                gg_same[
                    "PR_AUC"
                ]
                -
                r[
                    "PR_AUC"
                ],
        }
    )


pd.DataFrame(
    delta
).to_csv(
    OUT
    /
    "58A_same_strand_GenoGramma_deltas.csv",
    index=False,
)


# ----------------------------------------------------------------------
# Per-lineage / per-genome same-strand distributions
# ----------------------------------------------------------------------

group_rows=[]


for model_name in MODELS:

    q=pred[
        (
            pred[
                "model"
            ]
            ==
            model_name
        )
        &
        (
            pred[
                "same_strand"
            ]
            ==
            1
        )
    ]


    for level in [
        "lineage_95_block",
        "strain_entity_id",
    ]:

        for group_id,d in q.groupby(
            level,
            sort=False,
        ):

            if (
                len(d) < 20
                or
                d[
                    "y_true"
                ].nunique()
                < 2
            ):
                continue


            m=metrics(
                d[
                    "y_true"
                ],
                d[
                    "prob_positive"
                ],
            )


            group_rows.append(
                {
                    "model":
                        model_name,

                    "level":
                        (
                            "ANI95_lineage"
                            if level
                            ==
                            "lineage_95_block"
                            else
                            "genome"
                        ),

                    "group_id":
                        group_id,

                    **m,
                }
            )


group_df=pd.DataFrame(
    group_rows
)


group_df.to_csv(
    OUT
    /
    "58A_same_strand_group_metrics.csv.gz",
    index=False,
    compression="gzip",
)


group_summary=[]


for (
    model_name,
    level
),d in group_df.groupby(
    [
        "model",
        "level",
    ]
):

    for metric in [
        "ROC_AUC",
        "PR_AUC",
        "MCC",
    ]:

        values=d[
            metric
        ].dropna()


        group_summary.append(
            {
                "model":
                    model_name,

                "level":
                    level,

                "metric":
                    metric,

                "valid_groups":
                    len(values),

                "median":
                    float(
                        values.median()
                    ),

                "Q1":
                    float(
                        values.quantile(
                            0.25
                        )
                    ),

                "Q3":
                    float(
                        values.quantile(
                            0.75
                        )
                    ),

                "fraction_gt_0.70":
                    float(
                        (
                            values > 0.70
                        ).mean()
                    )
                    if metric
                    in {
                        "ROC_AUC",
                        "PR_AUC",
                    }
                    else np.nan,
            }
        )


pd.DataFrame(
    group_summary
).to_csv(
    OUT
    /
    "58A_same_strand_group_summary.csv",
    index=False,
)


# ----------------------------------------------------------------------
# Freeze
# ----------------------------------------------------------------------

freeze={
    "status":
        "STAGE58A_STRAND_SHORTCUT_AUDIT_COMPLETE",

    "formal_model_name":
        "GenoGramma",

    "formal_input_sha256":
        actual_sha,

    "target_pair_indices_0based":
        [
            LEFT,
            RIGHT,
        ],

    "analysis":[
        "strand-label composition",
        "StrandMatch-only baseline",
        "same-strand-only frozen OOF evaluation",
        "per-ANI95-lineage same-strand metrics",
        "per-genome same-strand metrics",
    ],

    "important_note":
        "Stage58A is a frozen-prediction shortcut audit. "
        "No model or downstream head is retrained.",

    "encoder_training":
        False,

    "new_model_fitting":
        False,
}


(
    OUT
    /
    "58A_result_freeze.json"
).write_text(
    json.dumps(
        freeze,
        indent=2,
    )
)


# ----------------------------------------------------------------------
# Final console report
# ----------------------------------------------------------------------

print(
    "="*110
)

print(
    "STAGE58A — STRAND SHORTCUT AUDIT"
)

print(
    "="*110
)


print(
    "\n===== LABEL COMPOSITION ====="
)

print(
    composition_df.to_string(
        index=False
    )
)


print(
    "\n===== STRANDMATCH-ONLY ====="
)

print(
    pd.DataFrame(
        [
            strand_row
        ]
    )[
        [
            "model",
            "ROC_AUC",
            "PR_AUC",
            "Balanced_Accuracy",
            "F1",
            "MCC",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n===== SAME-STRAND-ONLY OOF ====="
)

print(
    same_table[
        [
            "model",
            "N",
            "ROC_AUC",
            "PR_AUC",
            "Balanced_Accuracy",
            "F1",
            "MCC",
        ]
    ]
    .sort_values(
        "ROC_AUC",
        ascending=False,
    )
    .to_string(
        index=False
    )
)


print(
    "\n===== GENOGRAMMA SAME-STRAND DELTAS ====="
)

print(
    pd.DataFrame(
        delta
    ).to_string(
        index=False
    )
)


print(
    "\n[PASS] no retraining performed"
)

print(
    "[PASS] frozen OOF predictions only"
)

print(
    "[OUTPUT]",
    OUT
)

print(
    "STAGE58A_STRAND_SHORTCUT_AUDIT_COMPLETE"
)

print(
    "="*110
)
