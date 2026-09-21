from __future__ import annotations

import os

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(__file__).resolve().parents[1]

ROOT = (
    Path(os.environ.get("GENOGRAMMA_RESULT_ROOT", str(BASE / "runtime_assets" / "source_tree" / "new_paper_data" / "10_genogramma_pairaware")))
)

INPUT_NPZ = (
    BASE / "runtime_assets" / "source_tree" / "new_paper_data"
    / "08_stage50_order_dependent_downstream"
    / "07_pair_to_genogrammar_alignment"
    / "13_STAGE50_operon_formal_model_inputs.npz"
)

PRED_FILE = (
    ROOT
    / "11_stage59A_768d_ablation"
    / "59A_oof_predictions.csv.gz"
)

SUMMARY_FILE = (
    ROOT
    / "11_stage59A_768d_ablation"
    / "59A_FINAL_ablation_summary.csv"
)

OUT = (
    ROOT
    / "12_stage59B_ablation_lineage_bootstrap"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_INPUT_SHA = (
    "978e233098e4ccda12460e821d01fb076"
    "26f909d8afd4dafa2737c867081ab3e"
)

N_EXPECTED = 164302
N_SAME_EXPECTED = 123921

LEFT = 31
RIGHT = 32

N_BOOT = 5000
SEED = 42


VARIANTS = [
    "A1_CenterPair",
    "A2_PairInteraction",
    "A3_PairPlusEdge",
    "A4_PairPlusGlobal",
    "A5_Full768",
]


CONTRASTS = [
    (
        "Interaction_gain_over_center",
        "A2_PairInteraction",
        "A1_CenterPair",
    ),
    (
        "ContextualEdge_gain_over_pair",
        "A3_PairPlusEdge",
        "A2_PairInteraction",
    ),
    (
        "Global_gain_over_pair",
        "A4_PairPlusGlobal",
        "A2_PairInteraction",
    ),
    (
        "Global_gain_given_edge",
        "A5_Full768",
        "A3_PairPlusEdge",
    ),
    (
        "ContextualEdge_gain_given_global",
        "A5_Full768",
        "A4_PairPlusGlobal",
    ),
]


METRICS = [
    "ROC_AUC",
    "PR_AUC",
    "MCC",
]


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


def calc_metrics(y, p):

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    if len(y) < 2:
        return None

    if np.unique(y).size < 2:
        return None

    pred = (
        p >= 0.5
    ).astype(np.int64)

    return {
        "ROC_AUC":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "PR_AUC":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),

        "MCC":
            float(
                matthews_corrcoef(
                    y,
                    pred,
                )
            ),
    }


# =============================================================================
# INPUT AUDIT
# =============================================================================

print(
    "=" * 116,
    flush=True,
)

print(
    "STAGE59B — GENOGRAMMA ABLATION LINEAGE-LEVEL BOOTSTRAP",
    flush=True,
)

print(
    "FROZEN STAGE59A OOF | ANI95 EFFECTIVE UNIT | "
    "ALL-PAIRS + SAME-STRAND-ONLY | NO RETRAINING",
    flush=True,
)

print(
    "=" * 116,
    flush=True,
)


actual_sha = sha256(
    INPUT_NPZ
)

if actual_sha != EXPECTED_INPUT_SHA:
    raise RuntimeError(
        f"Formal input SHA mismatch: {actual_sha}"
    )


if not PRED_FILE.exists():
    raise RuntimeError(
        f"Missing Stage59A predictions: {PRED_FILE}"
    )


if not SUMMARY_FILE.exists():
    raise RuntimeError(
        f"Missing Stage59A summary: {SUMMARY_FILE}"
    )


z = np.load(
    INPUT_NPZ
)

row_index = np.asarray(
    z["row_index"],
    dtype=np.int64,
)

strand = np.asarray(
    z["strand"],
    dtype=np.int64,
)


if len(row_index) != N_EXPECTED:
    raise RuntimeError(
        f"Formal N={len(row_index)} != {N_EXPECTED}"
    )


same_strand = (
    strand[:, LEFT]
    ==
    strand[:, RIGHT]
)


if int(
    same_strand.sum()
) != N_SAME_EXPECTED:

    raise RuntimeError(
        f"Same-strand N={same_strand.sum()} "
        f"!= expected {N_SAME_EXPECTED}"
    )


strand_map = pd.DataFrame(
    {
        "row_index":
            row_index,

        "same_strand":
            same_strand.astype(
                np.int8
            ),
    }
)


pred = pd.read_csv(
    PRED_FILE
)


required = {
    "row_index",
    "variant",
    "y_true",
    "prob_positive",
    "strain_entity_id",
    "lineage_95_block",
}


if not required.issubset(
    pred.columns
):
    raise RuntimeError(
        f"Missing prediction columns: "
        f"{sorted(required - set(pred.columns))}"
    )


if set(
    pred["variant"].unique()
) != set(VARIANTS):

    raise RuntimeError(
        f"Unexpected variants: "
        f"{sorted(pred['variant'].unique())}"
    )


for variant in VARIANTS:

    q = pred[
        pred["variant"]
        ==
        variant
    ]

    if len(q) != N_EXPECTED:
        raise RuntimeError(
            f"{variant}: OOF N={len(q)}"
        )

    if q["row_index"].nunique() != N_EXPECTED:
        raise RuntimeError(
            f"{variant}: row_index not unique"
        )


pred = pred.merge(
    strand_map,
    on="row_index",
    how="left",
    validate="m:1",
)


if pred["same_strand"].isna().any():
    raise RuntimeError(
        "same-strand merge failed"
    )


print(
    f"[PASS] formal input SHA = {actual_sha}",
    flush=True,
)

print(
    f"[PASS] variants = {len(VARIANTS)}",
    flush=True,
)

print(
    f"[PASS] each variant OOF N = {N_EXPECTED}",
    flush=True,
)

print(
    f"[PASS] same-strand N = {N_SAME_EXPECTED}",
    flush=True,
)

print(
    "[PASS] no model fitting will be performed",
    flush=True,
)


# =============================================================================
# FREEZE PROTOCOL BEFORE ANALYSIS
# =============================================================================

protocol = {

    "status":
        "STAGE59B_PROTOCOL_FROZEN_BEFORE_ANALYSIS",

    "formal_model_name":
        "GenoGramma",

    "source":
        "Stage59A frozen OOF predictions",

    "effective_unit":
        "ANI95 lineage",

    "subsets":[
        "all_pairs",
        "same_strand_only",
    ],

    "metrics":[
        "ROC_AUC",
        "PR_AUC",
        "MCC",
    ],

    "bootstrap_replicates":
        N_BOOT,

    "bootstrap_seed":
        SEED,

    "bootstrap_strategy":
        "paired resampling of ANI95 lineage-level metrics",

    "contrasts":[
        {
            "name": name,
            "left": left,
            "right": right,
            "delta":
                "left minus right",
        }
        for name, left, right
        in CONTRASTS
    ],

    "encoder_training":
        False,

    "probe_training":
        False,

    "new_model_fitting":
        False,

    "direct_operon_features":
        False,
}


(
    OUT
    / "59B_protocol_freeze.json"
).write_text(
    json.dumps(
        protocol,
        indent=2,
    )
)


# =============================================================================
# PER-LINEAGE METRICS
# =============================================================================

lineage_rows = []


for subset_name in [
    "all_pairs",
    "same_strand_only",
]:

    if subset_name == "all_pairs":

        sub = pred

    else:

        sub = pred[
            pred["same_strand"]
            ==
            1
        ]


    for variant in VARIANTS:

        q = sub[
            sub["variant"]
            ==
            variant
        ]


        for lineage_id, d in q.groupby(
            "lineage_95_block",
            sort=False,
        ):

            m = calc_metrics(
                d["y_true"],
                d["prob_positive"],
            )


            if m is None:
                continue


            lineage_rows.append(
                {
                    "subset":
                        subset_name,

                    "variant":
                        variant,

                    "lineage_95_block":
                        str(
                            lineage_id
                        ),

                    "N":
                        int(
                            len(d)
                        ),

                    "Positive":
                        int(
                            d["y_true"].sum()
                        ),

                    "Negative":
                        int(
                            len(d)
                            -
                            d["y_true"].sum()
                        ),

                    "Positive_rate":
                        float(
                            d["y_true"].mean()
                        ),

                    **m,
                }
            )


lineage_df = pd.DataFrame(
    lineage_rows
)


lineage_df.to_csv(
    OUT
    / "59B_per_lineage_metrics.csv.gz",
    index=False,
    compression="gzip",
)


# =============================================================================
# VARIANT MACRO LINEAGE BOOTSTRAP
# =============================================================================

rng = np.random.default_rng(
    SEED
)

macro_rows = []


for subset_name in [
    "all_pairs",
    "same_strand_only",
]:

    for variant in VARIANTS:

        q = lineage_df[
            (
                lineage_df["subset"]
                ==
                subset_name
            )
            &
            (
                lineage_df["variant"]
                ==
                variant
            )
        ]


        for metric in METRICS:

            x = (
                q[metric]
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )


            if len(x) < 2:
                continue


            boot = np.empty(
                N_BOOT,
                dtype=float,
            )


            for b in range(
                N_BOOT
            ):

                idx = rng.integers(
                    0,
                    len(x),
                    size=len(x),
                )

                boot[b] = np.mean(
                    x[idx]
                )


            macro_rows.append(
                {
                    "subset":
                        subset_name,

                    "variant":
                        variant,

                    "metric":
                        metric,

                    "valid_lineages":
                        len(x),

                    "lineage_macro_mean":
                        float(
                            np.mean(x)
                        ),

                    "lineage_median":
                        float(
                            np.median(x)
                        ),

                    "Q1":
                        float(
                            np.quantile(
                                x,
                                0.25,
                            )
                        ),

                    "Q3":
                        float(
                            np.quantile(
                                x,
                                0.75,
                            )
                        ),

                    "bootstrap_CI_low":
                        float(
                            np.quantile(
                                boot,
                                0.025,
                            )
                        ),

                    "bootstrap_CI_high":
                        float(
                            np.quantile(
                                boot,
                                0.975,
                            )
                        ),
                }
            )


macro_df = pd.DataFrame(
    macro_rows
)


macro_df.to_csv(
    OUT
    / "59B_variant_lineage_macro_bootstrap.csv",
    index=False,
)


# =============================================================================
# PAIRED LINEAGE BOOTSTRAP CONTRASTS
# =============================================================================

paired_rows = []
detail_rows = []


for subset_name in [
    "all_pairs",
    "same_strand_only",
]:

    for contrast_name, left, right in CONTRASTS:

        for metric in METRICS:

            a = (
                lineage_df[
                    (
                        lineage_df["subset"]
                        ==
                        subset_name
                    )
                    &
                    (
                        lineage_df["variant"]
                        ==
                        left
                    )
                ][
                    [
                        "lineage_95_block",
                        metric,
                    ]
                ]
                .rename(
                    columns={
                        metric:
                            "left_value"
                    }
                )
            )


            b = (
                lineage_df[
                    (
                        lineage_df["subset"]
                        ==
                        subset_name
                    )
                    &
                    (
                        lineage_df["variant"]
                        ==
                        right
                    )
                ][
                    [
                        "lineage_95_block",
                        metric,
                    ]
                ]
                .rename(
                    columns={
                        metric:
                            "right_value"
                    }
                )
            )


            pair = (
                a.merge(
                    b,
                    on="lineage_95_block",
                    how="inner",
                    validate="1:1",
                )
                .dropna()
            )


            if len(pair) < 2:
                continue


            diff = (
                pair["left_value"]
                .to_numpy(
                    dtype=float
                )
                -
                pair["right_value"]
                .to_numpy(
                    dtype=float
                )
            )


            boot = np.empty(
                N_BOOT,
                dtype=float,
            )


            n = len(diff)


            for k in range(
                N_BOOT
            ):

                idx = rng.integers(
                    0,
                    n,
                    size=n,
                )

                boot[k] = np.mean(
                    diff[idx]
                )


            paired_rows.append(
                {
                    "subset":
                        subset_name,

                    "contrast":
                        contrast_name,

                    "left_variant":
                        left,

                    "right_variant":
                        right,

                    "metric":
                        metric,

                    "paired_lineages":
                        n,

                    "mean_delta":
                        float(
                            np.mean(diff)
                        ),

                    "median_delta":
                        float(
                            np.median(diff)
                        ),

                    "CI_low":
                        float(
                            np.quantile(
                                boot,
                                0.025,
                            )
                        ),

                    "CI_high":
                        float(
                            np.quantile(
                                boot,
                                0.975,
                            )
                        ),

                    "bootstrap_probability_delta_gt_0":
                        float(
                            np.mean(
                                boot > 0
                            )
                        ),

                    "fraction_lineages_delta_gt_0":
                        float(
                            np.mean(
                                diff > 0
                            )
                        ),

                    "fraction_lineages_delta_ge_0":
                        float(
                            np.mean(
                                diff >= 0
                            )
                        ),
                }
            )


            for lineage_id, delta in zip(
                pair[
                    "lineage_95_block"
                ],
                diff,
            ):

                detail_rows.append(
                    {
                        "subset":
                            subset_name,

                        "contrast":
                            contrast_name,

                        "metric":
                            metric,

                        "lineage_95_block":
                            lineage_id,

                        "delta":
                            float(
                                delta
                            ),
                    }
                )


paired_df = pd.DataFrame(
    paired_rows
)


paired_df.to_csv(
    OUT
    / "59B_paired_lineage_bootstrap_contrasts.csv",
    index=False,
)


pd.DataFrame(
    detail_rows
).to_csv(
    OUT
    / "59B_per_lineage_contrast_deltas.csv.gz",
    index=False,
    compression="gzip",
)


# =============================================================================
# CONTEXTUAL EDGE FOCUS TABLE
# =============================================================================

focus_names = {
    "ContextualEdge_gain_over_pair",
    "ContextualEdge_gain_given_global",
}


focus_df = paired_df[
    (
        paired_df["contrast"]
        .isin(
            focus_names
        )
    )
    &
    (
        paired_df["metric"]
        .isin(
            [
                "ROC_AUC",
                "PR_AUC",
            ]
        )
    )
].copy()


focus_df.to_csv(
    OUT
    / "59B_CONTEXTUAL_EDGE_focus.csv",
    index=False,
)


# =============================================================================
# RESULT FREEZE
# =============================================================================

result_freeze = {

    "status":
        "STAGE59B_ABLATION_LINEAGE_BOOTSTRAP_COMPLETE",

    "formal_model_name":
        "GenoGramma",

    "formal_input_sha256":
        actual_sha,

    "effective_unit":
        "ANI95 lineage",

    "bootstrap_replicates":
        N_BOOT,

    "subsets":[
        "all_pairs",
        "same_strand_only",
    ],

    "contextual_edge_primary_contrasts":[
        "A3_PairPlusEdge minus A2_PairInteraction",
        "A5_Full768 minus A4_PairPlusGlobal",
    ],

    "new_model_fitting":
        False,

    "encoder_training":
        False,

    "probe_training":
        False,
}


(
    OUT
    / "59B_result_freeze.json"
).write_text(
    json.dumps(
        result_freeze,
        indent=2,
    )
)


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

print()
print(
    "=" * 116,
    flush=True,
)

print(
    "STAGE59B — CONTEXTUAL EDGE LINEAGE-LEVEL RESULTS",
    flush=True,
)

print(
    "=" * 116,
    flush=True,
)


print()
print(
    "===== CONTEXTUAL EDGE PRIMARY CONTRASTS =====",
    flush=True,
)


print(
    focus_df[
        [
            "subset",
            "contrast",
            "metric",
            "paired_lineages",
            "mean_delta",
            "CI_low",
            "CI_high",
            "bootstrap_probability_delta_gt_0",
            "fraction_lineages_delta_gt_0",
        ]
    ].to_string(
        index=False
    ),
    flush=True,
)


print()
print(
    "===== ALL PRE-REGISTERED CONTRASTS =====",
    flush=True,
)


print(
    paired_df[
        paired_df[
            "metric"
        ].isin(
            [
                "ROC_AUC",
                "PR_AUC",
            ]
        )
    ][
        [
            "subset",
            "contrast",
            "metric",
            "paired_lineages",
            "mean_delta",
            "CI_low",
            "CI_high",
            "bootstrap_probability_delta_gt_0",
        ]
    ].to_string(
        index=False
    ),
    flush=True,
)


print()
print(
    "[PASS] no model retraining",
    flush=True,
)

print(
    "[PASS] effective unit = ANI95 lineage",
    flush=True,
)

print(
    "[PASS] 5000 paired lineage bootstrap resamples",
    flush=True,
)

print(
    "[PASS] all-pairs and same-strand-only both evaluated",
    flush=True,
)

print(
    "[OUTPUT]",
    OUT,
    flush=True,
)

print(
    "STAGE59B_ABLATION_LINEAGE_BOOTSTRAP_COMPLETE",
    flush=True,
)

print(
    "=" * 116,
    flush=True,
)

