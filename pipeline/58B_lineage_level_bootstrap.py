from __future__ import annotations

import os

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]

ROOT=(
    Path(os.environ.get("GENOGRAMMA_RESULT_ROOT", str(BASE / "runtime_assets" / "source_tree" / "new_paper_data" / "10_genogramma_pairaware")))
)

GROUP_FILE=(
    ROOT
    / "09_stage58A_strand_shortcut_audit"
    / "58A_same_strand_group_metrics.csv.gz"
)

OUT=(
    ROOT
    / "10_stage58B_lineage_bootstrap"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


MODELS=[
    "GenoGramma",
    "TransformerSmall",
    "MeanPool",
    "BiGRU",
    "CNN1D",
]

BASELINES=[
    "TransformerSmall",
    "MeanPool",
    "BiGRU",
    "CNN1D",
]

METRICS=[
    "ROC_AUC",
    "PR_AUC",
    "MCC",
]

N_BOOT=5000
SEED=42


# =============================================================================
# LOAD
# =============================================================================

df=pd.read_csv(
    GROUP_FILE
)

required={
    "model",
    "level",
    "group_id",
    "N",
    "ROC_AUC",
    "PR_AUC",
    "MCC",
}

if not required.issubset(
    df.columns
):
    raise RuntimeError(
        f"Missing columns: "
        f"{sorted(required-set(df.columns))}"
    )


if set(
    df["model"].unique()
) != set(MODELS):

    raise RuntimeError(
        f"Unexpected model set: "
        f"{sorted(df['model'].unique())}"
    )


lineage=df[
    df[
        "level"
    ]
    ==
    "ANI95_lineage"
].copy()


genome=df[
    df[
        "level"
    ]
    ==
    "genome"
].copy()


print(
    "="*112
)

print(
    "STAGE58B — EFFECTIVE-N / LINEAGE-LEVEL ROBUSTNESS"
)

print(
    "SAME-STRAND ONLY | EQUAL-LINEAGE BOOTSTRAP | "
    "NO RETRAINING"
)

print(
    "="*112
)


print(
    "[PASS] lineage-level rows =",
    len(lineage)
)

print(
    "[PASS] genome-level rows =",
    len(genome)
)

print(
    "[PASS] bootstrap replicates =",
    N_BOOT
)


# =============================================================================
# DESCRIPTIVE LINEAGE / GENOME DISTRIBUTIONS
# =============================================================================

summary=[]


for level_name,d0 in [
    (
        "ANI95_lineage",
        lineage,
    ),
    (
        "genome",
        genome,
    ),
]:

    for model in MODELS:

        d=d0[
            d0[
                "model"
            ]
            ==
            model
        ]

        for metric in METRICS:

            x=(
                d[
                    metric
                ]
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            if len(x)==0:
                continue

            row={
                "level":
                    level_name,

                "model":
                    model,

                "metric":
                    metric,

                "valid_groups":
                    len(x),

                "mean":
                    float(
                        np.mean(x)
                    ),

                "median":
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

                "min":
                    float(
                        np.min(x)
                    ),

                "max":
                    float(
                        np.max(x)
                    ),
            }

            if metric in {
                "ROC_AUC",
                "PR_AUC",
            }:

                row[
                    "fraction_gt_0.70"
                ]=float(
                    np.mean(
                        x > 0.70
                    )
                )

            else:

                row[
                    "fraction_gt_0.70"
                ]=np.nan

            summary.append(
                row
            )


summary_df=pd.DataFrame(
    summary
)


summary_df.to_csv(
    OUT
    /
    "58B_lineage_genome_descriptive_summary.csv",
    index=False,
)


# =============================================================================
# EQUAL-LINEAGE BOOTSTRAP
#
# Important:
# each ANI95 lineage contributes one biological unit.
# We bootstrap lineage-level metrics, NOT individual gene pairs.
# =============================================================================

rng=np.random.default_rng(
    SEED
)

bootstrap_rows=[]


for model in MODELS:

    d=lineage[
        lineage[
            "model"
        ]
        ==
        model
    ].copy()


    for metric in METRICS:

        x=(
            d[
                metric
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )


        if len(x)<2:
            continue


        boot=np.empty(
            N_BOOT,
            dtype=float,
        )


        for b in range(
            N_BOOT
        ):

            sample=rng.choice(
                x,
                size=len(x),
                replace=True,
            )

            boot[
                b
            ]=np.mean(
                sample
            )


        bootstrap_rows.append(
            {
                "model":
                    model,

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

                "bootstrap_mean":
                    float(
                        np.mean(boot)
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


bootstrap_df=pd.DataFrame(
    bootstrap_rows
)


bootstrap_df.to_csv(
    OUT
    /
    "58B_equal_lineage_bootstrap_CI.csv",
    index=False,
)


# =============================================================================
# PAIRED LINEAGE BOOTSTRAP:
# GenoGramma vs each baseline
# =============================================================================

paired_rows=[]

paired_detail=[]


for baseline in BASELINES:

    for metric in METRICS:

        gg=(
            lineage[
                lineage[
                    "model"
                ]
                ==
                "GenoGramma"
            ][
                [
                    "group_id",
                    metric,
                ]
            ]
            .rename(
                columns={
                    metric:
                        "GenoGramma"
                }
            )
        )


        bl=(
            lineage[
                lineage[
                    "model"
                ]
                ==
                baseline
            ][
                [
                    "group_id",
                    metric,
                ]
            ]
            .rename(
                columns={
                    metric:
                        "baseline_value"
                }
            )
        )


        pair=gg.merge(
            bl,
            on="group_id",
            how="inner",
            validate="1:1",
        ).dropna()


        if len(pair)<2:
            continue


        diff=(
            pair[
                "GenoGramma"
            ].to_numpy(
                dtype=float
            )
            -
            pair[
                "baseline_value"
            ].to_numpy(
                dtype=float
            )
        )


        n=len(diff)

        boot=np.empty(
            N_BOOT,
            dtype=float,
        )


        for b in range(
            N_BOOT
        ):

            idx=rng.integers(
                0,
                n,
                size=n,
            )

            boot[
                b
            ]=np.mean(
                diff[
                    idx
                ]
            )


        paired_rows.append(
            {
                "baseline":
                    baseline,

                "metric":
                    metric,

                "paired_lineages":
                    n,

                "mean_delta_GenoGramma_minus_baseline":
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

                "fraction_lineages_GenoGramma_gt_baseline":
                    float(
                        np.mean(
                            diff > 0
                        )
                    ),
            }
        )


        for gid,dv in zip(
            pair[
                "group_id"
            ],
            diff,
        ):

            paired_detail.append(
                {
                    "baseline":
                        baseline,

                    "metric":
                        metric,

                    "lineage":
                        gid,

                    "delta":
                        float(dv),
                }
            )


paired_df=pd.DataFrame(
    paired_rows
)


paired_df.to_csv(
    OUT
    /
    "58B_paired_lineage_bootstrap_deltas.csv",
    index=False,
)


pd.DataFrame(
    paired_detail
).to_csv(
    OUT
    /
    "58B_per_lineage_paired_deltas.csv.gz",
    index=False,
    compression="gzip",
)


# =============================================================================
# WORST / BEST LINEAGES — GenoGramma
# =============================================================================

gg_lineage=lineage[
    lineage[
        "model"
    ]
    ==
    "GenoGramma"
].copy()


cols=[
    "group_id",
    "N",
    "Positive",
    "Negative",
    "Positive_rate",
    "ROC_AUC",
    "PR_AUC",
    "MCC",
]


gg_lineage[
    cols
].sort_values(
    "ROC_AUC",
    ascending=True,
).head(
    10
).to_csv(
    OUT
    /
    "58B_GenoGramma_10_worst_lineages.csv",
    index=False,
)


gg_lineage[
    cols
].sort_values(
    "ROC_AUC",
    ascending=False,
).head(
    10
).to_csv(
    OUT
    /
    "58B_GenoGramma_10_best_lineages.csv",
    index=False,
)


# =============================================================================
# FREEZE
# =============================================================================

freeze={
    "status":
        "STAGE58B_LINEAGE_BOOTSTRAP_COMPLETE",

    "formal_model_name":
        "GenoGramma",

    "subset":
        "same-strand-only",

    "effective_biological_unit":
        "ANI95 lineage",

    "bootstrap_unit":
        "ANI95 lineage-level metric",

    "bootstrap_replicates":
        N_BOOT,

    "bootstrap_seed":
        SEED,

    "important_note":
        "No individual gene-pair bootstrap was used. "
        "Each lineage is treated as one biological unit.",

    "new_model_fitting":
        False,

    "encoder_training":
        False,
}


(
    OUT
    /
    "58B_result_freeze.json"
).write_text(
    json.dumps(
        freeze,
        indent=2,
    )
)


# =============================================================================
# CONSOLE
# =============================================================================

print()
print(
    "===== GENOGRAMMA SAME-STRAND LINEAGE DISTRIBUTION ====="
)


print(
    summary_df[
        (
            summary_df[
                "level"
            ]
            ==
            "ANI95_lineage"
        )
        &
        (
            summary_df[
                "model"
            ]
            ==
            "GenoGramma"
        )
        &
        (
            summary_df[
                "metric"
            ].isin(
                [
                    "ROC_AUC",
                    "PR_AUC",
                    "MCC",
                ]
            )
        )
    ][
        [
            "metric",
            "valid_groups",
            "mean",
            "median",
            "Q1",
            "Q3",
            "fraction_gt_0.70",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "===== 5000x EQUAL-LINEAGE BOOTSTRAP CI ====="
)


print(
    bootstrap_df[
        bootstrap_df[
            "model"
        ]
        ==
        "GenoGramma"
    ][
        [
            "metric",
            "valid_lineages",
            "lineage_macro_mean",
            "bootstrap_CI_low",
            "bootstrap_CI_high",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "===== PAIRED GENOGRAMMA - BASELINE LINEAGE BOOTSTRAP ====="
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
            "baseline",
            "metric",
            "paired_lineages",
            "mean_delta_GenoGramma_minus_baseline",
            "CI_low",
            "CI_high",
            "bootstrap_probability_delta_gt_0",
            "fraction_lineages_GenoGramma_gt_baseline",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "[PASS] effective unit = ANI95 lineage"
)

print(
    "[PASS] bootstrap = 5000 equal-lineage resamples"
)

print(
    "[PASS] no pair-level pseudoreplication used for CI"
)

print(
    "[PASS] no model retraining"
)

print(
    "[OUTPUT]",
    OUT
)

print(
    "STAGE58B_LINEAGE_BOOTSTRAP_COMPLETE"
)

