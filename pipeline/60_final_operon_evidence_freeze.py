from __future__ import annotations

import os

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(__file__).resolve().parents[1]

ROOT = (
    Path(os.environ.get("GENOGRAMMA_RESULT_ROOT", str(BASE / "runtime_assets" / "source_tree" / "new_paper_data" / "10_genogramma_pairaware")))
)

OUT = (
    ROOT
    / "13_stage60_final_operon_evidence"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


FILES = {

    "stage57C_comparison":
        ROOT
        / "08_stage57C_matched_comparison"
        / "57C_FINAL_matched_pairaware_comparison.csv",

    "stage57C_deltas":
        ROOT
        / "08_stage57C_matched_comparison"
        / "57C_GenoGramma_vs_baseline_deltas.csv",

    "stage58A_summary":
        ROOT
        / "09_stage58A_strand_shortcut_audit"
        / "58A_all_vs_same_strand_model_summary.csv",

    "stage58A_strandmatch":
        ROOT
        / "09_stage58A_strand_shortcut_audit"
        / "58A_StrandMatch_baseline.csv",

    "stage58B_macro":
        ROOT
        / "10_stage58B_lineage_bootstrap"
        / "58B_equal_lineage_bootstrap_CI.csv",

    "stage58B_paired":
        ROOT
        / "10_stage58B_lineage_bootstrap"
        / "58B_paired_lineage_bootstrap_deltas.csv",

    "stage59A_ablation":
        ROOT
        / "11_stage59A_768d_ablation"
        / "59A_FINAL_ablation_summary.csv",

    "stage59A_increment":
        ROOT
        / "11_stage59A_768d_ablation"
        / "59A_component_increment_summary.csv",

    "stage59B_contrasts":
        ROOT
        / "12_stage59B_ablation_lineage_bootstrap"
        / "59B_paired_lineage_bootstrap_contrasts.csv",

    "stage59B_edge_focus":
        ROOT
        / "12_stage59B_ablation_lineage_bootstrap"
        / "59B_CONTEXTUAL_EDGE_focus.csv",
}


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


def require(path: Path):

    if not path.exists():
        raise RuntimeError(
            f"Missing required formal result: {path}"
        )


for name, path in FILES.items():
    require(path)


print("=" * 118)
print("STAGE60 — FINAL GENOGRAMMA OPERON EVIDENCE FREEZE")
print("NO TRAINING | NO RETUNING | NO NEW STATISTICAL TESTS")
print("=" * 118)


# =============================================================================
# LOAD
# =============================================================================

comp57 = pd.read_csv(
    FILES["stage57C_comparison"]
)

delta57 = pd.read_csv(
    FILES["stage57C_deltas"]
)

sum58 = pd.read_csv(
    FILES["stage58A_summary"]
)

strand58 = pd.read_csv(
    FILES["stage58A_strandmatch"]
)

macro58 = pd.read_csv(
    FILES["stage58B_macro"]
)

paired58 = pd.read_csv(
    FILES["stage58B_paired"]
)

abl59 = pd.read_csv(
    FILES["stage59A_ablation"]
)

inc59 = pd.read_csv(
    FILES["stage59A_increment"]
)

contrast59 = pd.read_csv(
    FILES["stage59B_contrasts"]
)

edge59 = pd.read_csv(
    FILES["stage59B_edge_focus"]
)


# =============================================================================
# 1. MAIN MATCHED PAIR-AWARE MODEL TABLE
# =============================================================================

model_order = {
    "GenoGramma": 1,
    "TransformerSmall": 2,
    "MeanPool": 3,
    "BiGRU": 4,
    "CNN1D": 5,
}

main = comp57.copy()

main["display_order"] = (
    main["model"]
    .map(model_order)
)

main = (
    main
    .sort_values(
        "display_order"
    )
    .drop(
        columns=[
            "display_order"
        ]
    )
)

main.to_csv(
    OUT
    / "60A_FINAL_matched_pairaware_model_comparison.csv",
    index=False,
)


# =============================================================================
# 2. STRAND ROBUSTNESS TABLE
# =============================================================================

keep_metrics = [
    "model",
    "subset",
    "N",
    "ROC_AUC",
    "PR_AUC",
    "Balanced_Accuracy",
    "F1",
    "MCC",
]

strand_table = sum58[
    keep_metrics
].copy()

strand_table.to_csv(
    OUT
    / "60B_FINAL_strand_robustness.csv",
    index=False,
)


strand58.to_csv(
    OUT
    / "60B_FINAL_StrandMatch_control.csv",
    index=False,
)


# =============================================================================
# 3. EFFECTIVE-N / LINEAGE TABLE
# =============================================================================

gg_macro = macro58[
    macro58["model"]
    ==
    "GenoGramma"
].copy()


gg_macro.to_csv(
    OUT
    / "60C_FINAL_GenoGramma_lineage_bootstrap.csv",
    index=False,
)


paired58[
    paired58["metric"]
    .isin(
        [
            "ROC_AUC",
            "PR_AUC",
        ]
    )
].to_csv(
    OUT
    / "60C_FINAL_baseline_lineage_paired_deltas.csv",
    index=False,
)


# =============================================================================
# 4. REPRESENTATION ABLATION TABLE
# =============================================================================

abl_order = {
    "A1_CenterPair": 1,
    "A2_PairInteraction": 2,
    "A3_PairPlusEdge": 3,
    "A4_PairPlusGlobal": 4,
    "A5_Full768": 5,
}


ablation = abl59.copy()

ablation["display_order"] = (
    ablation["variant"]
    .map(
        abl_order
    )
)

ablation = (
    ablation
    .sort_values(
        "display_order"
    )
    .drop(
        columns=[
            "display_order"
        ]
    )
)

ablation.to_csv(
    OUT
    / "60D_FINAL_representation_ablation.csv",
    index=False,
)


inc59.to_csv(
    OUT
    / "60D_FINAL_component_increments.csv",
    index=False,
)


# =============================================================================
# 5. CONTEXTUAL EDGE LINEAGE ROBUSTNESS
# =============================================================================

edge59.to_csv(
    OUT
    / "60E_FINAL_contextual_edge_lineage_bootstrap.csv",
    index=False,
)


# =============================================================================
# 6. INTERNAL CONSISTENCY AUDIT
# Stage57C GenoGramma should match Stage59A A5 Full768.
# =============================================================================

gg57 = comp57[
    comp57["model"]
    ==
    "GenoGramma"
].iloc[0]


full59 = abl59[
    abl59["variant"]
    ==
    "A5_Full768"
].iloc[0]


roc_diff = abs(
    float(
        gg57["ROC_AUC"]
    )
    -
    float(
        full59["ROC_AUC"]
    )
)

pr_diff = abs(
    float(
        gg57["PR_AUC"]
    )
    -
    float(
        full59["PR_AUC"]
    )
)


consistency = pd.DataFrame(
    [
        {
            "check":
                "Stage57C_GenoGramma_vs_Stage59A_Full768",

            "Stage57C_ROC":
                float(
                    gg57["ROC_AUC"]
                ),

            "Stage59A_ROC":
                float(
                    full59["ROC_AUC"]
                ),

            "Absolute_ROC_difference":
                roc_diff,

            "Stage57C_PR":
                float(
                    gg57["PR_AUC"]
                ),

            "Stage59A_PR":
                float(
                    full59["PR_AUC"]
                ),

            "Absolute_PR_difference":
                pr_diff,

            "PASS":
                (
                    roc_diff < 1e-8
                    and
                    pr_diff < 1e-8
                ),
        }
    ]
)


consistency.to_csv(
    OUT
    / "60F_internal_consistency_audit.csv",
    index=False,
)


if not bool(
    consistency.iloc[0]["PASS"]
):
    raise RuntimeError(
        "Stage57C / Stage59A Full768 consistency failure"
    )


# =============================================================================
# 7. FINAL KEY-EVIDENCE TABLE
# =============================================================================

rows = []


# Main all-pairs result
rows.append(
    {
        "Evidence_block":
            "Matched pair-aware benchmark",

        "Analysis":
            "GenoGramma all-pairs",

        "Metric":
            "ROC_AUC",

        "Estimate":
            float(
                gg57["ROC_AUC"]
            ),

        "CI_low":
            np.nan,

        "CI_high":
            np.nan,

        "Interpretation":
            "Highest point estimate among matched pair-aware models",
    }
)

rows.append(
    {
        "Evidence_block":
            "Matched pair-aware benchmark",

        "Analysis":
            "GenoGramma all-pairs",

        "Metric":
            "PR_AUC",

        "Estimate":
            float(
                gg57["PR_AUC"]
            ),

        "CI_low":
            np.nan,

        "CI_high":
            np.nan,

        "Interpretation":
            "Highest point estimate among matched pair-aware models",
    }
)


# Same-strand robustness
gg_same = sum58[
    (
        sum58["model"]
        ==
        "GenoGramma"
    )
    &
    (
        sum58["subset"]
        ==
        "same_strand_only"
    )
].iloc[0]


for metric in [
    "ROC_AUC",
    "PR_AUC",
]:

    rows.append(
        {
            "Evidence_block":
                "Strand shortcut control",

            "Analysis":
                "GenoGramma same-strand-only",

            "Metric":
                metric,

            "Estimate":
                float(
                    gg_same[metric]
                ),

            "CI_low":
                np.nan,

            "CI_high":
                np.nan,

            "Interpretation":
                "Performance retained after removal of opposite-strand easy negatives",
        }
    )


# Lineage macro bootstrap
for metric in [
    "ROC_AUC",
    "PR_AUC",
]:

    q = gg_macro[
        gg_macro["metric"]
        ==
        metric
    ]

    if len(q) == 1:

        r = q.iloc[0]

        rows.append(
            {
                "Evidence_block":
                    "Effective-N robustness",

                "Analysis":
                    "Same-strand ANI95 lineage macro",

                "Metric":
                    metric,

                "Estimate":
                    float(
                        r["lineage_macro_mean"]
                    ),

                "CI_low":
                    float(
                        r["bootstrap_CI_low"]
                    ),

                "CI_high":
                    float(
                        r["bootstrap_CI_high"]
                    ),

                "Interpretation":
                    "54 ANI95 lineages treated as biological units",
            }
        )


# Contextual-edge primary contrasts
for subset in [
    "all_pairs",
    "same_strand_only",
]:

    for contrast in [
        "ContextualEdge_gain_over_pair",
        "ContextualEdge_gain_given_global",
    ]:

        for metric in [
            "ROC_AUC",
            "PR_AUC",
        ]:

            q = edge59[
                (
                    edge59["subset"]
                    ==
                    subset
                )
                &
                (
                    edge59["contrast"]
                    ==
                    contrast
                )
                &
                (
                    edge59["metric"]
                    ==
                    metric
                )
            ]

            if len(q) != 1:
                raise RuntimeError(
                    f"Missing contextual-edge result: "
                    f"{subset}/{contrast}/{metric}"
                )


            r = q.iloc[0]

            rows.append(
                {
                    "Evidence_block":
                        "Contextual-edge ablation",

                    "Analysis":
                        f"{subset}: {contrast}",

                    "Metric":
                        metric,

                    "Estimate":
                        float(
                            r["mean_delta"]
                        ),

                    "CI_low":
                        float(
                            r["CI_low"]
                        ),

                    "CI_high":
                        float(
                            r["CI_high"]
                        ),

                    "Interpretation":
                        (
                            "Positive lineage-level contextual-edge increment"
                            if float(
                                r["CI_low"]
                            ) > 0
                            else
                            "Directionally positive but CI overlaps zero"
                        ),
                }
            )


key = pd.DataFrame(
    rows
)


key.to_csv(
    OUT
    / "60G_FINAL_key_evidence.csv",
    index=False,
)


# =============================================================================
# 8. CLAIM-SAFETY TABLE
# =============================================================================

claim_table = pd.DataFrame(
    [
        {
            "Topic":
                "Model scope",

            "Supported_wording":
                "GenoGramma learns order-sensitive local genomic neighborhood representations.",

            "Avoid":
                "chromosome-scale genomic grammar",
        },

        {
            "Topic":
                "Matched baseline comparison",

            "Supported_wording":
                "GenoGramma achieved the highest point estimates among matched pair-aware baselines.",

            "Avoid":
                "GenoGramma significantly outperformed all baselines",
        },

        {
            "Topic":
                "Transformer comparison",

            "Supported_wording":
                "GenoGramma showed slightly higher point estimates than TransformerSmall.",

            "Avoid":
                "significantly superior to TransformerSmall",
        },

        {
            "Topic":
                "Strand shortcut",

            "Supported_wording":
                "Strand orientation was a strong shortcut, but same-strand-only evaluation retained substantial predictive performance.",

            "Avoid":
                "performance is independent of strand",
        },

        {
            "Topic":
                "Effective sample size",

            "Supported_wording":
                "Robustness was evaluated using 54 ANI95 lineages as biological resampling units.",

            "Avoid":
                "164,302 independent biological samples",
        },

        {
            "Topic":
                "Contextual edge",

            "Supported_wording":
                "The contextual-edge representation provided a modest but consistent lineage-level increment beyond center-gene content and pairwise interactions.",

            "Avoid":
                "contextual edge is the dominant source of predictive performance",
        },

        {
            "Topic":
                "Global context",

            "Supported_wording":
                "Global pooled context did not improve this local operon-pair task.",

            "Avoid":
                "global genomic information is generally uninformative",
        },
    ]
)


claim_table.to_csv(
    OUT
    / "60H_claim_safety_table.csv",
    index=False,
)


# =============================================================================
# 9. PROVENANCE MANIFEST
# =============================================================================

manifest_rows = []


for name, path in FILES.items():

    manifest_rows.append(
        {
            "source":
                name,

            "path":
                str(path),

            "size_bytes":
                int(
                    path.stat().st_size
                ),

            "sha256":
                sha256(
                    path
                ),
        }
    )


manifest = pd.DataFrame(
    manifest_rows
)


manifest.to_csv(
    OUT
    / "60I_source_provenance_manifest.csv",
    index=False,
)


# =============================================================================
# 10. FREEZE JSON
# =============================================================================

freeze = {

    "status":
        "STAGE60_FINAL_OPERON_EVIDENCE_FROZEN",

    "formal_model_name":
        "GenoGramma",

    "task":
        "DOOR2-derived adjacent-gene operon-status prediction",

    "scope":
        "order-sensitive local genomic neighborhood representation",

    "main_evaluation":
        "ANI95 lineage-blocked OOF-5CV",

    "formal_examples":
        164302,

    "genomes":
        96,

    "ANI95_lineages":
        54,

    "target_pair_zero_based":[
        31,
        32,
    ],

    "important_controls":[
        "matched pair-aware baselines",
        "same-strand-only evaluation",
        "StrandMatch-only baseline",
        "equal-lineage bootstrap",
        "768D representation ablation",
        "contextual-edge paired lineage bootstrap",
    ],

    "no_new_model_fitting":
        True,

    "no_posthoc_retuning":
        True,

    "main_formal_representation":
        "A5 Full768 / Stage56 preregistered GenoGramma pair-aware representation",

    "important_ablation_note":
        (
            "A3 PairPlusEdge achieved a higher point estimate than Full768, "
            "but Full768 remains the formal preregistered downstream representation. "
            "A3 is interpreted only as an ablation result."
        ),

    "stage57C_stage59A_full_consistency":
        True,
}


(
    OUT
    / "60_FINAL_RESULT_FREEZE.json"
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
print("===== MAIN MATCHED PAIR-AWARE COMPARISON =====")

print(
    main[
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


print()
print("===== SAME-STRAND GENOGRAMMA =====")

print(
    pd.DataFrame(
        [
            {
                "ROC_AUC":
                    gg_same["ROC_AUC"],

                "PR_AUC":
                    gg_same["PR_AUC"],

                "Balanced_Accuracy":
                    gg_same["Balanced_Accuracy"],

                "F1":
                    gg_same["F1"],

                "MCC":
                    gg_same["MCC"],
            }
        ]
    ).to_string(
        index=False
    )
)


print()
print("===== CONTEXTUAL EDGE LINEAGE BOOTSTRAP =====")

print(
    edge59[
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
    )
)


print()
print("===== INTERNAL CONSISTENCY =====")

print(
    consistency.to_string(
        index=False
    )
)


print()
print("[PASS] no new model training")
print("[PASS] no post-hoc retuning")
print("[PASS] formal Full768 result preserved")
print("[PASS] source SHA256 manifest written")
print("[OUTPUT]", OUT)

print(
    "STAGE60_FINAL_OPERON_EVIDENCE_FROZEN"
)

print("=" * 118)

