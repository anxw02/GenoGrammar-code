from __future__ import annotations

import os

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)

from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# PATHS
# =============================================================================

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

META=(
    BASE / "runtime_assets" / "source_tree" / "new_paper_data"
    / "08_stage50_order_dependent_downstream"
    / "07_pair_to_genogrammar_alignment"
    / "14B_STAGE50_operon_model_input_metadata_with_row_index.csv.gz"
)

REP_ROOT=(
    ROOT
    / "03_stage55_pairaware_representations"
)

REP_INDEX=(
    REP_ROOT
    / "55_pairaware_representation_index.csv"
)

SPLIT_JSON=(
    ROOT
    / "08_stage57C_matched_comparison"
    / "57C_inner_lineage_splits.json"
)

OUT=(
    ROOT
    / "11_stage59A_768d_ablation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONSTANTS
# =============================================================================

N=164302
DIM=768

BLOCKS={
    "left":
        (0,128),

    "right":
        (128,256),

    "diff":
        (256,384),

    "product":
        (384,512),

    "edge":
        (512,640),

    "global":
        (640,768),
}


VARIANTS={

    "A1_CenterPair":[
        "left",
        "right",
    ],

    "A2_PairInteraction":[
        "left",
        "right",
        "diff",
        "product",
    ],

    "A3_PairPlusEdge":[
        "left",
        "right",
        "diff",
        "product",
        "edge",
    ],

    "A4_PairPlusGlobal":[
        "left",
        "right",
        "diff",
        "product",
        "global",
    ],

    "A5_Full768":[
        "left",
        "right",
        "diff",
        "product",
        "edge",
        "global",
    ],
}


LR=0.001
WEIGHT_DECAY=0.0001
DROPOUT=0.20

MAX_EPOCHS=100
MIN_EPOCHS=10
PATIENCE=15

BATCH_SIZE=4096

SEEDS={
    1:42,
    2:142,
    3:242,
    4:342,
    5:442,
}


# =============================================================================
# UTILS
# =============================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def metrics(y,p):

    y=np.asarray(
        y,
        dtype=int,
    )

    p=np.asarray(
        p,
        dtype=float,
    )

    pred=(
        p >= 0.5
    ).astype(int)

    return {
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

        "Balanced_Accuracy":
            float(
                balanced_accuracy_score(
                    y,pred
                )
            ),

        "F1":
            float(
                f1_score(
                    y,pred
                )
            ),

        "MCC":
            float(
                matthews_corrcoef(
                    y,pred
                )
            ),
    }


def make_mask(
    active_blocks,
):

    mask=np.zeros(
        DIM,
        dtype=np.float32,
    )

    for name in active_blocks:

        a,b=BLOCKS[
            name
        ]

        mask[
            a:b
        ]=1.0

    return mask


# =============================================================================
# LOAD FORMAL DATA
# =============================================================================

z=np.load(
    INPUT_NPZ
)

y=np.asarray(
    z[
        "label_index"
    ],
    dtype=np.int64,
)

outer_fold=np.asarray(
    z[
        "outer_fold"
    ],
    dtype=np.int64,
)

row_index=np.asarray(
    z[
        "row_index"
    ],
    dtype=np.int64,
)


meta=pd.read_csv(
    META
)

meta=(
    meta
    .set_index(
        "row_index",
        drop=False,
    )
    .loc[
        row_index
    ]
    .reset_index(
        drop=True
    )
)

genome=(
    meta[
        "strain_entity_id"
    ]
    .astype(str)
    .to_numpy()
)

lineage=(
    meta[
        "lineage_95_block"
    ]
    .astype(str)
    .to_numpy()
)


if len(y) != N:
    raise RuntimeError(
        f"N={len(y)} != {N}"
    )


# =============================================================================
# LOAD STAGE57C FROZEN INNER SPLITS
# =============================================================================

split_def=json.loads(
    SPLIT_JSON.read_text()
)

splits={}


for fold in range(
    1,
    6,
):

    d=split_def[
        str(fold)
    ]

    train_lineages=set(
        map(
            str,
            d[
                "train_lineages"
            ],
        )
    )

    val_lineages=set(
        map(
            str,
            d[
                "val_lineages"
            ],
        )
    )


    te=np.where(
        outer_fold == fold
    )[0]


    outer_train=np.where(
        outer_fold != fold
    )[0]


    tr=np.asarray(
        [
            i
            for i in outer_train
            if str(
                lineage[i]
            )
            in train_lineages
        ],
        dtype=np.int64,
    )


    va=np.asarray(
        [
            i
            for i in outer_train
            if str(
                lineage[i]
            )
            in val_lineages
        ],
        dtype=np.int64,
    )


    if len(
        set(
            lineage[tr]
        )
        &
        set(
            lineage[va]
        )
    ):
        raise RuntimeError(
            f"F{fold}: train-val leakage"
        )


    if len(
        set(
            lineage[tr]
        )
        &
        set(
            lineage[te]
        )
    ):
        raise RuntimeError(
            f"F{fold}: train-test leakage"
        )


    splits[
        fold
    ]=(
        tr,
        va,
        te,
    )


# =============================================================================
# RESOLVE STAGE55 REPRESENTATIONS
# =============================================================================

idx=pd.read_csv(
    REP_INDEX
)


def resolve_representation(
    fold,
):

    fold_cols=[
        x
        for x in [
            "fold",
            "outer_fold",
            "formal_outer_fold",
        ]
        if x in idx.columns
    ]

    if not fold_cols:
        raise RuntimeError(
            "Cannot resolve fold column in Stage55 index"
        )


    fc=fold_cols[0]

    q=idx[
        idx[
            fc
        ].astype(int)
        ==
        fold
    ]


    if len(q)<1:
        raise RuntimeError(
            f"Stage55 fold {fold} missing"
        )


    r=q.iloc[0]


    for col in [
        "feature_file",
        "representation_file",
        "representation_path",
        "output_file",
        "npy_file",
        "path",
    ]:

        if (
            col in idx.columns
            and
            pd.notna(
                r[col]
            )
        ):

            p=Path(
                str(
                    r[col]
                )
            )

            if p.exists():
                return p


    # fallback: inspect fold directory
    candidates=[]

    for p in REP_ROOT.rglob(
        "*.npy"
    ):

        if (
            f"fold_{fold}"
            in str(p)
            or
            f"fold{fold}"
            in str(p).lower()
        ):

            try:
                a=np.load(
                    p,
                    mmap_mode="r",
                )

                if a.shape == (
                    N,
                    DIM,
                ):
                    candidates.append(
                        p
                    )

            except Exception:
                pass


    if len(candidates) != 1:

        raise RuntimeError(
            f"F{fold}: could not uniquely resolve "
            f"Stage55 representation. "
            f"Candidates={candidates}"
        )


    return candidates[0]


rep_files={
    fold:
        resolve_representation(
            fold
        )
    for fold
    in range(
        1,
        6,
    )
}


# =============================================================================
# MODEL
# =============================================================================

class Probe(nn.Module):

    def __init__(self):

        super().__init__()

        self.net=nn.Sequential(

            nn.Linear(
                768,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                128,
                2,
            ),
        )


    def forward(
        self,
        x,
    ):

        return self.net(
            x
        )


# =============================================================================
# PREDICT
# =============================================================================

def predict(
    model,
    X,
    indices,
    mean,
    sd,
    mask,
    device,
):

    model.eval()

    out=[]

    with torch.inference_mode():

        for start in range(
            0,
            len(indices),
            BATCH_SIZE,
        ):

            ids=indices[
                start:
                start+BATCH_SIZE
            ]

            xb=np.asarray(
                X[
                    ids
                ],
                dtype=np.float32,
            )

            xb=(
                xb
                -
                mean
            )/sd

            xb *= mask

            xt=torch.from_numpy(
                xb
            ).to(
                device
            )

            prob=torch.softmax(
                model(
                    xt
                ),
                dim=1,
            )[:,1]

            out.append(
                prob.cpu().numpy()
            )


    return np.concatenate(
        out
    )


# =============================================================================
# FREEZE PROTOCOL BEFORE FIT
# =============================================================================

protocol={

    "status":
        "STAGE59A_ABLATION_PROTOCOL_FROZEN_BEFORE_FIT",

    "formal_model_name":
        "GenoGramma",

    "representation_dimension":
        768,

    "block_order":[
        "left",
        "right",
        "absolute_difference",
        "elementwise_product",
        "contextual_edge",
        "global_context",
    ],

    "block_dimension":
        128,

    "variants":
        VARIANTS,

    "masking_strategy":
        "Inactive blocks are zeroed after train-only z-scoring. "
        "All variants retain the identical 768->128->2 probe.",

    "outer_split":
        "frozen ANI95 5-fold",

    "inner_split":
        "exact Stage57C frozen lineage split",

    "genome_balanced_training":
        True,

    "encoder_training":
        False,

    "direct_operon_features":
        False,

    "optimizer":
        "AdamW",

    "lr":
        LR,

    "weight_decay":
        WEIGHT_DECAY,

    "max_epochs":
        MAX_EPOCHS,

    "min_epochs":
        MIN_EPOCHS,

    "patience":
        PATIENCE,

    "selection":
        "0.5*(inner ROC-AUC + inner PR-AUC)",
}


(
    OUT
    /
    "59A_ablation_protocol_freeze.json"
).write_text(
    json.dumps(
        protocol,
        indent=2,
    )
)


# =============================================================================
# TRAIN
# =============================================================================

device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)


print(
    "="*112,
    flush=True,
)

print(
    "STAGE59A — GENOGRAMMA 768D COMPONENT ABLATION",
    flush=True,
)

print(
    "FROZEN REPRESENTATIONS | IDENTICAL 768→128→2 PROBE | "
    "ANI95 BLOCKED 5-FOLD",
    flush=True,
)

print(
    "="*112,
    flush=True,
)

print(
    "[PASS] examples =",
    N,
    flush=True,
)

print(
    "[PASS] ANI95 lineages =",
    len(
        np.unique(
            lineage
        )
    ),
    flush=True,
)

print(
    "[PASS] exact Stage57C inner splits reused",
    flush=True,
)

print(
    "[INFO] device =",
    device,
    flush=True,
)


fold_rows=[]
pred_rows=[]


for variant,active in VARIANTS.items():

    mask=make_mask(
        active
    )

    print(
        "\n"
        +
        "#"*112,
        flush=True,
    )

    print(
        variant,
        "| active blocks =",
        ",".join(
            active
        ),
        flush=True,
    )

    print(
        "#"*112,
        flush=True,
    )


    for fold in range(
        1,
        6,
    ):

        set_seed(
            SEEDS[
                fold
            ]
        )


        tr,va,te=splits[
            fold
        ]


        X=np.load(
            rep_files[
                fold
            ],
            mmap_mode="r",
        )


        if X.shape != (
            N,
            DIM,
        ):
            raise RuntimeError(
                f"F{fold}: "
                f"representation shape={X.shape}"
            )


        train_raw=np.asarray(
            X[
                tr
            ],
            dtype=np.float32,
        )


        mean=(
            train_raw
            .mean(
                axis=0,
                dtype=np.float64,
            )
            .astype(
                np.float32
            )
        )


        sd=(
            train_raw
            .std(
                axis=0,
                dtype=np.float64,
            )
            .astype(
                np.float32
            )
        )


        sd[
            sd < 1e-6
        ]=1.0


        train_x=(
            train_raw
            -
            mean
        )/sd

        train_x *= mask

        del train_raw


        train_genomes=genome[
            tr
        ]


        ug,cnt=np.unique(
            train_genomes,
            return_counts=True,
        )


        count_map={
            g:int(c)
            for g,c
            in zip(
                ug,
                cnt,
            )
        }


        weights=np.asarray(
            [
                1.0
                /
                count_map[g]
                for g
                in train_genomes
            ],
            dtype=np.float32,
        )


        weights *= (
            len(weights)
            /
            weights.sum()
        )


        ds=TensorDataset(

            torch.from_numpy(
                train_x
            ),

            torch.from_numpy(
                y[
                    tr
                ].astype(
                    np.int64
                )
            ),

            torch.from_numpy(
                weights
            ),
        )


        g=torch.Generator()

        g.manual_seed(
            SEEDS[
                fold
            ]
        )


        loader=DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            generator=g,
        )


        model=Probe().to(
            device
        )


        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )


        best_score=-np.inf
        best_state=None
        best_epoch=None
        stale=0


        for epoch in range(
            1,
            MAX_EPOCHS+1,
        ):

            model.train()


            for xb,yb,wb in loader:

                xb=xb.to(
                    device,
                    non_blocking=True,
                )

                yb=yb.to(
                    device,
                    non_blocking=True,
                )

                wb=wb.to(
                    device,
                    non_blocking=True,
                )


                optimizer.zero_grad(
                    set_to_none=True
                )


                logits=model(
                    xb
                )


                loss0=F.cross_entropy(
                    logits,
                    yb,
                    reduction="none",
                )


                loss=(
                    loss0*wb
                ).sum()/wb.sum()


                loss.backward()
                optimizer.step()


            pv=predict(
                model,
                X,
                va,
                mean,
                sd,
                mask,
                device,
            )


            vm=metrics(
                y[
                    va
                ],
                pv,
            )


            score=0.5*(
                vm[
                    "ROC_AUC"
                ]
                +
                vm[
                    "PR_AUC"
                ]
            )


            if score > best_score + 1e-12:

                best_score=score

                best_epoch=epoch

                best_state={
                    k:
                        v.detach()
                        .cpu()
                        .clone()
                    for k,v
                    in model.state_dict().items()
                }

                stale=0

            else:

                stale += 1


            if (
                epoch==1
                or
                epoch%5==0
                or
                stale==0
            ):

                print(
                    f"[{variant}/F{fold}] "
                    f"epoch={epoch:03d} "
                    f"valROC={vm['ROC_AUC']:.4f} "
                    f"valPR={vm['PR_AUC']:.4f} "
                    f"best={best_score:.4f} "
                    f"stale={stale}",
                    flush=True,
                )


            if (
                epoch >= MIN_EPOCHS
                and
                stale >= PATIENCE
            ):
                break


        model.load_state_dict(
            best_state,
            strict=True,
        )


        pt=predict(
            model,
            X,
            te,
            mean,
            sd,
            mask,
            device,
        )


        tm=metrics(
            y[
                te
            ],
            pt,
        )


        fold_rows.append(
            {
                "variant":
                    variant,

                "fold":
                    fold,

                "active_blocks":
                    ",".join(
                        active
                    ),

                "active_dimension":
                    int(
                        mask.sum()
                    ),

                "best_epoch":
                    best_epoch,

                "best_val_score":
                    best_score,

                **tm,
            }
        )


        pred_rows.append(
            pd.DataFrame(
                {
                    "row_index":
                        row_index[
                            te
                        ],

                    "variant":
                        variant,

                    "fold":
                        fold,

                    "y_true":
                        y[
                            te
                        ],

                    "prob_positive":
                        pt,

                    "strain_entity_id":
                        genome[
                            te
                        ],

                    "lineage_95_block":
                        lineage[
                            te
                        ],
                }
            )
        )


        print(
            f"[OUTER {variant}/F{fold}] "
            f"ROC={tm['ROC_AUC']:.4f} "
            f"PR={tm['PR_AUC']:.4f} "
            f"BalAcc={tm['Balanced_Accuracy']:.4f} "
            f"F1={tm['F1']:.4f} "
            f"MCC={tm['MCC']:.4f}",
            flush=True,
        )


        del X
        del train_x
        del ds
        del loader
        del model


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# SAVE + POOLED
# =============================================================================

fold_df=pd.DataFrame(
    fold_rows
)

fold_df.to_csv(
    OUT
    /
    "59A_fold_metrics.csv",
    index=False,
)


pred_df=pd.concat(
    pred_rows,
    ignore_index=True,
)

pred_df.to_csv(
    OUT
    /
    "59A_oof_predictions.csv.gz",
    index=False,
    compression="gzip",
)


summary=[]


for variant in VARIANTS:

    q=pred_df[
        pred_df[
            "variant"
        ]
        ==
        variant
    ]


    if len(q) != N:
        raise RuntimeError(
            f"{variant}: OOF N={len(q)}"
        )


    m=metrics(
        q[
            "y_true"
        ],
        q[
            "prob_positive"
        ],
    )


    f=fold_df[
        fold_df[
            "variant"
        ]
        ==
        variant
    ]


    summary.append(
        {
            "variant":
                variant,

            "active_dimension":
                int(
                    make_mask(
                        VARIANTS[
                            variant
                        ]
                    ).sum()
                ),

            **m,

            "ROC_fold_mean":
                f[
                    "ROC_AUC"
                ].mean(),

            "ROC_fold_sd":
                f[
                    "ROC_AUC"
                ].std(
                    ddof=1
                ),

            "PR_fold_mean":
                f[
                    "PR_AUC"
                ].mean(),

            "PR_fold_sd":
                f[
                    "PR_AUC"
                ].std(
                    ddof=1
                ),
        }
    )


summary_df=pd.DataFrame(
    summary
)


full=summary_df[
    summary_df[
        "variant"
    ]
    ==
    "A5_Full768"
].iloc[0]


summary_df[
    "Delta_ROC_vs_Full"
]=(
    summary_df[
        "ROC_AUC"
    ]
    -
    full[
        "ROC_AUC"
    ]
)


summary_df[
    "Delta_PR_vs_Full"
]=(
    summary_df[
        "PR_AUC"
    ]
    -
    full[
        "PR_AUC"
    ]
)


summary_df.to_csv(
    OUT
    /
    "59A_FINAL_ablation_summary.csv",
    index=False,
)


# =============================================================================
# COMPONENT INCREMENTS
# =============================================================================

S={
    r[
        "variant"
    ]:
        r
    for _,r
    in summary_df.iterrows()
}


increment_rows=[

    {
        "contrast":
            "Interaction_gain_over_center",

        "Delta_ROC":
            S[
                "A2_PairInteraction"
            ][
                "ROC_AUC"
            ]
            -
            S[
                "A1_CenterPair"
            ][
                "ROC_AUC"
            ],

        "Delta_PR":
            S[
                "A2_PairInteraction"
            ][
                "PR_AUC"
            ]
            -
            S[
                "A1_CenterPair"
            ][
                "PR_AUC"
            ],
    },

    {
        "contrast":
            "ContextualEdge_gain_over_pair",

        "Delta_ROC":
            S[
                "A3_PairPlusEdge"
            ][
                "ROC_AUC"
            ]
            -
            S[
                "A2_PairInteraction"
            ][
                "ROC_AUC"
            ],

        "Delta_PR":
            S[
                "A3_PairPlusEdge"
            ][
                "PR_AUC"
            ]
            -
            S[
                "A2_PairInteraction"
            ][
                "PR_AUC"
            ],
    },

    {
        "contrast":
            "Global_gain_over_pair",

        "Delta_ROC":
            S[
                "A4_PairPlusGlobal"
            ][
                "ROC_AUC"
            ]
            -
            S[
                "A2_PairInteraction"
            ][
                "ROC_AUC"
            ],

        "Delta_PR":
            S[
                "A4_PairPlusGlobal"
            ][
                "PR_AUC"
            ]
            -
            S[
                "A2_PairInteraction"
            ][
                "PR_AUC"
            ],
    },

    {
        "contrast":
            "Global_gain_given_edge",

        "Delta_ROC":
            S[
                "A5_Full768"
            ][
                "ROC_AUC"
            ]
            -
            S[
                "A3_PairPlusEdge"
            ][
                "ROC_AUC"
            ],

        "Delta_PR":
            S[
                "A5_Full768"
            ][
                "PR_AUC"
            ]
            -
            S[
                "A3_PairPlusEdge"
            ][
                "PR_AUC"
            ],
    },

    {
        "contrast":
            "ContextualEdge_gain_given_global",

        "Delta_ROC":
            S[
                "A5_Full768"
            ][
                "ROC_AUC"
            ]
            -
            S[
                "A4_PairPlusGlobal"
            ][
                "ROC_AUC"
            ],

        "Delta_PR":
            S[
                "A5_Full768"
            ][
                "PR_AUC"
            ]
            -
            S[
                "A4_PairPlusGlobal"
            ][
                "PR_AUC"
            ],
    },
]


increment_df=pd.DataFrame(
    increment_rows
)


increment_df.to_csv(
    OUT
    /
    "59A_component_increment_summary.csv",
    index=False,
)


# =============================================================================
# FINAL
# =============================================================================

print()
print(
    "="*112,
    flush=True,
)

print(
    "STAGE59A — FINAL 768D ABLATION",
    flush=True,
)

print(
    "="*112,
    flush=True,
)


print(
    summary_df[
        [
            "variant",
            "active_dimension",
            "ROC_AUC",
            "PR_AUC",
            "Balanced_Accuracy",
            "F1",
            "MCC",
            "Delta_ROC_vs_Full",
            "Delta_PR_vs_Full",
        ]
    ]
    .sort_values(
        "ROC_AUC",
        ascending=False,
    )
    .to_string(
        index=False
    ),
    flush=True,
)


print()
print(
    "===== COMPONENT INCREMENTS =====",
    flush=True,
)


print(
    increment_df.to_string(
        index=False
    ),
    flush=True,
)


print()
print(
    "[PASS] encoder frozen",
    flush=True,
)

print(
    "[PASS] identical 768→128→2 head across all variants",
    flush=True,
)

print(
    "[PASS] inactive representation blocks zero-masked",
    flush=True,
)

print(
    "[PASS] exact Stage57C inner lineage splits reused",
    flush=True,
)

print(
    "[OUTPUT]",
    OUT,
    flush=True,
)

print(
    "STAGE59A_GENOGRAMMA_768D_ABLATION_COMPLETE",
    flush=True,
)
