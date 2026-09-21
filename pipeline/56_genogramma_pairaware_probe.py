from __future__ import annotations

import os

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# FIXED PATHS
# =============================================================================

BASE = Path(__file__).resolve().parents[1]

ROOT=(
    Path(os.environ.get("GENOGRAMMA_RESULT_ROOT", str(BASE / "runtime_assets" / "source_tree" / "new_paper_data" / "10_genogramma_pairaware")))
)

REP_ROOT=(
    ROOT
    / "03_stage55_pairaware_representations"
)

REP_INDEX=(
    REP_ROOT
    / "55_pairaware_representation_index.csv"
)

REP_FREEZE=(
    REP_ROOT
    / "55_pairaware_representation_freeze.json"
)

PRE=(
    ROOT
    / "02_pairaware_preregistration.json"
)

INPUT_NPZ=(
    BASE / "runtime_assets" / "source_tree" / "new_paper_data"
    / "08_stage50_order_dependent_downstream"
    / "07_pair_to_genogrammar_alignment"
    / "13_STAGE50_operon_formal_model_inputs.npz"
)

META_CANDIDATES=[
    (
        BASE / "runtime_assets" / "source_tree" / "new_paper_data"
        / "08_stage50_order_dependent_downstream"
        / "07_pair_to_genogrammar_alignment"
        / "14B_STAGE50_operon_model_input_metadata_with_row_index.csv.gz"
    ),
    (
        BASE / "runtime_assets" / "source_tree" / "new_paper_data"
        / "08_stage50_order_dependent_downstream"
        / "07_pair_to_genogrammar_alignment"
        / "14_STAGE50_operon_model_input_metadata.csv.gz"
    ),
]

OUT=(
    ROOT
    / "04_stage56_pairaware_probe"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN PROTOCOL
# =============================================================================

N=164302
REP_DIM=768
HIDDEN_DIM=128
N_CLASSES=2

DROPOUT=0.20

LR=0.001
WEIGHT_DECAY=0.0001

MAX_EPOCHS=100
MIN_EPOCHS=10
PATIENCE=15

INNER_VAL_FRACTION=0.15

BATCH_SIZE=4096

FOLD_SEEDS={
    1:42,
    2:142,
    3:242,
    4:342,
    5:442,
}

EXPECTED_INPUT_SHA=(
    "978e233098e4ccda12460e821d01fb076"
    "26f909d8afd4dafa2737c867081ab3e"
)


# =============================================================================
# UTILITIES
# =============================================================================

def sha256(path: Path) -> str:
    h=hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda:f.read(1024*1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def metrics(y, prob):

    y=np.asarray(
        y,
        dtype=np.int64,
    )

    prob=np.asarray(
        prob,
        dtype=np.float64,
    )

    pred=(
        prob >= 0.5
    ).astype(np.int64)

    tn,fp,fn,tp=confusion_matrix(
        y,
        pred,
        labels=[0,1],
    ).ravel()

    specificity=(
        tn/(tn+fp)
        if (tn+fp)>0
        else float("nan")
    )

    return {
        "ROC_AUC":
            float(
                roc_auc_score(
                    y,
                    prob,
                )
            ),

        "PR_AUC":
            float(
                average_precision_score(
                    y,
                    prob,
                )
            ),

        "Accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "Balanced_Accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "Precision":
            float(
                precision_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "Recall":
            float(
                recall_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "Specificity":
            float(
                specificity
            ),

        "F1":
            float(
                f1_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "MCC":
            float(
                matthews_corrcoef(
                    y,
                    pred,
                )
            ),

        "TN":int(tn),
        "FP":int(fp),
        "FN":int(fn),
        "TP":int(tp),
    }


def predict_prob(
    model,
    X,
    mean,
    sd,
    device,
):

    model.eval()

    out=[]

    with torch.inference_mode():

        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):

            end=min(
                start+BATCH_SIZE,
                len(X),
            )

            xb=np.asarray(
                X[start:end],
                dtype=np.float32,
            )

            xb=(
                xb-mean
            )/sd

            xb=torch.from_numpy(
                xb
            ).to(
                device
            )

            logits=model(
                xb
            )

            prob=torch.softmax(
                logits,
                dim=1,
            )[:,1]

            out.append(
                prob.cpu().numpy()
            )

    return np.concatenate(
        out
    )


# =============================================================================
# FORMAL HEAD
# =============================================================================

class GenoGrammaPairHead(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1=nn.Linear(
            REP_DIM,
            HIDDEN_DIM,
        )

        self.dropout=nn.Dropout(
            DROPOUT
        )

        self.fc2=nn.Linear(
            HIDDEN_DIM,
            N_CLASSES,
        )

    def forward(
        self,
        x,
    ):

        x=self.fc1(
            x
        )

        x=F.gelu(
            x
        )

        x=self.dropout(
            x
        )

        return self.fc2(
            x
        )


# =============================================================================
# LOAD + AUDIT INPUTS
# =============================================================================

print(
    "="*112,
    flush=True,
)

print(
    "STAGE56 — GenoGramma PAIR-AWARE DOWNSTREAM EVALUATION",
    flush=True,
)

print(
    "ANI95 LINEAGE-BLOCKED OOF-5CV | GENOME-BALANCED TRAINING | "
    "NO DIRECT OPERON FEATURES",
    flush=True,
)

print(
    "="*112,
    flush=True,
)


if not PRE.exists():
    raise RuntimeError(
        f"Missing preregistration: {PRE}"
    )

pre=json.loads(
    PRE.read_text()
)

if (
    pre.get("status")
    !=
    "GENOGRAMMA_PAIRAWARE_PREREGISTERED_BEFORE_DOWNSTREAM_EVALUATION"
):
    raise RuntimeError(
        "Pair-aware preregistration status mismatch"
    )


if not REP_FREEZE.exists():
    raise RuntimeError(
        f"Missing Stage55 freeze: {REP_FREEZE}"
    )

rep_freeze=json.loads(
    REP_FREEZE.read_text()
)

if (
    rep_freeze.get("status")
    !=
    "ALL_5_GENOGRAMMA_PAIRAWARE_REPRESENTATIONS_PASS"
):
    raise RuntimeError(
        "Stage55 freeze status mismatch"
    )


actual_input_sha=sha256(
    INPUT_NPZ
)

if (
    actual_input_sha
    !=
    EXPECTED_INPUT_SHA
):
    raise RuntimeError(
        "Formal input SHA mismatch"
    )


with np.load(
    INPUT_NPZ
) as z:

    required={
        "label_index",
        "outer_fold",
        "row_index",
    }

    if not required.issubset(
        z.files
    ):
        raise RuntimeError(
            f"NPZ keys missing: "
            f"{sorted(required-set(z.files))}"
        )

    y=np.asarray(
        z["label_index"],
        dtype=np.int64,
    )

    outer_fold=np.asarray(
        z["outer_fold"],
        dtype=np.int64,
    )

    row_index=np.asarray(
        z["row_index"],
        dtype=np.int64,
    )


if len(y) != N:
    raise RuntimeError(
        f"N={len(y)} != {N}"
    )

if sorted(
    np.unique(
        y
    ).tolist()
) != [0,1]:
    raise RuntimeError(
        "Unexpected labels"
    )

if sorted(
    np.unique(
        outer_fold
    ).tolist()
) != [1,2,3,4,5]:
    raise RuntimeError(
        "Unexpected outer folds"
    )


# =============================================================================
# METADATA
# =============================================================================

META=None

for candidate in META_CANDIDATES:
    if candidate.exists():
        META=candidate
        break

if META is None:
    raise RuntimeError(
        "Cannot find formal Stage50 metadata"
    )

meta=pd.read_csv(
    META
)

required_meta={
    "strain_entity_id",
    "lineage_95_block",
}

if not required_meta.issubset(
    meta.columns
):
    raise RuntimeError(
        f"Metadata missing: "
        f"{sorted(required_meta-set(meta.columns))}"
    )


if "row_index" in meta.columns:

    if meta["row_index"].duplicated().any():
        raise RuntimeError(
            "metadata row_index duplicated"
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

else:

    if len(meta) != N:
        raise RuntimeError(
            "Metadata has no row_index "
            "and N mismatch"
        )

    meta=meta.reset_index(
        drop=True
    )


if len(meta) != N:
    raise RuntimeError(
        "metadata N mismatch"
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


print(
    "[PASS] formal input SHA =",
    actual_input_sha,
    flush=True,
)

print(
    "[PASS] examples / labels / folds =",
    N,
    sorted(np.unique(y).tolist()),
    sorted(np.unique(outer_fold).tolist()),
    flush=True,
)

print(
    "[PASS] genomes / lineages =",
    len(np.unique(genome)),
    "/",
    len(np.unique(lineage)),
    flush=True,
)

print(
    "[PASS] metadata =",
    META,
    flush=True,
)


# =============================================================================
# REPRESENTATION INDEX
# =============================================================================

ri=pd.read_csv(
    REP_INDEX
)

if len(ri) != 5:
    raise RuntimeError(
        f"Expected 5 Stage55 representations, "
        f"got {len(ri)}"
    )

if (
    set(
        ri["model"]
        .astype(str)
        .tolist()
    )
    !=
    {"GenoGramma"}
):
    raise RuntimeError(
        "Formal model name must be GenoGramma"
    )

if sorted(
    ri["fold"]
    .astype(int)
    .tolist()
) != [1,2,3,4,5]:
    raise RuntimeError(
        "Representation fold mismatch"
    )


rep_map={}

for _,r in ri.iterrows():

    fold=int(
        r["fold"]
    )

    p=Path(
        r[
            "representation_file"
        ]
    )

    if not p.exists():
        raise RuntimeError(
            f"Missing representation: {p}"
        )

    if (
        sha256(p)
        !=
        str(
            r[
                "representation_sha256"
            ]
        )
    ):
        raise RuntimeError(
            f"Representation SHA mismatch fold {fold}"
        )

    arr=np.load(
        p,
        mmap_mode="r",
    )

    if arr.shape != (
        N,
        REP_DIM,
    ):
        raise RuntimeError(
            f"Fold {fold}: "
            f"representation shape={arr.shape}"
        )

    rep_map[
        fold
    ]=p


print(
    "[PASS] Stage55 frozen representations = 5 / 5",
    flush=True,
)

print(
    "[PASS] representation dimension =",
    REP_DIM,
    flush=True,
)


# =============================================================================
# FREEZE INNER SPLITS BEFORE TRAINING
# =============================================================================

inner_splits={}

split_json={}

for fold in range(
    1,
    6,
):

    seed=FOLD_SEEDS[
        fold
    ]

    outer_train_idx=np.where(
        outer_fold != fold
    )[0]

    outer_test_idx=np.where(
        outer_fold == fold
    )[0]

    gss=GroupShuffleSplit(
        n_splits=1,
        test_size=INNER_VAL_FRACTION,
        random_state=seed,
    )

    tr_rel,va_rel=next(
        gss.split(
            np.zeros(
                len(
                    outer_train_idx
                )
            ),
            y[
                outer_train_idx
            ],
            groups=lineage[
                outer_train_idx
            ],
        )
    )

    tr=outer_train_idx[
        tr_rel
    ]

    va=outer_train_idx[
        va_rel
    ]

    te=outer_test_idx


    train_lineages=set(
        lineage[
            tr
        ]
    )

    val_lineages=set(
        lineage[
            va
        ]
    )

    test_lineages=set(
        lineage[
            te
        ]
    )

    if (
        train_lineages
        &
        val_lineages
    ):
        raise RuntimeError(
            f"Fold {fold}: train/val lineage leakage"
        )

    if (
        train_lineages
        &
        test_lineages
    ):
        raise RuntimeError(
            f"Fold {fold}: train/test lineage leakage"
        )

    if (
        val_lineages
        &
        test_lineages
    ):
        raise RuntimeError(
            f"Fold {fold}: val/test lineage leakage"
        )


    inner_splits[
        fold
    ]=(
        tr,
        va,
        te,
    )


    split_json[
        str(fold)
    ]={
        "seed":
            seed,

        "train_n":
            int(len(tr)),

        "val_n":
            int(len(va)),

        "test_n":
            int(len(te)),

        "train_lineages":
            sorted(
                train_lineages
            ),

        "val_lineages":
            sorted(
                val_lineages
            ),

        "test_lineages":
            sorted(
                test_lineages
            ),
    }


(
    OUT
    / "01_inner_lineage_splits.json"
).write_text(
    json.dumps(
        split_json,
        indent=2,
    )
)


print(
    "[PASS] deterministic lineage-blocked "
    "inner validation splits frozen",
    flush=True,
)


# =============================================================================
# DEVICE
# =============================================================================

device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

print(
    "[INFO] device =",
    device,
    flush=True,
)


# =============================================================================
# FORMAL 5-FOLD TRAINING
# =============================================================================

oof_prob=np.full(
    N,
    np.nan,
    dtype=np.float64,
)

oof_pred=np.full(
    N,
    -1,
    dtype=np.int64,
)

fold_rows=[]


for fold in range(
    1,
    6,
):

    print(
        "\n"
        +
        "="*100,
        flush=True,
    )

    print(
        f"GenoGramma | outer fold {fold}/5",
        flush=True,
    )

    print(
        "="*100,
        flush=True,
    )


    seed=FOLD_SEEDS[
        fold
    ]

    set_seed(
        seed
    )

    tr,va,te=inner_splits[
        fold
    ]


    X=np.load(
        rep_map[
            fold
        ],
        mmap_mode="r",
    )


    # -------------------------------------------------------------------------
    # TRAIN-ONLY Z-SCORE
    # -------------------------------------------------------------------------

    Xtr_raw=np.asarray(
        X[
            tr
        ],
        dtype=np.float32,
    )

    mean=(
        Xtr_raw
        .mean(
            axis=0,
            dtype=np.float64,
        )
        .astype(
            np.float32
        )
    )

    sd=(
        Xtr_raw
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


    Xtr=(
        Xtr_raw
        -
        mean
    )/sd

    del Xtr_raw


    # -------------------------------------------------------------------------
    # GENOME-BALANCED SAMPLE WEIGHTS
    #
    # Every training genome contributes equal total weight.
    # No class weighting is applied.
    # -------------------------------------------------------------------------

    train_genomes=genome[
        tr
    ]

    unique_g,counts=np.unique(
        train_genomes,
        return_counts=True,
    )

    count_map={
        g:int(c)
        for g,c in zip(
            unique_g,
            counts,
        )
    }

    sample_weight=np.asarray(
        [
            1.0
            /
            count_map[
                g
            ]
            for g in train_genomes
        ],
        dtype=np.float32,
    )

    # Normalize mean training weight to 1.
    sample_weight *= (
        len(
            sample_weight
        )
        /
        sample_weight.sum()
    )


    ytr=y[
        tr
    ].astype(
        np.int64
    )


    ds=TensorDataset(
        torch.from_numpy(
            Xtr
        ),
        torch.from_numpy(
            ytr
        ),
        torch.from_numpy(
            sample_weight
        ),
    )


    generator=torch.Generator()

    generator.manual_seed(
        seed
    )


    loader=DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
        drop_last=False,
    )


    # -------------------------------------------------------------------------
    # HEAD
    # -------------------------------------------------------------------------

    head=GenoGrammaPairHead().to(
        device
    )

    n_params=sum(
        p.numel()
        for p in head.parameters()
        if p.requires_grad
    )

    if n_params != 98690:
        raise RuntimeError(
            f"Unexpected head parameter count: "
            f"{n_params}"
        )


    optimizer=torch.optim.AdamW(
        head.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )


    best_score=-np.inf
    best_epoch=None
    best_state=None

    stale=0


    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS+1,
    ):

        head.train()

        running_num=0.0
        running_den=0.0


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


            logits=head(
                xb
            )

            per_sample=F.cross_entropy(
                logits,
                yb,
                reduction="none",
            )

            loss=(
                per_sample
                *
                wb
            ).sum()/wb.sum()


            loss.backward()

            optimizer.step()


            running_num += float(
                (
                    per_sample.detach()
                    *
                    wb
                ).sum().cpu()
            )

            running_den += float(
                wb.sum().cpu()
            )


        train_loss=(
            running_num
            /
            running_den
        )


        # ---------------------------------------------------------------------
        # INNER VALIDATION
        # ---------------------------------------------------------------------

        prob_val=predict_prob(
            head,
            X[
                va
            ],
            mean,
            sd,
            device,
        )

        val_met=metrics(
            y[
                va
            ],
            prob_val,
        )

        score=0.5*(
            val_met[
                "ROC_AUC"
            ]
            +
            val_met[
                "PR_AUC"
            ]
        )


        if (
            score
            >
            best_score
            +
            1e-12
        ):

            best_score=score
            best_epoch=epoch

            best_state={
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k,v
                in head.state_dict().items()
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
                f"[F{fold}] "
                f"epoch={epoch:03d} "
                f"loss={train_loss:.4f} "
                f"valROC={val_met['ROC_AUC']:.4f} "
                f"valPR={val_met['PR_AUC']:.4f} "
                f"select={score:.4f} "
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


    if best_state is None:
        raise RuntimeError(
            f"Fold {fold}: no best state"
        )


    # -------------------------------------------------------------------------
    # OUTER TEST FIRST/ONLY USE
    # -------------------------------------------------------------------------

    head.load_state_dict(
        best_state,
        strict=True,
    )

    prob_test=predict_prob(
        head,
        X[
            te
        ],
        mean,
        sd,
        device,
    )

    met=metrics(
        y[
            te
        ],
        prob_test,
    )


    oof_prob[
        te
    ]=prob_test

    oof_pred[
        te
    ]=(
        prob_test
        >=
        0.5
    ).astype(
        np.int64
    )


    fold_row={
        "model":
            "GenoGramma",

        "outer_fold":
            fold,

        "n_train":
            int(
                len(tr)
            ),

        "n_val":
            int(
                len(va)
            ),

        "n_test":
            int(
                len(te)
            ),

        "train_genomes":
            int(
                len(
                    np.unique(
                        genome[
                            tr
                        ]
                    )
                )
            ),

        "train_lineages":
            int(
                len(
                    np.unique(
                        lineage[
                            tr
                        ]
                    )
                )
            ),

        "val_lineages":
            int(
                len(
                    np.unique(
                        lineage[
                            va
                        ]
                    )
                )
            ),

        "test_lineages":
            int(
                len(
                    np.unique(
                        lineage[
                            te
                        ]
                    )
                )
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val_selection_score":
            float(
                best_score
            ),

        **met,
    }


    fold_rows.append(
        fold_row
    )


    fold_dir=(
        OUT
        / f"fold_{fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    torch.save(
        {
            "model_name":
                "GenoGramma",

            "outer_fold":
                fold,

            "best_epoch":
                best_epoch,

            "best_val_selection_score":
                best_score,

            "head_state_dict":
                best_state,

            "zscore_mean":
                mean,

            "zscore_sd":
                sd,

            "head_parameters":
                n_params,

            "protocol":{
                "representation_dim":
                    REP_DIM,

                "hidden_dim":
                    HIDDEN_DIM,

                "dropout":
                    DROPOUT,

                "learning_rate":
                    LR,

                "weight_decay":
                    WEIGHT_DECAY,

                "genome_balanced_training":
                    True,

                "class_weighting":
                    False,

                "checkpoint_selection":
                    "0.5*(inner_ROC_AUC+inner_PR_AUC)",
            },
        },
        fold_dir
        / "best_head.pt",
    )


    print(
        f"[OUTER F{fold}] "
        f"best_epoch={best_epoch} "
        f"ROC={met['ROC_AUC']:.4f} "
        f"PR={met['PR_AUC']:.4f} "
        f"BalAcc={met['Balanced_Accuracy']:.4f} "
        f"F1={met['F1']:.4f} "
        f"MCC={met['MCC']:.4f}",
        flush=True,
    )


    del X
    del Xtr
    del ds
    del loader
    del head

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# OOF AUDIT
# =============================================================================

if np.isnan(
    oof_prob
).any():
    raise RuntimeError(
        "Incomplete OOF probabilities"
    )

if (
    oof_pred < 0
).any():
    raise RuntimeError(
        "Incomplete OOF predictions"
    )


fold_df=pd.DataFrame(
    fold_rows
)

fold_df.to_csv(
    OUT
    / "02_fold_metrics.csv",
    index=False,
)


pred_df=pd.DataFrame(
    {
        "row_index":
            row_index,

        "model":
            "GenoGramma",

        "outer_fold":
            outer_fold,

        "y_true":
            y,

        "prob_positive":
            oof_prob,

        "y_pred":
            oof_pred,

        "strain_entity_id":
            genome,

        "lineage_95_block":
            lineage,
    }
)

pred_df.to_csv(
    OUT
    / "03_all_oof_predictions.csv.gz",
    index=False,
    compression="gzip",
)


# =============================================================================
# POOLED OOF SUMMARY
# =============================================================================

pooled=metrics(
    y,
    oof_prob,
)

summary={
    "model":
        "GenoGramma",

    **pooled,
}


for metric in [
    "ROC_AUC",
    "PR_AUC",
    "Accuracy",
    "Balanced_Accuracy",
    "F1",
    "MCC",
]:

    summary[
        metric
        +
        "_fold_mean"
    ]=float(
        fold_df[
            metric
        ].mean()
    )

    summary[
        metric
        +
        "_fold_sd"
    ]=float(
        fold_df[
            metric
        ].std(
            ddof=1
        )
    )


summary_df=pd.DataFrame(
    [
        summary
    ]
)

summary_df.to_csv(
    OUT
    / "04_model_summary.csv",
    index=False,
)


# =============================================================================
# GENOME / LINEAGE MACRO SUMMARY
# =============================================================================

group_rows=[]

for level in [
    "strain_entity_id",
    "lineage_95_block",
]:

    rows=[]

    for group_id,d in pred_df.groupby(
        level,
        sort=False,
    ):

        if d[
            "y_true"
        ].nunique() < 2:
            continue

        m=metrics(
            d[
                "y_true"
            ].to_numpy(),
            d[
                "prob_positive"
            ].to_numpy(),
        )

        rows.append(
            m
        )

    if rows:

        dd=pd.DataFrame(
            rows
        )

        group_rows.append(
            {
                "model":
                    "GenoGramma",

                "aggregation_level":
                    (
                        "genome"
                        if level
                        ==
                        "strain_entity_id"
                        else
                        "ANI95_lineage"
                    ),

                "valid_groups":
                    int(
                        len(dd)
                    ),

                "ROC_AUC_macro":
                    float(
                        dd[
                            "ROC_AUC"
                        ].mean()
                    ),

                "PR_AUC_macro":
                    float(
                        dd[
                            "PR_AUC"
                        ].mean()
                    ),

                "Balanced_Accuracy_macro":
                    float(
                        dd[
                            "Balanced_Accuracy"
                        ].mean()
                    ),

                "F1_macro":
                    float(
                        dd[
                            "F1"
                        ].mean()
                    ),

                "MCC_macro":
                    float(
                        dd[
                            "MCC"
                        ].mean()
                    ),
            }
        )


group_df=pd.DataFrame(
    group_rows
)

group_df.to_csv(
    OUT
    / "05_group_macro_summary.csv",
    index=False,
)


# =============================================================================
# SUCCESS GATE
# =============================================================================

roc=float(
    pooled[
        "ROC_AUC"
    ]
)

pr=float(
    pooled[
        "PR_AUC"
    ]
)

success=(
    roc >= 0.70
    and
    pr >= 0.70
)


freeze={
    "status":
        "STAGE56_GENOGRAMMA_PAIRAWARE_EVALUATION_COMPLETE",

    "model":
        "GenoGramma",

    "formal_input_sha256":
        actual_input_sha,

    "representation_index_sha256":
        sha256(
            REP_INDEX
        ),

    "preregistration_sha256":
        sha256(
            PRE
        ),

    "script_sha256":
        sha256(
            Path(
                __file__
            ).resolve()
        ),

    "n_examples":
        N,

    "n_folds":
        5,

    "outer_protocol":
        "ANI95 lineage-blocked five-fold OOF",

    "inner_protocol":
        "lineage-blocked deterministic validation",

    "representation_dimension":
        REP_DIM,

    "head":
        "Linear(768,128)->GELU->Dropout(0.20)->Linear(128,2)",

    "head_parameters":
        98690,

    "train_only_zscore":
        True,

    "genome_balanced_training":
        True,

    "class_weighting":
        False,

    "checkpoint_selection":
        "mean(inner ROC-AUC, inner PR-AUC)",

    "encoder_training":
        False,

    "direct_operon_features_used":
        False,

    "pooled_ROC_AUC":
        roc,

    "pooled_PR_AUC":
        pr,

    "success_gate":{
        "ROC_AUC_min":
            0.70,

        "PR_AUC_min":
            0.70,
    },

    "success_gate_pass":
        bool(
            success
        ),
}


(
    OUT
    / "06_stage56_result_freeze.json"
).write_text(
    json.dumps(
        freeze,
        indent=2,
    )
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print(
    "\n"
    +
    "="*112,
    flush=True,
)

print(
    "STAGE56 COMPLETE",
    flush=True,
)

print(
    "="*112,
    flush=True,
)

print(
    fold_df[
        [
            "outer_fold",
            "best_epoch",
            "ROC_AUC",
            "PR_AUC",
            "Balanced_Accuracy",
            "F1",
            "MCC",
        ]
    ].to_string(
        index=False
    ),
    flush=True,
)

print(
    "\nPOOLED OOF:",
    flush=True,
)

print(
    summary_df[
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
    ),
    flush=True,
)

print(
    "\nSUCCESS GATE:",
    flush=True,
)

print(
    f"ROC-AUC = {roc:.6f} "
    f"(target >= 0.70)",
    flush=True,
)

print(
    f"PR-AUC  = {pr:.6f} "
    f"(target >= 0.70)",
    flush=True,
)

print(
    "[PASS]"
    if success
    else
    "[NO-GO]",
    "GENOGRAMMA_PAIRAWARE_TARGET",
    flush=True,
)

print(
    "[OUTPUT]",
    OUT,
    flush=True,
)

print(
    "="*112,
    flush=True,
)
