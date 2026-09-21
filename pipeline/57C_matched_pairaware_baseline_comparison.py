from __future__ import annotations

import os

import json
import hashlib
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

EXPECTED_INPUT_SHA=(
    "978e233098e4ccda12460e821d01fb076"
    "26f909d8afd4dafa2737c867081ab3e"
)

META=(
    BASE / "runtime_assets" / "source_tree" / "new_paper_data"
    / "08_stage50_order_dependent_downstream"
    / "07_pair_to_genogrammar_alignment"
    / "14B_STAGE50_operon_model_input_metadata_with_row_index.csv.gz"
)

BASELINE_INDEX=(
    ROOT
    / "07_stage57B2_native_matched_features"
    / "57B2_native_pair_state_index.csv"
)

BASELINE_FREEZE=(
    ROOT
    / "07_stage57B2_native_matched_features"
    / "57B2_native_pair_state_freeze.json"
)

GENOGRAMMA_STAGE56=(
    ROOT
    / "04_stage56_pairaware_probe"
)

GENOGRAMMA_SUMMARY=(
    GENOGRAMMA_STAGE56
    / "04_model_summary.csv"
)

GENOGRAMMA_FOLD=(
    GENOGRAMMA_STAGE56
    / "02_fold_metrics.csv"
)

GENOGRAMMA_PRED=(
    GENOGRAMMA_STAGE56
    / "03_all_oof_predictions.csv.gz"
)

OUT=(
    ROOT
    / "08_stage57C_matched_comparison"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FIXED PROTOCOL
# =============================================================================

N=164302

MODELS=[
    "CNN1D",
    "BiGRU",
    "TransformerSmall",
    "MeanPool",
]

CONTEXT_DIMS={
    "CNN1D":128,
    "BiGRU":188,
    "TransformerSmall":128,
    "MeanPool":128,
}

GLOBAL_DIM=128
PAIR_DIM=128

HEAD_INPUT_DIM=768
HEAD_HIDDEN=128

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


# =============================================================================
# UTILS
# =============================================================================

def sha256(path:Path):

    h=hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda:f.read(1024*1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def calc_metrics(
    y,
    prob,
):

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
    ).astype(
        np.int64
    )

    tn,fp,fn,tp=confusion_matrix(
        y,
        pred,
        labels=[0,1],
    ).ravel()

    spec=(
        tn/(tn+fp)
        if (tn+fp)>0
        else np.nan
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
            float(spec),

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


# =============================================================================
# MATCHED BASELINE READOUT
# =============================================================================

class MatchedPairAwareHead(nn.Module):

    def __init__(
        self,
        context_dim,
    ):

        super().__init__()

        # Applied identically in concept to every baseline.
        # Even 128-D models receive a trainable 128->128 adapter.
        self.context_proj=nn.Sequential(
            nn.Linear(
                context_dim,
                PAIR_DIM,
            ),
            nn.LayerNorm(
                PAIR_DIM,
            ),
        )

        self.pair_adapter=nn.Sequential(
            nn.Linear(
                4*PAIR_DIM,
                PAIR_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(
                PAIR_DIM,
            ),
        )

        self.classifier=nn.Sequential(
            nn.Linear(
                HEAD_INPUT_DIM,
                HEAD_HIDDEN,
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                HEAD_HIDDEN,
                2,
            ),
        )


    def make_768(
        self,
        raw,
    ):

        D=(
            raw.shape[1]
            -
            GLOBAL_DIM
        )//2

        left_raw=raw[
            :,
            :D
        ]

        right_raw=raw[
            :,
            D:2*D
        ]

        global_z=raw[
            :,
            2*D:
        ]

        left=self.context_proj(
            left_raw
        )

        right=self.context_proj(
            right_raw
        )

        diff=torch.abs(
            right-left
        )

        prod=left*right

        local_pair=self.pair_adapter(
            torch.cat(
                [
                    left,
                    right,
                    diff,
                    prod,
                ],
                dim=1,
            )
        )

        z=torch.cat(
            [
                left,
                right,
                diff,
                prod,
                local_pair,
                global_z,
            ],
            dim=1,
        )

        if z.shape[1] != HEAD_INPUT_DIM:
            raise RuntimeError(
                f"matched representation "
                f"dim={z.shape[1]} != 768"
            )

        return z


    def forward(
        self,
        raw,
    ):

        return self.classifier(
            self.make_768(
                raw
            )
        )


# =============================================================================
# PREDICT
# =============================================================================

def predict_prob(
    model,
    X,
    mean,
    sd,
    device,
):

    model.eval()

    outputs=[]

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
                xb
                -
                mean
            )/sd

            xb=torch.from_numpy(
                xb
            ).to(
                device
            )

            logits=model(
                xb
            )

            p=torch.softmax(
                logits,
                dim=1,
            )[:,1]

            outputs.append(
                p.cpu().numpy()
            )

    return np.concatenate(
        outputs
    )


# =============================================================================
# AUDIT INPUTS
# =============================================================================

print(
    "="*118,
    flush=True,
)

print(
    "STAGE57C — MATCHED PAIR-AWARE BASELINE COMPARISON",
    flush=True,
)

print(
    "FROZEN ENCODERS | ANI95 LINEAGE-BLOCKED OOF-5CV | "
    "BASELINES RECEIVE CONSERVATIVE TRAINABLE PAIR ADAPTER",
    flush=True,
)

print(
    "="*118,
    flush=True,
)


actual_input_sha=sha256(
    INPUT_NPZ
)

if actual_input_sha != EXPECTED_INPUT_SHA:
    raise RuntimeError(
        "Formal input SHA mismatch"
    )


freeze=json.loads(
    BASELINE_FREEZE.read_text()
)

if (
    freeze.get("status")
    !=
    "ALL_20_NATIVE_MATCHED_PAIR_STATES_PASS"
):
    raise RuntimeError(
        "Stage57B2 freeze mismatch"
    )


if not GENOGRAMMA_SUMMARY.exists():
    raise RuntimeError(
        "Missing frozen GenoGramma Stage56 result"
    )


data=np.load(
    INPUT_NPZ
)

y=np.asarray(
    data[
        "label_index"
    ],
    dtype=np.int64,
)

outer_fold=np.asarray(
    data[
        "outer_fold"
    ],
    dtype=np.int64,
)

row_index=np.asarray(
    data[
        "row_index"
    ],
    dtype=np.int64,
)


if len(y) != N:
    raise RuntimeError(
        "Formal N mismatch"
    )


meta=pd.read_csv(
    META
)


if "row_index" in meta.columns:

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


print(
    "[PASS] formal input SHA =",
    actual_input_sha,
    flush=True,
)

print(
    "[PASS] examples =",
    N,
    flush=True,
)

print(
    "[PASS] genomes / ANI95 lineages =",
    len(np.unique(genome)),
    "/",
    len(np.unique(lineage)),
    flush=True,
)

print(
    "[PASS] Stage57B2 native pair states = 20 / 20",
    flush=True,
)


# =============================================================================
# FEATURE INDEX
# =============================================================================

idx=pd.read_csv(
    BASELINE_INDEX
)


if len(idx) != 20:
    raise RuntimeError(
        f"feature sets={len(idx)} != 20"
    )


feature_map={}


for _,r in idx.iterrows():

    model=str(
        r[
            "model"
        ]
    )

    fold=int(
        r[
            "fold"
        ]
    )

    p=Path(
        r[
            "feature_file"
        ]
    )

    if not p.exists():
        raise RuntimeError(
            f"Missing feature file: {p}"
        )

    if sha256(p) != str(
        r[
            "feature_sha256"
        ]
    ):
        raise RuntimeError(
            f"{model}/F{fold}: SHA mismatch"
        )

    feature_map[
        (
            model,
            fold,
        )
    ]=p


# =============================================================================
# FREEZE INNER SPLITS BEFORE TRAINING
# =============================================================================

splits={}

split_payload={}


for fold in range(
    1,
    6,
):

    outer_train=np.where(
        outer_fold != fold
    )[0]

    test=np.where(
        outer_fold == fold
    )[0]

    gss=GroupShuffleSplit(
        n_splits=1,
        test_size=INNER_VAL_FRACTION,
        random_state=FOLD_SEEDS[
            fold
        ],
    )

    tr_rel,va_rel=next(
        gss.split(
            np.zeros(
                len(
                    outer_train
                )
            ),
            y[
                outer_train
            ],
            groups=lineage[
                outer_train
            ],
        )
    )

    tr=outer_train[
        tr_rel
    ]

    va=outer_train[
        va_rel
    ]


    A=set(
        lineage[
            tr
        ]
    )

    B=set(
        lineage[
            va
        ]
    )

    C=set(
        lineage[
            test
        ]
    )


    if A & B:
        raise RuntimeError(
            f"F{fold}: train/val leakage"
        )

    if A & C:
        raise RuntimeError(
            f"F{fold}: train/test leakage"
        )

    if B & C:
        raise RuntimeError(
            f"F{fold}: val/test leakage"
        )


    splits[
        fold
    ]=(
        tr,
        va,
        test,
    )


    split_payload[
        str(
            fold
        )
    ]={
        "seed":
            FOLD_SEEDS[
                fold
            ],

        "train_n":
            int(
                len(tr)
            ),

        "val_n":
            int(
                len(va)
            ),

        "test_n":
            int(
                len(test)
            ),

        "train_lineages":
            sorted(A),

        "val_lineages":
            sorted(B),

        "test_lineages":
            sorted(C),
    }


(
    OUT
    /
    "57C_inner_lineage_splits.json"
).write_text(
    json.dumps(
        split_payload,
        indent=2,
    )
)


print(
    "[PASS] deterministic inner lineage splits frozen",
    flush=True,
)


# =============================================================================
# PREREGISTRATION/FREEZE BEFORE FIT
# =============================================================================

protocol={
    "status":
        "STAGE57C_PROTOCOL_FROZEN_BEFORE_BASELINE_FIT",

    "formal_model_name":
        "GenoGramma",

    "models":[
        "GenoGramma",
        "CNN1D",
        "BiGRU",
        "TransformerSmall",
        "MeanPool",
    ],

    "baseline_context_dimensions":
        CONTEXT_DIMS,

    "baseline_matched_representation":
        [
            "projected_left_128D",
            "projected_right_128D",
            "absolute_difference_128D",
            "elementwise_interaction_128D",
            "trainable_local_pair_adapter_128D",
            "frozen_global_context_128D",
        ],

    "baseline_final_dimension":
        768,

    "genogramma_representation":
        "frozen Stage55 pair-aware 768D",

    "fairness_note":
        "Baseline models receive an additional trainable "
        "context projection and local pair adapter. "
        "This is deliberately conservative in favor of baselines.",

    "classifier":
        "Linear(768,128)->GELU->Dropout(0.20)->Linear(128,2)",

    "outer_split":
        "frozen ANI95 lineage-blocked 5-fold",

    "inner_split":
        "lineage-blocked deterministic 15% outer-training validation",

    "train_only_zscore":
        True,

    "genome_balanced_training":
        True,

    "class_weighting":
        False,

    "optimizer":
        "AdamW",

    "learning_rate":
        LR,

    "weight_decay":
        WEIGHT_DECAY,

    "max_epochs":
        MAX_EPOCHS,

    "min_epochs":
        MIN_EPOCHS,

    "patience":
        PATIENCE,

    "checkpoint_selection":
        "0.5*(inner ROC-AUC + inner PR-AUC)",

    "decision_threshold":
        0.5,

    "direct_operon_features":
        False,

    "encoder_training":
        False,
}


PRE=(
    OUT
    /
    "57C_protocol_freeze.json"
)

PRE.write_text(
    json.dumps(
        protocol,
        indent=2,
    )
)


print(
    "[PASS] Stage57C protocol frozen before model fitting",
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
# TRAIN BASELINES
# =============================================================================

all_fold_rows=[]
all_pred=[]


for model_name in MODELS:

    D=CONTEXT_DIMS[
        model_name
    ]

    expected_raw_dim=(
        2*D
        +
        GLOBAL_DIM
    )


    print(
        "\n"
        +
        "#"*118,
        flush=True,
    )

    print(
        model_name,
        flush=True,
    )

    print(
        "#"*118,
        flush=True,
    )


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
            f"{model_name} | outer fold {fold}/5",
            flush=True,
        )

        print(
            "="*100,
            flush=True,
        )


        set_seed(
            FOLD_SEEDS[
                fold
            ]
        )


        tr,va,te=splits[
            fold
        ]


        X=np.load(
            feature_map[
                (
                    model_name,
                    fold,
                )
            ],
            mmap_mode="r",
        )


        if X.shape != (
            N,
            expected_raw_dim,
        ):
            raise RuntimeError(
                f"{model_name}/F{fold}: "
                f"raw shape={X.shape}, "
                f"expected={(N,expected_raw_dim)}"
            )


        # ---------------------------------------------------------------------
        # TRAIN-ONLY Z-SCORE
        # ---------------------------------------------------------------------

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


        # ---------------------------------------------------------------------
        # GENOME-BALANCED TRAINING
        # ---------------------------------------------------------------------

        train_genomes=genome[
            tr
        ]


        ug,cnt=np.unique(
            train_genomes,
            return_counts=True,
        )


        cmap={
            g:int(c)
            for g,c
            in zip(
                ug,
                cnt,
            )
        }


        w=np.asarray(
            [
                1.0/cmap[g]
                for g
                in train_genomes
            ],
            dtype=np.float32,
        )


        w *= (
            len(w)
            /
            w.sum()
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
                w
            ),
        )


        gen=torch.Generator()

        gen.manual_seed(
            FOLD_SEEDS[
                fold
            ]
        )


        loader=DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            generator=gen,
        )


        head=MatchedPairAwareHead(
            D
        ).to(
            device
        )


        n_params=sum(
            p.numel()
            for p in head.parameters()
            if p.requires_grad
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


        for epoch in range(
            1,
            MAX_EPOCHS+1,
        ):

            head.train()

            loss_num=0.0
            loss_den=0.0


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


                loss_num += float(
                    (
                        per_sample.detach()
                        *
                        wb
                    ).sum().cpu()
                )

                loss_den += float(
                    wb.sum().cpu()
                )


            train_loss=(
                loss_num
                /
                loss_den
            )


            prob_val=predict_prob(
                head,
                X[
                    va
                ],
                mean,
                sd,
                device,
            )


            vm=calc_metrics(
                y[
                    va
                ],
                prob_val,
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
                    f"[{model_name}/F{fold}] "
                    f"epoch={epoch:03d} "
                    f"loss={train_loss:.4f} "
                    f"valROC={vm['ROC_AUC']:.4f} "
                    f"valPR={vm['PR_AUC']:.4f} "
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
                f"{model_name}/F{fold}: "
                "no best checkpoint"
            )


        head.load_state_dict(
            best_state,
            strict=True,
        )


        prob=predict_prob(
            head,
            X[
                te
            ],
            mean,
            sd,
            device,
        )


        met=calc_metrics(
            y[
                te
            ],
            prob,
        )


        row={
            "model":
                model_name,

            "outer_fold":
                fold,

            "context_dim":
                D,

            "head_trainable_params":
                n_params,

            "best_epoch":
                best_epoch,

            "best_val_selection_score":
                best_score,

            "n_train":
                len(tr),

            "n_val":
                len(va),

            "n_test":
                len(te),

            **met,
        }


        all_fold_rows.append(
            row
        )


        all_pred.append(
            pd.DataFrame(
                {
                    "row_index":
                        row_index[
                            te
                        ],

                    "model":
                        model_name,

                    "outer_fold":
                        fold,

                    "y_true":
                        y[
                            te
                        ],

                    "prob_positive":
                        prob,

                    "y_pred":
                        (
                            prob >= 0.5
                        ).astype(
                            np.int64
                        ),

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


        fold_dir=(
            OUT
            /
            model_name
            /
            f"fold_{fold}"
        )


        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        torch.save(
            {
                "model":
                    model_name,

                "outer_fold":
                    fold,

                "context_dim":
                    D,

                "trainable_parameters":
                    n_params,

                "best_epoch":
                    best_epoch,

                "best_val_selection_score":
                    best_score,

                "head_state_dict":
                    best_state,

                "train_zscore_mean":
                    mean,

                "train_zscore_sd":
                    sd,
            },
            fold_dir
            /
            "best_head.pt",
        )


        print(
            f"[OUTER {model_name}/F{fold}] "
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
# SAVE BASELINE RESULTS
# =============================================================================

fold_df=pd.DataFrame(
    all_fold_rows
)


fold_df.to_csv(
    OUT
    /
    "57C_baseline_fold_metrics.csv",
    index=False,
)


pred_df=pd.concat(
    all_pred,
    ignore_index=True,
)


pred_df.to_csv(
    OUT
    /
    "57C_baseline_oof_predictions.csv.gz",
    index=False,
    compression="gzip",
)


# =============================================================================
# POOLED BASELINE SUMMARY
# =============================================================================

summary_rows=[]


for model_name in MODELS:

    q=pred_df[
        pred_df[
            "model"
        ]
        ==
        model_name
    ]


    if len(q) != N:
        raise RuntimeError(
            f"{model_name}: "
            f"OOF N={len(q)} != {N}"
        )


    met=calc_metrics(
        q[
            "y_true"
        ].to_numpy(),
        q[
            "prob_positive"
        ].to_numpy(),
    )


    f=fold_df[
        fold_df[
            "model"
        ]
        ==
        model_name
    ]


    row={
        "model":
            model_name,

        **met,
    }


    for metric in [
        "ROC_AUC",
        "PR_AUC",
        "Balanced_Accuracy",
        "F1",
        "MCC",
    ]:

        row[
            metric
            +
            "_fold_mean"
        ]=float(
            f[
                metric
            ].mean()
        )

        row[
            metric
            +
            "_fold_sd"
        ]=float(
            f[
                metric
            ].std(
                ddof=1
            )
        )


    summary_rows.append(
        row
    )


baseline_summary=pd.DataFrame(
    summary_rows
)


baseline_summary.to_csv(
    OUT
    /
    "57C_baseline_summary.csv",
    index=False,
)


# =============================================================================
# ADD FROZEN GENOGRAMMA RESULT
# =============================================================================

gg=pd.read_csv(
    GENOGRAMMA_SUMMARY
)


gg_row={
    "model":
        "GenoGramma",

    "ROC_AUC":
        float(
            gg.iloc[0][
                "ROC_AUC"
            ]
        ),

    "PR_AUC":
        float(
            gg.iloc[0][
                "PR_AUC"
            ]
        ),

    "Balanced_Accuracy":
        float(
            gg.iloc[0][
                "Balanced_Accuracy"
            ]
        ),

    "F1":
        float(
            gg.iloc[0][
                "F1"
            ]
        ),

    "MCC":
        float(
            gg.iloc[0][
                "MCC"
            ]
        ),
}


comparison=pd.concat(
    [
        pd.DataFrame(
            [
                gg_row
            ]
        ),
        baseline_summary[
            [
                "model",
                "ROC_AUC",
                "PR_AUC",
                "Balanced_Accuracy",
                "F1",
                "MCC",
            ]
        ],
    ],
    ignore_index=True,
)


comparison[
    "ROC_rank"
]=comparison[
    "ROC_AUC"
].rank(
    ascending=False,
    method="min",
).astype(int)


comparison[
    "PR_rank"
]=comparison[
    "PR_AUC"
].rank(
    ascending=False,
    method="min",
).astype(int)


comparison.to_csv(
    OUT
    /
    "57C_FINAL_matched_pairaware_comparison.csv",
    index=False,
)


# =============================================================================
# DELTAS VS GENOGRAMMA
# =============================================================================

delta=[]


for _,r in baseline_summary.iterrows():

    delta.append(
        {
            "baseline":
                r[
                    "model"
                ],

            "GenoGramma_ROC":
                gg_row[
                    "ROC_AUC"
                ],

            "baseline_ROC":
                r[
                    "ROC_AUC"
                ],

            "Delta_ROC_GenoGramma_minus_baseline":
                gg_row[
                    "ROC_AUC"
                ]
                -
                r[
                    "ROC_AUC"
                ],

            "GenoGramma_PR":
                gg_row[
                    "PR_AUC"
                ],

            "baseline_PR":
                r[
                    "PR_AUC"
                ],

            "Delta_PR_GenoGramma_minus_baseline":
                gg_row[
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
    "57C_GenoGramma_vs_baseline_deltas.csv",
    index=False,
)


# =============================================================================
# FREEZE
# =============================================================================

freeze_out={
    "status":
        "STAGE57C_MATCHED_PAIRAWARE_COMPARISON_COMPLETE",

    "formal_model_name":
        "GenoGramma",

    "formal_input_sha256":
        actual_input_sha,

    "baseline_feature_freeze_sha256":
        sha256(
            BASELINE_FREEZE
        ),

    "protocol_freeze_sha256":
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

    "n_genomes":
        int(
            len(
                np.unique(
                    genome
                )
            )
        ),

    "n_ANI95_lineages":
        int(
            len(
                np.unique(
                    lineage
                )
            )
        ),

    "encoder_training":
        False,

    "direct_operon_features":
        False,

    "fairness_note":
        "Baseline encoders receive trainable context projection "
        "and local pair adapter; GenoGramma result is the frozen "
        "pre-registered Stage56 pair-aware result.",

    "comparison_file":
        str(
            OUT
            /
            "57C_FINAL_matched_pairaware_comparison.csv"
        ),
}


(
    OUT
    /
    "57C_result_freeze.json"
).write_text(
    json.dumps(
        freeze_out,
        indent=2,
    )
)


# =============================================================================
# FINAL
# =============================================================================

print(
    "\n"
    +
    "="*118,
    flush=True,
)

print(
    "STAGE57C COMPLETE — FINAL MATCHED PAIR-AWARE COMPARISON",
    flush=True,
)

print(
    "="*118,
    flush=True,
)


print(
    comparison[
        [
            "model",
            "ROC_AUC",
            "PR_AUC",
            "Balanced_Accuracy",
            "F1",
            "MCC",
            "ROC_rank",
            "PR_rank",
        ]
    ]
    .sort_values(
        [
            "ROC_AUC",
            "PR_AUC",
        ],
        ascending=False,
    )
    .to_string(
        index=False
    ),
    flush=True,
)


print(
    "\n===== GenoGramma deltas =====",
    flush=True,
)


print(
    pd.DataFrame(
        delta
    )[
        [
            "baseline",
            "Delta_ROC_GenoGramma_minus_baseline",
            "Delta_PR_GenoGramma_minus_baseline",
        ]
    ].to_string(
        index=False
    ),
    flush=True,
)


print(
    "\n[PASS] encoder training downstream = NONE",
    flush=True,
)

print(
    "[PASS] direct operon features = NONE",
    flush=True,
)

print(
    "[PASS] ANI95 lineage-blocked OOF-5CV preserved",
    flush=True,
)

print(
    "[OUTPUT]",
    OUT,
    flush=True,
)

print(
    "STAGE57C_MATCHED_PAIRAWARE_COMPARISON_COMPLETE",
    flush=True,
)

print(
    "="*118,
    flush=True,
)
