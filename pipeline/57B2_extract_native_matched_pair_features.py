from __future__ import annotations

import os
import sys
import json
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(__file__).resolve().parents[1]
PAPER=BASE
PIPE=PAPER/"pipeline"

os.environ.setdefault(
    "GENOGRAMMAR_DATA_ROOT",
    str(BASE),
)

os.environ.setdefault(
    "GENOGRAMMAR_REVISION_OUTPUT",
    str(
        BASE / "runtime_assets" / "source_tree" / "new_paper_data"
        / "10_genogramma_pairaware"
        / "07_stage57B2_native_matched_features"
        / "_baseline_import_context"
    ),
)

OUT=(
    Path(os.environ.get("GENOGRAMMA_RESULT_ROOT", str(BASE / "runtime_assets" / "source_tree" / "new_paper_data" / "10_genogramma_pairaware"))) / '07_stage57B2_native_matched_features'
)

FEATURE_ROOT=OUT/"features"

INPUT_NPZ=(
    BASE / "runtime_assets" / "source_tree" / "new_paper_data"
    / "08_stage50_order_dependent_downstream"
    / "07_pair_to_genogrammar_alignment"
    / "13_STAGE50_operon_formal_model_inputs.npz"
)

FAMILY_EMBEDDINGS=(
    BASE / "model_assets" / "family_embeddings" / "00_family_embeddings_276517x480.npy"
)

EXPECTED_INPUT_SHA=(
    "978e233098e4ccda12460e821d01fb076"
    "26f909d8afd4dafa2737c867081ab3e"
)


MODEL_SPECS={

    "CNN1D":{
        "runner":
            PIPE/"lineage_runs"/
            "29G_cnn1d_lineage95_resume_fold1.py",

        "checkpoint_root":
            BASE / "runtime_assets" / "source_tree" / "new_paper_data"/
            "07_downstream_benchmark"/
            "04_lineage_clean_baseline_pretraining"/
            "CNN1D_29D",

        "expected_params":
            508310,

        "expected_context_dim":
            128,
    },

    "BiGRU":{
        "runner":
            PIPE/"lineage_runs"/
            "29G_bigru_lineage95_5fold.py",

        "checkpoint_root":
            BASE / "runtime_assets" / "source_tree" / "new_paper_data"/
            "07_downstream_benchmark"/
            "04_lineage_clean_baseline_pretraining"/
            "BiGRU_29G",

        "expected_params":
            506370,

        "expected_context_dim":
            188,
    },

    "TransformerSmall":{
        "runner":
            PIPE/"lineage_runs"/
            "29G_transformer_small_lineage95_5fold.py",

        "checkpoint_root":
            BASE / "runtime_assets" / "source_tree" / "new_paper_data"/
            "07_downstream_benchmark"/
            "04_lineage_clean_baseline_pretraining"/
            "TransformerSmall_29G",

        "expected_params":
            502294,

        "expected_context_dim":
            128,
    },

    "MeanPool":{
        "runner":
            PIPE/"lineage_runs"/
            "29G_mean_pool_lineage95_5fold.py",

        "checkpoint_root":
            BASE / "runtime_assets" / "source_tree" / "new_paper_data"/
            "07_downstream_benchmark"/
            "04_lineage_clean_baseline_pretraining"/
            "MeanPool_29G",

        "expected_params":
            228757,

        "expected_context_dim":
            128,
    },
}


N=164302
WINDOW_LEN=64
GLOBAL_DIM=128

LEFT=31
RIGHT=32

BATCH_SIZE=512


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


def load_module(
    name,
    path,
):

    spec=importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if (
        spec is None
        or
        spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot load {path}"
        )

    mod=importlib.util.module_from_spec(
        spec
    )

    sys.modules[name]=mod

    spec.loader.exec_module(
        mod
    )

    return mod


def build_model(
    model_name,
    fold,
    device,
):

    spec=MODEL_SPECS[
        model_name
    ]

    mod=load_module(
        f"stage57b2_{model_name.lower()}_f{fold}",
        spec["runner"],
    )

    cls=getattr(
        mod,
        "_BASELINE_CLASS",
        None,
    )

    if cls is None:
        raise RuntimeError(
            f"{model_name}: _BASELINE_CLASS missing"
        )

    model=cls()

    ckpt_path=(
        spec[
            "checkpoint_root"
        ]
        /
        f"fold_{fold}"
        /
        "best_checkpoint.pt"
    )

    if not ckpt_path.exists():
        raise RuntimeError(
            f"Missing checkpoint: {ckpt_path}"
        )

    ckpt=torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        not isinstance(ckpt,dict)
        or
        "model" not in ckpt
    ):
        raise RuntimeError(
            f"{model_name}/F{fold}: "
            "invalid checkpoint"
        )

    if int(
        ckpt.get(
            "formal_outer_fold",
            -1,
        )
    ) != fold:
        raise RuntimeError(
            f"{model_name}/F{fold}: "
            "outer-fold mismatch"
        )

    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    n_params=sum(
        p.numel()
        for p in model.parameters()
    )

    if n_params != spec[
        "expected_params"
    ]:
        raise RuntimeError(
            f"{model_name}: "
            f"{n_params} != "
            f"{spec['expected_params']}"
        )

    for p in model.parameters():
        p.requires_grad_(False)

    if any(
        p.requires_grad
        for p in model.parameters()
    ):
        raise RuntimeError(
            f"{model_name}: freeze failed"
        )

    model.to(device)
    model.eval()

    return (
        model,
        ckpt,
        ckpt_path,
        sha256(ckpt_path),
        n_params,
    )


# =============================================================================
# EXACT CONTEXTUAL STATE
# =============================================================================

def encode_with_context(
    model_name,
    model,
    token,
):

    """
    Returns:

        contextual sequence:
            exact per-position tensor immediately before
            the encoder's native global pooling operation.

        global representation:
            native frozen encode_tokens() output.

    CNN1D / BiGRU / TransformerSmall:
        capture pool_gate input.

    MeanPool:
        mean_ff(token), i.e. exact tensor being mean-pooled.
    """

    if model_name=="MeanPool":

        contextual=model.mean_ff(
            token
        )

        global_z=model.encode_tokens(
            token
        )

        return (
            contextual,
            global_z,
            "mean_ff_output_before_mean_pool"
        )


    if not hasattr(
        model,
        "pool_gate",
    ):
        raise RuntimeError(
            f"{model_name}: pool_gate missing"
        )


    capture={}


    def pre_hook(
        module,
        inputs,
    ):

        if not inputs:
            raise RuntimeError(
                f"{model_name}: empty pool_gate input"
            )

        capture[
            "context"
        ]=inputs[0]


    handle=model.pool_gate.register_forward_pre_hook(
        pre_hook
    )

    try:

        global_z=model.encode_tokens(
            token
        )

    finally:

        handle.remove()


    contextual=capture.get(
        "context"
    )

    if contextual is None:
        raise RuntimeError(
            f"{model_name}: "
            "pool_gate contextual state not captured"
        )


    return (
        contextual,
        global_z,
        "pool_gate_input"
    )


# =============================================================================
# EXTRACT ONE MODEL/FOLD
# =============================================================================

def extract_one(
    model_name,
    fold,
    family_rows,
    strand,
    embedding_matrix,
    input_sha,
    device,
):

    spec=MODEL_SPECS[
        model_name
    ]

    expected_d=spec[
        "expected_context_dim"
    ]

    # Save only:
    # left + right + global
    #
    # abs-diff/product are deterministic and will
    # be constructed inside Stage57C.
    #
    # dimension = 2*D + 128

    saved_dim=(
        2*expected_d
        +
        GLOBAL_DIM
    )


    print(
        "\n"
        +
        "="*100,
        flush=True,
    )

    print(
        f"{model_name} | fold {fold}/5",
        flush=True,
    )

    print(
        "="*100,
        flush=True,
    )


    (
        model,
        ckpt,
        ckpt_path,
        ckpt_sha,
        n_params,
    )=build_model(
        model_name,
        fold,
        device,
    )


    print(
        "[PASS] strict checkpoint load",
        flush=True,
    )

    print(
        "[PASS] encoder frozen params =",
        n_params,
        flush=True,
    )


    fold_dir=(
        FEATURE_ROOT
        /
        model_name
        /
        f"fold_{fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    feature_path=(
        fold_dir
        /
        "native_pair_states.npy"
    )

    tmp_path=Path(
        str(feature_path)
        +
        ".tmp"
    )

    meta_path=(
        fold_dir
        /
        "metadata.json"
    )


    # -------------------------------------------------------------------------
    # Resume only if exact current definition matches.
    # -------------------------------------------------------------------------

    if (
        feature_path.exists()
        and
        meta_path.exists()
    ):

        try:

            meta=json.loads(
                meta_path.read_text()
            )

            arr=np.load(
                feature_path,
                mmap_mode="r",
            )

            if (
                meta.get("status")
                ==
                "NATIVE_MATCHED_PAIR_STATE_PASS"
                and
                meta.get(
                    "checkpoint_sha256"
                )
                ==
                ckpt_sha
                and
                meta.get(
                    "formal_input_sha256"
                )
                ==
                input_sha
                and
                meta.get(
                    "contextual_dim"
                )
                ==
                expected_d
                and
                meta.get(
                    "capture_semantics"
                )
                in {
                    "pool_gate_input",
                    "mean_ff_output_before_mean_pool",
                }
                and
                arr.shape
                ==
                (
                    N,
                    saved_dim,
                )
                and
                sha256(
                    feature_path
                )
                ==
                meta.get(
                    "feature_sha256"
                )
            ):

                print(
                    "[SKIP] exact current PASS",
                    flush=True,
                )

                return meta

        except Exception:
            pass


    if tmp_path.exists():
        tmp_path.unlink()


    out=np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            N,
            saved_dim,
        ),
    )


    n_batches=(
        N
        +
        BATCH_SIZE
        -
        1
    )//BATCH_SIZE


    capture_semantics=None


    with torch.inference_mode():

        for batch_idx,start in enumerate(
            range(
                0,
                N,
                BATCH_SIZE,
            ),
            start=1,
        ):

            end=min(
                start+BATCH_SIZE,
                N,
            )

            bsz=end-start


            rows_np=np.asarray(
                family_rows[
                    start:end
                ],
                dtype=np.int64,
            )


            strand_np=np.asarray(
                strand[
                    start:end
                ],
                dtype=np.int64,
            )


            family_np=np.asarray(
                embedding_matrix[
                    rows_np
                ],
                dtype=np.float32,
            )


            family_t=(
                torch.from_numpy(
                    family_np
                )
                .to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            )


            strand_t=(
                torch.from_numpy(
                    strand_np
                )
                .to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True,
                )
            )


            token=model.prepare_tokens(
                family_t,
                strand_t,
            )


            if tuple(
                token.shape
            ) != (
                bsz,
                WINDOW_LEN,
                128,
            ):
                raise RuntimeError(
                    f"{model_name}/F{fold}: "
                    f"token={tuple(token.shape)}"
                )


            (
                contextual,
                global_z,
                capture_semantics,
            )=encode_with_context(
                model_name,
                model,
                token,
            )


            if tuple(
                contextual.shape
            ) != (
                bsz,
                WINDOW_LEN,
                expected_d,
            ):
                raise RuntimeError(
                    f"{model_name}/F{fold}: "
                    f"contextual="
                    f"{tuple(contextual.shape)}, "
                    f"expected "
                    f"{(bsz,WINDOW_LEN,expected_d)}"
                )


            if tuple(
                global_z.shape
            ) != (
                bsz,
                GLOBAL_DIM,
            ):
                raise RuntimeError(
                    f"{model_name}/F{fold}: "
                    f"global="
                    f"{tuple(global_z.shape)}"
                )


            left=contextual[
                :,
                LEFT,
                :
            ]


            right=contextual[
                :,
                RIGHT,
                :
            ]


            saved=torch.cat(
                [
                    left,
                    right,
                    global_z,
                ],
                dim=-1,
            )


            if tuple(
                saved.shape
            ) != (
                bsz,
                saved_dim,
            ):
                raise RuntimeError(
                    f"{model_name}/F{fold}: "
                    f"saved="
                    f"{tuple(saved.shape)}"
                )


            if not torch.isfinite(
                saved
            ).all():
                raise RuntimeError(
                    f"{model_name}/F{fold}: "
                    "non-finite state"
                )


            out[
                start:end
            ]=(
                saved
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )


            if (
                batch_idx==1
                or
                batch_idx%100==0
                or
                batch_idx==n_batches
            ):

                print(
                    f"[{model_name}/F{fold}] "
                    f"context={expected_d}D "
                    f"batch={batch_idx}/{n_batches} "
                    f"rows={end:,}/{N:,}",
                    flush=True,
                )


    out.flush()
    del out


    # NumPy appends .npy when a temporary filename does not end in .npy.
    # Recover that generated file before the atomic final rename.
    if not tmp_path.exists():
        numpy_appended_tmp = Path(str(tmp_path) + ".npy")
        if numpy_appended_tmp.exists():
            numpy_appended_tmp.replace(tmp_path)
    
    tmp_path.replace(
        feature_path
    )


    arr=np.load(
        feature_path,
        mmap_mode="r",
    )


    feature_sha=sha256(
        feature_path
    )


    meta={

        "status":
            "NATIVE_MATCHED_PAIR_STATE_PASS",

        "model":
            model_name,

        "fold":
            fold,

        "formal_outer_fold":
            int(
                ckpt.get(
                    "formal_outer_fold"
                )
            ),

        "checkpoint":
            str(
                ckpt_path
            ),

        "checkpoint_sha256":
            ckpt_sha,

        "formal_input_sha256":
            input_sha,

        "encoder_parameters":
            n_params,

        "encoder_trainable_parameters_downstream":
            0,

        "target_pair_indices_0based":
            [
                LEFT,
                RIGHT,
            ],

        "contextual_dim":
            expected_d,

        "global_dim":
            GLOBAL_DIM,

        "saved_dim":
            saved_dim,

        "capture_semantics":
            capture_semantics,

        "saved_components":[
            "left_native_contextual_state",
            "right_native_contextual_state",
            "frozen_global_representation",
        ],

        "derived_in_stage57C":[
            "absolute_pair_difference",
            "elementwise_pair_interaction",
        ],

        "direct_operon_features_used":
            False,

        "encoder_training_downstream":
            False,

        "feature_file":
            str(
                feature_path
            ),

        "feature_sha256":
            feature_sha,
    }


    meta_path.write_text(
        json.dumps(
            meta,
            indent=2,
        )
    )


    print(
        "[PASS] capture semantics =",
        capture_semantics,
        flush=True,
    )

    print(
        "[PASS] contextual dim =",
        expected_d,
        flush=True,
    )

    print(
        "[PASS] saved shape =",
        arr.shape,
        flush=True,
    )


    return meta


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "="*110,
        flush=True,
    )

    print(
        "STAGE57B2 — NATIVE MATCHED PAIR STATE EXTRACTION",
        flush=True,
    )

    print(
        "EXACT PRE-POOL CONTEXT | FROZEN ENCODERS | "
        "NO DOOR2-DERIVED INPUT FEATURES",
        flush=True,
    )

    print(
        "="*110,
        flush=True,
    )


    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    actual_sha=sha256(
        INPUT_NPZ
    )

    if actual_sha != EXPECTED_INPUT_SHA:
        raise RuntimeError(
            "Formal-input SHA mismatch"
        )


    data=np.load(
        INPUT_NPZ,
        mmap_mode="r",
    )


    family_rows=data[
        "family_rows"
    ]

    strand=data[
        "strand"
    ]


    if family_rows.shape != (
        N,
        WINDOW_LEN,
    ):
        raise RuntimeError(
            f"family_rows="
            f"{family_rows.shape}"
        )


    if strand.shape != (
        N,
        WINDOW_LEN,
    ):
        raise RuntimeError(
            f"strand="
            f"{strand.shape}"
        )


    embedding_matrix=np.load(
        FAMILY_EMBEDDINGS,
        mmap_mode="r",
    )


    if embedding_matrix.shape != (
        276517,
        480,
    ):
        raise RuntimeError(
            f"embeddings="
            f"{embedding_matrix.shape}"
        )


    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )


    print(
        "[PASS] formal input SHA =",
        actual_sha,
        flush=True,
    )

    print(
        "[PASS] examples =",
        N,
        flush=True,
    )

    print(
        "[PASS] target pair = [31,32]",
        flush=True,
    )

    print(
        "[PASS] semantic rule = "
        "exact tensor used by native global pooling",
        flush=True,
    )

    print(
        "[INFO] device =",
        device,
        flush=True,
    )


    rows=[]


    for model_name in [
        "CNN1D",
        "BiGRU",
        "TransformerSmall",
        "MeanPool",
    ]:

        for fold in range(
            1,
            6,
        ):

            meta=extract_one(
                model_name,
                fold,
                family_rows,
                strand,
                embedding_matrix,
                actual_sha,
                device,
            )

            rows.append(
                meta
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    index=pd.DataFrame(
        [
            {
                "model":
                    x["model"],

                "fold":
                    x["fold"],

                "status":
                    x["status"],

                "contextual_dim":
                    x["contextual_dim"],

                "global_dim":
                    x["global_dim"],

                "saved_dim":
                    x["saved_dim"],

                "capture_semantics":
                    x["capture_semantics"],

                "checkpoint":
                    x["checkpoint"],

                "checkpoint_sha256":
                    x["checkpoint_sha256"],

                "feature_file":
                    x["feature_file"],

                "feature_sha256":
                    x["feature_sha256"],
            }
            for x in rows
        ]
    )


    if len(index) != 20:
        raise RuntimeError(
            f"Expected 20 sets, got {len(index)}"
        )


    for model_name in MODEL_SPECS:

        q=index[
            index["model"]
            ==
            model_name
        ]

        if sorted(
            q["fold"].tolist()
        ) != [
            1,2,3,4,5
        ]:
            raise RuntimeError(
                f"{model_name}: incomplete folds"
            )


    index_path=(
        OUT
        /
        "57B2_native_pair_state_index.csv"
    )

    index.to_csv(
        index_path,
        index=False,
    )


    freeze={

        "status":
            "ALL_20_NATIVE_MATCHED_PAIR_STATES_PASS",

        "formal_model_name":
            "GenoGramma",

        "baseline_models":[
            "CNN1D",
            "BiGRU",
            "TransformerSmall",
            "MeanPool",
        ],

        "n_models":
            4,

        "n_folds":
            5,

        "n_feature_sets":
            20,

        "n_examples":
            N,

        "contextual_dimensions":{
            "CNN1D":128,
            "BiGRU":188,
            "TransformerSmall":128,
            "MeanPool":128,
        },

        "semantic_matching_rule":
            "Use the exact native per-position tensor "
            "immediately before each encoder's own "
            "global pooling operation.",

        "meanpool_rule":
            "mean_ff(token) before permutation-invariant mean pooling",

        "formal_input_sha256":
            actual_sha,

        "encoder_training":
            False,

        "direct_operon_features_used":
            False,

        "feature_index":
            str(
                index_path
            ),

        "feature_index_sha256":
            sha256(
                index_path
            ),

        "script":
            str(
                Path(
                    __file__
                ).resolve()
            ),

        "script_sha256":
            sha256(
                Path(
                    __file__
                ).resolve()
            ),
    }


    freeze_path=(
        OUT
        /
        "57B2_native_pair_state_freeze.json"
    )

    freeze_path.write_text(
        json.dumps(
            freeze,
            indent=2,
        )
    )


    print(
        "\n"
        +
        "="*110,
        flush=True,
    )

    print(
        "[PASS] native pair-state sets = 20 / 20",
        flush=True,
    )

    print(
        "[PASS] CNN1D contextual = 128D",
        flush=True,
    )

    print(
        "[PASS] BiGRU contextual = 188D",
        flush=True,
    )

    print(
        "[PASS] TransformerSmall contextual = 128D",
        flush=True,
    )

    print(
        "[PASS] MeanPool contextual = 128D "
        "(order-invariant)",
        flush=True,
    )

    print(
        "[PASS] encoder training downstream = NONE",
        flush=True,
    )

    print(
        "[PASS] direct operon features = NONE",
        flush=True,
    )

    print(
        "[OUTPUT]",
        index_path,
        flush=True,
    )

    print(
        "ALL_20_NATIVE_MATCHED_PAIR_STATES_PASS",
        flush=True,
    )

    print(
        "="*110,
        flush=True,
    )


if __name__=="__main__":
    main()
