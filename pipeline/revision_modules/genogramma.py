from __future__ import annotations

import torch
import torch.nn.functional as F

from genogrammar_v2_cmoc_rnoe import (
    make_v2_encoder as _make_encoder,
    forward_objectives_v2 as _forward_objectives,
)

MODEL_NAME = "GenoGramma"

PAIR_LEFT_INDEX = 31
PAIR_RIGHT_INDEX = 32

TOKEN_DIM = 128
PAIR_AWARE_DIM = 768


def make_encoder(base_encoder_cls):
    """
    Canonical GenoGramma encoder constructor.

    This is the public GenoGramma interface used by paper_code.
    """
    return _make_encoder(base_encoder_cls)


def forward_objectives(
    base_forward_objectives,
    *args,
    **kwargs,
):
    """
    Canonical GenoGramma representation-learning objective.
    """
    return _forward_objectives(
        base_forward_objectives,
        *args,
        **kwargs,
    )


def encode_pairaware(
    model,
    token,
    left_index: int = PAIR_LEFT_INDEX,
    right_index: int = PAIR_RIGHT_INDEX,
):
    """
    Extract the fixed GenoGramma pair-aware representation.

    Composition:
      1. contextual input token at left target gene
      2. contextual input token at right target gene
      3. absolute token difference
      4. element-wise token interaction
      5. contextual target edge after edge convolutions
      6. whole-window global GenoGramma representation

    No labels, genomic coordinates, intergenic distance,
    operon IDs, or annotation-bridge variables are used.
    """

    if token.ndim != 3:
        raise RuntimeError(
            f"token ndim={token.ndim}, expected 3"
        )

    B, L, D = token.shape

    if D != TOKEN_DIM:
        raise RuntimeError(
            f"token dim={D}, expected {TOKEN_DIM}"
        )

    if right_index != left_index + 1:
        raise RuntimeError(
            "Target pair must be adjacent."
        )

    if left_index < 0 or right_index >= L:
        raise RuntimeError(
            f"Target indices {(left_index,right_index)} "
            f"invalid for window length {L}"
        )

    # -------------------------------------------------------------------------
    # Same edge computation used by GenoGramma.encode_tokens()
    # -------------------------------------------------------------------------

    a = token[:, :-1, :]
    b = token[:, 1:, :]

    edge = model.edge_mlp(
        model.pair_feature(
            a,
            b,
        )
    )

    x = edge.transpose(
        1,
        2,
    )

    x = F.gelu(
        model.edge_conv1(x)
    )

    x = F.gelu(
        model.edge_conv2(x)
    )

    x = x.transpose(
        1,
        2,
    )

    edge_context = model.edge_norm(
        x + edge
    )

    weight = torch.softmax(
        model.edge_gate(
            edge_context
        ).squeeze(-1),
        dim=1,
    )

    pooled = (
        edge_context
        *
        weight.unsqueeze(-1)
    ).sum(dim=1)

    global_z = model.out_proj(
        pooled
    )

    global_z = F.normalize(
        global_z,
        dim=-1,
    )

    # -------------------------------------------------------------------------
    # Fixed target-pair readout
    # -------------------------------------------------------------------------

    left = token[
        :,
        left_index,
        :
    ]

    right = token[
        :,
        right_index,
        :
    ]

    target_edge = edge_context[
        :,
        left_index,
        :
    ]

    pair_rep = torch.cat(
        [
            left,
            right,
            torch.abs(
                right - left
            ),
            left * right,
            target_edge,
            global_z,
        ],
        dim=-1,
    )

    if pair_rep.shape != (
        B,
        PAIR_AWARE_DIM,
    ):
        raise RuntimeError(
            f"pair-aware shape={tuple(pair_rep.shape)}, "
            f"expected {(B,PAIR_AWARE_DIM)}"
        )

    return pair_rep
