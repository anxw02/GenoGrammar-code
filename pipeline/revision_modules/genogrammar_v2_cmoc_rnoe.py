#!/usr/bin/env python3

"""
GenoGrammar-v2
==============

Exactly two preregistered additions:

1. RNOE
   Relative Neighborhood Organization Encoding

2. CMOC
   Controlled Matched-Order Contrastive objective

No DOOR2 labels or downstream operon metadata are used here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# FROZEN V2 HYPERPARAMETERS
# ============================================================

RNOE_FEATURES = 5

CMOC_WEIGHT = 0.25
CMOC_TEMPERATURE = 0.20

CMOC_MIN_SEGMENT = 8
CMOC_MAX_SEGMENT = 16


# ============================================================
# RNOE
# ============================================================

_CLASS_CACHE = {}


def _normalize_strand(strand: torch.Tensor) -> torch.Tensor:
    """
    Normalize strand representation to {-1,+1}.

    Supports:
      {0,1}
      {-1,+1}
      other signed numeric coding
    """

    s = strand.float()

    # Common binary 0/1 coding.
    if torch.all(
        (s == 0)
        |
        (s == 1)
    ):
        return (
            s * 2.0
            -
            1.0
        )

    s = torch.sign(s)

    # Defensive handling of exact zeros.
    s = torch.where(
        s == 0,
        torch.ones_like(s),
        s,
    )

    return s


def _rnoe_features(
    token: torch.Tensor,
    strand: torch.Tensor,
) -> torch.Tensor:

    """
    Five explicit local-organization features:

    0 relative gene position
    1 signed relative offset
    2 left/right relation
    3 strand relation to center gene
    4 smooth local-order bias

    Shape:
        [B,L,5]
    """

    if token.ndim != 3:
        raise RuntimeError(
            f"RNOE token ndim={token.ndim}, expected 3"
        )

    B, L, _ = token.shape

    if strand.shape != (B, L):
        raise RuntimeError(
            f"RNOE strand shape={tuple(strand.shape)} "
            f"expected={(B,L)}"
        )

    dtype = token.dtype
    device = token.device

    pos = torch.linspace(
        -1.0,
        1.0,
        L,
        device=device,
        dtype=dtype,
    )

    # Feature 1: normalized relative position.
    rel_pos = (
        pos
        .view(1, L)
        .expand(B, L)
    )

    # Feature 2: signed nonlinear offset.
    signed_offset = (
        torch.sign(rel_pos)
        *
        torch.sqrt(
            torch.abs(rel_pos)
            +
            1e-8
        )
    )

    # Feature 3: left versus right.
    left_right = torch.sign(
        rel_pos
    )

    # Center position itself = 0.
    left_right = torch.where(
        torch.abs(rel_pos) < 1e-8,
        torch.zeros_like(left_right),
        left_right,
    )

    # Feature 4: strand relationship to center gene.
    s = _normalize_strand(
        strand
    ).to(dtype)

    center = L // 2

    center_strand = (
        s[:, center]
        .unsqueeze(1)
    )

    strand_relation = (
        s
        *
        center_strand
    )

    # Feature 5: smooth local relative-order bias.
    local_order = torch.sin(
        torch.pi
        *
        rel_pos
    )

    return torch.stack(
        [
            rel_pos,
            signed_offset,
            left_right,
            strand_relation,
            local_order,
        ],
        dim=-1,
    )


def _channel_basis(
    dim: int,
    device,
    dtype,
) -> torch.Tensor:

    """
    Deterministic 5 x D channel basis.

    No extra hidden trainable projection is introduced.
    RNOE only learns five scalar gates.
    """

    if dim < RNOE_FEATURES:
        raise RuntimeError(
            f"token dim {dim} < {RNOE_FEATURES}"
        )

    basis = torch.zeros(
        RNOE_FEATURES,
        dim,
        device=device,
        dtype=dtype,
    )

    idx = torch.arange(
        dim,
        device=device,
    )

    for k in range(
        RNOE_FEATURES
    ):
        mask = (
            idx % RNOE_FEATURES
            ==
            k
        )

        basis[
            k,
            mask,
        ] = 1.0

    return basis


def make_v2_encoder(
    BaseEncoder,
):

    """
    Create a GenoGrammar-v2 encoder from the exact
    v1 ExplicitNeighborhoodEncoder class supplied by
    the frozen training script.
    """

    if BaseEncoder in _CLASS_CACHE:
        V2Class = _CLASS_CACHE[
            BaseEncoder
        ]
        return V2Class()

    class ExplicitNeighborhoodEncoderV2(
        BaseEncoder
    ):

        def __init__(self):

            super().__init__()

            # Only 5 additional trainable parameters.
            #
            # Zero initialization means training begins
            # exactly from the v1 token representation.
            self.rnoe_gate = nn.Parameter(
                torch.zeros(
                    RNOE_FEATURES,
                    dtype=torch.float32,
                )
            )

        def prepare_tokens(
            self,
            family_embedding,
            strand,
        ):

            token = super().prepare_tokens(
                family_embedding,
                strand,
            )

            feat = _rnoe_features(
                token,
                strand,
            )

            basis = _channel_basis(
                token.shape[-1],
                token.device,
                token.dtype,
            )

            gate = torch.tanh(
                self.rnoe_gate
            ).to(
                dtype=token.dtype,
                device=token.device,
            )

            # [B,L,5] x [5,D] -> [B,L,D]
            rnoe = torch.einsum(
                "blf,fd->bld",
                feat * gate.view(1, 1, -1),
                basis,
            )

            return (
                token
                +
                rnoe
            )

    ExplicitNeighborhoodEncoderV2.__name__ = (
        "ExplicitNeighborhoodEncoderV2"
    )

    _CLASS_CACHE[
        BaseEncoder
    ] = ExplicitNeighborhoodEncoderV2

    return ExplicitNeighborhoodEncoderV2()


# ============================================================
# FAMILY EMBEDDING RESOLUTION
# ============================================================

def _as_numpy_fam(
    fam
):

    if torch.is_tensor(fam):
        return (
            fam.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        fam
    )


def _valid_embedding_output(
    x,
    fam_shape,
):

    if torch.is_tensor(x):
        shape = tuple(
            x.shape
        )
    else:
        try:
            shape = tuple(
                np.asarray(x).shape
            )
        except Exception:
            return False

    return (
        len(shape) == 3
        and
        shape[:2]
        ==
        tuple(fam_shape)
    )


def _lookup_family_embedding(
    store,
    fam,
    device,
):

    """
    Resolve the frozen FamilyEmbeddingStore without
    assuming one implementation-specific attribute name.
    """

    fam_np = _as_numpy_fam(
        fam
    ).astype(
        np.int64,
        copy=False,
    )

    fam_shape = fam_np.shape

    if len(
        fam_shape
    ) != 2:
        raise RuntimeError(
            f"CMOC fam shape={fam_shape}, expected 2D"
        )

    # --------------------------------------------------------
    # First try explicit store methods.
    # --------------------------------------------------------

    method_names = [
        "lookup",
        "get_embeddings",
        "get_embedding",
        "get_rows",
        "fetch",
    ]

    for name in method_names:

        if not hasattr(
            store,
            name,
        ):
            continue

        fn = getattr(
            store,
            name,
        )

        if not callable(
            fn
        ):
            continue

        try:
            out = fn(
                fam_np
            )
        except Exception:
            continue

        if not _valid_embedding_output(
            out,
            fam_shape,
        ):
            continue

        return torch.as_tensor(
            out,
            dtype=torch.float32,
            device=device,
        )

    # --------------------------------------------------------
    # Then inspect 2-D embedding-like store attributes.
    # --------------------------------------------------------

    candidates = []

    for name, obj in vars(
        store
    ).items():

        try:

            if torch.is_tensor(
                obj
            ):
                shape = tuple(
                    obj.shape
                )

            elif isinstance(
                obj,
                np.ndarray,
            ):
                shape = tuple(
                    obj.shape
                )

            else:
                continue

        except Exception:
            continue

        if (
            len(shape) == 2
            and
            shape[1] >= 32
        ):
            candidates.append(
                (
                    name,
                    obj,
                    shape,
                )
            )

    max_id = int(
        fam_np[
            fam_np >= 0
        ].max()
    ) if np.any(
        fam_np >= 0
    ) else 0

    candidates = [
        x
        for x in candidates
        if x[2][0] > max_id
    ]

    if not candidates:
        raise RuntimeError(
            "CMOC could not resolve FamilyEmbeddingStore "
            f"2D embedding matrix. store attrs="
            f"{sorted(vars(store).keys())}"
        )

    # Prefer the widest matrix, then largest number of rows.
    candidates.sort(
        key=lambda x:
        (
            x[2][1],
            x[2][0],
        ),
        reverse=True,
    )

    _, matrix, _ = candidates[0]

    safe = fam_np.copy()

    invalid = (
        safe < 0
    )

    safe[
        invalid
    ] = 0

    if torch.is_tensor(
        matrix
    ):

        idx = torch.as_tensor(
            safe,
            dtype=torch.long,
            device=matrix.device,
        )

        out = matrix[
            idx
        ]

        out = out.to(
            device=device,
            dtype=torch.float32,
        )

    else:

        out = torch.as_tensor(
            np.asarray(
                matrix[
                    safe
                ]
            ),
            dtype=torch.float32,
            device=device,
        )

    if np.any(
        invalid
    ):

        mask = torch.as_tensor(
            invalid,
            dtype=torch.bool,
            device=device,
        )

        out = out.clone()
        out[
            mask
        ] = 0.0

    return out


# ============================================================
# CMOC CONTROLLED MATCHED-ORDER NEGATIVE
# ============================================================

def _controlled_order_perturbation(
    fam,
    strand,
):

    """
    Deterministic content-matched local order perturbation.

    Same genes.
    Same copy counts.
    Same per-gene strand assignment.
    Only local order changes.

    Each row receives one deterministic local segment reversal.
    """

    f = _as_numpy_fam(
        fam
    ).copy()

    if torch.is_tensor(
        strand
    ):
        s = (
            strand.detach()
            .cpu()
            .numpy()
            .copy()
        )
    else:
        s = np.asarray(
            strand
        ).copy()

    if (
        f.ndim != 2
        or
        s.shape != f.shape
    ):
        raise RuntimeError(
            f"CMOC shapes fam={f.shape} strand={s.shape}"
        )

    B, L = f.shape

    min_len = min(
        CMOC_MIN_SEGMENT,
        L,
    )

    max_len = min(
        CMOC_MAX_SEGMENT,
        L,
    )

    if min_len < 2:
        raise RuntimeError(
            "CMOC window too short"
        )

    for i in range(
        B
    ):

        row = f[
            i
        ]

        # Deterministic seed derived ONLY from family content.
        #
        # It does not consume the base ContextSSL RNG and
        # therefore does not change v1 objective sampling.
        key = int(
            np.abs(
                np.int64(
                    row[:min(8, L)]
                    .astype(np.int64)
                    .sum()
                )
            )
        )

        seg_len = (
            min_len
            +
            key
            %
            (
                max_len
                -
                min_len
                +
                1
            )
        )

        max_start = (
            L
            -
            seg_len
        )

        if max_start <= 0:
            start = 0
        else:
            key2 = int(
                np.abs(
                    np.int64(
                        row[-min(8, L):]
                        .astype(np.int64)
                        .sum()
                    )
                )
            )

            start = (
                key2
                %
                (
                    max_start
                    +
                    1
                )
            )

        end = (
            start
            +
            seg_len
        )

        # Reverse family+strand jointly:
        # content and per-gene strand identity are preserved.
        f[
            i,
            start:end,
        ] = f[
            i,
            start:end,
        ][::-1]

        s[
            i,
            start:end,
        ] = s[
            i,
            start:end,
        ][::-1]

    return f, s



def _orientation_equivalent_view(
    fam,
    strand,
):
    """
    Same biological neighborhood viewed from the
    opposite genomic orientation.

    Gene-family content is unchanged.
    Full order is reversed and strand orientation flipped.
    """

    f = _as_numpy_fam(fam).copy()

    if torch.is_tensor(strand):
        s = strand.detach().cpu().numpy().copy()
    else:
        s = np.asarray(strand).copy()

    f = f[:, ::-1].copy()
    s = s[:, ::-1].copy()

    uniq = set(
        np.unique(s).tolist()
    )

    if uniq.issubset({0, 1}):
        s = 1 - s
    else:
        s = -s

    return f, s


def cmoc_loss_only(
    model,
    store,
    fam,
    strand,
):
    """
    CMOC InfoNCE-style matched comparison.

    Anchor:
      original biological neighborhood

    Positive:
      orientation-equivalent whole-window view
      (reverse order + strand flip)

    Negative:
      same gene-family composition under a
      controlled local-order perturbation

    No downstream labels are used.
    """

    device = next(
        model.parameters()
    ).device

    fam_pos, strand_pos = (
        _orientation_equivalent_view(
            fam,
            strand,
        )
    )

    fam_neg, strand_neg = (
        _controlled_order_perturbation(
            fam,
            strand,
        )
    )

    emb_anchor = _lookup_family_embedding(
        store,
        fam,
        device,
    )

    emb_pos = _lookup_family_embedding(
        store,
        fam_pos,
        device,
    )

    emb_neg = _lookup_family_embedding(
        store,
        fam_neg,
        device,
    )

    strand_anchor = torch.as_tensor(
        strand,
        device=device,
    )

    strand_pos_t = torch.as_tensor(
        strand_pos,
        device=device,
    )

    strand_neg_t = torch.as_tensor(
        strand_neg,
        device=device,
    )

    z_anchor = model.encode_tokens(
        model.prepare_tokens(
            emb_anchor,
            strand_anchor,
        )
    )

    z_pos = model.encode_tokens(
        model.prepare_tokens(
            emb_pos,
            strand_pos_t,
        )
    )

    z_neg = model.encode_tokens(
        model.prepare_tokens(
            emb_neg,
            strand_neg_t,
        )
    )

    z_anchor = F.normalize(
        z_anchor,
        dim=-1,
    )

    z_pos = F.normalize(
        z_pos,
        dim=-1,
    )

    z_neg = F.normalize(
        z_neg,
        dim=-1,
    )

    sim_pos = (
        z_anchor * z_pos
    ).sum(dim=-1)

    sim_neg = (
        z_anchor * z_neg
    ).sum(dim=-1)

    logits = torch.stack(
        [
            sim_pos,
            sim_neg,
        ],
        dim=1,
    ) / CMOC_TEMPERATURE

    target = torch.zeros(
        logits.shape[0],
        dtype=torch.long,
        device=device,
    )

    return F.cross_entropy(
        logits,
        target,
    )


# ============================================================
# WRAPPED TRAINING OBJECTIVE
# ============================================================

def forward_objectives_v2(
    base_forward_objectives,
    model,
    store,
    fam,
    strand,
    rng,
):

    """
    Original frozen ContextSSL objective
    +
    preregistered CMOC regularizer.

    Validation/checkpoint selection remains the existing
    lineage-clean v1 validation objective, so downstream labels
    never participate in model selection.
    """

    base_loss, metrics = (
        base_forward_objectives(
            model,
            store,
            fam,
            strand,
            rng,
        )
    )

    cmoc = cmoc_loss_only(
        model,
        store,
        fam,
        strand,
    )

    total = (
        base_loss
        +
        CMOC_WEIGHT
        *
        cmoc
    )

    # Preserve EXACT metric-key contract expected by the
    # frozen training loop. Only the reported training loss
    # is updated to the total v2 objective.
    metrics = dict(
        metrics
    )

    if "loss" in metrics:
        metrics[
            "loss"
        ] = float(
            total.detach()
            .float()
            .cpu()
            .item()
        )

    return total, metrics
