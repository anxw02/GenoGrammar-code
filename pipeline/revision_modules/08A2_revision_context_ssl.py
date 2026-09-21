"""
R4 ContextSSL objective module.

Training objectives
-------------------
1. token-level adjacency discrimination
2. token-level neighbor retrieval
3. masked-view contextual contrastive consistency

The ContextSSL branch passes both masked views through
ExplicitNeighborhoodEncoder.encode_tokens().

Excluded objectives
-------------------
- shuffle classification
- GLOBAL > LOCAL ranking
- GLOBAL-LOCAL separation
- phenotype labels

No trainable parameters are added relative to the canonical
508,182-parameter GenoGrammar architecture.
"""



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage08A2
Explicit Gene-Neighborhood / Order SSL smoke test

IMPORTANT
---------
Stage08A remains frozen as FAIL.
This is a NEW architecture qualification experiment.

No phenotype labels are used.

Core idea
---------
Instead of hoping a generic Transformer retains gene order, explicitly encode
ordered adjacent-gene edges:

gene_i -> gene_(i+1)

The final representation is generated from the ordered edge sequence.

Training objectives:
1. TRUE / LOCAL / GLOBAL perturbation classification
2. TRUE-GLOBAL > TRUE-LOCAL embedding-distance ranking
3. explicit global-separation / local-consistency objective
4. biological adjacency discrimination
5. next-neighbor contrastive prediction

Outer Fold 1 remains untouched until final smoke evaluation.
"""

import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CANON_EMB = (
    ROOT /
    "08_explicit_order_ssl" /
    "00_family_embeddings_276517x480.npy"
)

CACHE_DIR = (
    ROOT /
    "07_1_ssl_encoder" /
    "data_cache"
)

SSL_ROOT = (
    ROOT /
    "07_1_ssl_encoder"
)

ANI_TABLE = (
    ROOT /
    "01_1_ani_robustness" /
    "06_FINAL_robust_ANI_blocks.csv"
)

OUT = (
    ROOT /
    "08_explicit_order_ssl_A2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Frozen configuration
# ============================================================

SEED = 42

OUTER_FOLD = 1

WINDOW_LEN = 64
MIN_CONTIG_GENES = 64

ESM_DIM = 480

MODEL_DIM = 128
STRAND_DIM = 8

BATCH_SIZE = 48

EPOCHS = 5
STEPS_PER_EPOCH = 400

VAL_BATCHES = 30
OUTER_BATCHES = 40

LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Ranking/separation
RANK_MARGIN = 0.05

TARGET_GLOBAL_DISP = 0.08
MAX_LOCAL_DISP = 0.04

# Loss weights
W_SHUFFLE = 0.0
W_RANK = 0.0
W_SEPARATION = 0.0
W_ADJ = 0.75
W_NEIGHBOR = 0.20

W_CONTEXT = 0.20
CONTEXT_MASK_PROB = 0.15
CONTEXT_TEMP = 0.10
NEIGHBOR_TEMP = 0.10


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(
    True,
    warn_only=False
)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Generic helpers
# ============================================================

def read_csv(path):

    with open(
        path,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


def stable_hash(text):

    return int(
        hashlib.sha256(
            str(text).encode()
        ).hexdigest()[:16],
        16
    )


def detect_sid_column(cols):

    preferred = [
        "strain_entity_id",
        "sample_id",
        "strain_id",
        "sid",
    ]

    lower = {
        c.lower(): c
        for c in cols
    }

    for x in preferred:
        if x in lower:
            return lower[x]

    for c in cols:

        low = c.lower()

        if (
            "strain" in low
            and
            "id" in low
        ):
            return c

    return cols[0]


# ============================================================
# Family embeddings
# ============================================================

def verify_embeddings():

    if not CANON_EMB.exists():

        raise RuntimeError(
            f"Missing canonical embeddings: "
            f"{CANON_EMB}"
        )

    x = np.load(
        CANON_EMB,
        mmap_mode="r",
        allow_pickle=False,
    )

    if tuple(x.shape) != (
        276517,
        ESM_DIM,
    ):
        raise RuntimeError(
            f"Unexpected embedding shape: "
            f"{x.shape}"
        )

    if x.dtype not in (
        np.float16,
        np.float32,
    ):
        raise RuntimeError(
            f"Unexpected dtype: {x.dtype}"
        )

    probe = np.asarray(
        x[
            [
                0,
                1,
                1000,
                276516,
            ]
        ],
        dtype=np.float32,
    )

    if not np.isfinite(
        probe
    ).all():
        raise RuntimeError(
            "Non-finite family embeddings."
        )

    print(
        "[PASS] canonical embeddings:",
        CANON_EMB
    )

    print(
        "[PASS] embedding shape:",
        x.shape,
        "| dtype=",
        x.dtype,
    )

    return CANON_EMB


class FamilyEmbeddingStore:

    def __init__(self, path):

        self.matrix = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

    def lookup(self, rows):

        x = np.asarray(
            self.matrix[rows],
            dtype=np.float32,
        )

        return torch.from_numpy(
            x
        ).to(
            DEVICE,
            non_blocking=True,
        )


# ============================================================
# Fold assignments
# ============================================================

def load_outer_folds():

    mapping = {}

    for fold in range(1, 6):

        p = (
            SSL_ROOT /
            f"fold_{fold}" /
            "outer_holdout_embedding_index.csv"
        )

        if not p.exists():
            raise RuntimeError(
                f"Missing {p}"
            )

        rows = read_csv(p)

        if not rows:
            raise RuntimeError(
                f"Empty {p}"
            )

        sid_col = detect_sid_column(
            list(rows[0].keys())
        )

        for r in rows:

            sid = (
                r[sid_col]
                .strip()
            )

            if not sid:
                continue

            if sid in mapping:
                raise RuntimeError(
                    f"Duplicate fold assignment: "
                    f"{sid}"
                )

            mapping[sid] = fold

    if len(mapping) != 979:
        raise RuntimeError(
            f"Expected 979 genomes; "
            f"found {len(mapping)}"
        )

    counts = defaultdict(int)

    for f in mapping.values():
        counts[f] += 1

    print(
        "[PASS] outer folds:",
        dict(sorted(counts.items()))
    )

    return mapping


# ============================================================
# ANI near-clone blocks
# ============================================================

def load_nearclone_groups():

    rows = read_csv(
        ANI_TABLE
    )

    if not rows:
        raise RuntimeError(
            "ANI table empty."
        )

    cols = list(
        rows[0].keys()
    )

    sid_col = detect_sid_column(
        cols
    )

    candidates = [
        c for c in cols
        if (
            "near" in c.lower()
            or
            "99_5" in c.lower()
            or
            "99.5" in c.lower()
            or
            "995" in c.lower()
        )
    ]

    if not candidates:

        candidates = [
            c for c in cols
            if "block" in c.lower()
        ]

    if not candidates:
        raise RuntimeError(
            "No near-clone block column."
        )

    candidates.sort(
        key=lambda c: (
            0
            if "near" in c.lower()
            else 1,
            c,
        )
    )

    block_col = candidates[0]

    out = {}

    for r in rows:

        sid = r[
            sid_col
        ].strip()

        block = r[
            block_col
        ].strip()

        if sid and block:
            out[sid] = block

    print(
        "[PASS] near-clone mapping:",
        len(out),
        "| column=",
        block_col,
    )

    return out


# ============================================================
# Genome cache
# ============================================================

def load_genome(sid):

    p = (
        CACHE_DIR /
        f"{sid}.npz"
    )

    if not p.exists():
        raise RuntimeError(
            f"Missing cache: {p}"
        )

    with np.load(
        p,
        allow_pickle=False,
    ) as z:

        return {
            "family":
                np.asarray(
                    z["family_rows"],
                    dtype=np.int32,
                ),

            "strand":
                np.asarray(
                    z["strand"],
                    dtype=np.int64,
                ),

            "offsets":
                np.asarray(
                    z["contig_offsets"],
                    dtype=np.int32,
                ),
        }


def build_eligible_contigs(sids):

    out = {}

    total = 0

    for sid in sids:

        g = load_genome(sid)

        offsets = g["offsets"]

        rr = []

        for i in range(
            len(offsets) - 1
        ):

            a = int(offsets[i])
            b = int(offsets[i + 1])

            if (
                b - a
                >= MIN_CONTIG_GENES
            ):

                rr.append(
                    (a, b)
                )

        if rr:

            out[sid] = rr
            total += len(rr)

    print(
        "[PASS] eligible genomes:",
        len(out),
        "/",
        len(sids),
        "| eligible contigs:",
        total,
    )

    return out


# ============================================================
# Leakage-free split
# ============================================================

def build_split(
    outer_folds,
    nearclone,
):

    outer_test = sorted(
        sid
        for sid, fold
        in outer_folds.items()
        if fold == OUTER_FOLD
    )

    outer_train = sorted(
        sid
        for sid, fold
        in outer_folds.items()
        if fold != OUTER_FOLD
    )

    test_blocks = {
        nearclone[s]
        for s in outer_test
        if s in nearclone
    }

    train_blocks = {
        nearclone[s]
        for s in outer_train
        if s in nearclone
    }

    overlap = (
        test_blocks
        &
        train_blocks
    )

    if overlap:
        raise RuntimeError(
            f"Outer near-clone leakage: "
            f"{len(overlap)}"
        )

    groups = defaultdict(list)

    for sid in outer_train:

        if sid in nearclone:
            groups[
                nearclone[sid]
            ].append(sid)

    inner_train = []
    inner_val = []

    for block, members in groups.items():

        h = (
            stable_hash(
                f"A2::{SEED}::{block}"
            )
            % 10000
        )

        target = (
            inner_val
            if h < 1500
            else inner_train
        )

        target.extend(
            members
        )

    inner_train.sort()
    inner_val.sort()

    tr_blocks = {
        nearclone[s]
        for s in inner_train
    }

    va_blocks = {
        nearclone[s]
        for s in inner_val
    }

    if (
        tr_blocks
        &
        va_blocks
    ):
        raise RuntimeError(
            "Inner near-clone leakage."
        )

    print(
        "[PASS] leakage-free split:"
    )

    print(
        "       inner train =",
        len(inner_train)
    )

    print(
        "       inner val   =",
        len(inner_val)
    )

    print(
        "       outer test  =",
        len(outer_test)
    )

    return (
        inner_train,
        inner_val,
        outer_test,
    )


# ============================================================
# Window sampler
# ============================================================

class WindowSampler:

    def __init__(
        self,
        sids,
        eligible,
        seed,
    ):

        self.sids = [
            s
            for s in sids
            if s in eligible
        ]

        if not self.sids:
            raise RuntimeError(
                "No eligible genomes."
            )

        self.eligible = eligible

        self.rng = (
            np.random.default_rng(
                seed
            )
        )

        self.cache = {}

    def genome(self, sid):

        if sid not in self.cache:

            self.cache[sid] = (
                load_genome(sid)
            )

        return self.cache[sid]

    def sample_one(self):

        sid = self.sids[
            int(
                self.rng.integers(
                    len(self.sids)
                )
            )
        ]

        g = self.genome(sid)

        contigs = (
            self.eligible[sid]
        )

        a, b = contigs[
            int(
                self.rng.integers(
                    len(contigs)
                )
            )
        ]

        max_start = (
            b - WINDOW_LEN
        )

        if max_start == a:
            start = a
        else:
            start = int(
                self.rng.integers(
                    a,
                    max_start + 1,
                )
            )

        end = (
            start
            +
            WINDOW_LEN
        )

        return (
            g["family"][
                start:end
            ].copy(),

            g["strand"][
                start:end
            ].copy(),
        )

    def sample_batch(
        self,
        batch_size,
    ):

        fam = []
        strand = []

        for _ in range(
            batch_size
        ):

            f, s = self.sample_one()

            fam.append(f)
            strand.append(s)

        return (
            np.stack(fam),
            np.stack(strand),
        )


# ============================================================
# Perturbations
# ============================================================

def make_permutations(
    B,
    L,
    rng,
):

    local = []
    global_ = []

    for _ in range(B):

        # LOCAL_W10-like
        p = np.arange(
            L,
            dtype=np.int64,
        )

        for a in range(
            0,
            L,
            10
        ):

            b = min(
                a + 10,
                L
            )

            block = (
                p[a:b]
                .copy()
            )

            rng.shuffle(
                block
            )

            p[a:b] = block

        local.append(p)

        g = np.arange(
            L,
            dtype=np.int64,
        )

        rng.shuffle(g)

        global_.append(g)

    return (
        np.stack(local),
        np.stack(global_),
    )


def permute_tokens(
    x,
    p,
):

    B, L = p.shape

    batch = (
        torch.arange(
            B,
            device=x.device,
        )
        .unsqueeze(1)
        .expand(B, L)
    )

    return x[
        batch,
        p,
    ]


# ============================================================
# Explicit neighborhood encoder
# ============================================================

class ExplicitNeighborhoodEncoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.family_proj = (
            nn.Sequential(
                nn.Linear(
                    ESM_DIM,
                    MODEL_DIM,
                ),
                nn.LayerNorm(
                    MODEL_DIM
                ),
                nn.GELU(),
            )
        )

        self.strand_emb = (
            nn.Embedding(
                2,
                STRAND_DIM,
            )
        )

        self.token_fuse = (
            nn.Sequential(
                nn.Linear(
                    MODEL_DIM
                    +
                    STRAND_DIM,
                    MODEL_DIM,
                ),
                nn.LayerNorm(
                    MODEL_DIM
                ),
                nn.GELU(),
            )
        )

        # Ordered adjacent pair:
        # [a, b, b-a, a*b]
        self.edge_mlp = (
            nn.Sequential(
                nn.Linear(
                    MODEL_DIM * 4,
                    MODEL_DIM * 2,
                ),
                nn.GELU(),

                nn.Linear(
                    MODEL_DIM * 2,
                    MODEL_DIM,
                ),
                nn.LayerNorm(
                    MODEL_DIM
                ),
                nn.GELU(),
            )
        )

        # Explicitly preserve sequence of edges.
        self.edge_conv1 = (
            nn.Conv1d(
                MODEL_DIM,
                MODEL_DIM,
                kernel_size=3,
                padding=1,
            )
        )

        self.edge_conv2 = (
            nn.Conv1d(
                MODEL_DIM,
                MODEL_DIM,
                kernel_size=5,
                padding=2,
            )
        )

        self.edge_norm = (
            nn.LayerNorm(
                MODEL_DIM
            )
        )

        self.edge_gate = (
            nn.Linear(
                MODEL_DIM,
                1,
            )
        )

        self.out_proj = (
            nn.Sequential(
                nn.Linear(
                    MODEL_DIM,
                    MODEL_DIM,
                ),
                nn.LayerNorm(
                    MODEL_DIM
                ),
            )
        )

        # TRUE / LOCAL / GLOBAL
        self.shuffle_head = (
            nn.Sequential(
                nn.Linear(
                    MODEL_DIM,
                    MODEL_DIM,
                ),
                nn.GELU(),

                nn.Linear(
                    MODEL_DIM,
                    3,
                ),
            )
        )

        # Biological adjacency
        self.adj_head = (
            nn.Sequential(
                nn.Linear(
                    MODEL_DIM * 4,
                    MODEL_DIM,
                ),
                nn.GELU(),

                nn.Linear(
                    MODEL_DIM,
                    2,
                ),
            )
        )

        # Contrastive next-neighbor
        self.query_proj = (
            nn.Linear(
                MODEL_DIM,
                MODEL_DIM,
                bias=False,
            )
        )

        self.key_proj = (
            nn.Linear(
                MODEL_DIM,
                MODEL_DIM,
                bias=False,
            )
        )

    def prepare_tokens(
        self,
        family_embedding,
        strand,
    ):

        x = self.family_proj(
            family_embedding
        )

        s = self.strand_emb(
            strand
        )

        x = self.token_fuse(
            torch.cat(
                [x, s],
                dim=-1,
            )
        )

        return x

    @staticmethod
    def pair_feature(
        a,
        b,
    ):

        return torch.cat(
            [
                a,
                b,
                b - a,
                a * b,
            ],
            dim=-1,
        )

    def encode_tokens(
        self,
        token,
    ):

        # Adjacent ordered edges
        a = token[:, :-1]
        b = token[:, 1:]

        edge = self.edge_mlp(
            self.pair_feature(
                a,
                b,
            )
        )

        x = edge.transpose(
            1,
            2
        )

        x = F.gelu(
            self.edge_conv1(x)
        )

        x = F.gelu(
            self.edge_conv2(x)
        )

        x = x.transpose(
            1,
            2
        )

        x = self.edge_norm(
            x + edge
        )

        weight = torch.softmax(
            self.edge_gate(x)
            .squeeze(-1),
            dim=1,
        )

        pooled = (
            x
            *
            weight.unsqueeze(-1)
        ).sum(dim=1)

        z = self.out_proj(
            pooled
        )

        z = F.normalize(
            z,
            dim=-1,
        )

        return z


# ============================================================
# Pair sampling
# ============================================================

def sample_pair_indices(
    B,
    L,
    rng,
):

    # adjacency task
    labels = rng.integers(
        0,
        2,
        size=B,
        dtype=np.int64,
    )

    i_adj = np.zeros(
        B,
        dtype=np.int64,
    )

    j_adj = np.zeros(
        B,
        dtype=np.int64,
    )

    for b in range(B):

        if labels[b] == 1:

            delta = 1

        else:

            delta = int(
                rng.integers(
                    3,
                    17
                )
            )

        i = int(
            rng.integers(
                0,
                L - delta
            )
        )

        i_adj[b] = i
        j_adj[b] = (
            i + delta
        )

    # next-neighbor contrastive task
    anchor = rng.integers(
        0,
        L - 1,
        size=B,
        dtype=np.int64,
    )

    nxt = anchor + 1

    return (
        labels,
        i_adj,
        j_adj,
        anchor,
        nxt,
    )


# ============================================================
# Forward objectives
# ============================================================

def forward_objectives(
    model,
    store,
    fam_np,
    strand_np,
    rng,
):

    B, L = fam_np.shape

    family = store.lookup(
        fam_np
    )

    strand = (
        torch.from_numpy(
            strand_np.astype(
                np.int64,
                copy=False,
            )
        )
        .to(DEVICE)
    )

    token = model.prepare_tokens(
        family,
        strand,
    )

    (
        p_local_np,
        p_global_np,
    ) = make_permutations(
        B,
        L,
        rng,
    )

    p_local = (
        torch.from_numpy(
            p_local_np
        )
        .to(DEVICE)
    )

    p_global = (
        torch.from_numpy(
            p_global_np
        )
        .to(DEVICE)
    )

    token_local = (
        permute_tokens(
            token,
            p_local,
        )
    )

    token_global = (
        permute_tokens(
            token,
            p_global,
        )
    )

    all_token = torch.cat(
        [
            token,
            token_local,
            token_global,
        ],
        dim=0,
    )

    all_z = model.encode_tokens(
        all_token
    )

    z_true = all_z[:B]

    z_local = all_z[
        B:2 * B
    ]

    z_global = all_z[
        2 * B:
    ]

    # --------------------------------------------------------
    # Shuffle classification
    # --------------------------------------------------------

    logits_shuffle = (
        model.shuffle_head(
            all_z
        )
    )

    target_shuffle = torch.cat(
        [
            torch.zeros(
                B,
                dtype=torch.long,
                device=DEVICE,
            ),

            torch.ones(
                B,
                dtype=torch.long,
                device=DEVICE,
            ),

            torch.full(
                (B,),
                2,
                dtype=torch.long,
                device=DEVICE,
            ),
        ]
    )

    loss_shuffle = (
        F.cross_entropy(
            logits_shuffle,
            target_shuffle,
        )
    )

    # --------------------------------------------------------
    # Embedding perturbation
    # --------------------------------------------------------

    d_local = (
        1.0
        -
        F.cosine_similarity(
            z_true,
            z_local,
            dim=-1,
        )
    )

    d_global = (
        1.0
        -
        F.cosine_similarity(
            z_true,
            z_global,
            dim=-1,
        )
    )

    # Explicit ordering
    loss_rank = (
        F.relu(
            RANK_MARGIN
            +
            d_local
            -
            d_global
        )
        .mean()
    )

    # Enforce meaningful separation
    loss_global_sep = (
        F.relu(
            TARGET_GLOBAL_DISP
            -
            d_global
        )
        .mean()
    )

    loss_local_consistency = (
        F.relu(
            d_local
            -
            MAX_LOCAL_DISP
        )
        .mean()
    )

    loss_separation = (
        loss_global_sep
        +
        0.5
        *
        loss_local_consistency
    )

    # --------------------------------------------------------
    # Biological adjacency discrimination
    # --------------------------------------------------------

    (
        labels_np,
        ia_np,
        ja_np,
        anchor_np,
        next_np,
    ) = sample_pair_indices(
        B,
        L,
        rng,
    )

    labels = (
        torch.from_numpy(
            labels_np
        )
        .to(
            DEVICE,
            dtype=torch.long,
        )
    )

    ia = (
        torch.from_numpy(
            ia_np
        )
        .to(DEVICE)
    )

    ja = (
        torch.from_numpy(
            ja_np
        )
        .to(DEVICE)
    )

    batch_idx = torch.arange(
        B,
        device=DEVICE,
    )

    a = token[
        batch_idx,
        ia,
    ]

    b = token[
        batch_idx,
        ja,
    ]

    adj_logits = (
        model.adj_head(
            model.pair_feature(
                a,
                b,
            )
        )
    )

    loss_adj = (
        F.cross_entropy(
            adj_logits,
            labels,
        )
    )

    # --------------------------------------------------------
    # Next-neighbor contrastive prediction
    # --------------------------------------------------------

    anchor = (
        torch.from_numpy(
            anchor_np
        )
        .to(DEVICE)
    )

    nxt = (
        torch.from_numpy(
            next_np
        )
        .to(DEVICE)
    )

    q = token[
        batch_idx,
        anchor,
    ]

    k = token[
        batch_idx,
        nxt,
    ]

    q = F.normalize(
        model.query_proj(q),
        dim=-1,
    )

    k = F.normalize(
        model.key_proj(k),
        dim=-1,
    )

    contrast_logits = (
        q @ k.T
    ) / NEIGHBOR_TEMP

    contrast_target = (
        torch.arange(
            B,
            device=DEVICE,
        )
    )

    loss_neighbor = (
        F.cross_entropy(
            contrast_logits,
            contrast_target,
        )
    )


    # --------------------------------------------------------
    # R4 ContextSSL
    # --------------------------------------------------------
    #
    # Two independently token-masked views.
    #
    # IMPORTANT:
    # - original gene order is preserved
    # - no shuffle is used
    # - no GLOBAL/LOCAL perturbation is used
    # - no rank/separation target is used
    # - no phenotype label is used
    #
    # Both views MUST pass through encode_tokens(), therefore
    # this loss trains the explicit sequence-encoding core.
    # --------------------------------------------------------

    context_mask_a = (
        torch.rand(
            (
                token.shape[0],
                token.shape[1],
            ),
            device=token.device,
        )
        <
        CONTEXT_MASK_PROB
    )

    context_mask_b = (
        torch.rand(
            (
                token.shape[0],
                token.shape[1],
            ),
            device=token.device,
        )
        <
        CONTEXT_MASK_PROB
    )

    # Guarantee at least one masked position in each view.
    # With L=64 and p=0.15 this is rarely needed, but making
    # it explicit prevents accidental identity views.
    no_mask_a = ~context_mask_a.any(
        dim=1
    )

    no_mask_b = ~context_mask_b.any(
        dim=1
    )

    if no_mask_a.any():

        rows = torch.where(
            no_mask_a
        )[0]

        pos = torch.randint(
            low=0,
            high=token.shape[1],
            size=(rows.numel(),),
            device=token.device,
        )

        context_mask_a[
            rows,
            pos,
        ] = True

    if no_mask_b.any():

        rows = torch.where(
            no_mask_b
        )[0]

        pos = torch.randint(
            low=0,
            high=token.shape[1],
            size=(rows.numel(),),
            device=token.device,
        )

        context_mask_b[
            rows,
            pos,
        ] = True


    context_token_a = (
        token.masked_fill(
            context_mask_a.unsqueeze(-1),
            0.0,
        )
    )

    context_token_b = (
        token.masked_fill(
            context_mask_b.unsqueeze(-1),
            0.0,
        )
    )


    context_z_a = (
        model.encode_tokens(
            context_token_a
        )
    )

    context_z_b = (
        model.encode_tokens(
            context_token_b
        )
    )


    context_logits = (
        context_z_a
        @
        context_z_b.T
    ) / CONTEXT_TEMP


    context_target = (
        torch.arange(
            B,
            device=token.device,
        )
    )


    loss_context_ab = (
        F.cross_entropy(
            context_logits,
            context_target,
        )
    )

    loss_context_ba = (
        F.cross_entropy(
            context_logits.T,
            context_target,
        )
    )

    loss_context = (
        0.5
        *
        (
            loss_context_ab
            +
            loss_context_ba
        )
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    loss = (
        W_SHUFFLE
        *
        loss_shuffle

        +
        W_RANK
        *
        loss_rank

        +
        W_SEPARATION
        *
        loss_separation

        +
        W_ADJ
        *
        loss_adj

        +
        W_NEIGHBOR
        *
        loss_neighbor

        +
        W_CONTEXT
        *
        loss_context
    )

    metrics = {
        "loss":
            float(
                loss.detach()
            ),

        "loss_shuffle":
            float(
                loss_shuffle.detach()
            ),

        "loss_rank":
            float(
                loss_rank.detach()
            ),

        "loss_separation":
            float(
                loss_separation.detach()
            ),

        "loss_adj":
            float(
                loss_adj.detach()
            ),

        "loss_neighbor":
            float(
                loss_neighbor.detach()
            ),

        "shuffle_correct":
            int(
                (
                    logits_shuffle
                    .argmax(dim=1)
                    ==
                    target_shuffle
                )
                .sum()
                .item()
            ),

        "shuffle_total":
            3 * B,

        "rank_correct":
            int(
                (
                    d_global
                    >
                    d_local
                )
                .sum()
                .item()
            ),

        "rank_total":
            B,

        "strong_rank_correct":
            int(
                (
                    d_global
                    >
                    d_local
                    + 0.02
                )
                .sum()
                .item()
            ),

        "d_local_sum":
            float(
                d_local
                .sum()
                .item()
            ),

        "d_global_sum":
            float(
                d_global
                .sum()
                .item()
            ),

        "disp_total":
            B,

        "adj_correct":
            int(
                (
                    adj_logits
                    .argmax(dim=1)
                    ==
                    labels
                )
                .sum()
                .item()
            ),

        "adj_total":
            B,

        "neighbor_correct":
            int(
                (
                    contrast_logits
                    .argmax(dim=1)
                    ==
                    contrast_target
                )
                .sum()
                .item()
            ),

        "neighbor_total":
            B,
    }

    return (
        loss,
        metrics,
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    store,
    sids,
    eligible,
    n_batches,
    seed,
):

    model.eval()

    sampler = WindowSampler(
        sids,
        eligible,
        seed,
    )

    rng = np.random.default_rng(
        seed + 100000
    )

    acc = defaultdict(float)

    for _ in range(
        n_batches
    ):

        fam, strand = (
            sampler.sample_batch(
                BATCH_SIZE
            )
        )

        _, m = forward_objectives(
            model,
            store,
            fam,
            strand,
            rng,
        )

        for k, v in m.items():
            acc[k] += v

    d_local = (
        acc["d_local_sum"]
        /
        acc["disp_total"]
    )

    d_global = (
        acc["d_global_sum"]
        /
        acc["disp_total"]
    )

    return {
        "loss":
            acc["loss"]
            /
            n_batches,

        "shuffle_accuracy":
            acc["shuffle_correct"]
            /
            acc["shuffle_total"],

        "rank_fraction_global_gt_local":
            acc["rank_correct"]
            /
            acc["rank_total"],

        "strong_rank_fraction":
            acc["strong_rank_correct"]
            /
            acc["rank_total"],

        "local_displacement_mean":
            d_local,

        "global_displacement_mean":
            d_global,

        "global_minus_local":
            (
                d_global
                -
                d_local
            ),

        "adjacency_accuracy":
            acc["adj_correct"]
            /
            acc["adj_total"],

        "neighbor_top1_accuracy":
            acc[
                "neighbor_correct"
            ]
            /
            acc[
                "neighbor_total"
            ],
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 88
    )

    print(
        " Stage08A2 EXPLICIT "
        "GENE-NEIGHBORHOOD SSL SMOKE"
    )

    print(
        "=" * 88
    )

    print(
        "[INFO] device:",
        DEVICE
    )

    if DEVICE.type != "cuda":

        raise RuntimeError(
            "CUDA required."
        )

    emb_path = (
        verify_embeddings()
    )

    outer_folds = (
        load_outer_folds()
    )

    nearclone = (
        load_nearclone_groups()
    )

    (
        inner_train,
        inner_val,
        outer_test,
    ) = build_split(
        outer_folds,
        nearclone,
    )

    eligible = (
        build_eligible_contigs(
            sorted(
                outer_folds.keys()
            )
        )
    )

    train_sids = [
        s for s in inner_train
        if s in eligible
    ]

    val_sids = [
        s for s in inner_val
        if s in eligible
    ]

    test_sids = [
        s for s in outer_test
        if s in eligible
    ]

    print(
        "[PASS] usable:"
    )

    print(
        "       train =",
        len(train_sids)
    )

    print(
        "       val   =",
        len(val_sids)
    )

    print(
        "       test  =",
        len(test_sids)
    )

    if len(train_sids) < 500:
        raise RuntimeError(
            "Too few training genomes."
        )

    if len(val_sids) < 50:
        raise RuntimeError(
            "Too few validation genomes."
        )

    if len(test_sids) < 100:
        raise RuntimeError(
            "Too few test genomes."
        )

    # --------------------------------------------------------
    # Freeze prospective acceptance criteria
    # --------------------------------------------------------

    definition = {
        "stage":
            "Stage08A2",

        "timestamp":
            datetime.now().isoformat(),

        "previous_stage08A_status":
            "FAIL_FROZEN",

        "phenotype_labels_used":
            False,

        "outer_fold":
            OUTER_FOLD,

        "outer_holdout_used_for_training":
            False,

        "outer_holdout_used_for_early_stopping":
            False,

        "nearclone_grouped_inner_split":
            True,

        "architecture":
            (
                "ordered adjacent-gene edge MLP "
                "+ edge-sequence 1D convolution "
                "+ attention pooling"
            ),

        "objectives": [
            "TRUE_LOCAL_GLOBAL classification",
            "GLOBAL_gt_LOCAL embedding ranking",
            "global separation and local consistency",
            "biological adjacency discrimination",
            "next-neighbor contrastive prediction",
        ],

        "acceptance_criteria": {
            "shuffle_accuracy":
                ">=0.60",

            "rank_fraction_global_gt_local":
                ">=0.80",

            "global_displacement":
                ">=0.05",

            "global_minus_local":
                ">=0.03",

            "adjacency_accuracy":
                ">=0.65",
        },

        "supporting_metric_only": {
            "neighbor_top1_accuracy":
                (
                    "reported but not required "
                    "for Stage08A2 PASS"
                )
        },

        "scientific_boundary":
            (
                "Architecture qualification only. "
                "No phenotype inference."
            ),
    }

    (
        OUT /
        "00_STAGE08A2_definition.json"
    ).write_text(
        json.dumps(
            definition,
            indent=2,
        ),
        encoding="utf-8",
    )

    store = FamilyEmbeddingStore(
        emb_path
    )

    train_sampler = WindowSampler(
        train_sids,
        eligible,
        SEED + 1,
    )

    model = (
        ExplicitNeighborhoodEncoder()
        .to(DEVICE)
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "[INFO] trainable parameters:",
        n_params
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    # --------------------------------------------------------
    # Pre-training inner-val baseline
    # Outer fold remains untouched.
    # --------------------------------------------------------

    pre_val = evaluate(
        model,
        store,
        val_sids,
        eligible,
        VAL_BATCHES,
        SEED + 500,
    )

    print()
    print(
        "[PRE-VAL]"
        f" shuffle={pre_val['shuffle_accuracy']:.3f}"
        f" rank={pre_val['rank_fraction_global_gt_local']:.3f}"
        f" dL={pre_val['local_displacement_mean']:.4f}"
        f" dG={pre_val['global_displacement_mean']:.4f}"
        f" delta={pre_val['global_minus_local']:.4f}"
        f" adj={pre_val['adjacency_accuracy']:.3f}"
        f" next={pre_val['neighbor_top1_accuracy']:.3f}"
    )

    history = []

    best_score = None
    best_epoch = None
    best_state = None

    train_rng = (
        np.random.default_rng(
            SEED + 1000
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running = defaultdict(float)

        for step in range(
            1,
            STEPS_PER_EPOCH + 1
        ):

            fam, strand = (
                train_sampler.sample_batch(
                    BATCH_SIZE
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):

                loss, metrics = (
                    forward_objectives(
                        model,
                        store,
                        fam,
                        strand,
                        train_rng,
                    )
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            for k, v in metrics.items():
                running[k] += v

            if (
                step % 50 == 0
                or
                step == STEPS_PER_EPOCH
            ):

                print(
                    f"[TRAIN] "
                    f"epoch={epoch}/{EPOCHS} "
                    f"step={step}/"
                    f"{STEPS_PER_EPOCH} "
                    f"loss="
                    f"{running['loss']/step:.4f}"
                )

        # fixed inner validation sample
        val = evaluate(
            model,
            store,
            val_sids,
            eligible,
            VAL_BATCHES,
            SEED + 500,
        )

        print()
        print(
            f"[VAL] "
            f"epoch={epoch} "
            f"loss={val['loss']:.4f} "
            f"shuffle="
            f"{val['shuffle_accuracy']:.3f} "
            f"rank="
            f"{val['rank_fraction_global_gt_local']:.3f} "
            f"strong_rank="
            f"{val['strong_rank_fraction']:.3f} "
            f"dL="
            f"{val['local_displacement_mean']:.4f} "
            f"dG="
            f"{val['global_displacement_mean']:.4f} "
            f"delta="
            f"{val['global_minus_local']:.4f} "
            f"adj="
            f"{val['adjacency_accuracy']:.3f} "
            f"next="
            f"{val['neighbor_top1_accuracy']:.3f}"
        )

        history.append({
            "epoch": epoch,
            **{
                f"val_{k}": v
                for k, v in val.items()
            },
        })

        # Inner-only selection score
        score = (
            2.0
            *
            val[
                "shuffle_accuracy"
            ]

            +
            2.0
            *
            val[
                "rank_fraction_global_gt_local"
            ]

            +
            5.0
            *
            max(
                0.0,
                val[
                    "global_minus_local"
                ],
            )

            +
            val[
                "adjacency_accuracy"
            ]

            +
            0.25
            *
            val[
                "neighbor_top1_accuracy"
            ]
        )

        if (
            best_score is None
            or
            score > best_score
        ):

            best_score = score
            best_epoch = epoch

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()

                for k, v
                in model.state_dict().items()
            }

    if best_state is None:

        raise RuntimeError(
            "No best state."
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    # --------------------------------------------------------
    # Final untouched Fold-1 smoke evaluation
    # --------------------------------------------------------

    outer = evaluate(
        model,
        store,
        test_sids,
        eligible,
        OUTER_BATCHES,
        SEED + 9000,
    )

    print()
    print(
        "=" * 88
    )

    print(
        " OUTER FOLD-1 A2 SMOKE TEST"
    )

    print(
        "=" * 88
    )

    for k, v in outer.items():

        print(
            f"{k}: {v}"
        )

    criteria = {
        "shuffle_accuracy_pass":
            (
                outer[
                    "shuffle_accuracy"
                ]
                >= 0.60
            ),

        "rank_fraction_pass":
            (
                outer[
                    "rank_fraction_global_gt_local"
                ]
                >= 0.80
            ),

        "global_displacement_pass":
            (
                outer[
                    "global_displacement_mean"
                ]
                >= 0.05
            ),

        "global_minus_local_pass":
            (
                outer[
                    "global_minus_local"
                ]
                >= 0.03
            ),

        "adjacency_pass":
            (
                outer[
                    "adjacency_accuracy"
                ]
                >= 0.65
            ),
    }

    overall_pass = all(
        criteria.values()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUT /
        "01_training_history.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                history[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(history)

    torch.save(
        {
            "model":
                model.state_dict(),

            "best_epoch":
                best_epoch,

            "best_inner_score":
                best_score,

            "phenotype_labels_used":
                False,

            "outer_holdout_used_for_training":
                False,

            "outer_holdout_used_for_early_stopping":
                False,

            "architecture":
                "ExplicitNeighborhoodEncoder",
        },
        OUT /
        "02_best_A2_checkpoint.pt",
    )

    result = {
        "status":
            (
                "PASS"
                if overall_pass
                else "FAIL"
            ),

        "stage":
            "Stage08A2",

        "timestamp":
            datetime.now().isoformat(),

        "stage08A_status":
            "FAIL_FROZEN",

        "best_epoch":
            best_epoch,

        "train_genomes":
            len(train_sids),

        "inner_val_genomes":
            len(val_sids),

        "outer_test_genomes":
            len(test_sids),

        "pretraining_inner_val":
            pre_val,

        "outer_metrics":
            outer,

        "criteria":
            criteria,

        "all_acceptance_criteria_pass":
            overall_pass,

        "phenotype_labels_used":
            False,

        "outer_holdout_used_for_training":
            False,

        "outer_holdout_used_for_early_stopping":
            False,

        "next_step":
            (
                "Proceed to Stage08B formal 5-fold "
                "order-aware SSL"
                if overall_pass
                else
                "Do NOT start Stage08B; "
                "audit which explicit order objective failed."
            ),
    }

    (
        OUT /
        "03_STAGE08A2_result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 88
    )

    if overall_pass:

        print(
            "[FINAL PASS] Stage08A2"
        )

        print(
            "[NEXT] Formal Stage08B "
            "5-fold is qualified."
        )

    else:

        print(
            "[FINAL FAIL] Stage08A2"
        )

        print(
            "[STOP] Do not launch "
            "Stage08B."
        )

    print(
        "=" * 88
    )

    print(
        "[OUTPUT]",
        OUT
    )


if __name__ == "__main__":
    main()

