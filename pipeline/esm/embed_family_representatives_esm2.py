from pathlib import Path
from datetime import datetime

import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import esm


ROOT = Path(os.environ["GENOGRAMMA_WORK_ROOT"]).resolve()
OUT = ROOT / "06_family_esm2"
S051 = ROOT / "05_1_canonical_grammar_atlas"

FASTA = OUT / "01_family_representatives.faa.gz"
INDEX = OUT / "02_family_representative_index.csv"

EMBED = OUT / "03_ESM2_t12_35M_embeddings.f16.mmap"
DONE = OUT / "04_ESM2_embedding_done.u8.mmap"
EDGE_MAP = OUT / "05_canonical_edge_embedding_index.csv.gz"
SUMMARY = OUT / "06_STAGE06_summary.json"

CANONICAL = (
    S051
    /
    "02_CANONICAL_GENOME_GRAMMAR_ATLAS.csv.gz"
)

SUMMARY051 = (
    S051
    /
    "08_STAGE05_1_summary.json"
)


EXPECTED_FAMILIES = 276_517
EXPECTED_EDGES = 22_008

MODEL_NAME = "esm2_t12_35M_UR50D"
MODEL_LAYER = 12
EMBED_DIM = 480

MAX_RESIDUES = 1022
CHUNK_OVERLAP = 128

CHECKPOINT_EVERY = 500


print("=" * 124)
print("STAGE 06B — ESM-2 FAMILY REPRESENTATIVE EMBEDDINGS")
print("=" * 124)


# ==============================================================================================
# Verify Stage05.1
# ==============================================================================================

with open(
    SUMMARY051,
    encoding="utf-8",
) as fh:

    s051 = json.load(
        fh
    )


if (
    s051.get(
        "status"
    )
    !=
    "PASS"
):

    raise RuntimeError(
        "Stage05.1 is not PASS."
    )


if int(
    s051[
        "final_canonical_grammar_edges"
    ]
) != EXPECTED_EDGES:

    raise RuntimeError(
        "Stage05.1 canonical-edge lock failed."
    )


# ==============================================================================================
# Load representative index
# ==============================================================================================

rows = []


with open(
    INDEX,
    encoding="utf-8",
) as fh:

    reader = csv.DictReader(
        fh
    )

    for row in reader:

        rows.append({
            "embedding_row":
                int(
                    row[
                        "embedding_row"
                    ]
                ),

            "family_id":
                row[
                    "family_id"
                ],

            "representative_gene_uid":
                row[
                    "representative_gene_uid"
                ],

            "protein_length":
                int(
                    row[
                        "protein_length"
                    ]
                ),

            "sequence_sha1":
                row[
                    "sequence_sha1"
                ],
        })


if len(
    rows
) != EXPECTED_FAMILIES:

    raise RuntimeError(
        f"Representative index rows="
        f"{len(rows)}"
    )


for i, row in enumerate(
    rows
):

    if (
        row[
            "embedding_row"
        ]
        !=
        i
    ):

        raise RuntimeError(
            "embedding_row is not contiguous."
        )


family_to_row = {
    r[
        "family_id"
    ]:
        r[
            "embedding_row"
        ]
    for r in rows
}


# ==============================================================================================
# Read representative FASTA
# ==============================================================================================

sequences = {}


current_header = None
current_seq = []


def flush_fasta():

    global current_header
    global current_seq

    if current_header is None:
        return

    family = (
        current_header.split(
            "|",
            1,
        )[0]
    )

    seq = "".join(
        current_seq
    ).upper()

    sequences[
        family
    ] = seq


with gzip.open(
    FASTA,
    "rt",
    encoding="utf-8",
) as fh:

    for line in fh:

        line = line.strip()

        if not line:
            continue

        if line.startswith(
            ">"
        ):

            flush_fasta()

            current_header = (
                line[
                    1:
                ]
            )

            current_seq = []

        else:

            current_seq.append(
                line
            )


flush_fasta()


if len(
    sequences
) != EXPECTED_FAMILIES:

    raise RuntimeError(
        f"Representative FASTA="
        f"{len(sequences)}"
    )


for row in rows:

    family = row[
        "family_id"
    ]

    seq = sequences.get(
        family
    )

    if seq is None:

        raise RuntimeError(
            f"Missing sequence: {family}"
        )

    if len(
        seq
    ) != row[
        "protein_length"
    ]:

        raise RuntimeError(
            f"Length mismatch: {family}"
        )


print(
    "[PASS] representative proteins:",
    len(
        sequences
    ),
)


# ==============================================================================================
# Device
# ==============================================================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is not available. "
        "Stage06 is intentionally not falling "
        "back silently to CPU."
    )


device = torch.device(
    "cuda:0"
)


print(
    "[INFO] torch:",
    torch.__version__,
)

print(
    "[INFO] CUDA:",
    torch.version.cuda,
)

print(
    "[INFO] GPU:",
    torch.cuda.get_device_name(
        0
    ),
)


# ==============================================================================================
# Load ESM-2
# ==============================================================================================

print()
print(
    "[INFO] Loading:",
    MODEL_NAME,
)


model, alphabet = (
    esm.pretrained.esm2_t12_35M_UR50D()
)

model.eval()
model = model.to(
    device
)

batch_converter = (
    alphabet.get_batch_converter()
)


if int(
    model.embed_dim
) != EMBED_DIM:

    raise RuntimeError(
        f"Unexpected ESM dimension: "
        f"{model.embed_dim}"
    )


print(
    "[PASS] ESM-2 loaded | "
    f"layer={MODEL_LAYER} | "
    f"dim={EMBED_DIM}"
)


# ==============================================================================================
# Persistent resume-safe arrays
# ==============================================================================================

expected_embedding_bytes = (
    EXPECTED_FAMILIES
    *
    EMBED_DIM
    *
    np.dtype(
        np.float16
    ).itemsize
)


if EMBED.exists():

    if EMBED.stat().st_size != expected_embedding_bytes:

        raise RuntimeError(
            "Existing embedding memmap has "
            "unexpected byte size."
        )

    embedding = np.memmap(
        EMBED,
        dtype=np.float16,
        mode="r+",
        shape=(
            EXPECTED_FAMILIES,
            EMBED_DIM,
        ),
    )

else:

    embedding = np.memmap(
        EMBED,
        dtype=np.float16,
        mode="w+",
        shape=(
            EXPECTED_FAMILIES,
            EMBED_DIM,
        ),
    )

    embedding[:] = 0
    embedding.flush()


if DONE.exists():

    if DONE.stat().st_size != EXPECTED_FAMILIES:

        raise RuntimeError(
            "Existing DONE bitmap has wrong size."
        )

    done = np.memmap(
        DONE,
        dtype=np.uint8,
        mode="r+",
        shape=(
            EXPECTED_FAMILIES,
        ),
    )

else:

    done = np.memmap(
        DONE,
        dtype=np.uint8,
        mode="w+",
        shape=(
            EXPECTED_FAMILIES,
        ),
    )

    done[:] = 0
    done.flush()


already_done = int(
    done.sum()
)


print()
print(
    "[RESUME] embeddings already complete:",
    f"{already_done}/{EXPECTED_FAMILIES}",
)


# ==============================================================================================
# Embedding utilities
# ==============================================================================================

def batch_size_for_length(length):

    if length <= 180:
        return 32

    if length <= 280:
        return 24

    if length <= 400:
        return 12

    if length <= 550:
        return 6

    if length <= 750:
        return 3

    return 2


@torch.inference_mode()
def embed_batch(
    batch_items,
):
    """
    batch_items:
        [(embedding_row, family_id, sequence), ...]
    """

    converter_input = [
        (
            family,
            seq,
        )
        for _, family, seq
        in batch_items
    ]


    _, _, tokens = batch_converter(
        converter_input
    )

    tokens = tokens.to(
        device,
        non_blocking=True,
    )


    # Mixed precision is used only for forward inference.
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    ):

        result = model(
            tokens,
            repr_layers=[
                MODEL_LAYER
            ],
            return_contacts=False,
        )

        reps = result[
            "representations"
        ][
            MODEL_LAYER
        ]


    for batch_idx, (
        row_idx,
        family,
        seq,
    ) in enumerate(
        batch_items
    ):

        length = len(
            seq
        )

        vec = (
            reps[
                batch_idx,
                1:
                length + 1,
                :
            ]
            .float()
            .mean(
                dim=0
            )
            .cpu()
            .numpy()
        )


        if not np.isfinite(
            vec
        ).all():

            raise RuntimeError(
                f"Non-finite embedding: "
                f"{family}"
            )


        embedding[
            row_idx,
            :
        ] = vec.astype(
            np.float16
        )

        done[
            row_idx
        ] = 1


    del tokens
    del reps
    del result


@torch.inference_mode()
def embed_long_protein(
    row_idx,
    family,
    seq,
):

    length = len(
        seq
    )

    step = (
        MAX_RESIDUES
        -
        CHUNK_OVERLAP
    )


    starts = list(
        range(
            0,
            length,
            step,
        )
    )


    if (
        starts
        and
        starts[
            -1
        ]
        +
        MAX_RESIDUES
        <
        length
    ):

        starts.append(
            length
            -
            MAX_RESIDUES
        )


    starts = sorted(
        set(
            min(
                s,
                max(
                    0,
                    length
                    -
                    MAX_RESIDUES
                ),
            )
            for s in starts
        )
    )


    residue_sum = np.zeros(
        (
            length,
            EMBED_DIM,
        ),
        dtype=np.float32,
    )

    residue_count = np.zeros(
        (
            length,
        ),
        dtype=np.float32,
    )


    for chunk_idx, start in enumerate(
        starts
    ):

        end = min(
            length,
            start
            +
            MAX_RESIDUES,
        )

        chunk = seq[
            start:
            end
        ]


        _, _, tokens = batch_converter([
            (
                f"{family}_chunk{chunk_idx}",
                chunk,
            )
        ])


        tokens = tokens.to(
            device
        )


        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):

            result = model(
                tokens,
                repr_layers=[
                    MODEL_LAYER
                ],
                return_contacts=False,
            )

            rep = (
                result[
                    "representations"
                ][
                    MODEL_LAYER
                ][
                    0,
                    1:
                    len(
                        chunk
                    )
                    +
                    1,
                    :
                ]
                .float()
                .cpu()
                .numpy()
            )


        residue_sum[
            start:
            end
        ] += rep

        residue_count[
            start:
            end
        ] += 1.0


        del tokens
        del result
        del rep


    if np.any(
        residue_count
        ==
        0
    ):

        raise RuntimeError(
            f"Long-protein chunk coverage gap: "
            f"{family}"
        )


    residue_mean = (
        residue_sum
        /
        residue_count[
            :,
            None
        ]
    )


    vec = residue_mean.mean(
        axis=0
    )


    if not np.isfinite(
        vec
    ).all():

        raise RuntimeError(
            f"Non-finite long embedding: "
            f"{family}"
        )


    embedding[
        row_idx,
        :
    ] = vec.astype(
        np.float16
    )

    done[
        row_idx
    ] = 1


# ==============================================================================================
# Embed shortest-to-longest to reduce padding
# ==============================================================================================

pending = [
    (
        row[
            "embedding_row"
        ],
        row[
            "family_id"
        ],
        row[
            "protein_length"
        ],
    )
    for row in rows
    if done[
        row[
            "embedding_row"
        ]
    ] == 0
]


pending.sort(
    key=lambda x: (
        x[
            2
        ],
        x[
            1
        ],
    )
)


print(
    "[INFO] pending embeddings:",
    len(
        pending
    ),
)


completed_this_run = 0
long_count_this_run = 0

batch = []
current_limit = None


def flush_batch():

    global batch
    global completed_this_run

    if not batch:
        return

    embed_batch(
        batch
    )

    completed_this_run += len(
        batch
    )

    batch = []


for row_idx, family, length in pending:

    seq = sequences[
        family
    ]


    if length > MAX_RESIDUES:

        flush_batch()

        embed_long_protein(
            row_idx,
            family,
            seq,
        )

        completed_this_run += 1
        long_count_this_run += 1


    else:

        limit = (
            batch_size_for_length(
                length
            )
        )


        if not batch:

            current_limit = limit


        if (
            batch
            and
            (
                len(
                    batch
                )
                >=
                current_limit
                or
                limit
                !=
                current_limit
            )
        ):

            flush_batch()

            current_limit = limit


        batch.append(
            (
                row_idx,
                family,
                seq,
            )
        )


    if (
        completed_this_run
        > 0
        and
        completed_this_run
        %
        CHECKPOINT_EVERY
        ==
        0
    ):

        embedding.flush()
        done.flush()

        total_done = int(
            done.sum()
        )

        print(
            "[PROGRESS] "
            f"{total_done}/"
            f"{EXPECTED_FAMILIES} "
            f"({100.0 * total_done / EXPECTED_FAMILIES:.2f}%)"
        )


flush_batch()

embedding.flush()
done.flush()


n_done = int(
    done.sum()
)


if n_done != EXPECTED_FAMILIES:

    raise RuntimeError(
        f"Embedding incomplete: "
        f"{n_done}/"
        f"{EXPECTED_FAMILIES}"
    )


print()
print(
    "[PASS] all family embeddings complete:",
    n_done,
)


# ==============================================================================================
# Finite-value audit in chunks
# ==============================================================================================

print()
print(
    "[AUDIT] finite-value scan"
)


finite_ok = True

block = 10_000


for start in range(
    0,
    EXPECTED_FAMILIES,
    block,
):

    end = min(
        EXPECTED_FAMILIES,
        start
        +
        block,
    )

    x = np.asarray(
        embedding[
            start:
            end
        ],
        dtype=np.float32,
    )

    if not np.isfinite(
        x
    ).all():

        finite_ok = False
        break


if not finite_ok:

    raise RuntimeError(
        "Embedding memmap contains "
        "NaN/Inf values."
    )


print(
    "[PASS] finite-value audit"
)


# ==============================================================================================
# Canonical grammar -> embedding row map
# ==============================================================================================

print()
print(
    "[MAP] canonical grammar edges "
    "to family embedding rows"
)


edge_rows = 0
missing_family_edges = 0


with gzip.open(
    CANONICAL,
    "rt",
    encoding="utf-8",
) as fin, gzip.open(
    EDGE_MAP,
    "wt",
    encoding="utf-8",
    newline="",
) as fout:

    reader = csv.DictReader(
        fin
    )

    fieldnames = [
        "canonical_edge_id",
        "family_A",
        "family_A_embedding_row",
        "strand_A",
        "family_B",
        "family_B_embedding_row",
        "strand_B",
        "relative_orientation",
        "true_genomes_order979",
        "true_species_order979",
        "true_nearclone_blocks_order979",
        "true_lineage95_blocks_order979",
        "log2_true_vs_GLOBAL",
    ]


    writer = csv.DictWriter(
        fout,
        fieldnames=fieldnames,
    )

    writer.writeheader()


    for row in reader:

        family_a = row[
            "family_A"
        ]

        family_b = row[
            "family_B"
        ]


        if (
            family_a
            not in
            family_to_row
            or
            family_b
            not in
            family_to_row
        ):

            missing_family_edges += 1
            continue


        writer.writerow({
            "canonical_edge_id":
                row[
                    "canonical_edge_id"
                ],

            "family_A":
                family_a,

            "family_A_embedding_row":
                family_to_row[
                    family_a
                ],

            "strand_A":
                row[
                    "strand_A"
                ],

            "family_B":
                family_b,

            "family_B_embedding_row":
                family_to_row[
                    family_b
                ],

            "strand_B":
                row[
                    "strand_B"
                ],

            "relative_orientation":
                row[
                    "relative_orientation"
                ],

            "true_genomes_order979":
                row[
                    "true_genomes_order979"
                ],

            "true_species_order979":
                row[
                    "true_species_order979"
                ],

            "true_nearclone_blocks_order979":
                row[
                    "true_nearclone_blocks_order979"
                ],

            "true_lineage95_blocks_order979":
                row[
                    "true_lineage95_blocks_order979"
                ],

            "log2_true_vs_GLOBAL":
                row[
                    "log2_true_vs_GLOBAL"
                ],
        })


        edge_rows += 1


if edge_rows != EXPECTED_EDGES:

    raise RuntimeError(
        f"Canonical edge mapping rows="
        f"{edge_rows}; "
        f"expected={EXPECTED_EDGES}; "
        f"missing={missing_family_edges}"
    )


print(
    "[PASS] canonical edges mapped:",
    edge_rows,
)


# ==============================================================================================
# Hash final embedding cache
# ==============================================================================================

print()
print(
    "[AUDIT] SHA256 embedding cache"
)


sha = hashlib.sha256()


with open(
    EMBED,
    "rb",
) as fh:

    while True:

        chunk = fh.read(
            16
            *
            1024
            *
            1024
        )

        if not chunk:
            break

        sha.update(
            chunk
        )


embedding_sha256 = (
    sha.hexdigest()
)


# ==============================================================================================
# Summary
# ==============================================================================================

lengths = [
    r[
        "protein_length"
    ]
    for r in rows
]


summary = {
    "status":
        "PASS",

    "timestamp":
        datetime.now().isoformat(),

    "phenotype_labels_used":
        False,

    "model":
        MODEL_NAME,

    "model_layer":
        MODEL_LAYER,

    "embedding_dimension":
        EMBED_DIM,

    "pooling":
        "mean of residue embeddings",

    "storage_dtype":
        "float16",

    "n_protein_families":
        EXPECTED_FAMILIES,

    "representative_policy":
        "actual frozen MMseqs representative protein",

    "arbitrary_representative_substitution":
        False,

    "protein_length_max":
        max(
            lengths
        ),

    "proteins_gt_1022aa":
        sum(
            x > MAX_RESIDUES
            for x in lengths
        ),

    "long_protein_policy":
        (
            "overlapping residue windows of <=1022 aa; "
            "128-aa overlap; overlapping residue "
            "representations averaged before whole-protein "
            "mean pooling"
        ),

    "cuda_required":
        True,

    "torch_version":
        torch.__version__,

    "cuda_runtime":
        torch.version.cuda,

    "gpu":
        torch.cuda.get_device_name(
            0
        ),

    "embedding_cache":
        str(
            EMBED
        ),

    "embedding_sha256":
        embedding_sha256,

    "embedding_done":
        EXPECTED_FAMILIES,

    "all_embeddings_finite":
        True,

    "canonical_grammar_edges":
        EXPECTED_EDGES,

    "canonical_edges_mapped_to_embeddings":
        edge_rows,

    "missing_edge_family_mappings":
        missing_family_edges,

    "resume_safe":
        True,

    "important_boundary":
        (
            "No phenotype labels are used. "
            "These are protein-sequence representations, "
            "not probiotic-function predictions."
        ),
}


with open(
    SUMMARY,
    "w",
    encoding="utf-8",
) as fh:

    json.dump(
        summary,
        fh,
        indent=2,
        ensure_ascii=False,
    )


print()
print("=" * 124)
print("FINAL STAGE 06 SUMMARY")
print("=" * 124)

for k, v in summary.items():

    print(
        f"{k:46s}: {v}"
    )


print()
print(
    "[FINAL PASS] Stage 06 family ESM-2 "
    "embedding cache completed."
)
