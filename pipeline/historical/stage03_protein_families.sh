#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/axw
BIOENV="${ROOT}/bioEnv"

S02="${ROOT}/02_gene_order"
S021="${ROOT}/02_1_gene_call_qc"
S011="${ROOT}/01_1_ani_robustness"

OUT="${ROOT}/03_protein_families"

ENV="${BIOENV}/envs/gene_order"
PY="${ENV}/bin/python"
MMSEQS="${ENV}/bin/mmseqs"

PROTEINS_GZ="${S02}/05_all_proteins.faa.gz"
GENES="${S02}/07_all_genes.tsv.gz"
QCDEC="${S021}/10_STAGE02_1_QC_DECISIONS.csv"
BLOCKS="${S011}/06_FINAL_robust_ANI_blocks.csv"
PRIMARY="${S011}/07_PRIMARY_FINAL_robust_ANI_blocks.csv"
S02SUMMARY="${S02}/12_stage02_summary.json"

INPUTDIR="${OUT}/00_input"
MMOUT="${OUT}/01_mmseqs"
TMP="${BIOENV}/tmp/stage03_mmseqs"

mkdir -p \
    "$OUT" \
    "$INPUTDIR" \
    "$MMOUT" \
    "$TMP" \
    "${BIOENV}/logs"

export TMPDIR="${BIOENV}/tmp"
export XDG_CACHE_HOME="${BIOENV}/cache"

THREADS=$(nproc)

if [ "$THREADS" -gt 20 ]; then
    THREADS=20
fi

echo "================================================================================"
echo "AXW STAGE 03"
echo "Protein families + family-token genome grammar"
echo "================================================================================"
echo "ROOT    = $ROOT"
echo "OUT     = $OUT"
echo "TMP     = $TMP"
echo "THREADS = $THREADS"
echo

# =============================================================================
# 0. INPUT + STORAGE CHECK
# =============================================================================

echo "[0/7] Input and storage validation"

for f in \
    "$PY" \
    "$MMSEQS" \
    "$PROTEINS_GZ" \
    "$GENES" \
    "$QCDEC" \
    "$BLOCKS" \
    "$PRIMARY" \
    "$S02SUMMARY"
do
    if [ ! -s "$f" ]; then
        echo "[FATAL] Missing/empty:"
        echo "$f"
        exit 1
    fi
done

echo "[MMSEQS2]"
"$MMSEQS" version

AVAIL_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
MIN_KB=$((30 * 1024 * 1024))

echo "[DISK] available = $((AVAIL_KB / 1024 / 1024)) GB"

if [ "$AVAIL_KB" -lt "$MIN_KB" ]; then
    echo "[FATAL] <30 GB free on data disk."
    exit 1
fi

EXPECTED=$(
"$PY" - <<'PY'
import json
with open("/root/autodl-tmp/axw/02_gene_order/12_stage02_summary.json") as f:
    print(json.load(f)["total_genes"])
PY
)

echo "[LOCK] expected proteins = $EXPECTED"

# =============================================================================
# 1. PREPARE FASTA
# =============================================================================

echo
echo "[1/7] Preparing MMseqs2 FASTA"

FASTA="${INPUTDIR}/all_proteins.faa"

if [ ! -s "$FASTA" ]; then
    gzip -cd "$PROTEINS_GZ" > "$FASTA"
else
    echo "[CACHE] Reusing decompressed protein FASTA."
fi

NSEQ=$(grep -c '^>' "$FASTA")

echo "[INFO] FASTA proteins = $NSEQ"

if [ "$NSEQ" -ne "$EXPECTED" ]; then
    echo "[FATAL] FASTA count != Stage 02 total genes."
    exit 1
fi

# =============================================================================
# 2. MMSEQS2 EASY-LINCLUST
# =============================================================================

echo
echo "[2/7] MMseqs2 easy-linclust"
echo "[PARAM] min-seq-id = 0.50"
echo "[PARAM] coverage   = 0.80"
echo "[PARAM] cov-mode   = 0 (bidirectional/full-length)"

PREFIX="${MMOUT}/GF50C80"

CLUSTER="${PREFIX}_cluster.tsv"
REP="${PREFIX}_rep_seq.fasta"
ALLSEQ="${PREFIX}_all_seqs.fasta"

if [ ! -s "$CLUSTER" ]; then

    rm -rf "$TMP"
    mkdir -p "$TMP"

    "$MMSEQS" easy-linclust \
        "$FASTA" \
        "$PREFIX" \
        "$TMP" \
        --min-seq-id 0.50 \
        -c 0.80 \
        --cov-mode 0 \
        --threads "$THREADS"

else
    echo "[CACHE] Existing MMseqs cluster result found."
fi

for f in "$CLUSTER" "$REP"; do
    if [ ! -s "$f" ]; then
        echo "[FATAL] MMseqs output missing: $f"
        exit 1
    fi
done

echo "[PASS] MMseqs2 clustering completed."

echo -n "[INFO] membership rows = "
wc -l < "$CLUSTER"

echo -n "[INFO] representatives = "
grep -c '^>' "$REP"

# =============================================================================
# 3. FAMILY IDs + GRAMMAR
# =============================================================================

echo
echo "[3/7] Building stable family IDs and grammar tables"

"$PY" -u - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
import gzip
import gc
import re

ROOT = Path("/root/autodl-tmp/axw")
OUT = ROOT / "03_protein_families"

S02 = ROOT / "02_gene_order"
S021 = ROOT / "02_1_gene_call_qc"
S011 = ROOT / "01_1_ani_robustness"

CLUSTER = OUT / "01_mmseqs/GF50C80_cluster.tsv"
REP_FASTA = OUT / "01_mmseqs/GF50C80_rep_seq.fasta"

GENES = S02 / "07_all_genes.tsv.gz"
QCDEC = S021 / "10_STAGE02_1_QC_DECISIONS.csv"
BLOCKS = S011 / "06_FINAL_robust_ANI_blocks.csv"
PRIMARY = S011 / "07_PRIMARY_FINAL_robust_ANI_blocks.csv"

with open(S02 / "12_stage02_summary.json") as f:
    s02 = json.load(f)

EXPECTED = int(s02["total_genes"])

print("=" * 110)
print("STAGE 03 FAMILY-TOKEN CONSTRUCTION")
print("=" * 110)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def fasta_iter(path):
    header = None
    seq = []

    with open(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)

                header = line[1:].strip()
                seq = []
            else:
                seq.append(line)

        if header is not None:
            yield header, "".join(seq)


# ---------------------------------------------------------------------
# 1. Representative sequence -> stable family ID
#
# Family ID is derived from SHA1 of representative AA sequence.
# This avoids meaningless sequential family IDs.
# ---------------------------------------------------------------------

print("[1/6] Reading family representatives")

rep_to_family = {}
family_to_rep = {}
family_length = {}

for header, seq in fasta_iter(REP_FASTA):

    rep_id = header.split()[0]
    seq = seq.rstrip("*").upper()

    # -------------------------------------------------------------
    # FAMILY ID POLICY
    #
    # MMseqs2 cluster identity must be preserved even when two
    # independent clusters happen to have identical representative
    # amino-acid sequences.
    #
    # Therefore family ID is generated from:
    #
    #   representative_gene_uid + representative_sequence
    #
    # rather than representative sequence alone.
    #
    # This is deterministic for the frozen MMseqs clustering and
    # prevents accidental merging of distinct clusters.
    # -------------------------------------------------------------

    cluster_identity = (
        rep_id
        + "\\n"
        + seq
    )

    digest = hashlib.sha1(
        cluster_identity.encode("ascii")
    ).hexdigest()[:20].upper()

    family_id = f"GF_{digest}"

    if (
        family_id in family_to_rep
        and
        family_to_rep[family_id] != rep_id
    ):
        raise RuntimeError(
            f"True cluster-aware family-ID collision: "
            f"{family_id}"
        )

    rep_to_family[rep_id] = family_id
    family_to_rep[family_id] = rep_id
    family_length[family_id] = len(seq)

print("[INFO] representatives/families:", len(rep_to_family))

# ---------------------------------------------------------------------
# 2. Raw MMseqs membership
# ---------------------------------------------------------------------

print("[2/6] Reading MMseqs membership")

membership = pd.read_csv(
    CLUSTER,
    sep="\t",
    header=None,
    names=[
        "representative_gene_uid",
        "gene_uid",
    ],
    dtype=str,
)

if len(membership) != EXPECTED:
    raise RuntimeError(
        f"Expected {EXPECTED} protein memberships; "
        f"observed {len(membership)}."
    )

if membership["gene_uid"].duplicated().any():
    raise RuntimeError(
        "A protein occurs in more than one MMseqs cluster."
    )

membership["family_id"] = membership[
    "representative_gene_uid"
].map(rep_to_family)

if membership["family_id"].isna().any():
    raise RuntimeError(
        "Some representatives cannot be mapped to stable family IDs."
    )

membership["strain_entity_id"] = membership[
    "gene_uid"
].str.extract(
    r"^(STRAIN_\d+)_g\d+$",
    expand=False,
)

if membership["strain_entity_id"].isna().any():
    raise RuntimeError(
        "Cannot recover strain ID from some gene_uid values."
    )

membership[
    [
        "gene_uid",
        "strain_entity_id",
        "representative_gene_uid",
        "family_id",
    ]
].to_csv(
    OUT / "02_protein_family_membership.tsv.gz",
    sep="\t",
    index=False,
    compression="gzip",
)

print("[PASS] Membership rows:", len(membership))
print("[INFO] Families       :", membership["family_id"].nunique())

# ---------------------------------------------------------------------
# 3. QC cohorts and metadata
# ---------------------------------------------------------------------

print("[3/6] Loading gene/QC metadata")

qc = pd.read_csv(
    QCDEC,
    dtype=str,
    keep_default_na=False,
    low_memory=False,
)

blocks = pd.read_csv(
    BLOCKS,
    dtype=str,
    keep_default_na=False,
    low_memory=False,
)

primary = pd.read_csv(
    PRIMARY,
    dtype=str,
    keep_default_na=False,
    low_memory=False,
)

if len(qc) != 994:
    raise RuntimeError(
        f"Expected 994 Stage 02.1 QC rows; observed {len(qc)}"
    )

qc["order_primary"] = (
    qc["stage02_1_decision"] == "KEEP"
)

if int(qc["order_primary"].sum()) != 979:
    raise RuntimeError(
        "Expected 979 main order-QC genomes."
    )

order_map = dict(
    zip(
        qc["strain_entity_id"],
        qc["order_primary"],
    )
)

decision_map = dict(
    zip(
        qc["strain_entity_id"],
        qc["stage02_1_decision"],
    )
)

species_map = dict(
    zip(
        blocks["strain_entity_id"],
        blocks["species"],
    )
) if "species" in blocks.columns else {}

membership["order_primary"] = membership[
    "strain_entity_id"
].map(order_map).fillna(False)

membership["species"] = membership[
    "strain_entity_id"
].map(species_map).fillna("")

# ---------------------------------------------------------------------
# 4. Compact gene-family master
# ---------------------------------------------------------------------

usecols = [
    "gene_uid",
    "strain_entity_id",
    "contig_id",
    "contig_order",
    "contig_length_bp",
    "start",
    "end",
    "strand",
    "gene_order_within_contig",
    "partial",
    "protein_length_aa",
]

genes = pd.read_csv(
    GENES,
    sep="\t",
    usecols=usecols,
    dtype={
        "gene_uid": str,
        "strain_entity_id": str,
        "contig_id": str,
        "strand": str,
        "partial": str,
    },
    low_memory=False,
)

if len(genes) != EXPECTED:
    raise RuntimeError(
        f"Gene table has {len(genes)} rows; expected {EXPECTED}"
    )

genes = genes.merge(
    membership[
        [
            "gene_uid",
            "representative_gene_uid",
            "family_id",
        ]
    ],
    on="gene_uid",
    how="left",
    validate="one_to_one",
)

if genes["family_id"].isna().any():
    raise RuntimeError(
        "Some genes have no protein-family assignment."
    )

genes["order_primary"] = genes[
    "strain_entity_id"
].map(order_map).fillna(False)

genes["species"] = genes[
    "strain_entity_id"
].map(species_map).fillna("")

genes["is_partial"] = (
    genes["partial"].astype(str) != "00"
)

for c in [
    "contig_order",
    "contig_length_bp",
    "start",
    "end",
    "gene_order_within_contig",
    "protein_length_aa",
]:
    genes[c] = pd.to_numeric(
        genes[c],
        errors="coerce",
    )

genes = genes.sort_values(
    [
        "strain_entity_id",
        "contig_order",
        "gene_order_within_contig",
    ]
).reset_index(drop=True)

genes.to_csv(
    OUT / "03_gene_family_master.tsv.gz",
    sep="\t",
    index=False,
    compression="gzip",
)

# ---------------------------------------------------------------------
# 5. Family summary
# ---------------------------------------------------------------------

print("[4/6] Family prevalence summary")

fam = (
    genes.groupby(
        "family_id",
        sort=False,
    )
    .agg(
        representative_gene_uid=(
            "representative_gene_uid",
            "first",
        ),
        n_proteins=(
            "gene_uid",
            "size",
        ),
        n_strains=(
            "strain_entity_id",
            "nunique",
        ),
        n_species=(
            "species",
            lambda x: x[x != ""].nunique(),
        ),
        n_partial_proteins=(
            "is_partial",
            "sum",
        ),
        median_protein_length_aa=(
            "protein_length_aa",
            "median",
        ),
    )
    .reset_index()
)

primary_gene = genes[
    genes["order_primary"]
]

order_prev = (
    primary_gene.groupby(
        "family_id"
    )["strain_entity_id"]
    .nunique()
)

fam[
    "n_order_primary_strains"
] = fam[
    "family_id"
].map(order_prev).fillna(0).astype(int)

fam[
    "representative_length_aa"
] = fam[
    "family_id"
].map(family_length)

fam[
    "prevalence_all_994"
] = fam[
    "n_strains"
] / 994.0

fam[
    "prevalence_order_979"
] = fam[
    "n_order_primary_strains"
] / 979.0

fam[
    "singleton_protein_family"
] = (
    fam["n_proteins"] == 1
)

fam[
    "single_strain_family"
] = (
    fam["n_strains"] == 1
)

fam = fam.sort_values(
    [
        "n_strains",
        "n_proteins",
        "family_id",
    ],
    ascending=[
        False,
        False,
        True,
    ],
)

fam.to_csv(
    OUT / "04_family_summary.csv.gz",
    index=False,
    compression="gzip",
)

# ---------------------------------------------------------------------
# 6. Family-order grammar
# ---------------------------------------------------------------------

print("[5/6] Contig family-token grammar")

genes["family_strand_token"] = (
    genes["family_id"].astype(str)
    +
    genes["strand"].astype(str)
)

order_table = (
    genes.groupby(
        [
            "strain_entity_id",
            "contig_id",
            "contig_order",
            "contig_length_bp",
            "order_primary",
        ],
        sort=False,
        dropna=False,
    )
    .agg(
        n_genes=(
            "gene_uid",
            "size",
        ),
        family_id_sequence=(
            "family_id",
            lambda x: " ".join(x.astype(str)),
        ),
        strand_sequence=(
            "strand",
            lambda x: "".join(x.astype(str)),
        ),
        family_strand_token_sequence=(
            "family_strand_token",
            lambda x: " ".join(x.astype(str)),
        ),
    )
    .reset_index()
)

order_table.to_csv(
    OUT / "05_contig_family_order.tsv.gz",
    sep="\t",
    index=False,
    compression="gzip",
)

# ---------------------------------------------------------------------
# 7. Family adjacency
# ---------------------------------------------------------------------

print("[6/6] Family adjacency + genome content")

grp = genes.groupby(
    [
        "strain_entity_id",
        "contig_id",
    ],
    sort=False,
)

genes["right_gene_uid"] = grp[
    "gene_uid"
].shift(-1)

genes["right_family_id"] = grp[
    "family_id"
].shift(-1)

genes["right_strand"] = grp[
    "strand"
].shift(-1)

genes["right_start"] = grp[
    "start"
].shift(-1)

adj = genes[
    genes["right_gene_uid"].notna()
].copy()

adj["intergenic_bp"] = (
    adj["right_start"]
    -
    adj["end"]
    -
    1
)

adj["gap_bp"] = adj[
    "intergenic_bp"
].clip(lower=0)

adj["overlap_bp"] = (
    -adj["intergenic_bp"]
).clip(lower=0)

adj["orientation"] = (
    adj["strand"].astype(str)
    +
    adj["right_strand"].astype(str)
)

adj_out = adj[
    [
        "strain_entity_id",
        "species",
        "contig_id",
        "contig_order",
        "order_primary",
        "gene_uid",
        "right_gene_uid",
        "family_id",
        "right_family_id",
        "orientation",
        "intergenic_bp",
        "gap_bp",
        "overlap_bp",
    ]
].rename(
    columns={
        "gene_uid":
            "left_gene_uid",
        "family_id":
            "left_family_id",
    }
)

adj_out.to_csv(
    OUT / "06_family_adjacency.tsv.gz",
    sep="\t",
    index=False,
    compression="gzip",
)

# ---------------------------------------------------------------------
# 8. Order-invariant family-content baseline
# ---------------------------------------------------------------------

content = (
    genes.groupby(
        [
            "strain_entity_id",
            "family_id",
        ],
        sort=False,
    )
    .size()
    .rename("copy_number")
    .reset_index()
)

content["species"] = content[
    "strain_entity_id"
].map(species_map).fillna("")

content["stage02_1_decision"] = content[
    "strain_entity_id"
].map(decision_map).fillna("")

content["order_primary"] = content[
    "strain_entity_id"
].map(order_map).fillna(False)

content.to_csv(
    OUT / "07_genome_family_content.tsv.gz",
    sep="\t",
    index=False,
    compression="gzip",
)

genome_summary = (
    content.groupby(
        "strain_entity_id",
        sort=False,
    )
    .agg(
        n_unique_families=(
            "family_id",
            "nunique",
        ),
        total_family_copies=(
            "copy_number",
            "sum",
        ),
        single_copy_families=(
            "copy_number",
            lambda x: int((x == 1).sum()),
        ),
        multicopy_families=(
            "copy_number",
            lambda x: int((x > 1).sum()),
        ),
    )
    .reset_index()
)

genome_summary["species"] = genome_summary[
    "strain_entity_id"
].map(species_map).fillna("")

genome_summary["stage02_1_decision"] = genome_summary[
    "strain_entity_id"
].map(decision_map).fillna("")

genome_summary["order_primary"] = genome_summary[
    "strain_entity_id"
].map(order_map).fillna(False)

genome_summary.to_csv(
    OUT / "08_genome_family_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# ---------------------------------------------------------------------
# 9. Cohort manifests
# ---------------------------------------------------------------------

meta_cols = [
    c
    for c in [
        "strain_entity_id",
        "species",
        "strain",
        "canonical_genome_path",
        "FINAL_nearclone_99_5_block",
        "FINAL_lineage_95_block",
    ]
    if c in blocks.columns
]

meta = blocks[
    meta_cols
].drop_duplicates(
    "strain_entity_id"
)

qc_small = qc[
    [
        "strain_entity_id",
        "gene_call_qc",
        "stage02_1_decision",
    ]
].drop_duplicates(
    "strain_entity_id"
)

manifest = meta.merge(
    qc_small,
    on="strain_entity_id",
    how="left",
)

content_manifest = manifest.copy()

content_manifest[
    "analysis_role"
] = "CONTENT_PRIMARY"

content_manifest.to_csv(
    OUT / "09_content_primary_994_manifest.csv",
    index=False,
    encoding="utf-8-sig",
)

order_primary_manifest = manifest[
    manifest[
        "stage02_1_decision"
    ] == "KEEP"
].copy()

order_primary_manifest[
    "analysis_role"
] = "ORDER_PRIMARY"

order_primary_manifest.to_csv(
    OUT / "10_order_primary_979_manifest.csv",
    index=False,
    encoding="utf-8-sig",
)

order_sens_manifest = manifest.copy()

order_sens_manifest[
    "analysis_role"
] = "ORDER_SENSITIVITY_ALL_994"

order_sens_manifest.to_csv(
    OUT / "11_order_sensitivity_994_manifest.csv",
    index=False,
    encoding="utf-8-sig",
)

# ---------------------------------------------------------------------
# 10. PRIMARY phenotype-strain family QC
# ---------------------------------------------------------------------

primary_ids = set(
    primary[
        "strain_entity_id"
    ]
)

primary_family = genome_summary[
    genome_summary[
        "strain_entity_id"
    ].isin(primary_ids)
].copy()

if primary_family[
    "strain_entity_id"
].nunique() != 6:
    raise RuntimeError(
        "Expected all 6 PRIMARY strains in Stage 03."
    )

if not primary_family[
    "order_primary"
].all():
    raise RuntimeError(
        "At least one PRIMARY strain failed main order QC."
    )

primary_family.to_csv(
    OUT / "12_PRIMARY_6_family_QC.csv",
    index=False,
    encoding="utf-8-sig",
)

# ---------------------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------------------

n_families = int(
    fam["family_id"].nunique()
)

n_singleton = int(
    fam["singleton_protein_family"].sum()
)

n_single_strain = int(
    fam["single_strain_family"].sum()
)

n_multi_strain = int(
    (fam["n_strains"] >= 2).sum()
)

n_prev_1pct = int(
    (fam["n_strains"] >= 10).sum()
)

n_prev_5pct = int(
    (fam["n_strains"] >= 50).sum()
)

n_prev_50pct = int(
    (fam["n_strains"] >= 497).sum()
)

summary = {
    "protein_sequences":
        int(len(membership)),

    "protein_families":
        n_families,

    "singleton_protein_families":
        n_singleton,

    "single_strain_families":
        n_single_strain,

    "multi_strain_families":
        n_multi_strain,

    "families_prevalence_ge_1pct":
        n_prev_1pct,

    "families_prevalence_ge_5pct":
        n_prev_5pct,

    "families_prevalence_ge_50pct":
        n_prev_50pct,

    "content_primary_genomes":
        int(len(content_manifest)),

    "order_primary_genomes":
        int(len(order_primary_manifest)),

    "order_sensitivity_genomes":
        int(len(order_sens_manifest)),

    "contig_family_sequences":
        int(len(order_table)),

    "family_adjacencies":
        int(len(adj_out)),

    "PRIMARY_unique_strains":
        int(primary_family[
            "strain_entity_id"
        ].nunique()),

    "family_definition":
        {
            "MMseqs2_workflow":
                "easy-linclust",
            "min_seq_id":
                0.50,
            "coverage":
                0.80,
            "coverage_mode":
                0,
        },
}

with open(
    OUT / "13_STAGE03_summary.json",
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
print("=" * 110)
print("STAGE 03 FAMILY-GRAMMAR SUMMARY")
print("=" * 110)

for k, v in summary.items():
    print(f"{k:42s}: {v}")

print()
print(
    "[POLICY] Content primary cohort     = 994 genomes"
)

print(
    "[POLICY] Gene-order primary cohort  = 979 PASS genomes"
)

print(
    "[POLICY] Order sensitivity cohort   = all 994 genomes"
)

print()
print(
    "[IMPORTANT] No phenotype labels were used "
    "to construct protein families."
)

print(
    "[IMPORTANT] No adjacency crosses a contig boundary."
)

del genes, membership, adj, adj_out, content
gc.collect()
PY

# =============================================================================
# 4. INTEGRITY CHECK
# =============================================================================

echo
echo "[4/7] Stage 03 integrity"

for f in \
    "${OUT}/02_protein_family_membership.tsv.gz" \
    "${OUT}/03_gene_family_master.tsv.gz" \
    "${OUT}/04_family_summary.csv.gz" \
    "${OUT}/05_contig_family_order.tsv.gz" \
    "${OUT}/06_family_adjacency.tsv.gz" \
    "${OUT}/07_genome_family_content.tsv.gz" \
    "${OUT}/08_genome_family_summary.csv" \
    "${OUT}/09_content_primary_994_manifest.csv" \
    "${OUT}/10_order_primary_979_manifest.csv" \
    "${OUT}/11_order_sensitivity_994_manifest.csv" \
    "${OUT}/12_PRIMARY_6_family_QC.csv" \
    "${OUT}/13_STAGE03_summary.json"
do
    if [ ! -s "$f" ]; then
        echo "[FATAL] Missing/empty:"
        echo "$f"
        exit 1
    fi
done

echo "[PASS] Required Stage 03 outputs exist."

# =============================================================================
# 5. RAW CLUSTER AUDIT + COMPRESSION
# =============================================================================

echo
echo "[5/7] Raw MMseqs output audit"

N_MEMBERS=$(wc -l < "$CLUSTER")
N_REPS=$(grep -c '^>' "$REP")

echo "Membership records : $N_MEMBERS"
echo "Family reps        : $N_REPS"

if [ "$N_MEMBERS" -ne "$EXPECTED" ]; then
    echo "[FATAL] MMseqs membership count mismatch."
    exit 1
fi

gzip -f "$CLUSTER"

# `_all_seqs.fasta` duplicates the source proteins and is not needed
# after family membership has been frozen.
if [ -f "$ALLSEQ" ]; then
    rm -f "$ALLSEQ"
fi

# =============================================================================
# 6. CLEAN LARGE TEMPORARY FILES
# =============================================================================

echo
echo "[6/7] Cleaning temporary files"

rm -rf "$TMP"
rm -f "$FASTA"

echo "[PASS] Temporary MMseqs workspace removed."

du -sh "$OUT" "${BIOENV}/envs/gene_order"

df -h "$ROOT"

# =============================================================================
# 7. FINAL REPORT
# =============================================================================

echo
echo "[7/7] Final Stage 03 report"

"$PY" - <<'PY'
from pathlib import Path
import json
import pandas as pd

OUT = Path(
    "/root/autodl-tmp/axw/03_protein_families"
)

with open(
    OUT / "13_STAGE03_summary.json",
    encoding="utf-8",
) as f:
    s = json.load(f)

print()
print("=" * 115)
print("FINAL STAGE 03 SUMMARY")
print("=" * 115)

for k, v in s.items():
    print(f"{k:45s}: {v}")

print()
print("[PRIMARY 6 FAMILY QC]")

p = pd.read_csv(
    OUT / "12_PRIMARY_6_family_QC.csv",
    keep_default_na=False,
)

print(
    p.to_string(
        index=False
    )
)

print()
print("[LARGEST 20 PROTEIN FAMILIES]")

fam = pd.read_csv(
    OUT / "04_family_summary.csv.gz",
)

print(
    fam.head(20)[
        [
            "family_id",
            "representative_gene_uid",
            "n_proteins",
            "n_strains",
            "n_species",
            "n_partial_proteins",
            "prevalence_all_994",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "[FINAL PASS] Stage 03 protein-family "
    "+ family-token grammar completed."
)
PY

