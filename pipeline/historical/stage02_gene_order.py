#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
import subprocess
import gzip
import shutil
import os
import re
import json
import sys

# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path("/root/autodl-tmp/axw")
BIOENV = ROOT / "bioEnv"
OUT = ROOT / "02_gene_order"

ROBUST = (
    ROOT /
    "01_1_ani_robustness" /
    "06_FINAL_robust_ANI_blocks.csv"
)

PRIMARY = (
    ROOT /
    "01_1_ani_robustness" /
    "07_PRIMARY_FINAL_robust_ANI_blocks.csv"
)

ENV = BIOENV / "envs" / "gene_order"
PRODIGAL = ENV / "bin" / "prodigal"

N_JOBS = min(
    20,
    os.cpu_count() or 4
)

# Directories
NORM = OUT / "01_normalized_genomes"
CONTIG = OUT / "02_contig_maps"

RAW_GFF = OUT / "03_prodigal_gff"
RAW_FAA = OUT / "tmp_raw_faa"
RAW_FFN = OUT / "tmp_raw_ffn"

GENETAB = OUT / "04_gene_tables"
PROTEINS = OUT / "05_proteins"
CDS = OUT / "06_cds"
ORDER = OUT / "07_gene_order"
ADJ = OUT / "08_gene_adjacency"

TMP = BIOENV / "tmp" / "stage02"

for d in [
    NORM,
    CONTIG,
    RAW_GFF,
    RAW_FAA,
    RAW_FFN,
    GENETAB,
    PROTEINS,
    CDS,
    ORDER,
    ADJ,
    TMP,
]:
    d.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================================
# HELPERS
# =============================================================================

def clean(x):

    if pd.isna(x):
        return ""

    s = str(x).strip()

    if s.lower() in {
        "",
        "nan",
        "none",
        "na",
        "n/a",
    }:
        return ""

    return s


def open_text(path):

    path = Path(path)

    if path.name.lower().endswith(".gz"):

        return gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="ignore",
        )

    return open(
        path,
        "rt",
        encoding="utf-8",
        errors="ignore",
    )


def read_fasta(path):

    header = None
    seq = []

    with open_text(path) as fh:

        for raw in fh:

            line = raw.strip()

            if not line:
                continue

            if line.startswith(">"):

                if header is not None:

                    yield (
                        header,
                        "".join(seq),
                    )

                header = line[1:].strip()
                seq = []

            else:

                seq.append(line)

        if header is not None:

            yield (
                header,
                "".join(seq),
            )


def write_wrapped(
    fh,
    sequence,
    width=80,
):

    for i in range(
        0,
        len(sequence),
        width,
    ):

        fh.write(
            sequence[i:i + width]
            + "\n"
        )


def parse_attributes(text):

    out = {}

    for token in clean(text).split(";"):

        token = token.strip()

        if not token:
            continue

        if "=" in token:

            k, v = token.split(
                "=",
                1,
            )

            out[
                k.strip()
            ] = v.strip()

    return out


def fasta_dict_by_prodigal_id(path):

    records = {}

    for header, seq in read_fasta(path):

        m = re.search(
            r"(?:^|[;\s])ID=([^;\s]+)",
            header,
        )

        if not m:

            raise RuntimeError(
                f"Cannot parse Prodigal ID from header: "
                f"{header[:300]}"
            )

        pid = m.group(1)

        if pid in records:

            raise RuntimeError(
                f"Duplicate Prodigal ID {pid} "
                f"in {path}"
            )

        records[pid] = {
            "header": header,
            "sequence": seq.upper(),
        }

    return records


# =============================================================================
# LOAD FROZEN GENOME SET
# =============================================================================

blocks = pd.read_csv(
    ROBUST,
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

required = [
    "strain_entity_id",
    "canonical_genome_path",
    "FINAL_nearclone_99_5_block",
    "FINAL_lineage_95_block",
]

missing = [
    c
    for c in required
    if c not in blocks.columns
]

if missing:

    raise RuntimeError(
        f"Missing block columns: {missing}"
    )


# Stage 01.1 must already have removed STRAIN_000003.
if (
    "STRAIN_000003"
    in set(blocks["strain_entity_id"])
):

    raise RuntimeError(
        "STRAIN_000003 unexpectedly present "
        "after Stage 01.1 QC."
    )


if len(blocks) != 994:

    raise RuntimeError(
        f"Expected exactly 994 QC-pass genomes; "
        f"observed {len(blocks)}."
    )


if blocks["strain_entity_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate strain_entity_id in frozen "
        "Stage 01.1 genome set."
    )


print("=" * 110)
print("STAGE 02 DATASET LOCK")
print("=" * 110)

print(
    "QC-pass genomes               :",
    len(blocks)
)

print(
    "FINAL near-clone blocks       :",
    blocks[
        "FINAL_nearclone_99_5_block"
    ].nunique()
)

print(
    "FINAL ANI95 lineage blocks    :",
    blocks[
        "FINAL_lineage_95_block"
    ].nunique()
)

print(
    "PRIMARY unique strains        :",
    primary[
        "strain_entity_id"
    ].nunique()
)


# =============================================================================
# 1. NORMALIZE CONTIG IDs
# =============================================================================

print()
print("[1/5] Normalizing contig IDs")


def normalize_genome(record):

    sid = clean(
        record["strain_entity_id"]
    )

    source = Path(
        clean(
            record[
                "canonical_genome_path"
            ]
        )
    )

    out_fna = (
        NORM /
        f"{sid}.fna.gz"
    )

    out_map = (
        CONTIG /
        f"{sid}.contigs.tsv.gz"
    )

    if (
        out_fna.exists()
        and out_fna.stat().st_size > 0
        and out_map.exists()
        and out_map.stat().st_size > 0
    ):

        m = pd.read_csv(
            out_map,
            sep="\t",
            keep_default_na=False,
        )

        return {
            "strain_entity_id":
                sid,

            "source_genome_path":
                str(source),

            "normalized_genome_path":
                str(out_fna),

            "n_contigs":
                len(m),

            "total_bp":
                int(
                    pd.to_numeric(
                        m["contig_length_bp"]
                    ).sum()
                ),

            "largest_contig_bp":
                int(
                    pd.to_numeric(
                        m["contig_length_bp"]
                    ).max()
                ),

            "normalization_status":
                "CACHED",
        }


    if not source.exists():

        raise FileNotFoundError(
            source
        )


    tmp_fna = Path(
        str(out_fna) + ".tmp"
    )

    tmp_map = Path(
        str(out_map) + ".tmp"
    )


    contig_rows = []

    total_bp = 0
    largest = 0

    with gzip.open(
        tmp_fna,
        "wt",
        encoding="utf-8",
    ) as fout:

        for i, (
            original_header,
            seq,
        ) in enumerate(
            read_fasta(source),
            start=1,
        ):

            seq = (
                seq
                .replace(" ", "")
                .replace("\t", "")
                .upper()
            )

            if not seq:
                continue

            contig_id = (
                f"{sid}_ctg{i:06d}"
            )

            length = len(seq)

            total_bp += length
            largest = max(
                largest,
                length,
            )

            n_count = seq.count(
                "N"
            )

            contig_rows.append({
                "strain_entity_id":
                    sid,

                "contig_order":
                    i,

                "contig_id":
                    contig_id,

                "original_header":
                    original_header,

                "contig_length_bp":
                    length,

                "N_count":
                    n_count,

                "N_fraction":
                    (
                        n_count / length
                        if length
                        else 0.0
                    ),
            })

            fout.write(
                f">{contig_id}\n"
            )

            write_wrapped(
                fout,
                seq,
            )


    if not contig_rows:

        if tmp_fna.exists():
            tmp_fna.unlink()

        raise RuntimeError(
            f"No contigs found for {sid}"
        )


    pd.DataFrame(
        contig_rows
    ).to_csv(
        tmp_map,
        sep="\t",
        index=False,
        compression="gzip",
    )


    tmp_fna.replace(
        out_fna
    )

    tmp_map.replace(
        out_map
    )


    return {
        "strain_entity_id":
            sid,

        "source_genome_path":
            str(source),

        "normalized_genome_path":
            str(out_fna),

        "n_contigs":
            len(contig_rows),

        "total_bp":
            total_bp,

        "largest_contig_bp":
            largest,

        "normalization_status":
            "PASS",
    }


norm_rows = []

with ThreadPoolExecutor(
    max_workers=N_JOBS
) as ex:

    futures = {
        ex.submit(
            normalize_genome,
            row,
        ):
        clean(
            row["strain_entity_id"]
        )

        for row in blocks.to_dict(
            orient="records"
        )
    }

    done = 0

    for fut in as_completed(
        futures
    ):

        sid = futures[fut]

        try:

            norm_rows.append(
                fut.result()
            )

        except Exception as e:

            norm_rows.append({
                "strain_entity_id":
                    sid,

                "normalization_status":
                    "FAIL",

                "error":
                    repr(e),
            })

        done += 1

        if (
            done % 100 == 0
            or
            done == len(futures)
        ):

            print(
                f"[NORMALIZE] "
                f"{done}/{len(futures)}",
                flush=True,
            )


norm_status = pd.DataFrame(
    norm_rows
)

norm_status.to_csv(
    OUT /
    "01_normalization_status.csv",
    index=False,
    encoding="utf-8-sig",
)


failed_norm = norm_status[
    norm_status[
        "normalization_status"
    ] == "FAIL"
]


if len(failed_norm):

    failed_norm.to_csv(
        OUT /
        "ERROR_normalization_failures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raise RuntimeError(
        f"Normalization failed for "
        f"{len(failed_norm)} genomes."
    )


# Combine contig maps without loading all genome sequences.
all_contig_maps = []

for sid in sorted(
    blocks[
        "strain_entity_id"
    ]
):

    p = (
        CONTIG /
        f"{sid}.contigs.tsv.gz"
    )

    all_contig_maps.append(
        pd.read_csv(
            p,
            sep="\t",
            keep_default_na=False,
        )
    )


contig_manifest = pd.concat(
    all_contig_maps,
    ignore_index=True,
)


contig_manifest.to_csv(
    OUT /
    "02_contig_manifest.csv.gz",
    index=False,
    compression="gzip",
)


print(
    "[PASS] Normalized genomes:",
    len(norm_status)
)

print(
    "[INFO] Total contigs      :",
    len(contig_manifest)
)


# =============================================================================
# 2. PRODIGAL
# =============================================================================

print()
print("[2/5] Prodigal gene calling")


def run_prodigal(sid):

    norm = (
        NORM /
        f"{sid}.fna.gz"
    )

    gff = (
        RAW_GFF /
        f"{sid}.gff"
    )

    faa = (
        RAW_FAA /
        f"{sid}.faa"
    )

    ffn = (
        RAW_FFN /
        f"{sid}.ffn"
    )

    final_gene = (
        GENETAB /
        f"{sid}.genes.tsv.gz"
    )

    final_faa = (
        PROTEINS /
        f"{sid}.faa.gz"
    )

    final_ffn = (
        CDS /
        f"{sid}.ffn.gz"
    )

    final_order = (
        ORDER /
        f"{sid}.order.tsv.gz"
    )

    final_adj = (
        ADJ /
        f"{sid}.adjacency.tsv.gz"
    )

    final_gff = (
        RAW_GFF /
        f"{sid}.gff.gz"
    )


    # Fully completed cache.
    if all(
        p.exists()
        and p.stat().st_size > 0

        for p in [
            final_gene,
            final_faa,
            final_ffn,
            final_order,
            final_adj,
            final_gff,
        ]
    ):

        return {
            "strain_entity_id":
                sid,

            "prodigal_status":
                "CACHED_FINAL",

            "return_code":
                0,
        }


    temp_fna = (
        TMP /
        f"{sid}.fna"
    )

    try:

        # Decompress normalized input.
        with gzip.open(
            norm,
            "rt",
        ) as fi, open(
            temp_fna,
            "wt",
        ) as fo:

            shutil.copyfileobj(
                fi,
                fo,
            )


        cmd = [
            str(PRODIGAL),
            "-i",
            str(temp_fna),
            "-o",
            str(gff),
            "-a",
            str(faa),
            "-d",
            str(ffn),
            "-f",
            "gff",
            "-p",
            "single",
            "-q",
        ]


        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
            },
        )


        return {
            "strain_entity_id":
                sid,

            "prodigal_status":
                (
                    "PASS"
                    if result.returncode == 0
                    else "FAIL"
                ),

            "return_code":
                result.returncode,

            "stderr":
                result.stderr[-3000:],

            "stdout":
                result.stdout[-1000:],
        }


    finally:

        if temp_fna.exists():

            temp_fna.unlink()


call_rows = []

with ThreadPoolExecutor(
    max_workers=N_JOBS
) as ex:

    futures = {
        ex.submit(
            run_prodigal,
            sid,
        ):
        sid

        for sid in sorted(
            blocks[
                "strain_entity_id"
            ]
        )
    }

    done = 0

    for fut in as_completed(
        futures
    ):

        sid = futures[fut]

        try:

            call_rows.append(
                fut.result()
            )

        except Exception as e:

            call_rows.append({
                "strain_entity_id":
                    sid,

                "prodigal_status":
                    "FAIL",

                "return_code":
                    -1,

                "stderr":
                    repr(e),
            })

        done += 1

        if (
            done % 100 == 0
            or
            done == len(futures)
        ):

            print(
                f"[PRODIGAL] "
                f"{done}/{len(futures)}",
                flush=True,
            )


call_status = pd.DataFrame(
    call_rows
)

call_status.to_csv(
    OUT /
    "03_prodigal_status.csv",
    index=False,
    encoding="utf-8-sig",
)


fail_call = call_status[
    call_status[
        "prodigal_status"
    ] == "FAIL"
]


if len(fail_call):

    fail_call.to_csv(
        OUT /
        "ERROR_prodigal_failures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raise RuntimeError(
        f"Prodigal failed for "
        f"{len(fail_call)} genomes."
    )


print(
    "[PASS] Prodigal completed/cached:",
    len(call_status)
)


# =============================================================================
# 3. PARSE / STABLE GENE IDs / ORDER
# =============================================================================

print()
print(
    "[3/5] Stable gene IDs + "
    "contig-aware gene order"
)


def parse_genome(sid):

    final_gene = (
        GENETAB /
        f"{sid}.genes.tsv.gz"
    )

    final_faa = (
        PROTEINS /
        f"{sid}.faa.gz"
    )

    final_ffn = (
        CDS /
        f"{sid}.ffn.gz"
    )

    final_order = (
        ORDER /
        f"{sid}.order.tsv.gz"
    )

    final_adj = (
        ADJ /
        f"{sid}.adjacency.tsv.gz"
    )

    final_gff = (
        RAW_GFF /
        f"{sid}.gff.gz"
    )


    # Cache
    if all(
        p.exists()
        and p.stat().st_size > 0

        for p in [
            final_gene,
            final_faa,
            final_ffn,
            final_order,
            final_adj,
            final_gff,
        ]
    ):

        gt = pd.read_csv(
            final_gene,
            sep="\t",
            keep_default_na=False,
        )

        return {
            "strain_entity_id":
                sid,

            "parse_status":
                "CACHED",

            "n_genes":
                len(gt),

            "n_contigs_with_genes":
                gt[
                    "contig_id"
                ].nunique(),

            "partial_genes":
                int(
                    (
                        gt[
                            "partial"
                        ]
                        != "00"
                    ).sum()
                ),

            "internal_stop_proteins":
                int(
                    pd.to_numeric(
                        gt[
                            "internal_stop"
                        ],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                ),
        }


    gff = (
        RAW_GFF /
        f"{sid}.gff"
    )

    faa = (
        RAW_FAA /
        f"{sid}.faa"
    )

    ffn = (
        RAW_FFN /
        f"{sid}.ffn"
    )

    cmap = pd.read_csv(
        CONTIG /
        f"{sid}.contigs.tsv.gz",
        sep="\t",
        keep_default_na=False,
    )

    contig_order = dict(
        zip(
            cmap["contig_id"],
            pd.to_numeric(
                cmap["contig_order"]
            ).astype(int),
        )
    )

    contig_length = dict(
        zip(
            cmap["contig_id"],
            pd.to_numeric(
                cmap["contig_length_bp"]
            ).astype(int),
        )
    )


    genes = []

    with open(
        gff,
        "rt",
        encoding="utf-8",
        errors="ignore",
    ) as fh:

        for raw in fh:

            if (
                not raw.strip()
                or
                raw.startswith("#")
            ):
                continue

            fields = raw.rstrip(
                "\n"
            ).split("\t")

            if len(fields) != 9:
                continue

            (
                seqid,
                source,
                feature_type,
                start,
                end,
                score,
                strand,
                phase,
                attrs_text,
            ) = fields

            if feature_type != "CDS":
                continue

            attrs = parse_attributes(
                attrs_text
            )

            pid = clean(
                attrs.get(
                    "ID",
                    "",
                )
            )

            if not pid:

                raise RuntimeError(
                    f"{sid}: CDS without "
                    f"Prodigal ID."
                )

            genes.append({
                "strain_entity_id":
                    sid,

                "contig_id":
                    seqid,

                "contig_order":
                    contig_order.get(
                        seqid,
                        -1,
                    ),

                "contig_length_bp":
                    contig_length.get(
                        seqid,
                        np.nan,
                    ),

                "start":
                    int(start),

                "end":
                    int(end),

                "strand":
                    strand,

                "phase":
                    phase,

                "prodigal_id":
                    pid,

                "partial":
                    clean(
                        attrs.get(
                            "partial",
                            "",
                        )
                    ),

                "start_type":
                    clean(
                        attrs.get(
                            "start_type",
                            "",
                        )
                    ),

                "rbs_motif":
                    clean(
                        attrs.get(
                            "rbs_motif",
                            "",
                        )
                    ),

                "rbs_spacer":
                    clean(
                        attrs.get(
                            "rbs_spacer",
                            "",
                        )
                    ),
            })


    if not genes:

        raise RuntimeError(
            f"{sid}: no CDS predicted."
        )


    g = pd.DataFrame(
        genes
    )


    if (
        g["contig_order"] < 1
    ).any():

        raise RuntimeError(
            f"{sid}: GFF contig cannot be "
            f"mapped to normalized contig."
        )


    g = g.sort_values(
        [
            "contig_order",
            "start",
            "end",
            "strand",
        ]
    ).reset_index(
        drop=True
    )


    # Stable global gene IDs.
    g[
        "global_gene_order"
    ] = np.arange(
        1,
        len(g) + 1,
    )


    g[
        "gene_order_within_contig"
    ] = (
        g
        .groupby(
            "contig_id"
        )
        .cumcount()
        + 1
    )


    g[
        "gene_uid"
    ] = [
        f"{sid}_g{i:06d}"

        for i in g[
            "global_gene_order"
        ]
    ]


    faa_dict = fasta_dict_by_prodigal_id(
        faa
    )

    ffn_dict = fasta_dict_by_prodigal_id(
        ffn
    )


    gff_ids = set(
        g["prodigal_id"]
    )

    if gff_ids != set(
        faa_dict
    ):

        missing_faa = (
            gff_ids
            -
            set(faa_dict)
        )

        extra_faa = (
            set(faa_dict)
            -
            gff_ids
        )

        raise RuntimeError(
            f"{sid}: GFF/FAA ID mismatch. "
            f"missing={len(missing_faa)} "
            f"extra={len(extra_faa)}"
        )


    if gff_ids != set(
        ffn_dict
    ):

        raise RuntimeError(
            f"{sid}: GFF/FFN ID mismatch."
        )


    protein_lengths = []
    cds_lengths = []
    internal_stops = []


    with gzip.open(
        final_faa,
        "wt",
        encoding="utf-8",
    ) as protein_out, gzip.open(
        final_ffn,
        "wt",
        encoding="utf-8",
    ) as cds_out:

        for _, row in g.iterrows():

            pid = row[
                "prodigal_id"
            ]

            uid = row[
                "gene_uid"
            ]

            prot = (
                faa_dict[pid][
                    "sequence"
                ]
                .strip()
                .upper()
            )

            # Prodigal commonly emits terminal '*'.
            prot_clean = (
                prot[:-1]
                if prot.endswith("*")
                else prot
            )

            internal_stop = (
                "*" in prot_clean
            )

            cds_seq = (
                ffn_dict[pid][
                    "sequence"
                ]
                .strip()
                .upper()
            )


            protein_lengths.append(
                len(prot_clean)
            )

            cds_lengths.append(
                len(cds_seq)
            )

            internal_stops.append(
                int(internal_stop)
            )


            protein_out.write(
                f">{uid} "
                f"strain={sid} "
                f"contig={row['contig_id']} "
                f"start={row['start']} "
                f"end={row['end']} "
                f"strand={row['strand']} "
                f"prodigal_id={pid}\n"
            )

            write_wrapped(
                protein_out,
                prot_clean,
            )


            cds_out.write(
                f">{uid} "
                f"strain={sid} "
                f"contig={row['contig_id']} "
                f"start={row['start']} "
                f"end={row['end']} "
                f"strand={row['strand']} "
                f"prodigal_id={pid}\n"
            )

            write_wrapped(
                cds_out,
                cds_seq,
            )


    g[
        "protein_length_aa"
    ] = protein_lengths

    g[
        "cds_length_nt"
    ] = cds_lengths

    g[
        "internal_stop"
    ] = internal_stops


    # Consistency check.
    expected_nt = (
        g["end"]
        -
        g["start"]
        +
        1
    )

    g[
        "coordinate_length_nt"
    ] = expected_nt

    g[
        "cds_coordinate_length_match"
    ] = (
        g[
            "cds_length_nt"
        ]
        ==
        expected_nt
    )


    g.to_csv(
        final_gene,
        sep="\t",
        index=False,
        compression="gzip",
    )


    # -----------------------------------------------------------------
    # Contig-level raw gene-order grammar
    # -----------------------------------------------------------------

    order_rows = []

    adjacency_rows = []


    for contig_id, cg in g.groupby(
        "contig_id",
        sort=False,
    ):

        cg = cg.sort_values(
            [
                "start",
                "end",
            ]
        ).reset_index(
            drop=True
        )


        gene_tokens = list(
            cg[
                "gene_uid"
            ]
        )

        strands = list(
            cg[
                "strand"
            ]
        )


        order_rows.append({
            "strain_entity_id":
                sid,

            "contig_id":
                contig_id,

            "contig_order":
                int(
                    cg[
                        "contig_order"
                    ].iloc[0]
                ),

            "contig_length_bp":
                int(
                    cg[
                        "contig_length_bp"
                    ].iloc[0]
                ),

            "n_genes":
                len(cg),

            "gene_uid_sequence":
                " ".join(
                    gene_tokens
                ),

            "strand_sequence":
                "".join(
                    strands
                ),
        })


        # --------------------------------------------------------------
        # Adjacency grammar:
        # current gene -> next gene within same contig only.
        # NEVER bridge across contig boundaries.
        # --------------------------------------------------------------

        for i in range(
            len(cg) - 1
        ):

            a = cg.iloc[i]
            b = cg.iloc[i + 1]

            intergenic = (
                int(b["start"])
                -
                int(a["end"])
                -
                1
            )

            adjacency_rows.append({
                "strain_entity_id":
                    sid,

                "contig_id":
                    contig_id,

                "contig_order":
                    int(
                        a[
                            "contig_order"
                        ]
                    ),

                "left_gene_uid":
                    a[
                        "gene_uid"
                    ],

                "right_gene_uid":
                    b[
                        "gene_uid"
                    ],

                "left_gene_order":
                    int(
                        a[
                            "gene_order_within_contig"
                        ]
                    ),

                "right_gene_order":
                    int(
                        b[
                            "gene_order_within_contig"
                        ]
                    ),

                "orientation":
                    str(
                        a["strand"]
                    )
                    +
                    str(
                        b["strand"]
                    ),

                "intergenic_bp":
                    intergenic,

                "overlap_bp":
                    max(
                        0,
                        -intergenic,
                    ),

                "gap_bp":
                    max(
                        0,
                        intergenic,
                    ),
            })


    pd.DataFrame(
        order_rows
    ).to_csv(
        final_order,
        sep="\t",
        index=False,
        compression="gzip",
    )


    pd.DataFrame(
        adjacency_rows,
        columns=[
            "strain_entity_id",
            "contig_id",
            "contig_order",
            "left_gene_uid",
            "right_gene_uid",
            "left_gene_order",
            "right_gene_order",
            "orientation",
            "intergenic_bp",
            "overlap_bp",
            "gap_bp",
        ]
    ).to_csv(
        final_adj,
        sep="\t",
        index=False,
        compression="gzip",
    )


    # -----------------------------------------------------------------
    # Preserve compressed GFF, remove temporary duplicate FASTAs.
    # -----------------------------------------------------------------

    with open(
        gff,
        "rb",
    ) as fi, gzip.open(
        final_gff,
        "wb",
    ) as fo:

        shutil.copyfileobj(
            fi,
            fo,
        )


    for p in [
        gff,
        faa,
        ffn,
    ]:

        if p.exists():
            p.unlink()


    return {
        "strain_entity_id":
            sid,

        "parse_status":
            "PASS",

        "n_genes":
            len(g),

        "n_contigs_with_genes":
            g[
                "contig_id"
            ].nunique(),

        "partial_genes":
            int(
                (
                    g[
                        "partial"
                    ]
                    != "00"
                ).sum()
            ),

        "internal_stop_proteins":
            int(
                g[
                    "internal_stop"
                ].sum()
            ),

        "coordinate_length_mismatches":
            int(
                (
                    ~g[
                        "cds_coordinate_length_match"
                    ]
                ).sum()
            ),

        "median_protein_length_aa":
            float(
                g[
                    "protein_length_aa"
                ].median()
            ),
    }


parse_rows = []

with ThreadPoolExecutor(
    max_workers=N_JOBS
) as ex:

    futures = {
        ex.submit(
            parse_genome,
            sid,
        ):
        sid

        for sid in sorted(
            blocks[
                "strain_entity_id"
            ]
        )
    }

    done = 0

    for fut in as_completed(
        futures
    ):

        sid = futures[fut]

        try:

            parse_rows.append(
                fut.result()
            )

        except Exception as e:

            parse_rows.append({
                "strain_entity_id":
                    sid,

                "parse_status":
                    "FAIL",

                "error":
                    repr(e),
            })

        done += 1

        if (
            done % 100 == 0
            or
            done == len(futures)
        ):

            print(
                f"[PARSE] "
                f"{done}/{len(futures)}",
                flush=True,
            )


parse_status = pd.DataFrame(
    parse_rows
)


parse_status.to_csv(
    OUT /
    "04_gene_parsing_status.csv",
    index=False,
    encoding="utf-8-sig",
)


parse_fail = parse_status[
    parse_status[
        "parse_status"
    ] == "FAIL"
]


if len(parse_fail):

    parse_fail.to_csv(
        OUT /
        "ERROR_gene_parsing_failures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raise RuntimeError(
        f"Gene parsing failed for "
        f"{len(parse_fail)} genomes."
    )


print(
    "[PASS] Stable gene tables:",
    len(parse_status)
)


# =============================================================================
# 4. AGGREGATE WITHOUT LOADING EVERYTHING INTO RAM
# =============================================================================

print()
print("[4/5] Building combined protein/CDS/order resources")


def concatenate_text_gz(
    inputs,
    output,
):

    with gzip.open(
        output,
        "wt",
        encoding="utf-8",
    ) as out:

        for p in inputs:

            with gzip.open(
                p,
                "rt",
                encoding="utf-8",
                errors="ignore",
            ) as fh:

                shutil.copyfileobj(
                    fh,
                    out,
                )


def concatenate_tsv_gz(
    inputs,
    output,
):

    first = True

    with gzip.open(
        output,
        "wt",
        encoding="utf-8",
    ) as out:

        for p in inputs:

            with gzip.open(
                p,
                "rt",
                encoding="utf-8",
                errors="ignore",
            ) as fh:

                header = fh.readline()

                if first:

                    out.write(
                        header
                    )

                    first = False

                for line in fh:

                    out.write(
                        line
                    )


sids = sorted(
    blocks[
        "strain_entity_id"
    ]
)


concatenate_text_gz(
    [
        PROTEINS /
        f"{sid}.faa.gz"

        for sid in sids
    ],

    OUT /
    "05_all_proteins.faa.gz",
)


concatenate_text_gz(
    [
        CDS /
        f"{sid}.ffn.gz"

        for sid in sids
    ],

    OUT /
    "06_all_cds.ffn.gz",
)


concatenate_tsv_gz(
    [
        GENETAB /
        f"{sid}.genes.tsv.gz"

        for sid in sids
    ],

    OUT /
    "07_all_genes.tsv.gz",
)


concatenate_tsv_gz(
    [
        ORDER /
        f"{sid}.order.tsv.gz"

        for sid in sids
    ],

    OUT /
    "08_contig_gene_order.tsv.gz",
)


concatenate_tsv_gz(
    [
        ADJ /
        f"{sid}.adjacency.tsv.gz"

        for sid in sids
    ],

    OUT /
    "09_gene_adjacency.tsv.gz",
)


# =============================================================================
# 5. QC + MANIFEST
# =============================================================================

print()
print("[5/5] Stage 02 QC and manifest freeze")


qc = blocks.merge(
    norm_status,
    on="strain_entity_id",
    how="left",
)


qc = qc.merge(
    parse_status,
    on="strain_entity_id",
    how="left",
)


for c in [
    "n_genes",
    "n_contigs",
    "partial_genes",
    "internal_stop_proteins",
    "coordinate_length_mismatches",
    "total_bp",
    "largest_contig_bp",
]:

    if c in qc.columns:

        qc[c] = pd.to_numeric(
            qc[c],
            errors="coerce",
        )


qc[
    "partial_gene_fraction"
] = (
    qc[
        "partial_genes"
    ]
    /
    qc[
        "n_genes"
    ]
)


# Non-destructive warning flags only.
def qc_flag(r):

    flags = []

    n_gene = r.get(
        "n_genes",
        np.nan,
    )

    total_bp = r.get(
        "total_bp",
        np.nan,
    )

    partial = r.get(
        "partial_gene_fraction",
        np.nan,
    )

    internal = r.get(
        "internal_stop_proteins",
        np.nan,
    )

    mismatch = r.get(
        "coordinate_length_mismatches",
        np.nan,
    )


    if pd.notna(
        n_gene
    ) and n_gene < 500:

        flags.append(
            "LOW_GENE_COUNT"
        )


    if pd.notna(
        total_bp
    ) and total_bp < 1_000_000:

        flags.append(
            "SMALL_ASSEMBLY_LT_1MB"
        )


    if pd.notna(
        partial
    ) and partial > 0.30:

        flags.append(
            "HIGH_PARTIAL_GENE_FRACTION"
        )


    if pd.notna(
        internal
    ) and internal > 0:

        flags.append(
            "INTERNAL_STOP_PROTEIN"
        )


    if pd.notna(
        mismatch
    ) and mismatch > 0:

        flags.append(
            "CDS_COORDINATE_MISMATCH"
        )


    return (
        "PASS"
        if not flags
        else "WARN:"
        + "|".join(flags)
    )


qc[
    "gene_call_qc"
] = qc.apply(
    qc_flag,
    axis=1,
)


qc.to_csv(
    OUT /
    "10_genome_gene_call_qc.csv",
    index=False,
    encoding="utf-8-sig",
)


# --------------------------------------------------------------
# Explicit PRIMARY audit
# --------------------------------------------------------------

primary_ids = set(
    primary[
        "strain_entity_id"
    ]
)


stage02_ids = set(
    qc[
        "strain_entity_id"
    ]
)


missing_primary = sorted(
    primary_ids
    -
    stage02_ids
)


if missing_primary:

    raise RuntimeError(
        "PRIMARY strains missing after Stage 02: "
        + str(
            missing_primary
        )
    )


primary_qc = qc[
    qc[
        "strain_entity_id"
    ].isin(
        primary_ids
    )
].copy()


primary_qc.to_csv(
    OUT /
    "11_PRIMARY_gene_call_qc.csv",
    index=False,
    encoding="utf-8-sig",
)


# --------------------------------------------------------------
# Combined gene count without reading one giant DataFrame
# --------------------------------------------------------------

total_genes = int(
    pd.to_numeric(
        parse_status[
            "n_genes"
        ],
        errors="coerce",
    ).sum()
)


total_partial = int(
    pd.to_numeric(
        parse_status[
            "partial_genes"
        ],
        errors="coerce",
    ).sum()
)


total_internal_stop = int(
    pd.to_numeric(
        parse_status[
            "internal_stop_proteins"
        ],
        errors="coerce",
    ).sum()
)


summary = {
    "genomes_input":
        int(len(blocks)),

    "genomes_gene_called":
        int(len(parse_status)),

    "total_genes":
        total_genes,

    "median_genes_per_genome":
        float(
            pd.to_numeric(
                parse_status[
                    "n_genes"
                ],
                errors="coerce",
            ).median()
        ),

    "min_genes_per_genome":
        int(
            pd.to_numeric(
                parse_status[
                    "n_genes"
                ],
                errors="coerce",
            ).min()
        ),

    "max_genes_per_genome":
        int(
            pd.to_numeric(
                parse_status[
                    "n_genes"
                ],
                errors="coerce",
            ).max()
        ),

    "total_contigs":
        int(
            len(
                contig_manifest
            )
        ),

    "total_partial_genes":
        total_partial,

    "partial_gene_fraction":
        (
            total_partial
            /
            total_genes
            if total_genes
            else None
        ),

    "internal_stop_proteins":
        total_internal_stop,

    "genomes_PASS_without_warning":
        int(
            (
                qc[
                    "gene_call_qc"
                ]
                ==
                "PASS"
            ).sum()
        ),

    "genomes_with_warning":
        int(
            qc[
                "gene_call_qc"
            ]
            .str.startswith(
                "WARN"
            )
            .sum()
        ),

    "nearclone_blocks":
        int(
            blocks[
                "FINAL_nearclone_99_5_block"
            ].nunique()
        ),

    "lineage95_blocks":
        int(
            blocks[
                "FINAL_lineage_95_block"
            ].nunique()
        ),

    "PRIMARY_unique_strains":
        int(
            primary[
                "strain_entity_id"
            ].nunique()
        ),

    "timestamp":
        datetime.now().isoformat(),
}


with open(
    OUT /
    "12_stage02_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print("=" * 110)
print("STAGE 02 GENE-ORDER SUMMARY")
print("=" * 110)

for k, v in summary.items():

    print(
        f"{k:38s}: {v}"
    )


print()
print(
    "[IMPORTANT] Gene-order edges are strictly "
    "within-contig. No adjacency is created "
    "across contig boundaries."
)

print()
print(
    "[IMPORTANT] These are raw gene-order identities. "
    "Protein-family grammar tokens will be created "
    "in Stage 03 using MMseqs2."
)
