import os
from pathlib import Path
from collections import defaultdict

import csv
import gzip
import hashlib
import json
import re
import sys


ROOT = Path(os.environ["GENOGRAMMA_WORK_ROOT"]).resolve()

S02 = ROOT / "02_gene_order"
S03 = ROOT / "03_protein_families"
OUT = ROOT / "06_family_esm2"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PROTEINS = S02 / "05_all_proteins.faa.gz"

SOURCES = [
    S03 / "04_family_summary.csv.gz",
    S03 / "02_protein_family_membership.tsv.gz",
    S03 / "03_gene_family_master.tsv.gz",
]

EXPECTED_FAMILIES = 276_517
EXPECTED_PROTEINS = 2_583_281

OUT_FASTA = OUT / "01_family_representatives.faa.gz"
OUT_INDEX = OUT / "02_family_representative_index.csv"
OUT_AUDIT = OUT / "00_representative_extraction_audit.json"


def opener(path):

    if str(path).endswith(".gz"):
        return gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="replace",
        )

    return open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    )


def delimiter(path):

    name = path.name.lower()

    if ".csv" in name:
        return ","

    return "\t"


def header(path):

    with opener(path) as fh:

        reader = csv.reader(
            fh,
            delimiter=delimiter(path),
        )

        return next(reader)


def detect(
    columns,
    candidates,
):

    lower = {
        str(c).strip().lower(): c
        for c in columns
    }

    for c in candidates:

        if c in columns:
            return c

        if c.lower() in lower:
            return lower[
                c.lower()
            ]

    return None


FAMILY_CANDIDATES = [
    "family_id",
    "protein_family_id",
    "cluster_id",
    "gene_family_id",
]

REP_CANDIDATES = [
    "representative_gene_uid",
    "representative_gene_id",
    "representative_protein_uid",
    "representative_protein_id",
    "representative_member",
    "representative_id",
    "rep_gene_uid",
    "rep_gene_id",
    "cluster_representative",
    "representative",
]

MEMBER_CANDIDATES = [
    "gene_uid",
    "gene_id",
    "protein_uid",
    "protein_id",
    "member_gene_uid",
    "member_gene_id",
    "member",
]

FLAG_CANDIDATES = [
    "is_representative",
    "representative_flag",
    "is_rep",
]


def looks_like_gene_id(x):

    x = str(x).strip()

    return bool(
        re.search(
            r"STRAIN_\d+_g\d+",
            x,
        )
    )


def parse_bool(x):

    return str(x).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def try_direct_representative_source(path):

    cols = header(path)

    fam_col = detect(
        cols,
        FAMILY_CANDIDATES,
    )

    rep_col = detect(
        cols,
        REP_CANDIDATES,
    )

    if not fam_col or not rep_col:
        return None


    mapping = {}
    bad = 0


    with opener(path) as fh:

        reader = csv.DictReader(
            fh,
            delimiter=delimiter(path),
        )

        for row in reader:

            fam = str(
                row.get(
                    fam_col,
                    "",
                )
            ).strip()

            rep = str(
                row.get(
                    rep_col,
                    "",
                )
            ).strip()


            if not fam or not rep:
                continue


            if not looks_like_gene_id(
                rep
            ):
                bad += 1
                continue


            old = mapping.get(
                fam
            )

            if (
                old is not None
                and
                old != rep
            ):
                raise RuntimeError(
                    f"Conflicting representatives "
                    f"for {fam}: {old} vs {rep}"
                )

            mapping[
                fam
            ] = rep


    if len(
        mapping
    ) == EXPECTED_FAMILIES:

        return {
            "path":
                str(path),

            "method":
                "direct_representative_column",

            "family_column":
                fam_col,

            "representative_column":
                rep_col,

            "mapping":
                mapping,

            "bad_non_gene_values":
                bad,
        }

    return None


def try_flagged_representative_source(path):

    cols = header(path)

    fam_col = detect(
        cols,
        FAMILY_CANDIDATES,
    )

    member_col = detect(
        cols,
        MEMBER_CANDIDATES,
    )

    flag_col = detect(
        cols,
        FLAG_CANDIDATES,
    )


    if not (
        fam_col
        and
        member_col
        and
        flag_col
    ):
        return None


    mapping = {}


    with opener(path) as fh:

        reader = csv.DictReader(
            fh,
            delimiter=delimiter(path),
        )

        for row in reader:

            if not parse_bool(
                row.get(
                    flag_col,
                    "",
                )
            ):
                continue


            fam = str(
                row[
                    fam_col
                ]
            ).strip()

            rep = str(
                row[
                    member_col
                ]
            ).strip()


            if fam and rep:

                if fam in mapping:

                    raise RuntimeError(
                        f"Multiple representative rows "
                        f"for family {fam}"
                    )

                mapping[
                    fam
                ] = rep


    if len(
        mapping
    ) == EXPECTED_FAMILIES:

        return {
            "path":
                str(path),

            "method":
                "representative_flag",

            "family_column":
                fam_col,

            "representative_column":
                member_col,

            "representative_flag_column":
                flag_col,

            "mapping":
                mapping,
        }

    return None


print("=" * 120)
print("STAGE 06A — EXTRACT TRUE MMSEQS FAMILY REPRESENTATIVES")
print("=" * 120)


for src in SOURCES:

    print()
    print(
        "[INFO] source:",
        src,
    )

    print(
        "[INFO] header:",
        header(src),
    )


result = None


for src in SOURCES:

    result = (
        try_direct_representative_source(
            src
        )
    )

    if result is not None:
        break


if result is None:

    for src in SOURCES:

        result = (
            try_flagged_representative_source(
                src
            )
        )

        if result is not None:
            break


if result is None:

    raise RuntimeError(
        "Could not identify the actual MMseqs "
        "representative gene for all 276,517 families. "
        "No arbitrary substitute representative will be used."
    )


rep_map = result.pop(
    "mapping"
)


print()
print(
    "[PASS] Representative source:",
    result,
)

print(
    "[PASS] Families mapped:",
    len(
        rep_map
    ),
)


if len(
    set(
        rep_map.values()
    )
) != EXPECTED_FAMILIES:

    raise RuntimeError(
        "Representative genes are not unique across families."
    )


rep_to_family = {
    gene:
        family
    for family, gene in rep_map.items()
}


# ==============================================================================================
# Stream all 2.58M proteins and extract only true representatives
# ==============================================================================================

seq_by_family = {}

protein_count = 0

current_id = None
current_seq = []


def flush():

    global protein_count
    global current_id
    global current_seq

    if current_id is None:
        return

    protein_count += 1

    if current_id in rep_to_family:

        family = rep_to_family[
            current_id
        ]

        seq = "".join(
            current_seq
        ).strip().upper()

        # Terminal '*' is not a biological residue.
        seq = seq.rstrip(
            "*"
        )

        if not seq:

            raise RuntimeError(
                f"Empty representative sequence: "
                f"{current_id}"
            )

        if "*" in seq:

            raise RuntimeError(
                f"Internal stop in representative: "
                f"{current_id}"
            )

        if family in seq_by_family:

            raise RuntimeError(
                f"Representative encountered twice: "
                f"{family}"
            )

        seq_by_family[
            family
        ] = (
            current_id,
            seq,
        )


with gzip.open(
    PROTEINS,
    "rt",
    encoding="utf-8",
    errors="replace",
) as fh:

    for line in fh:

        line = line.rstrip(
            "\n\r"
        )

        if line.startswith(
            ">"
        ):

            flush()

            current_id = (
                line[
                    1:
                ]
                .split(
                    None,
                    1,
                )[0]
            )

            current_seq = []

        else:

            current_seq.append(
                line.strip()
            )


flush()


print()
print(
    "Protein FASTA records scanned :",
    protein_count,
)

print(
    "Representative sequences found:",
    len(
        seq_by_family
    ),
)


if protein_count != EXPECTED_PROTEINS:

    raise RuntimeError(
        f"Protein FASTA count={protein_count}; "
        f"expected={EXPECTED_PROTEINS}"
    )


missing = sorted(
    set(
        rep_map
    )
    -
    set(
        seq_by_family
    )
)


if missing:

    print(
        "[ERROR] First missing families:",
        missing[
            :20
        ],
    )

    raise RuntimeError(
        f"Missing representative sequences: "
        f"{len(missing)}"
    )


if len(
    seq_by_family
) != EXPECTED_FAMILIES:

    raise RuntimeError(
        "Representative sequence count mismatch."
    )


# ==============================================================================================
# Stable family order
# ==============================================================================================

families = sorted(
    seq_by_family
)


lengths = []


with gzip.open(
    OUT_FASTA,
    "wt",
    encoding="utf-8",
) as fout, open(
    OUT_INDEX,
    "w",
    encoding="utf-8",
    newline="",
) as idx:

    writer = csv.writer(
        idx
    )

    writer.writerow([
        "embedding_row",
        "family_id",
        "representative_gene_uid",
        "protein_length",
        "sequence_sha1",
    ])


    for row_idx, family in enumerate(
        families
    ):

        gene, seq = seq_by_family[
            family
        ]

        sha1 = hashlib.sha1(
            seq.encode()
        ).hexdigest()

        lengths.append(
            len(
                seq
            )
        )

        fout.write(
            f">{family}|{gene}\n"
        )

        for i in range(
            0,
            len(
                seq
            ),
            80,
        ):

            fout.write(
                seq[
                    i:
                    i + 80
                ]
                +
                "\n"
            )


        writer.writerow([
            row_idx,
            family,
            gene,
            len(
                seq
            ),
            sha1,
        ])


lengths_sorted = sorted(
    lengths
)


def quantile(
    arr,
    q,
):

    pos = int(
        round(
            q
            *
            (
                len(
                    arr
                )
                -
                1
            )
        )
    )

    return arr[
        pos
    ]


audit = {
    "status":
        "PASS",

    "expected_families":
        EXPECTED_FAMILIES,

    "representative_sequences":
        len(
            families
        ),

    "source":
        result,

    "proteins_scanned":
        protein_count,

    "protein_length_min":
        min(
            lengths
        ),

    "protein_length_median":
        quantile(
            lengths_sorted,
            0.50,
        ),

    "protein_length_q95":
        quantile(
            lengths_sorted,
            0.95,
        ),

    "protein_length_q99":
        quantile(
            lengths_sorted,
            0.99,
        ),

    "protein_length_max":
        max(
            lengths
        ),

    "proteins_gt_1022aa":
        sum(
            x > 1022
            for x in lengths
        ),

    "arbitrary_representative_substitution":
        False,
}


with open(
    OUT_AUDIT,
    "w",
    encoding="utf-8",
) as fh:

    json.dump(
        audit,
        fh,
        indent=2,
        ensure_ascii=False,
    )


print()
print("=" * 120)
print("REPRESENTATIVE EXTRACTION SUMMARY")
print("=" * 120)

for k, v in audit.items():

    print(
        f"{k:42s}: {v}"
    )


print()
print(
    "[FINAL PASS] 276,517 MMseqs "
    "family representative proteins frozen."
)
