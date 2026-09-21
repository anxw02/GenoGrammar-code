# GenoGramma Raw Data

This directory contains the source genome collection and associated provenance
metadata used in the GenoGramma study.

## Structure

    raw_data/
    ├── genomes/
    ├── metadata/
    ├── checksums/
    ├── genome_accessions.txt
    ├── all_candidate_accessions.txt
    └── README.md

## `genomes/`

Contains a complete copy of the source genome archive used to construct the
GenoGramma datasets.

The files are retained exactly as present in the source archive. The directory may
therefore also contain small source-side metadata files distributed together with the
genome collection.

The authoritative classification of every file is provided in:

    metadata/raw_data_file_manifest.tsv

Each file is classified as one of:

- `public_accession_genome`
- `local_named_genome`
- `metadata_sidecar`

### Public-accession genomes

These genome files could be directly associated with a GCA or GCF assembly accession.

Their accession list is provided in:

    genome_accessions.txt

and their file-level manifest is:

    metadata/public_accession_genomes.tsv

### Local named genomes

Some source genomes are identified by strain names rather than GCA/GCF accessions.

These files are retained as part of the original source-data collection and are listed
in:

    metadata/local_named_genomes.tsv

The absence of a GCA/GCF accession for these files should not be interpreted as a
missing-data error.

### Metadata sidecars

Small metadata files that were present within the original source genome archive are
retained for provenance.

They are classified in:

    metadata/metadata_sidecars.tsv

Copies are also placed under `metadata/` for easier inspection.

## `genome_accessions.txt`

Contains the public GCA/GCF identifiers that were directly resolved to genome files
included in this raw-data release.

## `all_candidate_accessions.txt`

Contains the broader set of accession candidates encountered during historical identity
resolution.

This file is retained for provenance only and must not be interpreted as the set of
genomes used in the final released dataset.

## `metadata/`

Contains source-level provenance information including genome/accession mappings,
dataset manifests, contig metadata, DOOR2 mappings, and formal downstream metadata.

The primary authoritative file inventory is:

    metadata/raw_data_file_manifest.tsv

## `checksums/`

Contains SHA256 manifests for the source genome archive and released metadata.

    checksums/genomes_SHA256SUMS.txt
    checksums/metadata_SHA256SUMS.txt

## Relationship to runtime data

`raw_data/` is an additional Data Availability copy.

It does not replace or modify the runtime directories:

    data/
    inputs/
    checkpoints/
    model_assets/
    pipeline/

These directories remain unchanged so that `run.py` continues to use the same frozen
paths and assets used for the reported analyses.

## DOOR2-derived labels

The downstream operon-status task uses DOOR2-derived computational annotations.

These labels are external computational annotations and should not be interpreted as
experimentally determined operon ground truth.

## Data Availability

This directory provides the source genome collection, public accession identifiers,
local strain-labelled genome files, provenance metadata, and integrity checksums
associated with the GenoGramma study.

Processed model inputs and analysis outputs are maintained separately from this
raw/source-data release.
