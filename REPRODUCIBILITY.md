# GenoGrammar reproducibility

## Formal scope

GenoGrammar learns order-sensitive local genomic neighborhood
representations.

The formal downstream task is DOOR2-derived adjacent-gene
operon-status prediction.

The DOOR2-derived labels are external computational annotations
rather than experimentally established ground truth.

## Formal dataset

- 164,302 adjacent-gene examples
- 96 genomes
- 54 ANI95 lineages
- 64-gene local genomic-neighborhood windows
- target adjacent pair: zero-based positions 31 and 32

## Evaluation

ANI95 lineage-blocked five-fold out-of-fold evaluation.

## Formal result

- ROC-AUC: 0.860914
- PR-AUC: 0.803928
- Balanced Accuracy: 0.769288
- F1: 0.743938
- MCC: 0.536842

## Reproducibility boundary

The public fresh workflow starts from:

- packaged pretrained lineage-clean GenoGrammar checkpoints
- packaged family embeddings
- formal Stage50 downstream inputs

It reproduces the formal Stage55-60 downstream workflow.

It does not retrain the complete historical self-supervised
pretraining workflow from raw genomes.

The raw_data directory is included for data availability and
provenance.

## Commands

Audit the package:

    python run.py --audit-only

Evaluate packaged frozen results:

    python run.py

Inspect the fresh execution plan:

    python run.py --fresh --plan-only

Run fresh Stage55-60:

    python run.py --fresh

Generated outputs are written to:

    results/
