# GenoGramma Training

## Purpose

`train.py` is the public training entry point for GenoGramma.

GenoGramma learns order-sensitive representations of 64-gene local genomic neighborhoods.

## Formal configuration

- Architecture: `ExplicitNeighborhoodEncoder`
- Window length: 64 genes
- Family embedding dimension: 480
- Internal model dimension: 128
- Trainable parameters: 508,187
- ANI95 lineage-blocked five-fold training
- Fold seeds: 42, 142, 242, 342, 442
- Epochs: 5
- Steps per epoch: 400
- Batch size: 48
- Learning rate: 3e-4

The self-supervised pretraining does not use phenotype labels. The outer lineage holdout is not used for training, early stopping, or checkpoint selection.

## Inspect the training plan

```bash
python train.py --plan
```

## Current verified training route

```bash
python train.py --from-family-embeddings
```

This route starts from the packaged 276,517 − 480 family embedding matrix and reruns the lineage-controlled GenoGramma self-supervised training workflow.

## Complete from-raw route

The final intended workflow is:

```text
raw genomes / proteins
        ↓
ESM-2 protein representations
        ↓
480D gene-family embeddings
        ↓
ordered 64-gene neighborhoods
        ↓
GenoGramma self-supervised pretraining
        ↓
lineage-controlled downstream evaluation
```

The raw-data ESM producer route remains provenance-locked and must not be described as verified until the exact historical producer is packaged and tested.

<!-- FINAL_TRAINING_CONTRACT -->

## Final training contract



Default full route:



    python train.py



Plan only:



    python train.py --plan



Start from the packaged byte-verified family embedding matrix:



    python train.py --from-family-embeddings



Train only the five GenoGramma SSL folds:



    python train.py --from-family-embeddings --skip-downstream



The raw route performs Prodigal gene calling, MMseqs2 protein-family clustering, deterministic family representative selection, pretrained ESM-2 t12 35M inference, lineage-clean GenoGramma SSL training, Stage55 representation extraction, and Stage56 downstream evaluation.



ESM-2 is pretrained; GenoGramma SSL encoders and downstream heads are newly trained.



The formal downstream head is Linear(768,128) -> GELU -> Dropout(0.2) -> Linear(128,2), with fold-specific z-score parameters stored in each checkpoint.



The raw route requires Prodigal and MMseqs2 and should have approximately 60 GB or more free working space.
