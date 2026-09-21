# GenoGramma

**GenoGramma** is a genome-representation framework for learning **order-sensitive local genomic neighborhood representations**.

This repository contains the code, pretrained model assets, frozen downstream inputs, reference results, and reproducibility workflow used for the formal GenoGramma evaluation on **DOOR2-derived adjacent-gene operon-status prediction**.

The public workflow is organized around a single main entry point:

```bash
python run.py
```

Users normally do **not** need to execute the individual stage scripts or shell wrappers manually.

---

## 1. What does GenoGramma do?

For each target adjacent-gene pair, GenoGramma represents the surrounding **local genomic neighborhood** while preserving information related to gene order and local context.

The formal downstream task asks whether a target pair of adjacent genes is annotated as belonging to the same operon according to DOOR2-derived computational annotations.

The formal representation evaluated in this repository is the preregistered **768-dimensional GenoGramma pair-aware representation**.

GenoGramma should therefore be interpreted as a model of **local genomic organization** within the neighborhood scope evaluated here.

---

## 2. Formal downstream task

**Task:** DOOR2-derived adjacent-gene operon-status prediction.

**Formal dataset:**

- 164,302 adjacent-gene examples
- 96 genomes
- 54 ANI95 lineages
- 64-gene local genomic-neighborhood windows
- target adjacent-gene pair at zero-based positions `[31, 32]`

DOOR2-derived operon-status labels are **external computational annotations** and are **not experimentally established ground truth**.

---

## 3. Evaluation protocol

The formal evaluation uses **ANI95 lineage-blocked five-fold out-of-fold (OOF) cross-validation**, rather than random example-level splitting.

Genomes assigned to the same ANI95 lineage are kept within the same evaluation block. This reduces the risk that highly related genomes are distributed across training and validation partitions.

The formal evaluation therefore covers:

- 5 outer folds
- 96 genomes
- 54 ANI95 lineage blocks
- 164,302 formal adjacent-gene examples

---

## 4. Frozen formal GenoGramma result

| Metric | GenoGramma |
|---|---:|
| ROC-AUC | 0.860914 |
| PR-AUC | 0.803928 |
| Balanced Accuracy | 0.769288 |
| F1 | 0.743938 |
| MCC | 0.536842 |

The frozen result corresponds to the formal GenoGramma pair-aware representation evaluated under ANI95 lineage-blocked OOF-5CV.

---

## 5. Matched pair-aware baseline comparison

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| GenoGramma | 0.860914 | 0.803928 |
| TransformerSmall | 0.853748 | 0.787762 |
| MeanPool | 0.844809 | 0.780923 |
| BiGRU | 0.804038 | 0.735335 |
| CNN1D | 0.733892 | 0.688869 |

GenoGramma produced the highest point estimates among these matched pair-aware baselines.

However, **statistical superiority over TransformerSmall is not claimed**, because the corresponding same-strand equal-lineage bootstrap confidence interval for the GenoGramma-versus-TransformerSmall difference crosses zero.

---

## 6. Same-strand contextual-edge analysis

A separate robustness analysis evaluates the incremental contribution of contextual-edge information after restricting evaluation to same-strand adjacent-gene pairs.

| Metric | Increment | 95% CI |
|---|---:|---:|
| ROC-AUC | +0.005841 | [0.003181, 0.008608] |
| PR-AUC | +0.003963 | [0.001741, 0.006279] |

These results support a modest but measurable contribution from local contextual-edge information.

They should not be generalized beyond the local-neighborhood scope evaluated here.

---

## 7. Reproducibility scope

The public package provides manuscript reproduction, end-to-end GenoGramma retraining, and compatible-window inference.

The fresh downstream workflow starts from:

- lineage-clean pretrained GenoGramma checkpoints;
- packaged family embeddings;
- the frozen formal Stage50 downstream inputs;
- supporting model assets required by the downstream workflow.

It reproduces:

1. Stage55 — GenoGramma pair-aware representation extraction;
2. Stage56 — formal GenoGramma downstream probe;
3. Stage57B2 — native matched baseline pair-state extraction;
4. Stage57C — matched pair-aware baseline comparison;
5. Stage58A — strand-shortcut robustness audit;
6. Stage58B — ANI95 lineage-level baseline bootstrap;
7. Stage59A — GenoGramma 768D representation ablation;
8. Stage59B — ablation ANI95 lineage bootstrap;
9. Stage60 — final operon evidence freeze.

The manuscript reproduction route uses packaged pretrained assets; train.py separately provides new GenoGramma SSL and downstream training.

This distinction is intentional and should be preserved when describing reproducibility.

---

## 8. Repository structure

```text
paper_code/
├── README.md
├── REPRODUCIBILITY.md
├── MANIFEST.sha256
├── requirements.txt
├── requirements_lock.txt
├── run.py
├── baseline_models/
├── checkpoints/
├── data/
├── inputs/
├── manifests/
├── model_assets/
├── pipeline/
├── raw_data/
├── reference_results/
├── results/
└── runtime_assets/
```

### Directory descriptions

**`baseline_models/`**  
Model definitions used for matched downstream baseline comparisons.

**`checkpoints/`**  
Packaged pretrained checkpoints required by the reproducibility workflow.

**`data/`**  
Supporting processed datasets and intermediate scientific assets retained for reproducibility and provenance.

**`inputs/`**  
Packaged downstream input files used by the formal workflow.

**`manifests/`**  
Provenance, integrity, and packaging metadata.

**`model_assets/`**  
Large model resources, including family embeddings and pretrained model assets.

**`pipeline/`**  
Formal Stage55-60 analysis scripts, supporting modules, and matched-baseline scripts.

**`raw_data/`**  
Project-generated original data owned by the authors. These files are included for data availability, provenance, and independent inspection. They are not required by the default Stage55-60 downstream reproduction path.

**`reference_results/`**  
Frozen reference outputs used to validate the packaged workflow.

**`results/`**  
Destination for formal run records and isolated fresh reproduction outputs.

**`runtime_assets/`**  
Minimal frozen assets required by the packaged Stage55-60 runtime workflow.

---

## 9. Installation

Install the Python dependencies listed in:

```text
requirements.txt
```

For a more tightly specified environment, see:

```text
requirements_lock.txt
```

PyTorch/CUDA installation can depend on local hardware and CUDA versions, so users should ensure that the installed PyTorch build is compatible with their system.

---

## 10. Quick start

Run the following commands from the repository root.

### 10.1 Audit the package

```bash
python run.py --audit-only
```

This checks the packaged assets and formal Stage55-60 workflow without rerunning downstream analyses.

A valid package should report all formal stages as `PASS`.

### 10.2 Verify the packaged frozen result

```bash
python run.py
```

When frozen formal results are already present, individual stages may report:

```text
[SKIP/PASS] frozen result already exists
```

This is expected behavior.

The command should finish with:

```text
[FINAL PASS] GenoGramma training/evaluation orchestration complete
```

and report:

```text
ROC-AUC = 0.860914
PR-AUC  = 0.803928
BalAcc  = 0.769288
F1      = 0.743938
MCC     = 0.536842
```

### 10.3 Inspect the fresh downstream plan

```bash
python run.py --fresh --plan-only
```

This validates that all assets required for an isolated fresh Stage55-60 downstream reproduction are available, without executing training/evaluation.

Expected final checks include:

```text
[PASS] all packaged assets available
[PASS] all formal Stage55-60 scripts availe
[PASS] fresh result root override supported
[PASS] no training executed
```

### 10.4 Run a fresh Stage55-60 downstream reproduction

```bash
python run.py --fresh
```

This executes an isolated fresh downstream reproduction using the packaged checkpoints, family embeddings, Stage50 inputs, and supporting assets.

Fresh outputs are written under:

```text
results/reproduction_runs/
```

The packaged frozen reference results are not intended to be overwritten by this mode.

---

## 11. Frozen versus fresh execution

### Frozen evaluation

```bash
python run.py
```

This verifies and reports the packaged frozen manuscript results.

### Fresh downstream reproduction

```bash
python run.py --fresh
```

This creates a new isolated Stage55-60 downstream run from the packaged pretrained assets.

Neither mode should be interpreted as rerunning the complete historical self-supervised pretraining workflow from raw genomes.

---

## 12. Why are there `run_stage*.sh` files?

The repository contains nine small shell wrappers:

```text
run_stage55_genogramma_pairaware.sh
run_stage56_genogramma_pairaware_probe.sh
run_stage57B2_native_matched_features.sh
run_stage57C_matched_pairaware_comparison.sh
run_stage58A_strand_shortcut_audit.sh
run_stage58B_lineage_bootstrap.sh
run_stage59A_genogramma_768d_ablation.sh
run_stage59B_ablation_lineage_bootstrap.sh
run_stage60_final_operon_evidence_freeze.sh
```

These are **internal stage launchers used by `run.py`**. They are intentionally minimal wrappers around the corresponding Python scripts under `pipeline/`.

For normal use, invoke `python run.py` rather than running these shell wrappers manually.

---

## 13. Formal Stage55-60 workflow

```text
Stage55   GenoGramma pair-aware representation extraction
Stage56   Formal GenoGramma downstream probe
Stage57B2 Native matched baseline pair-state extraction
Stage57C  Matched pair-aware baseline comparison
Stage58A  Strand-shortcut robustness audit
Stage58B  ANI95 lineage-level baseline bootstrap
Stage59A  GenoGramma 768D representation ablation
Stage59B  Ablation ANI95 lineage bootstrap
Stage60   Final operon evidence freeze
```

A valid frozen package should report all nine formal stages as `PASS`.

---

## 14. Integrity verification

The repository includes:

```text
MANIFEST.sha256
```

To verify distributed files from the repository root:

```bash
sha256sum -c MANIFEST.sha256
```

Canonical GenoGramma implementation:

```text
pipeline/revision_modules/genogramma.py
```

Canonical SHA256:

```text
7579820f2324f101fa7412285233f9ab1d80548f234269a5ad7ef47a76be2e01
```

Frozen formal Stage50 input SHA256:

```text
978e233098e4ccda12460e821d01fb07626f909d8afd4dafa2737c867081ab3e
```

---

## 15. Interpretation and limitations

1. **DOOR2 labels are computational annotations.**  
   They are not direct experimental measurements.

2. **The formal task is local.**  
   GenoGramma is evaluated as an order-sensitive local genomic neighborhood representation model.

3. **Lineage structure is explicitly controlled.**  
   The principal evaluation uses ANI95 lineage-blocked cross-validation rather than random example splitting.

4. **Strand is a strong predictive shortcut.**  
   A dedicated same-strand analysis is therefore included to assess residual predictive signal.

5. **The TransformerSmall comparison should not be overstated.**  
   GenoGramma has higher point estimates, but statistical superiority over TransformerSmall is not established by the lineage-level bootstrap analysis.

6. **The public fresh workflow begins from pretrained assets.**  
   Stage55-60 downstream reproducibility should not be described as complete retraining from raw genomes.

---

## 16. Data and provenance

The package contains project-generated raw data, processed data, intermediate assets, frozen outputs, and provenance records required to document or reproduce the distributed downstream workflow.

The `raw_data/` directory contains original data generated or owned by the project authors and is included as part of this public research package.

Some metadata files retain historical provenance describing their original computational origin. These records are retained for traceability and should not be interpreted as runtime path requirements.

The executable Stage55-60 workflow uses repository-relative paths.

---

## 17. Additional reproducibility information

See:

```text
REPRODUCIBILITy.md
```

for additional details about the packaged reproduction boundary and workflow.

<!-- FINAL_PUBLIC_INTERFACE -->

## Final public interfaces



- run.py: reproduce and audit the formal manuscript Stage55-60 workflow.

- train.py: train GenoGramma SSL encoders and downstream heads; the default route starts from raw genomes.

- test.py: smoke/audit plus compatible-window embedding and task-specific prediction.



Embedding input must contain family_rows and strand arrays with shape (N, 64) using the packaged 276,517-family vocabulary.



The bundled prediction heads are specific to DOOR2-derived adjacent-gene operon-status prediction. They are not universal zero-shot phenotype predictors.



For exact CUDA reproduction of formal Stage55 representations, use batch size 256.



ESM-2 is pretrained. GenoGramma SSL encoders and downstream heads are trained separately in the train.py workflow.
