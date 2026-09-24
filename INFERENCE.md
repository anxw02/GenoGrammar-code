# GenoGrammar Inference

`test.py` is the public pretrained-model interface.

## Audit packaged model assets

```bash
python test.py --audit-only
```

The package contains five formal lineage-controlled GenoGrammar pretrained checkpoints.

## Intended custom-data modes

Extract GenoGrammar representations:

```bash
python test.py --mode embed --input <input>
```

Use a compatible downstream task head:

```bash
python test.py --mode predict --input <input>
```

A pretrained GenoGrammar encoder does not automatically provide zero-shot predictions for arbitrary biological phenotypes. A compatible downstream task head is required.

Raw-genome custom-data preprocessing will be enabled together with the verified ESM2/protein-family preprocessing route.

<!-- FINAL_INFERENCE_CONTRACT -->

## Final inference contract



Compatible NPZ input:



    family_rows : integer array, shape (N,64)

    strand      : integer array, shape (N,64), values 0/1

    row_index   : optional array, shape (N,)



Extract the 768D GenoGrammar representation:



    python test.py --mode embed --input user_windows.npz --output user_768D.npy --fold 1 --batch-size 256



Run the bundled formal DOOR2-derived operon-status predictor:



    python test.py --mode predict --input user_windows.npz --output predictions.csv --fold 1 --batch-size 256



Prediction uses the matching GenoGrammar fold, the checkpoint zscore_mean and zscore_sd, and the formal 768 -> 128 -> GELU -> Dropout -> 2 head.



The bundled head predicts the DOOR2-derived adjacent-gene operon-status task only. DOOR2 labels are computational annotations, not experimental ground truth.



Direct raw FASTA inference is not exposed through test.py; input windows must already be mapped to the compatible frozen family vocabulary.



For exact formal Stage55 CUDA reproduction, use batch size 256. Different batch geometries may introduce very small floating-point rounding differences.
