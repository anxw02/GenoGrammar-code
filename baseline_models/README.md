# GenoGrammar matched baseline models

Formal baselines are intentionally lightweight and locally deployable.

Planned models
--------------
1. MeanPool
   - order-invariant sanity/content control
   - not treated as a strong competing model

2. CNN1D
   - standard local sequential encoder

3. BiGRU
   - standard recurrent sequential encoder

4. SmallTransformer
   - moderate-capacity positional sequence encoder

Fair-comparison rules
---------------------
- Same frozen 480-d family representations
- Same strand information
- Same 64-gene windows
- Same sample-wise five folds
- Same lineage-blocked evaluation when formally tested
- Same adjacency + neighbor IndependentSSL objectives as R1
- Same perturbation-independent checkpoint selection
- No shuffle, rank, or separation objective in the primary matched benchmark
- Target capacity close to GenoGrammar (~0.508 M trainable parameters)
- PyTorch-native implementation
- No external pretrained checkpoints required
