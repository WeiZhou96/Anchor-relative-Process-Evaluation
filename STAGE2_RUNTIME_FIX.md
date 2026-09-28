# Deterministic CUDA compatibility repair

> Release copy. Apart from this note the text is unchanged. The SHA-256 of the file as committed and hashed by the original run is listed in `docs/PROVENANCE.md`.

The first full training launch (code fac6677) stopped before completing its first fit: PyTorch 2.5.1 rejects CUDA cumsum under strict deterministic algorithms. Its output directory `outputs/stage2_training` and log are retained. No held-out result from this failed launch was used.

Replace prefix cumulative averaging by multiplication with a lower-triangular averaging matrix. This is the same causal mean operator, supported by the configured deterministic cuBLAS path. Tests now explicitly enable deterministic algorithms, in addition to setting CUBLAS_WORKSPACE_CONFIG, so the actual restriction is exercised. Rerun the unchanged 250-fit plan into `outputs/stage2_training_v2`. No data, hyperparameters, stopping rule or loss objective is selected using results.
