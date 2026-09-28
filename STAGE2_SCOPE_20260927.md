# Development experiments, stage 2

> Release copy. Internal process references (review sessions, internal names, local paths) were removed for publication; the analysis plan itself is unchanged. The SHA-256 of the file as committed and hashed by the original run is listed in `docs/PROVENANCE.md`.

This stage continues the recommended next step. Its design, implementation and results were reviewed at fixed checkpoints. Old baseline and run002 are preserved. This file extends the earlier scope to new stage-2 scripts and tests; no historical frozen protocol is overwritten.

## Calibration stress tests

Use explicit finite distributions of three paired binary prediction trajectories on 21 timestamps spanning 0 to 10 seconds. Enumerate class and latent trajectory states to obtain exact population values of all six contrasts for all three pairs. Each independent source cluster contributes two identical clips, deliberately testing correct cluster weighting without claiming realistic within-cluster heterogeneity.

Four conditions: (1) 300 clusters, class probabilities [0.06,0.16,0.35,0.11,0.32]; (2) 60 clusters, same probabilities; (3) 120 clusters, probabilities [0.01,0.04,0.15,0.30,0.50]; (4) 300 clusters, first probabilities, with an event of probability 0.002 causing a late or terminal error. The first three use eight finite trajectory templates and class-dependent joint mappings; the last directly challenges empirical zero-variance inference.

Run 500 independent datasets per condition with 600 shared cluster bootstrap draws per dataset, seed 20260929. Report macro and micro simultaneous coverage over 18 coordinates, unestimable samples (missing classes or insufficient class support), Monte Carlo uncertainty, zero-variance coordinates and interval widths. Missing classes must not silently change the five-class macro target. Conditional coverage among estimable samples and the frequency of unestimable samples must both be reported. Include the original guarded interval and a diagnostic variant replacing zero-variance intervals by the known contrast support [-10,10]. Neither construction is asserted to provide distribution-free coverage. Do not tune critical values until simulations look favorable.

## Training feasibility and design checkpoint

Use only existing train/dev rows and frozen CLIP-B/16 features, with no test predictions loaded. Preflight found 188 train clips (187 existing source clusters) and 52 dev clips (50 clusters) with full 10-second support; class counts are train [8,41,17,22,100], dev [5,13,5,4,25]. No train/dev cluster overlap under existing identifiers. Feature timestamps precede readout targets, with a maximum observed staleness of approximately 0.333 seconds. Existing CLIP pretraining data overlap is unknown.

The prior R2 mean and GRU training recipes differ in supervision, temporal support and optimization budget; their difference is not an isolated architecture effect. The new pilot must use a common causal input, normalization fit on train only, the same supervised 41 readout targets, class-balanced per-clip loss, identical batch orders and fixed training budget. Development evidence is exploratory; this repeatedly used dev set is not a new independent test. Pending the design review, the intended minimum is GRU with ordinary CE, a time-weighted CE control and suffix-max CE, plus fair prefix-mean and causal temporal baselines trained with CE. Exact architecture, optimization, epoch and seed settings will be committed in a training amendment before fitting models. No architecture is called empirically stronger before measurements exist.

No paper revision, new official test, MM-AU expansion or unbounded hyperparameter search is part of this stage. Save all seeds and outcomes, including negative results. Stop a model only for explicit numerical or resource failure and record it.
