# Training amendment after the design checkpoint, before model fitting

> Release copy. Internal process references (review sessions, internal names, local paths) were removed for publication; the analysis plan itself is unchanged. The SHA-256 of the file as committed and hashed by the original run is listed in `docs/PROVENANCE.md`.

The design review completed successfully. It was collaborative oversight, not a blinded review. This amendment supersedes the tentative full-H10 train/dev restriction in STAGE2_SCOPE.

## Fixed development matrix

- Pool original train+dev (513 clips), never original test, for five-fold stratified source-group cross-fitting (split seed 20260929). All clips train through min(observed post-anchor duration,10 seconds); readout times are anchor plus multiples of 0.25 seconds within actual support. Report out-of-fold H10 primary and H4 secondary on eligible rows. These are reused development data, not independent test results. Cross-fitting evaluates a training recipe with overlapping training folds; a bootstrap over fixed OOF predictions does not capture retraining uncertainty and is descriptive only.
- Common input begins at anchor minus 4 seconds or clip start, whichever is later, rounded backward to the latest available cached feature. No future interpolation. Stop at anchor+10 seconds or available duration. Record actual feature timestamps and staleness. Freeze CLIP-B/16 features, whose original pretraining overlap is unknown.
- Fit feature mean/std and inverse-frequency class weights separately inside each training fold. Use all training input frames for normalization, identically across arms. No augmentation.
- Five trained arms: prefix-mean linear CE; GRU128 CE; GRU128 increasing-time CE; GRU128 suffix-max CE; one-layer causal Transformer CE (128-dimensional projection, 4 heads, FFN 256, sinusoidal positions, no dropout). The Transformer is a temporal comparator, not presumed stronger. GRU objective arms share exact initialization and batch orders per seed/fold/LR. Different architectures cannot share identical parameters.
- Ordinary CE uses normalized trapezoidal readout weights per clip; the increasing-time control multiplies those weights by readout index+1 and renormalizes. Suffix-max uses the same base trapezoidal weights on the reverse cumulative maximum of per-readout CE. All losses apply the same class weight after per-clip aggregation. No log(2) scaling in optimization; any theoretical upper-bound interpretation requires that explicit scaling separately.
- AdamW, weight decay 1e-4, learning rates [3e-4,1e-3], batch 64, 60 fixed epochs, gradient norm cap 5, float32. Both learning rates are reported, not selected. Five seeds [20260930,20260931,20260932,20260933,20260934]. Each integer is an RNG seed, not a calendar date. Final epoch is primary; log training loss each epoch and held-out-fold metrics every 10 epochs. No early stopping or best-epoch selection.
- 5 folds × 5 seeds × 2 learning rates × 5 arms = 250 fits. Sequential jobs use GPU 0 only after checking availability. Checkpoints stay outside Git. Held-out predictions and histories are kept; no test predictions are generated.
- Add fixed EMA(alpha=0.5) of GRU-CE softmax vectors on the complete input feature stream, initialized with its first vector, then read on the common target grid. EMA is causal, carries no abstention/commitment mechanism, and is not tuned. This is a sixth evaluation recipe, with no extra fits.

## Readout and decision rules

Primary predeclared comparisons: suffix-max versus ordinary CE and versus increasing-time CE. Also report suffix-max versus fixed EMA, prefix mean and Transformer. Report endpoint macro accuracy, ordinary error area, retracted-correct area and stable delay, paired by seed/LR and common OOF cohort; include all five seeds and both LRs. Do not infer endpoint equivalence/noninferiority from close point estimates. Any terminal-accuracy cost is reported alongside process gains.

Do not pursue a new-loss paper claim if improvements fail to repeat across seeds/LRs, disappear against the increasing-time or fixed EMA control, or are dominated by the simple mean baseline. Do not declare success merely because bootstrap intervals on fixed OOF predictions exclude zero. No post-result expansion of the hyperparameter search in this stage.

## Verification gates

Before fitting, assert development/test ID and source-group separation, all five classes in training folds, finite features, causal readout indices and per-fold scaling provenance. Test each architecture by changing/deleting future input and comparing past outputs; test all losses ignore padded targets and permit finite gradients. Record GRU initialization hashes and first-batch IDs. Save code SHA, plan hashes, feature hashes, actual runtime/device, complete fit index and exceptions. Preserve previous reports and inputs.

## Statistical scope refinement

### Implementation review amendment (before full training)

The implementation review requests a broader EMA control, actual CUDA-path causality tests, train-only convergence diagnostics and explicit practical criteria. The two-epoch smoke run was used only for execution checks; held-out metric values were not inspected or used to select settings. This amendment is recorded before the full 250-fit run.

- Replace the single EMA by fixed alpha values [0.5,0.25,0.1,0.05], all reported with their endpoint/process tradeoff. There are now nine evaluated recipes and still 250 fits. Do not choose one alpha after viewing outcomes.
- A candidate effect repeats if at least four of five paired seeds agree at each of both LRs. Report practical thresholds 0.125 and 0.25 seconds for the mean stable-delay gain separately. The comparator dominates on endpoint/delay when its mean endpoint accuracy is no lower and its mean delay no higher. Do not infer statistical equivalence or a scientific success merely from these development rules.
- Report all class counts, micro and macro, and input-start truncation strata. A macro/micro reversal is a tradeoff between estimands, not grounds to erase either result or claim a universal gain.
- Fixed budget stays 60 epochs, with no post-outcome extension. Compare mean training loss over epochs 41–50 and 51–60: relative absolute change <=1% is a plateau heuristic, not a convergence proof. Otherwise label convergence unresolved and do not interpret absence of gains as a definitive loss-function failure. Report gradient clipping frequency/max norm.
- Require a clean Git tree, repository-absolute plan hashes, deterministic Torch algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8; run actual CUDA/inference-mode/512-dimensional/padded-batch causal tests. Independently rerun the first fold/first seed/first LR GRU-CE fit to check numerical reproducibility.
- New coordinate-level inference, if used, flags zero variance or fewer than 20 source clusters with nonzero paired contributions as insufficient information; this threshold is a conservative diagnostic, not a proven coverage guarantee. OOF comparisons remain descriptive without inferential confidence intervals.

### Coverage scope

The initial finite-template calibration is retained as an exact, deliberately restricted test, including rare unseen errors and zero empirical variance. It does not represent the full 108-pair family. A second empirical-population calibration will use the actual 27-model H10 joint trajectories and all 108 pairs × 6 contrasts, reporting finite-population coverage rather than new external evidence. Full Markov and heterogeneous cluster generators are deferred, not claimed completed. Zero-variance coordinates in new inferential readouts must be marked uninformative; a zero-radius bootstrap interval is not evidence of an exact population effect.

The empirical-population calibration uses 200 outer trials at each of 1109 and 300 source draws, 500 inner bootstrap draws, seed 20260930. Each sampled source copy gets its own cluster identity; clips within the source stay together. Population targets are the empirical clip-weighted and class-balanced means. Report simultaneous coverage of all 648 coordinates, Monte Carlo error, strict opposite-direction claims and mean interval width. This checks the decomposition family only; the 972-coordinate protocol family remains uncalibrated by this new exercise.
