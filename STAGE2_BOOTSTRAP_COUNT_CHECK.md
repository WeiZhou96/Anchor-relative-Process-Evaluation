# Resolve inner-bootstrap-count mismatch before final interpretation

> Release copy. Apart from this note the text is unchanged. The SHA-256 of the file as committed and hashed by the original run is listed in `docs/PROVENANCE.md`.

The 1000-trial empirical-population extension uses 500 inner replicates for runtime, while run002 used 2000. Its first H10 decomposition result is 93.0% macro and 95.0% micro simultaneous coverage; this flags concern but is not an exact replication of the operational resampling count.

Run one additional H10 decomposition condition at the operational 2000 inner replicates and 1000 outer trials, using the same empirical population, estimand, critical-value rule and seed 20260931. A different inner count advances the single RNG differently, so outer samples are not paired with the 500-replicate run; do not attribute any difference causally to replicate count. Save all results separately in `outputs/stage2_exact_bootstrap` and do not tune thresholds or critical values. This is the final count-matched check for this stage, irrespective of whether it reaches 95%. Other 500-replicate conditions remain explicitly identified as such.
