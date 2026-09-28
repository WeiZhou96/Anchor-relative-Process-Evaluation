# Final calibration checkpoint after the result review

> Release copy. Internal process references (review sessions, internal names, local paths) were removed for publication; the analysis plan itself is unchanged. The SHA-256 of the file as committed and hashed by the original run is listed in `docs/PROVENANCE.md`.

The result review concluded that the suffix objective has no demonstrated benefit in the fixed-budget pilot; no training settings are changed and no rescue training is scheduled. The review also identified the remaining mismatch between the calibrated 648-coordinate H10 family and the previously reported 972-coordinate protocol comparison.

Before computing this extension, fix four empirical-population conditions: H10 and H4 decomposition families (108 pairs × 6 contrasts), and H10 and H4 common-cohort protocol families (108 pairs × 9 protocols). Sample the full observed source-cluster count for each condition; 1000 outer trials and 500 inner draws per trial, seed 20260931 with fixed offsets by condition. Report micro and macro, all-coordinate coverage with Monte Carlo uncertainty, wrong-direction frequency and interval width. Retain the previous n=300 failure results. Do not retune the statistical critical values.

Also report the previously specified information diagnostic: zero variance or fewer than 20 nonzero-contributing sampled source units for a coordinate marks that coordinate uninformative, represented by its known support [-H,H]. This is a disclosed conservative information rule, not a guarantee for the remaining coordinates. Full-coordinate coverage before and after this diagnostic are both retained. Include the actual-population support count for each coordinate. Any favorable empirical calibration remains conditional on these empirical populations and is not proof of independent source identification or distribution-free inference.

No manuscript changes are made. Existing run002 numbers remain immutable; any future manuscript use must identify the calibrated family and exact inference rule.
