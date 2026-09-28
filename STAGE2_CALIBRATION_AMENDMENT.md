# Conditional versus unconditional micro bootstrap diagnostic

> Release copy. Internal process references (review sessions, internal names, local paths) were removed for publication; the analysis plan itself is unchanged. The SHA-256 of the file as committed and hashed by the original run is listed in `docs/PROVENANCE.md`.

After a code review identified that the shared class-preserving resampling also conditions the micro bootstrap, retain the original `outputs/stage2_calibration` and run the same finite populations and outer seeds in a new `outputs/stage2_calibration_v2`. Add an unconditional source-cluster micro bootstrap alongside the original conditioned micro and macro. Do not replace the original output or change true population values. Report redraw counts and counts of coordinates supported by fewer than 20 nonzero-contributing source clusters. This diagnostic does not tune confidence levels or critical values.

The whole 250-fit training recipe was fixed before training and is unaffected. The 648-coordinate empirical-population calibration is retained unchanged and uses the earlier shared bootstrap; its small-sample undercoverage is not corrected by relabeling it. No general coverage theorem is claimed.
