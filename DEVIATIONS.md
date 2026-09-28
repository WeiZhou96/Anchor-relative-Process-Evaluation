# M3 deviations log

> Release copy. Internal process references (review sessions, internal names, local paths) were removed for publication; the analysis plan itself is unchanged. The SHA-256 of the file as committed and hashed by the original run is listed in `docs/PROVENANCE.md`.

## 1. 2026-09-28: tie tolerance in the "delay-dominant" comparison (before any M3 result was interpreted)

**What happened.** The first launch (commit `59c36da`, tag `plan/m3-reanalysis-20260928`) stopped at the reproduction check of section A, as the scope requires. At H = 4 s it counted 359 delay-dominant heterogeneous pairs; the paper's Table 2 reports 358. All other Table 2 columns at all three horizons had matched up to that point.

**Cause.** The mismatch comes from exactly one heterogeneous pair at H = 4 s:

- `r2__r18__gru512__msp0p7__seed20260904`
- `r2__r18__gru512__msp0p8__seed20260904`

Its RMSCD difference is −0.0662651 s. The endpoint and delay components are each −0.0331325 s, and their absolute values differ by 2.8e−17. That is an exact tie up to floating-point rounding.

The paper's figure was computed from a CSV of per-system terms; this run computes directly from the JSON records, so rounding placed the tie on different sides of a strict ">". No other heterogeneous pair at any horizon has | |delay| − |endpoint| | < 1e-6.

**Change.** "Dominant" is now evaluated as |delay| > |endpoint| + 1e-9. This applies the scope's general rule that absolute differences below 1e-9 are ties; no definition in the scope changes.

**Diff** (`scripts/m3_reanalysis.py`, function `audit_tied.summary`):

```diff
-                "delay_dominant": int((frame.dly.abs() > frame.lvl.abs()).sum()),
+                "delay_dominant": int((frame.dly.abs() > frame.lvl.abs() + 1e-9).sum()),
```

A diagnostic count of components with an absolute value below 1e-9 is added to the section A output, for transparency.

**Output of the stopped launch.** The partial output (`run_state.json` only; no M3 quantity was written) and its log are kept as `outputs/m3_20260928_attempt1_stopped/` and `outputs/m3_20260928_attempt1_stopped.log`.

## 2. 2026-09-28: identifier format in the "first-round prefix-mean" check (before any M3 result was interpreted)

**What happened.** The second launch (commit `60434c4`) reproduced every Table 2 column at all horizons and seven of the eight H = 10 s text figures. It then stopped at the last one: "14 of the 19 noncommitment replicated groups involve the first-round prefix-mean classifier".

**Cause.** The check was a bug in this code, not a data mismatch. The frozen seed groups name variants in `group_key` format (for example `R1|r18|mean|mean_linear|None`), but the check searched for the system-identifier fragment `clip__r18mean`, which never occurs in that format. Listing the 19 frozen groups shows exactly 14 containing `R1|r18|mean|`, as the paper states. The broader count, which also covers the second-round prefix-mean classifier, uses `|mean|`.

**Change** (`scripts/m3_reanalysis.py`): the constants `FIRST_ROUND_MEAN = "R1|r18|mean|"` and `ANY_MEAN = "|mean|"` replace the literal fragments in the check and in the two output counts. No definition in the scope changes.

**Output of the stopped launch.** It wrote only `run_state.json` and the section A files, which were never opened for interpretation. It is kept as `outputs/m3_20260928_attempt2_stopped/` with its log.

## 3. 2026-09-28: output-format corrections after the results were reviewed (no analysis change)

**What happened.** A review of the completed run (commit `d2740d0`, output `outputs/m3_20260928/`) asked for the following. None of them changes an analysis definition, a selection rule or a tie rule.

**Corrections.**

1. Section headings #1 and #2 now read "before any M3 result was interpreted" instead of "before any M3 output was produced". The second stopped launch had already written the section A files before it stopped, although nobody opened them.
2. `C3_dev_selection.csv` stored interval triples as list cells, which numpy 2 prints as `np.float64(...)`. They are now written as separate numeric `*_lo` / `*_hi` columns.
3. `SUMMARY.json` / `A_tied_noncommit.json` report the tied count after excluding zero-difference pairs, as the scope specifies. The raw counts, which include those pairs and match Table 2, are now added:
   - zero-difference pairs: 77, 6 and 7 at H = 4, 10 and 21.5 s;
   - raw tied counts: 5427, 7815 and 9021.
4. `C1_test_selection.csv` gains `n_accuracy_ties_at_max`. In set A at H = 10 s, the Ringel adaptation of `r2__clipb16__gru128__seed20260903` has exactly the same window-end correctness as its base classifier, because it commits at the forced 10-s point with the base classifier's label. The locked identifier rule therefore selects the Ringel adaptation. The 3.44-s regret in that row is a consequence of the tie and is reported only with this note.
5. C2 now also writes `C2_split_records.csv`, which holds every evaluation (split, direction, selected identifiers and metrics), so that its summaries can be reconstructed.

**Verification.** The corrected code is rerun with the same seeds into `outputs/m3_20260928_r2/`. Every numeric value present in both runs must agree to 1e-12; the comparison script and its result are kept with the outputs. The first run is kept unchanged.

## 4. 2026-09-28: post hoc per-class reading (after the results were reviewed; no new computation)

**What happened.** While the results were being written into the paper, one summary that the scope does not list was read from `outputs/m3_20260928_r2/D_per_class_configurations.csv` at H = 10 s.
- For each of the three round × backbone configurations and each class, the retracted-correct area of the prefix-mean classifier was compared with those of the GRU-128 and the GRU-512 of the same configuration (30 comparisons).
- The prefix-mean value was lower in 29 of them. The exception is head-on clips in round 2 with ResNet-18: 0.301 s for the prefix-mean classifier against 0.272 s for the GRU-512.

**Status.** No code was run and no output was changed. The reading is reported only as post hoc and supports no primary statement.
