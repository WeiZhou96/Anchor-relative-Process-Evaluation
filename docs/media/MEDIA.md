# Media files

This folder holds the visuals shown in the repository's README. One command writes all of them:

    python figures/make_media.py            # GIF + MP4 (MP4 only when ffmpeg is on PATH)
    python figures/make_media.py --no-mp4   # GIF and PNG only

`APE_DATA` defaults to the repository's `data/` folder. To include the copies of the paper figures, run `figures/run_all.sh` first.

## Rules that apply to every file

- **Stored records only.** Every value, curve, tile, count and label is read from the records named below.
- **No dataset imagery.** No video frame, image or thumbnail from ACCIDENT or MM-AU is used.
- **Computed areas are checked.** Where a frame shows an area computed for display, the script recomputes it from the stored values. Where a stored metric exists, the script asserts agreement (tolerance 1e-6 s or tighter) and prints the comparison.
- **Animation reveals real values only.** It reveals stored values progressively, offset by offset or horizon by horizon on the stored grids. Nothing is interpolated, smoothed or extrapolated. Straight segments between grid points are drawing aids; the areas between them are the trapezoidal areas used by the metric.
- **Colours follow the paper.**
  - Teal-green: correct / stable correct.
  - Ochre: correct now but later wrong, and the retracted-correct area.
  - Grey: wrong, and the error area.
  - Slate blue-grey: structure.
  - System A: blue `#0072B2`. System B: vermillion `#D55E00`.

  The A and B colours are fixed hex values. Their area fills are opaque light tints of the same colours (16% on white).
- **Fonts differ by platform.** Text is drawn in Calibri on Windows and in DejaVu Sans elsewhere. Text widths therefore differ slightly between platforms; the data are the same.
- **Animation format.** Every animation loops and ends on a held final frame that reads as a complete static figure.

## Files

| File | Content |
|---|---|
| `ape_reverse_and.gif`, `ape_reverse_and.mp4` | Offline reverse AND on one stored ACCIDENT trajectory pair |
| `ape_tied_pair_rmscd.gif`, `ape_tied_pair_rmscd.mp4` | S_H(δ) and the RMSCD area of a window-end-tied pair at H = 10 s |
| `ape_eligibility_cohort.gif`, `ape_eligibility_cohort.mp4` | Eligibility cohort against the horizon; per-clip-end vs fixed-cohort accounting |
| `results_at_a_glance.png` | Three-panel static summary of the same records |
| `paper_fig3_accounting.png`, `paper_fig4_tied_pairs.png`, `paper_fig6_knobs.png`, `paper_figS1_comparability.png`, `paper_figS2_mmau.png` | Unchanged copies of the paper's quantitative figures (present only if they were generated first) |

### `ape_reverse_and.gif` / `.mp4`

**Shown.** The stored predictions of systems A and B on one ACCIDENT test clip, H = 10 s, Δ = 0.5 s:

- A is the prefix-mean ResNet-18 classifier.
- B is the GRU-512 with EMA smoothing; its answers are the recorded replay at the 0.5-s step.

The system labels and their running notes are drawn in the system colours.

The animation runs in three steps:

1. Answers appear offset by offset as class tiles (HO head-on, RE rear-end, TB t-bone, SS sideswipe, SI single). They are shaded green when correct now.
2. An offline reverse AND runs from the window end backwards. Tiles in the stable-correct suffix stay green and are underlined. Tiles that are correct now but followed by a later error turn ochre.
3. The area above the stable indicator fills as the offset advances. This area is the clip's contribution d_i to RMSCD@10. It splits into:
   - the area above the correctness indicator (e, grey);
   - the area between correctness and stable correctness (r, ochre).

**Reads.** `figures/qualitative/accident_cases.json`, which is a derived, non-image record:

- the case of the post hoc stratum `early_correct_late_error` (clip `-NgnSm_oEB4_00`);
- the stored `pred`, `correct`, `stable` and `delay` of both systems;
- the class code of the clip.

**Checks.**

- Correctness and the stable suffix are recomputed from the predictions and must equal the stored flags.
- d_i is recomputed with the trapezoidal rule and must equal the stored `delay`.
- e + r = d_i holds exactly.

**Selection.**

- The clip is the stored display case of that stratum, which the paper selected as the lexicographically first identifier of the stratum.
- It was chosen here because it contains both stable-correct and correct-now-later-wrong tiles.
- Systems A and B are the fixed pair of the paper's Figs. 4 and 5.

**Limits.**

- This is one clip and one pair. It illustrates the mechanism and does not estimate how often the pattern occurs.
- e and r are per-clip display quantities computed here; they are not stored metrics. The cohort-level E_H and R_H of any system are not shown.

### `ape_tied_pair_rmscd.gif` / `.mp4`

**Shown.**

- **Curves.** The stable-correct curves S_H(δ) of systems A and B on the ACCIDENT test eligibility cohort at H = 10 s. They are drawn grid point by grid point as solid lines with small markers. The first frame is δ = 0 (area 0.00 s).
- **Area.** The area above each curve fills as δ advances; at δ = H that area is RMSCD@10. The running value is printed in a fixed white box at the top left of each panel.
- **Bars.** A bar panel accumulates both areas side by side; at the end, a thin mark gives the stored RMSCD.
- **Context.** Grey lines are the S_H curves of all non-trivial systems at H = 10 s (A and B included). No subsampling is applied.
- **End-point tie.** The window-end macro-accuracies and their 95% cluster-bootstrap intervals stay on screen throughout. The pair is window-end-tied: the intervals overlap, which the script verifies before writing that statement.

**Reads.**

- `data/accident/outputs/r2b/metrics/8ac32aae418b/<system>.json`:
  - for A and B: `S_H`, `delta_s`, `RMSCD`, `N_H`, `end_window_macro_acc`, `bootstrap`;
  - for the grey curves: `S_H` and `family` of every per-system record. Families `block` and `trivial` are excluded, and files starting with `_` are skipped.
- The two system identifiers come from `figures/qualitative/accident_cases.json`.

**Checks.**

- The trapezoidal area above `S_H` must equal the stored `RMSCD`.
- The area above `S_H_macro` must equal `RMSCD_macro`.
- A and B must share the same `N_H`.

**Limits.**

- S_H is stored on the 0.5-s grid only.
- The paper's tied-pair statistics (Holm-corrected comparisons, Fig. 4 and Table 2) are not recomputed or displayed here.

### `ape_eligibility_cohort.gif` / `.mp4`

**Shown.**

- **Left panel.**
  - The horizon H steps from 0 to 21.5 s on the 0.5-s grid; 21.5 s is the largest registered horizon.
  - The ACCIDENT test clips are sorted by post-anchor length L⁺. A clip leaves the eligibility cohort as soon as H exceeds its L⁺.
  - The animation pauses at the registered horizons 4, 10 and 21.5 s, which stay marked with their N_H.
- **Right panel.** When the sweep reaches a registered horizon, the right panel shows that horizon's long-stratum minus short-stratum mean stable-correct delay under two kinds of accounting:
  - the fixed cohort (blue);
  - per-clip-end accounting (orange), using the colours of the paper's Fig. 3.

  Each panel row shows:

  - one tick per non-trivial system;
  - the random-answer blocks with 95% intervals;
  - a shaded band of ± the RMSCD ruler of that horizon.

  The "random:" labels give the range of the random-block gaps. They use a true minus sign, two decimals when a value is below 1 s in magnitude, and never a negative zero.

**Reads.**

- `data/accident/manifest/manifest_real.csv` (`split`, `post_anchor_length_s`).
- `data/accident/outputs/r2c/gates/g3_v5.json`, primary pool, rule `v5_22p0`:
  - per-system `gap_long_minus_short` with `ci_lo`/`ci_hi` for `fixed` and `own`;
  - `ruler`.
- `N_H` of system A's per-system record at each registered horizon.

**Checks.**

- At every registered horizon, the manifest count of test clips with L⁺ ≥ H must equal the stored `N_H`.
- At 10 s it must also equal the `cohort_n` of `accident_cases.json`.

**Limits.**

- Cohort sizes between the registered horizons are exact counts from the manifest, but they are not stored metrics.
- The strata are the registered tertiles of post-anchor length computed on development clips of the same cohort; they are not drawn.
- The oracle block shown in the paper's Fig. 3 is not drawn.
- Ticks of different systems at nearly equal values overlap.

### `results_at_a_glance.png`

**Shown.** Three static panels:

| Panel | Content |
|---|---|
| (a) | Fraction of ACCIDENT test clips with L⁺ ≥ H against the horizon, with N_H at the three registered horizons |
| (b) | Window-end macro-accuracy against RMSCD@10 for all non-trivial systems (grey); A and B with 95% intervals; dashed line at RMSCD = H. The legend is placed by matplotlib's least-overlap rule (`loc="best"`) |
| (c) | Long-minus-short gaps of the random-answer blocks at each registered horizon, fixed cohort vs per-clip end, with ± ruler bands |

**Reads.** The same records as the three animations, with the same checks.

**Limits.** It is a summary of the same quantities as the paper's Figs. 3 and 4, not a replacement for them. The curve in (a) is evaluated on a 0.1-s grid of H.

### `paper_*.png`

These are byte-for-byte copies of the 300-dpi PNG files that `figures/run_all.sh` writes to `$APE_FIG_OUT/figures/`. They are copied only when those files exist; otherwise the script reports them as skipped.

| Copy | Source file | Paper item |
|---|---|---|
| `paper_fig3_accounting.png` | `fig3_accounting.png` | Fig. 3 |
| `paper_fig4_tied_pairs.png` | `fig2_tied_pairs.png` | Fig. 4 |
| `paper_fig6_knobs.png` | `fig5_knobs.png` | Fig. 6 |
| `paper_figS1_comparability.png` | `fig4_comparability.png` | Fig. S1 |
| `paper_figS2_mmau.png` | `fig6_mmau.png` | Fig. S2 |

Their captions are those of the paper.

## Format

- **Frames.**
  - Frames are rendered 1100 px wide.
  - Pixels whose channels are all ≥ 248 (anti-aliasing next to the page) are set to pure white.
  - Blank rows above and below the content are trimmed to a 14-px margin; heights are even.
- **GIF.**
  - **Palette.** One fixed palette of at most 128 colours per file. It starts with pure white (#FFFFFF) and the exact colours of the colour semantics above; median-cut colours of five evenly spaced frames fill the rest.
  - **Mapping and check.** Every pixel is mapped to its exact nearest palette colour. After writing, the script reads the file back and checks that the page background is #FFFFFF.
  - **Timing.** Looping; at most 120 frames; frame durations are multiples of 100 ms.
  - **Size.** The script warns above 4 MB.
- **MP4.**
  - Encoding: H.264, yuv420p, 10 fps, even frame dimensions, from the same frames as the GIF.
  - Timing: each GIF frame is repeated for its duration.
  - Size: the script warns above 2 MB.
- **Static PNG.** Written at 200 dpi.
