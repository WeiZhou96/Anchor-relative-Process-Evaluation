# Media files

This folder holds the visuals shown in the repository's README. Two scripts write them:

    python figures/make_media.py            # stored-record media: GIF + MP4 (MP4 only when ffmpeg is on PATH)
    python figures/make_media.py --no-mp4   # GIF and PNG only
    APE_ACCIDENT=/path/to/ACCIDENT APE_MMAU=/path/to/MM-AU python figures/make_qual_media.py   # qualitative replays
    python figures/make_qual_media.py --no-mp4 --cases accident_2 mmau                         # subset, GIF only

`APE_DATA` defaults to the repository's `data/` folder. To include the copies of the paper figures, run `figures/run_all.sh` first.

`make_qual_media.py` needs local copies of the two datasets:

- **`APE_ACCIDENT`.** The ACCIDENT dataset root, which contains `real_videos/`.
- **`APE_MMAU`.** The MM-AU dataset root, which contains `extracted/CAP-DATA/`.
- **OpenCV.** It decodes the ACCIDENT videos. The published frames are reproduced bit for bit by `opencv-python-headless==4.11.0.86` (see `figures/extract_qual_frames.py`).

## Rules that apply to every file

- **Stored records only.** Every value, curve, tile, count and label is read from the records named below.
- **No dataset imagery, with one exception.**
  - **Exception.** The qualitative replays (`qual_*.gif`, `qual_*.mp4`) show scaled real frames of three ACCIDENT clips and three MM-AU clips. See [Qualitative replays on real footage](#qualitative-replays-on-real-footage) and its [licence note](#licence-of-the-frames).
  - **Everything else.** No other file uses a video frame, image or thumbnail from ACCIDENT or MM-AU.
- **Computed areas are checked.** Where a frame shows an area computed for display, the script recomputes it from the stored values. Where a stored metric exists, the script asserts agreement (tolerance 1e-6 s or tighter) and prints the comparison.
- **Animation reveals real values only.** It reveals stored values progressively, offset by offset or horizon by horizon on the stored grids. Nothing is interpolated, smoothed or extrapolated. Straight segments between grid points are drawing aids; the areas between them are the trapezoidal areas used by the metric.
- **Colours follow the paper.**
  - Teal-green: correct / stable correct.
  - Ochre: correct now but later wrong, and the retracted-correct area.
  - Grey: wrong, and the error area.
  - Slate blue-grey: structure.
  - System A: blue `#0072B2`. System B: vermillion `#D55E00`.

  The A and B colours are fixed hex values. Their area fills are opaque light tints of the same colours (16% on white).

  The qualitative replays use the colours of the paper's Figs. 5 and S3 (`figures/fig_qualitative_cases.py`):

  - **Tiles.** Correct: fill `#D3ECE2`, edge `#8CC2AE`. Wrong: white fill, edge `#CBD2D8`.
  - **Markers.** Filled marker: correct class. Hollow marker: wrong class.
  - **Stable-correct suffix.** It is drawn as underlines in the system or cadence colour; tiles outside it get a grey `#E7EAED` underline.
  - **Cadences.** q = 2 blue `#0072B2`, q = 4 vermillion `#D55E00`, q = 8 reddish purple `#CC79A7`.
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
| `qual_accident_1.gif`, `qual_accident_2.gif`, `qual_accident_3.gif` | Fig. 5 cases (a)–(c), one per file: real ACCIDENT footage with the stored A/B predictions (contains dataset frames) |
| `qual_accident_replay.mp4` | The three Fig. 5 cases in sequence (contains dataset frames) |
| `qual_mmau_1.gif`, `qual_mmau_2.gif`, `qual_mmau_3.gif` | Fig. S3 cases (a)–(c), one per file: real MM-AU frames with the stored traces per input cadence (contains dataset frames) |
| `qual_mmau_replay.mp4` | The three Fig. S3 cases in sequence (contains dataset frames) |
| `qual_media_record.json` | For the qualitative replays: source hashes, the decoded frame or image file shown at every display position, checks, read-outs, display steps and file sizes |

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

### Qualitative replays on real footage

Files:

- **ACCIDENT.** `qual_accident_1.gif` to `qual_accident_3.gif` and `qual_accident_replay.mp4`.
- **MM-AU.** `qual_mmau_1.gif` to `qual_mmau_3.gif` and `qual_mmau_replay.mp4`.
- **Record.** `qual_media_record.json`.

Script: `figures/make_qual_media.py`.

**Shown.** The six qualitative cases of the paper, in the paper's order, replayed on the real dataset frames and synchronised with the stored predictions.

- **Fig. 5 (ACCIDENT test clips).**
  - Systems: A, prefix-mean ResNet-18; B, GRU-512 with EMA smoothing, replayed at the 0.5-s step.
  - Settings: seed 20260903, H = 10 s, Δ = 0.5 s.
- **Fig. S3 (MM-AU development clips).**
  - Model: one frozen class-weighted GRU checkpoint, with input every q = 2, 4, 8 frames.
  - Settings: seed 20260920, output step 8 frames, H = 64 frames.
  - Tiles show native class IDs.

Each GIF holds one case. Each MP4 holds the three cases of its dataset in sequence.

| n | Panel | Clip | Stratum (clips in stratum) | Reference | Stored note |
|---|---|---|---|---|---|
| 1 | Fig. 5 (a) | `-RrDtLjWsT4_00` | `same_endpoint_different_onset` (99) | t-bone (TB) | B answers SS until 1 s, then TB; both end on TB. |
| 2 | Fig. 5 (b) | `-NgnSm_oEB4_00` | `early_correct_late_error` (57) | rear-end (RE) | B is right at 0 s and from 1.5 to 8 s, then ends on SI. |
| 3 | Fig. 5 (c) | `-PpBteU0p3Q_00` | `both_wrong_at_end` (524) | single (SI) | Both predict SS at every offset. |
| 1 | Fig. S3 (a) | `0d37451c` | `denser_input_loses` (6) | native class 12 | q=2 ends on 8; q=4 and q=8 return to 12 from 56 f. |
| 2 | Fig. S3 (b) | `1c08612f` | `denser_input_gains` (3) | native class 43 | Only q=2 switches to 43, at the last offset. |
| 3 | Fig. S3 (c) | `00786b67` | `all_cadences_wrong_at_end` (193) | native class 43 | All cadences change from 12 to 14 at 32 f. |

**Layout of every frame.**

- **Video panel (top left).** The real frame, letterboxed.
- **Info column (right).**
  - Case header, clip and reference class.
  - The clock "t − anchor = … s" (ACCIDENT) or "frame offset = …" (MM-AU).
  - The decoded frame index and timestamp, or the image file name, of the frame on screen.
  - During playback, the latest read of each system or cadence.
- **Prediction panel (bottom).** The p(reference class) trace and the tile lanes against the offset after the published anchor.
- **Bottom lines.** The legend, the key of the systems, and the burned-in source line:
  - "Frames: ACCIDENT dataset (Picek & Hanzl, 2026), Kaggle picekl/accident · predictions: stored outputs of this study", or
  - "Frames: MM-AU dataset (Fang et al., 2024), Hugging Face JeffreyChou/MM-AU · predictions: stored outputs of this study".

**Video panel.**

- **What is done to the frames.**
  - Frames are scaled only, by area averaging, into a black 512×288 px letterbox at their own aspect ratio. ACCIDENT 1584×1080 becomes 422×288, 1080×720 becomes 432×288 and 1280×720 becomes 512×288; MM-AU 1280×720 becomes 512×288.
  - There is no crop, no enhancement, no frame interpolation and no generated or retouched content.
  - The frame is pasted into the page pixel for pixel after the page is drawn. No text or graphic is drawn on the footage.
- **ACCIDENT.**
  - The video plays from 1 s before the anchor to anchor + 10 s at the native timestamps (real time).
  - The videos are decoded sequentially with OpenCV, with the frame-selection rule of the published figure: for each display time, the last decoded frame whose timestamp is at or before anchor + t is shown.
  - Display rates:
    - MP4: 10 display frames per second (every 0.1 s).
    - GIF: 4 per second (every 0.25 s). If a GIF would exceed 6 MB, it is rewritten at 2 per second (every 0.5 s); the rate used is listed in `qual_media_record.json`.
  - Native rates are 29.73, 14.43 and 14.98 fps. A decoded frame can therefore be skipped, or repeated over two display times.
- **MM-AU.**
  - The image files from anchor − 8 to anchor + 64 are shown in order (frame numbers are the 1-based file names).
  - MP4: every image file, 100 ms each.
  - GIF: every second image file, 200 ms each. If a GIF would exceed 6 MB, every fourth file is shown, 400 ms each.
  - The display pace of 10 image files per second is not the recording rate of the clips.

**Prediction panel, synchronised.**

- **Reads appear in time.** A read at offset δ uses only the prefix up to anchor + δ. Its tile and its point on the p(reference class) trace appear when the display reaches anchor + δ.
- **Between reads.** Nothing new is shown between reads; answers and probabilities are never interpolated. Tiles that are not yet read are dashed outlines, and a slate cursor marks the display time.
- **Reading cadence.**
  - ACCIDENT shows systems A and B, read every 0.5 s.
  - MM-AU shows the three input cadences q = 2, 4, 8. Every cadence is read every 8 frames, and its inputs are the frames 0, q, 2q, … up to the read offset.
- **Offline suffix.** The stable-correct suffix, the endpoint marker, "stable from" and d_i are computed offline, after the window. They are not shown during playback.

**Final frame.** Each case ends in two holds:

- a 1.5-s hold at the window end;
- a 6-s held final frame that matches the paper's panel:
  - the full trace and tiles;
  - one underline per tile (system or cadence colour if the tile is in the stable-correct suffix, grey otherwise);
  - the endpoint marker (filled: correct, hollow: wrong);
  - "stable from" and d_i;
  - the stored case note.

The video panel keeps the frame at anchor + 10 s, or anchor + 64 frames, which is the right-hand photo of the paper's panel. The final read-outs are:

| Case | System | End | Stable from | d_i |
|---|---|---|---|---|
| ACCIDENT 1 `-RrDtLjWsT4_00` | A | correct | 0 s | 0 s |
| | B | correct | 1.5 s | 1.25 s |
| ACCIDENT 2 `-NgnSm_oEB4_00` | A | correct | 2 s | 1.75 s |
| | B | wrong (SI) | none | 10 s |
| ACCIDENT 3 `-PpBteU0p3Q_00` | A | wrong (SS) | none | 10 s |
| | B | wrong (SS) | none | 10 s |
| MM-AU 1 `0d37451c` | q=2 | wrong (8) | none | 64 f |
| | q=4 | correct | 56 f | 52 f |
| | q=8 | correct | 56 f | 52 f |
| MM-AU 2 `1c08612f` | q=2 | correct | 64 f | 60 f |
| | q=4 | wrong (14) | none | 64 f |
| | q=8 | wrong (14) | none | 64 f |
| MM-AU 3 `00786b67` | q=2, 4, 8 | wrong (14) | none | 64 f |

**Reads.**

- `figures/qualitative/accident_cases.json` and `mmau_cases.json`: stored `pred`, `p_true`, `correct`, `stable`, `delay` (ACCIDENT) and `input_offsets` (MM-AU), together with the case metadata (anchor, class, path or image directory).
- `figures/qualitative/accident_videos.json`, `accident_frames.json`, `accident_frame_pixels.json` and `mmau_frames.json`: hashes of the sources and of the published frames.
- `$APE_ACCIDENT/real_videos/<video_id>.mp4` and `$APE_MMAU/<relative_image_directory>/000NNN.jpg`.
- Optional, with `--verify-sources`: the ACCIDENT answer-set CSVs (`$APE_QUAL_SOURCES`) and the MM-AU prediction arrays (`$APE_DATA`).

**Checks.** All of these run before any frame is rendered; any failure stops the script.

- **Case read-outs.**
  - The case notes are asserted against the stored arrays with `check_notes` of `fig_qualitative_cases.py`.
  - Correctness and the stable-correct suffix are recomputed from `pred` and the reference class, and must equal the stored flags.
  - d_i is recomputed with the trapezoidal rule. For ACCIDENT it must equal the stored `delay` (tolerance 1e-12 s).
  - Probabilities must lie inside the plotted range.
  - MM-AU input offsets must be the prefix grid of each cadence.
- **Videos.** The SHA-256 and byte size of the three ACCIDENT videos must equal `accident_videos.json`.
- **MM-AU frames.** The nine recorded MM-AU frames must equal `mmau_frames.json` (SHA-256). Each recorded path must equal `relative_image_directory/anchor + offset`.
- **Published ACCIDENT frames.** At the display times anchor + 0, + 3 and + 10 s, the decoded frame index, timestamp and RGB pixel hash must equal the frames of the published figure (`accident_frames.json`, `accident_frame_pixels.json`).
  - With a different OpenCV build the pixels can differ by a few grey levels. The script then stops unless `--allow-decoder-drift` is given, and the result is recorded.
- **Record.** The SHA-256 of every MM-AU image file shown, and the frame index and timestamp of every ACCIDENT frame shown, are written to `qual_media_record.json`.
- **GIF background.** Each GIF is read back and its page background is checked to be #FFFFFF.

**Selection.** Copied from the stored records:

- ACCIDENT (`accident_cases.json`): "Post hoc display: lexicographically first eligible clip per outcome stratum, fixed manuscript A/B systems and seed; no largest-effect/visual-quality selection. Delay-difference stratum requires >=1 s solely for legible contrast. Cases are illustrative, not prevalence estimates."
- MM-AU (`mmau_cases.json`): "Post hoc display: first identifier per outcome stratum in all 269 development clips, fixed weighted GRU checkpoint and first seed 20260920. Includes denser-input gain, loss and shared failure; not representative frequencies or evidence for preferred cadence."

**Limits.**

- **Illustrative cases.** These six clips are post hoc illustrations. They are not prevalence estimates, and they are not evidence that one system or cadence is preferable.
- **System B's answers.** The B answer set of ACCIDENT is the recorded replay at the 0.5-s step. Subsampling the 0.25-s EMA output instead would give different answers at some offsets (`fig_qualitative_cases.py` counts these differences under `--verify-sources`).
- **MM-AU development clips.** They are not a source-independent audit (`usage: development_only_not_source_independent_audit`).
- **Offline read-outs.** The stable-correct suffix, onset and d_i depend on the whole window. They are not available to a streaming system at the time of a read.
- **Frame selection.**
  - The display grid subsamples the video; the frames between display times are not shown.
  - When a display rate exceeds the native rate, the same decoded frame is shown at consecutive display times.
- **Format limits.** These are limits of the formats, not edits of the footage:
  - GIF frames are reduced to one 256-colour palette per file without dithering, so the footage shows colour banding.
  - The MP4 uses H.264 with 4:2:0 chroma subsampling.
- **Content.** The clips show real road collisions.

#### Licence of the frames

These files contain third-party frames:

- `qual_accident_1.gif` to `qual_accident_3.gif` and `qual_accident_replay.mp4`: frames of the ACCIDENT dataset (Picek & Hanzl, 2026), Kaggle `picekl/accident`. The dataset is listed on Kaggle under CC BY-NC-SA 4.0; clarification from the dataset authors is pending.
- `qual_mmau_1.gif` to `qual_mmau_3.gif` and `qual_mmau_replay.mp4`: frames of the MM-AU dataset (Fang et al., 2024), Hugging Face `JeffreyChou/MM-AU`, under CC BY-NC 4.0.

They are shown for non-commercial research illustration, with attribution. The attribution line is burned into every frame. These files are not covered by the repository's code licence. The original videos and image files are not redistributed.

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
- **Qualitative replays (`qual_*`).** These settings replace the ones above for these files only.
  - **Frames.** 960 px wide; 650 px high (ACCIDENT) or 686 px high (MM-AU), with a fixed layout and no trimming. The near-white snapping applies to the page only; the video panel is pasted afterwards, unchanged apart from scaling.
  - **GIF palette.** One palette of at most 256 colours per file. It starts with pure white and the exact colours of the paper's tile, marker and system semantics. Median-cut colours of the page (24) and of the video panel of eight evenly spaced frames fill the rest.
  - **GIF mapping and check.** Exact nearest-colour mapping, no dithering. The background is read back and checked to be #FFFFFF.
  - **GIF timing.** ACCIDENT frames last 250 ms (500 ms in the fallback); MM-AU frames last 200 ms (400 ms). The window-end hold is 1.5 s and the final hold is 6 s.
  - **GIF size.** Target at most 6 MB per GIF, with the deterministic fallback described above.
  - **MP4.** H.264, yuv420p, 10 fps, CRF 24, streamed from the rendered frames. Target at most 10 MB per file; the script warns above it.
  - **Determinism.** Output is deterministic for a given set of library versions. The display step actually used, the frame counts and the sizes are written to `qual_media_record.json`.
