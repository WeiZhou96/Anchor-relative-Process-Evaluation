# Provenance of the code and data

This file records where every part of this repository and of the derived-data package comes from, what was left
out and which edits were made for the public release. The original repositories were private working
repositories with a long commit history; the release is a fresh repository without that history.

## 1. Source versions

| Part | Source | Version |
|---|---|---|
| ACCIDENT audit (APE library, manifest builders, system library, audit stages K-a/K-b/K-c, audit tables, tests) | audit repository, branch `master` | `2835c86542934fd470d375692aa9929849d787af` (2026-09-09, tag `baseline/accident-before-method-20260927`) |
| Re-analyses (first re-analysis `run002`, stage-two calibration and training comparison, M3) | clone of the audit repository, branch `research/reanalysis-m3-20260928` | `a80cf23` (2026-09-28, tags `results/method-diagnostics-20260927`, `results/method-stage2-20260927`, `plan/m3-reanalysis-20260928`, `results/m3-reanalysis-20260928`). Relative to `2835c86` it only adds files (15 code/test files and the analysis plans at the root) |
| MM-AU interface (frame axis, task vocabularies, G0 floor, G6 draft, `data/mmau/`) | branch `r3/mmau-preflight` of the audit repository | `f99e5cab26f28246e6b6e037a80b854b11c030b9` (2026-09-20, tag `baseline/mmau-interface-before-method-20260927`). Relative to `2835c86` it adds 35 files and changes `ape/metrics.py` |
| MM-AU development study (`mmau/study/`) | MM-AU experiment repository | `55891de` (2026-09-27, tag `baseline/mmau-source-20260927`) |
| MM-AU APE evaluation scripts | run from the evaluation workspace | byte-identical to `mmau/study/pixel_pilot_20260920/evaluate_pilot.py` (SHA-256 `24d29f57…`), `mmau/study/anchor_shift_20260921/evaluate_anchor.py` (`ed642668…`) and `mmau/study/overlap_dense_20260920/protocol_sensitivity.py` (`ebf58eb1…`); used under the names `evaluate_native_pixel_pilot_20260920.py`, `evaluate_anchor_20260921.py`, `protocol_sensitivity_20260920.py` |
| Article and ESM figure/table scripts (`figures/`) | manuscript working directory, version of 2026-09-21, plus the authors' plotting style package | see `figures/README.md` |

**Merge.** The three code lines share the base `2835c86`. The re-analysis branch adds files only, and the MM-AU
branch adds files and changes one shared file, `ape/metrics.py`. The released `ape/metrics.py` is the MM-AU
version: it adds an optional `n_classes` argument (default: the five ACCIDENT classes), input checks on the
probability shape and formatting changes, and leaves five-class results unchanged. This was verified on the
release: the per-system audit records of K-b at the reference protocol (552 files), the audit tables of K-c
(178 files), `run002` (30 files) and the stage-two calibration outputs were regenerated with the released code
from the released data and are byte-identical or numerically identical to the originals; M3 agrees to
2.1e-13 (see the release verification record).

## 2. History of the ACCIDENT audit (summary of the internal run reports)

* **Data (S0).** Manifest v1 from the official split; manifest v2 moves clips so that no source video (cluster)
  crosses train/dev/test ("leak repair"; `data/build_manifest.py`, `--no-repair-leaks` gives v1). Real subset:
  2,027 clips, train 411 / dev 102 / test 1,514. The G0 note on the length shift between the official train and
  test splits is `report/text/g0_length_shift_stats.json`.
* **Freeze S1 v1 (2026-09-03).** Reference protocol `pi0` (step 0.5 s, horizons 4/10/21.5 s, grid to 21.5 s,
  rank-preservation floor 0.9, 1,000 source-cluster bootstrap replicates, seed 20260903), perturbation grid,
  plausible sub-grid, metric family, gauge blocks and system library.
* **v2 (2026-09-03).** A1 post-processing arms registered by rule, not by parameter value; A2 per-metric rulers
  in the metric's own units; A3 floating-point tolerance of the eligibility test (corrected N_H at H = 10 s);
  A4 coarse-step re-generation of `trivial__random`; A5 scan size, p-value method and resolution guard. No
  result under v1 entered the paper.
* **First round (R1).** 52 non-trivial and trivial systems on frozen ResNet-18 features plus 12 gauge blocks;
  first audit pass (`scripts/a_*`).
* **v3 (2026-09-08), second round (R2 S).** Additive registration of 120 new fine-step systems (CLIP-B/16 and
  ResNet-18 backbones, prefix-mean and GRU classifiers, post-processing arms, MSP/margin, Ringel and TEASER
  commitment rules) and 204 coarse-step re-generations (`scripts/b_r2_run.py`). Test predictions were generated
  only after selection on development data was sealed (`outputs/r2_S/selection_sealed.json`).
* **K-a.** First audit of the merged library under v3 (`scripts/k_all.py`; output `outputs/r2_Ka`, not part of
  the data package).
* **v4 (2026-09-09), K-b.** A4-1 corrected reading of gate G2 after an implementation error in the first-round
  calibration; A4-2 within-cohort (fixed-cohort) G3 rule next to the original rule; A4-3 registration of the
  read-only VLM applicant. K-b recomputed the primary pool of 184 answer sets (168 non-trivial, 4 trivial,
  12 gauge blocks) into `outputs/r2b` (`scripts/k_b_all.py`).
* **v5 (2026-09-09).** A5-1 visible-end cap of the per-clip-end metric in G3 corrected from 23.5 s to 22.0 s
  (the answer cache ends at 22.0 s; the 23.5-s cap produced a spurious oracle gap); A5-2 run-time discipline of
  the VLM run.
* **v6 (2026-09-09).** Frame-selector fix of the VLM applicant (duplicate index at a floating-point boundary).
* **VLM (V).** Qwen2.5-VL-7B-Instruct, read-only, one prefix per request, 166,448 requests on dev and test,
  10.3 h on one GPU (`systems/vlm_full.py`).
* **K-c.** Secondary pool of 185 answer sets (primary pool plus the VLM) and the v5 G3 rule; the 552 primary
  per-system records of K-b are reused byte for byte (`scripts/k_c_all.py`; output `outputs/r2c`;
  `scripts/k_c_verify.py`, 5,677 assertions; `scripts/k_c_report.py` writes `final_conclusions.json`).

The run reports were written in Chinese for the internal record and are not part of the release. Their
operational rules are implemented in `ape/r2.py`, `ape/r2b.py` and `ape/r2c.py`, and are summarised above.

## 3. History of the re-analyses

* `run001` (2026-09-27): first run of `scripts/method_experiments.py` under `EXPERIMENT_SCOPE_20260927.md`;
  it reproduced 81 historical RMSCD values exactly. Its simultaneous-interval approximation under-covered in the
  declared simulation (91.5 %), so `EXPERIMENT_AMENDMENT_20260927.md` changed only the critical value.
  `run001` is kept by the authors but is not in the data package.
* `run002` (2026-09-27): the run reported in the paper.
* Stage two (2026-09-27): `STAGE2_SCOPE_20260927.md`, `STAGE2_TRAINING_20260927.md`, `STAGE2_RUNTIME_FIX.md`,
  `STAGE2_CALIBRATION_AMENDMENT.md`, `STAGE2_FINAL_CALIBRATION.md`, `STAGE2_BOOTSTRAP_COUNT_CHECK.md`. The
  first training launch (`stage2_training`) stopped on a deterministic-CUDA error before its first fit
  (`STAGE2_RUNTIME_FIX.md`); `stage2_training_v2` is the complete run. `stage2_calibration` (original) and
  `stage2_calibration_v2` (with an unconditional micro bootstrap) are both kept; the data package contains v2.
* M3 (2026-09-28): `M3_SCOPE_20260928.md`; `DEVIATIONS.md` #1–#4. Two launches stopped at the reproduction
  checks (#1, #2) before any M3 result was interpreted; `m3_20260928` is the first complete run and
  `m3_20260928_r2` the rerun with the output-format corrections of #3, which agrees with the first run to
  2.1e-13 (`COMPARE_WITH_FIRST_RUN.json`).

## 4. History of the MM-AU development study

Preflight of the MM-AU release (2026-09-12); full decoding of 11,730 image sequences, content-overlap audit and
split constraints (2026-09-20); native nine-class development manifest (`data/mmau/native_development.py`,
2026-09-20); pixel pilot and controls (2026-09-20); dense overlap and output-reading sensitivity (2026-09-20);
anchor shift and input stride (2026-09-21). The MM-AU study is a development study on reused development data,
not a confirmatory test; G6 is drafted but not registered (`prereg/G6_draft_2026-09-20.yaml`).

## 5. What the release leaves out

Code and documents:

| Left out | Reason |
|---|---|
| Git history, commit messages | private working history; the release is a fresh repository |
| Root `README.md`, `CONTRACT*.md`, `REPORT_R2_*.md`, `ape/REPORT.md`, `data/REPORT.md`, `systems/REPORT.md`, `data/verification_r2.md`, `report/r2*_method.md`, `prereg/README.md` (Chinese) | internal task contracts and run reports; the provenance they carry is summarised in §2–§4 and the prereg README is replaced by an English one |
| `report/text/email_*.md`, `report/text/g0_length_shift.md` | e-mail drafts; Chinese note whose numbers are in `g0_length_shift_stats.json` |
| `scripts/k_write_report.py`, `scripts/k_b_write_report.py`, `systems/_write_report.py`, `systems/r2_report.py`, `data/make_report_r2_d.py` | write only the internal Chinese reports (`k_c_report.py` is kept because it also writes `final_conclusions.json`) |
| `scripts/c_08_net_probe.sh`, `c_09_pip_pypi_probe.sh`, `c_15_git_init.sh`, `c_16_git_cleanup.sh`, `c_21_git_s1v2.sh`, `c_23_git_plaus.sh` | network probes of the original server and repository housekeeping |
| `systems/vlm_pilot/download_via_relay.py` | download through a site-specific network relay; `download.py` downloads the same pinned revision directly |
| `data/mmau/mmau_ai_mapping.py`, `data/mmau/test_mmau_ai_mapping.py`, `data/mmau/mmau_frame_axis_smoke.py` | automatically generated proposal for mapping the 58 MM-AU classes to the five ACCIDENT types, and the smoke test that depends on it; the paper uses the native MM-AU classes instead. `mmau_candidates.py --proposal` is optional and runs without it |
| `data/mapping/` | superseded 58-to-5 mapping template |
| `data/verification_sources/` | logs of web retrievals made while checking the literature |
| MM-AU study `preflight/` directory | byte-identical to `data/mmau/` |
| Annotator workbooks of the MM-AU mapping (`*.xlsx`) | personal data of the annotators; not used by the paper |
| Internal working notes and hand-over material | internal |

Data (see also the data package README): raw ACCIDENT videos, MM-AU frames, cached frame tensors of the VLM run,
the three raw MM-AU QA frames and the frames of the qualitative figures are not redistributed; neither are
pretrained third-party weights. Optional extras that the authors can provide on request are listed in the data
package README (coarse-step answer matrices, frozen features, model checkpoints, VLM per-clip responses, `run001`,
`outputs/r2_Ka`).

## 6. Edits made for the release

* **Paths.** Absolute paths of the original servers were replaced by environment variables with defaults
  relative to the repository (`scripts/env.sh`; `APE_ROOT`, `APE_TMP`, `APE_ACCIDENT`, `APE_PY`, `HF_HOME`,
  `TORCH_HOME` in Python and shell scripts; `APE_MMAU`, `APE_MMAU_PROJECT` in
  `mmau/study/overlap_dense_20260920/select_raw_qa.py`). Scripts that forced `HF_HOME` to a server cache now
  default to it only when unset. The `Makefile` is parameterised. Behaviour with the same inputs is unchanged.
* **Recorded paths in data files.** The data package rewrites absolute paths inside JSON/CSV/YAML records to
  repository-relative paths or placeholders such as `<APE_TMP>`; every rewritten file is listed with its original
  and released SHA-256 in `REWRITTEN_FILES.tsv` of the package. Three records are read back by code
  (`outputs/r2_S/feature_backend.json`, `outputs/r2c/library.json`, `outputs/r2_S/all_answers_paths.*`); for the
  first, `scripts/stage2_train.py` now resolves a relative `features_dir` against `--source-root`.
* **Frozen records.** `protocol/pi0.yaml` (hash-checked by `protocol/pi0.frozen.sha256`) and the
  pre-registration files keep every registered value of the originals. In this edition the operator identifiers
  in the attribution fields (`frozen_by`, `drafted_by`) and in comments naming the operator were replaced by
  `authors`, one open-precondition note of the unregistered G6 draft was reworded without changing its meaning,
  and `protocol/pi0.frozen.sha256` and `protocol/pi0.frozen.json` were regenerated with `ape.cli freeze` for the
  edited `pi0.yaml`. The `pi_hash` values depend only on protocol values and are unchanged. Digests recorded in
  historical run and validation records (for example `data/validation_r2_d.json`) refer to the original files.
  The copies under `data/accident/` are identical to the files below. The records contain Chinese comments and
  one absolute cache path (`hf_home` in v3, unchanged).

  | File | SHA-256 of the original | SHA-256 in this edition |
  |---|---|---|
  | `prereg/S1_freeze_2026-09-03.yaml` | `8f5be89334ec2231036ed8e3364e3a3dd129111d1bd5434e33bc08bba1ec37a7` | `4ea6645fc5daa48616895bd1b231b7cba897ce05c5cfd99d5369e3712d04f7b0` |
  | `prereg/S1_freeze_2026-09-03_v2.yaml` | `48d5a88a98a626a56f18cbc51f76ffc979b3e31e5eaa88e13eef89d91bd32181` | `d51cb93b9d9ae8f60a6548519049b1f71468ba5bd2aa490d0e9fe34df64acb43` |
  | `prereg/S1_freeze_2026-09-09_v4.yaml` | `ceeae7e13090b3206f885bbea3adbfc9731abe39359ba5da6537591e80fb3035` | `bd06e668454ce737c5367d1a8ec3b9f7f1f18d0f4334b9cf66d5e8baa1656f57` |
  | `prereg/S1_freeze_2026-09-09_v5.yaml` | `3e9d32ea073e34fcbc411e64c8f4f5abb9a553d34ca3e5704654fb681a96bc80` | `84b316d80da0150ed9e33387f547b9f1f9aa715277cd3fc55104a292c0d2c137` |
  | `prereg/S1_freeze_2026-09-09_v6.yaml` | `21464f047769d23442cbe0fa28bca3851f3131340339fa64a2e3cf190a1091ea` | `9d1100074dfde7e844d295594efade85e1fe746dba7cbc0e70a045564bd0c5b0` |
  | `prereg/G6_draft_2026-09-20.yaml` | `0e8e7b7a53ba88f237ea08842e4e92e52f244b822be2f527228872c40955b006` | `05d4c875124d880b1aec50594fadf2ea401aadb6ba702d17b177f25ee87c2be2` |
  | `protocol/pi0.yaml` | `fa82a7112a74d9e422cda2671d0721522970fa9a23ea0406583919130e7d897b` | `ba30adfc18707ea52c9c54b9d8837225f4d7d80becbd4abedae53cfd9d27cdbf` |
  | `protocol/pi0.frozen.json` | `d91aaeb5aad2602b936266afad0384d7104752a42e805c89c1699620f2366b1e` | `c1f42a4babec29a5276cfb6c7416b2482115bda5a5e59b8054e67d1c18cf6502` |
  | `protocol/pi0.frozen.sha256` | `9443f530ee79a797ecff7d87c5d48885186caa2e3228cb411efceefca1150c85` | `d280d2a21cf09d1d9a12051f54a55e39c5dbc9d1fb414ad356eba4e1b2b0e544` |
* **Analysis plans at the root.** Internal process references (review sessions, internal names, local
  paths, a quoted instruction) were removed from `EXPERIMENT_SCOPE_20260927.md`, `DEVIATIONS.md`,
  `M3_SCOPE_20260928.md`, `STAGE2_SCOPE_20260927.md`, `STAGE2_TRAINING_20260927.md`,
  `STAGE2_CALIBRATION_AMENDMENT.md` and `STAGE2_FINAL_CALIBRATION.md`; each carries a note. The scientific
  content is unchanged. `method_experiments.py`, `m3_reanalysis.py` and `stage2_train.py` hash these files into
  their run records, so a rerun records a different plan hash than the original run. SHA-256 of the files as
  hashed by the original runs:

  | File | SHA-256 of the original |
  |---|---|
  | `EXPERIMENT_SCOPE_20260927.md` | `6f333b5d0ee7cb16ef49933f390b67a20410190f2c328ce4580d38e75d9328d4` |
  | `EXPERIMENT_AMENDMENT_20260927.md` | `31a5a25e3b4c5af4ea10bfe370e5766693c71996e1187eb3c366dd8ce3823ff5` |
  | `DEVIATIONS.md` | `5de6d604fa3b0c3ce8791c3d3d0a7958b52fbeeebd10eb3c4957c9fbf7ce6001` |
  | `M3_SCOPE_20260928.md` | `425be1b36468655329f264dbd9d8cc4c277817938aff75a3693abeeb4aaf29d2` |
  | `STAGE2_SCOPE_20260927.md` | `a5fcfd1479c5753d5ecaff5e153c0a657dbcf840933a80859b189fa31e6b96dc` |
  | `STAGE2_TRAINING_20260927.md` | `2872d859e61205a105938d26198cac4f7acf994b1e68d216dd064dceb3a397c2` |
  | `STAGE2_RUNTIME_FIX.md` | `fef7df3fedaf8ce2c19c624ac8aed14d4154f63c845823d043a7f073a16c0c25` |
  | `STAGE2_CALIBRATION_AMENDMENT.md` | `bb356d1dc836e1223fc9acb27a1495928c8a938f2aaddec32db2ef2a62500512` |
  | `STAGE2_FINAL_CALIBRATION.md` | `a70eac72cd0c004b12de11ad0f378c16b335036a46a5cf8110746605b718f759` |
  | `STAGE2_BOOTSTRAP_COUNT_CHECK.md` | `2ecdc84a5480ce6e9c09dec9f403f5e13635f8635d0f6988bece142ad2f198e1` |

* **Other non-functional edits.** A regression test module was renamed to
  `tests/test_mmau_regressions.py` (docstring adjusted); in `systems/vlm_full.py` the internal report text no longer
  quotes internal commit metadata or a server cache path; `systems/vlm_pilot/model_download_verification.json` no longer
  describes the site-specific download route (file digests unchanged); `data/mmau/README.md`,
  `systems/vlm_pilot/README.md` and `prereg/README.md` were rewritten or edited for public readers; a comment in
  `scripts/b_00_deps.sh` no longer names the server environment.
* **Added.** `README.md`, `REPLICATION.md`, `LICENSE` (draft), `environment.yml`, `requirements.txt`,
  `mmau/requirements-training.txt`, `mmau/README.md`, `scripts/env.sh`, `scripts/release/install_data.py`,
  `docs/PROVENANCE.md`, `.gitignore`, `figures/`.

## 7. Notes

* A stale pytest cache in the MM-AU working tree recorded
  `tests/test_g6.py::test_shared_systems_widen_the_interval_relative_to_independent_pairs` as last failed
  (2026-09-20 09:32 +08:00). No commit contains a test of that name: it existed only in an uncommitted state and
  was replaced before commit `7bc6c79` (09:49 the same day), which switched the family resampling to strictly
  positive Dirichlet weights and documents the opposite, known behaviour in
  `test_a_family_shared_by_every_pair_contributes_no_uncertainty` and
  `test_no_pair_is_destroyed_by_the_resampling`. All 31 tests of `tests/test_g6.py` pass on the release.
* The MM-AU development manifest files in the data package have Windows line endings (CRLF) because they passed
  through a Windows machine; regenerating them with `data/mmau/native_development.py` from the released ledgers
  gives the same content with LF line endings.
* Code comments and docstrings keep references to the internal task contracts (for example "CONTRACT 5.4", the
  answer-matrix format) and to the original work tracks (A: protocol core, B: system library, C: data; D, S, V, K:
  second-round data, systems, VLM and audit). The contracts are not distributed; the rules they refer to are
  implemented in the code and summarised in section 2. Some tables and reports written by the pipeline carry
  Chinese-language labels and notes (`report/*tables.py`, `data/stats.py`, `data/clusters.py`,
  `scripts/k_c_report.py`, `systems/vlm_full.py`).
* `tests/test_freeze_guard.py` checks the `frozen_by` field of `protocol/pi0.yaml` as recorded in this edition.
