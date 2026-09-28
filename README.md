> **Private pre-release snapshot.** The code and data licences are drafts awaiting author confirmation. This repository contains only key derived data; the complete derived-data package will be provided with a DOI archive (DOI: [TO BE ASSIGNED]). See [data/README.md](data/README.md) for this subset's scope, checksums and limitations. The original replication instructions below describe the larger release package; `install_data.py install`, full audit recomputation and training are not supported by this subset alone.

# Anchor-relative process evaluation (APE): code for "Stable correctness after impact"

This repository contains the code for the paper

> *Stable correctness after impact: Evaluating collision-type recognition beyond final accuracy.*

It implements anchor-relative process evaluation (APE) of streaming collision-type classifiers and every
computation behind the paper:

* the frozen audit on the ACCIDENT real-video subset (manifest and cohorts, 185 answer sets, gauge blocks,
  characterization of the metric family, gates G2 to G5, window-end-tied pairs, the VLM secondary pool);
* the re-analyses of the stored answers (E/R components, paired decomposition, sparse-output bounds, reading
  protocols, coverage calibration, conditional delay, selection and per-class components);
* the training-objective comparison on the development data;
* the MM-AU development study (frame-axis adapter, native nine-class task, pixel and control pilots, anchor shift,
  input stride, output-reading sensitivity);
* the scripts that draw the figures and tables of the article and its electronic supplementary material (ESM).

Derived data (manifests, stored answers of all systems, audit and re-analysis outputs, MM-AU predictions and
evaluations) are distributed separately as the **APE derived-data package** (see [Data](#data)). Raw videos,
frames and pretrained weights are not redistributed.

**Licence (draft, to be confirmed by the authors).** The code is released under the Apache License 2.0 (`LICENSE`).
The derived-data package carries its own licence (`LICENSE-DATA` in the package; draft CC BY-NC-SA 4.0).
`REPLICATION.md` is the replication explanatory file required by the journal.

## Repository layout

| Path | Content |
|---|---|
| `ape/` | APE library: protocol vectors and cohorts (`protocol.py`, `cohort.py`), scoring (`metrics.py`), calibration and rank preservation (`calib.py`), gauge blocks (`blocks.py`), statistics (`stats.py`), CLI (`cli.py`); audit stages K-a/K-b/K-c (`r2.py`, `r2b.py`, `r2c.py`); re-analyses (`method_analysis.py`, `m3_analysis.py`); MM-AU frame axis (`frame_axis.py`, `vocabulary.py`, `g0_floor.py`, `g6.py`, `g6_pipeline.py`) |
| `protocol/` | Frozen reference protocol `pi0.yaml`, its digest `pi0.frozen.json` / `pi0.frozen.sha256` |
| `prereg/` | Frozen pre-registration files S1 v1–v6, template, and the (unregistered) MM-AU G6 draft |
| `data/` | Manifest builders (`build_manifest.py`, `clusters.py`, `stats.py`, `g0_length_shift_r2.py`, ...). `data/manifest/` is filled by the data package. `data/mmau/`: MM-AU preflight, decoding, overlap audit, split constraints and native development manifest |
| `systems/`, `configs/systems/` | System library: frozen feature extractors, training, inference, post-processing arms, commitment rules, trivial systems, the read-only VLM (`vlm_full.py`) and its pilot (`vlm_pilot/`), stage-two learning recipes (`stage2_learning.py`) |
| `scripts/` | Run-order entry points: `b_*` system library (first round), `b_r2_run.py` (second round), `a_*` first audit pass, `c_*` data track, `k_*` audit stages K-a/K-b/K-c, `method_experiments.py` (first re-analysis), `m3_*.py` (M3 re-analysis), `stage2_*.py` (calibration and training comparison), `release/install_data.py` |
| `report/` | Audit tables and internal audit figures (`make_tables.py`, `make_figs.py`, `r2*_tables.py`, `r2*_figs.py`) |
| `figures/` | Scripts for the article/ESM figures and tables, with the vendored plotting style (`figures/README.md`) |
| `mmau/study/` | MM-AU development-study scripts (pixel pilot, controls, dense overlap and reading sensitivity, anchor shift, input stride) |
| `tests/` | Unit and integration tests (`python -m pytest tests data/mmau`) |
| `*.md` at the root | Analysis plans and the deviation log of the re-analyses (`EXPERIMENT_*`, `M3_SCOPE_*`, `STAGE2_*`, `DEVIATIONS.md`). Several scripts hash these files into their run records, so they stay at the root |
| `docs/PROVENANCE.md` | Where every part of this repository and of the data package comes from; release edits |

## Installation

The computations were run with Python 3.11.14 on Linux.

```bash
conda env create -f environment.yml      # CPU+GPU environment as used (PyTorch 2.5.1, CUDA 12.1)
conda activate ape
# or: python -m pip install -r requirements.txt
```

All evaluation, audit, re-analysis, calibration and figure steps are CPU-only. A CUDA GPU is needed only for
feature extraction, system training, the VLM run and the stage-two training comparison. The MM-AU model runs
used a separate environment (Python 3.10.21, `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`,
`numpy==2.2.6`, `Pillow==12.3.0`), see `mmau/requirements-training.txt`.

### Paths and environment variables

No absolute path is hard-coded. Scripts resolve the repository root from their own location; external locations
are set with environment variables (shell scripts read the defaults from `scripts/env.sh`):

| Variable | Meaning | Default |
|---|---|---|
| `APE_ROOT` | repository root | parent of `scripts/` |
| `APE_DATA` | root of the derived-data package | none (required by `install_data.py` and `figures/`) |
| `APE_ACCIDENT` | ACCIDENT dataset root (contains `metadata-real.csv`, `real_videos/`) | `$APE_ROOT/external/ACCIDENT_2026` |
| `APE_MMAU` | MM-AU release root (contains `extracted/`, `official_metadata/`) | none; MM-AU scripts take `--root` |
| `APE_TMP` | scratch space for logs, pid files and run records | `$APE_ROOT/tmp` |
| `APE_PY` | interpreter used by the shell scripts | `python` |
| `HF_HOME` | Hugging Face cache (CLIP, Qwen2.5-VL) | `~/.cache/huggingface` |
| `TORCH_HOME` | torchvision weight cache (ResNet) | `~/.cache/torch` |
| `APE_FIG_OUT` | output directory of `figures/` | `figures/out` |

Some scripts pin GPUs exactly as in the original runs (for example `CUDA_VISIBLE_DEVICES=5` in
`scripts/b_04_train.sh`, physical GPU 1 in `systems/vlm_full.py`); adjust them to your machine.

Several entry points record `git rev-parse HEAD` in their run records, and `scripts/stage2_train.py` refuses to
start from a dirty working tree. Run them from a git checkout of this repository.

## Data

### 1. Derived-data package (required for everything except the raw-data steps)

The package (`ape-derived-data`, deposited with the article; DOI to be added) contains the manifests, the stored
answers of the 185 library systems, the audit results, the re-analysis outputs and the MM-AU predictions and
evaluations. Its `README.md` describes every file. Install it into this repository:

```bash
export APE_DATA=/path/to/ape-derived-data
python scripts/release/install_data.py verify  --data "$APE_DATA"   # SHA-256 of every file
python scripts/release/install_data.py install --data "$APE_DATA"   # copies into data/manifest/ and outputs/
```

`install` decompresses the 185 answer matrices (`answers.csv.gz`, 1.2 GB) to `outputs/answers/<system>/answers.csv`
(4.7 GB) and checks each restored file against the SHA-256 of the file used in the original analyses, so restored
files are byte-identical to the originals. The frozen protocol and pre-registration files of the package are
checked against `protocol/` and `prereg/` of this repository.

### 2. ACCIDENT (raw videos; needed only to rebuild manifests, features, systems or the VLM run)

Download the ACCIDENT release from Kaggle (`picekl/accident`), for example with `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("picekl/accident")
```

Point `APE_ACCIDENT` to the directory that holds `metadata-real.csv` and `real_videos/` (2,027 mp4 files,
4.6 GB). The copy used for the paper was downloaded on 2026-05-14; its `metadata-real.csv` has SHA-256
`a3d06073353a7fedff140b652d025135f017860a240cad693990f5288c6c0ec9`. Every row of
`data/manifest/manifest_real.csv` stores the MD5 of the video file in `decode_hash`; check your copy with

```bash
python - <<'EOF'
import hashlib, os, pandas as pd
root = os.environ["APE_ACCIDENT"]; m = pd.read_csv("data/manifest/manifest_real.csv")
bad = [r.path for r in m.itertuples() if hashlib.md5(open(os.path.join(root, r.path), "rb").read()).hexdigest() != r.decode_hash]
print(len(m), "videos,", len(bad), "mismatches")
EOF
```

The synthetic (CARLA) part of the release is not used.

### 3. MM-AU (raw frames; needed only for the MM-AU model runs and overlap audits)

MM-AU is available from Hugging Face, `JeffreyChou/MM-AU`, revision `540cb1277cb70e91a7022abe852decb3ee9adb0a`
(CAP-DATA and DADA-2000 image sequences plus `official_metadata/`). The released native development manifest,
decode ledger and planning ledger (`mmau/deliverables/` in the data package) record relative image directories and
SHA-256 digests of the frames that were read.

### 4. Pretrained models (not redistributed)

| Model | Source and revision | Used for |
|---|---|---|
| ResNet-18 | torchvision `IMAGENET1K_V1` | first-round frame features; MM-AU pilots |
| ResNet-50 | torchvision `IMAGENET1K_V2` | feature check (`r50`) |
| CLIP ViT-B/16 | Hugging Face `openai/clip-vit-base-patch16`, revision `57c216476eefef5ab752ec549e440a49ae4ae5f3` | second-round features; stage-two training |
| Qwen2.5-VL-7B-Instruct | Hugging Face `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` | read-only VLM applicant (`systems/vlm_pilot/download.py` fetches and verifies it) |

## Reproducing the results

Unless stated otherwise the commands start from the stored answers of the data package and run on a CPU.
Runtimes were measured on a 2×Intel Xeon Gold 6530 server (4 threads unless noted).

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest -q tests data/mmau   # about 45 s
# with a visible GPU: 628 passed, 12 skipped; CPU only (CUDA_VISIBLE_DEVICES=""): 618 passed, 22 skipped.
# The GPU tests of the stage-two recipes enable deterministic algorithms and fail without CUBLAS_WORKSPACE_CONFIG.
```

### Re-analyses (start from the released answers)

```bash
python scripts/method_experiments.py --source-root . --out outputs/run002_rerun              # about 1.5 min
python scripts/m3_reanalysis.py --source-root . --run002 outputs/run002 \
       --out outputs/m3_rerun --bootstrap 2000                                                # about 2.5 min
python scripts/m3_compare_runs.py outputs/m3_20260928 outputs/m3_rerun                        # rerun check
python scripts/stage2_calibration.py --out outputs/stage2_calibration_rerun                  # about 1 min
python scripts/stage2_plasmode.py --source-root . --out outputs/stage2_plasmode_rerun        # under 1 min
python scripts/stage2_final_calibration.py --source-root . --out outputs/stage2_final_rerun  # about 6 min
python scripts/stage2_final_calibration.py --source-root . --out outputs/stage2_exact_rerun \
       --only-h10-decomposition --bootstrap 2000                                              # about 5 min
python scripts/stage2_readout.py outputs/stage2_training_v2   # rewrites the paired readouts from the released out-of-fold predictions
```

`scripts/stage2_calibration.py` at this version is the one that produced `outputs/stage2_calibration_v2`
(the first calibration run, `stage2_calibration`, used the version before the unconditional micro bootstrap was
added; see `STAGE2_CALIBRATION_AMENDMENT.md`).

### Audit (start from the released audit results)

```bash
python -X utf8 report/make_tables.py && python -X utf8 report/make_figs.py   # audit tables/figures in outputs/r2c/, about 30 s
python scripts/k_c_verify.py                                                   # 5,677 consistency assertions, about 10 s
```

Recomputing the audit itself from the answer matrices also needs the coarse-step answer matrices of the 141
stateful systems (`outputs/answers/<system>__stride2|4`, 284 directories, 2.2 GB): the per-system evaluation and
the protocol scans read them for the 0.5-s and 1.0-s steps. They are an optional extra of the data package, or can
be regenerated with the system library (`scripts/b_06_postproc.sh`, `b_07_commit.sh`, `b_08b_trivial_stride.sh`,
`b_r2_run.py generate`). With them, in a checkout whose `outputs/` holds only `answers/`, `prefix_lists/`, `r2_S/` and
`report_index.json` of the data package (`k_c_all.py prepare` refuses to overwrite an existing K-c index):

```bash
python scripts/k_b_all.py prepare && python scripts/k_b_all.py eval && python scripts/k_b_all.py remaining   # K-b -> outputs/r2b, about 14 min, 1 thread
python scripts/k_c_validate_vlm.py        # checks the VLM run records (optional extra) -> outputs/r2c/vlm_acceptance.json
python scripts/k_c_all.py prepare && python scripts/k_c_all.py all                                           # K-c -> outputs/r2c, about 7 min
python -X utf8 report/make_tables.py && python -X utf8 report/make_figs.py
python scripts/k_c_verify.py
```

`scripts/k_c_report.py` renders the internal K-c run report and the final-conclusions table
(`outputs/r2c/final_conclusions.json`, `tables/final_conclusions.csv`); it reads the K-c run records
(`$APE_TMP/r2/Kc/`), which are not distributed, so the released `final_conclusions.json` is provided as produced.
The release verification recomputed K-b and K-c this way; all metric records, calibrations, gates, ablations,
mechanisms and tables were byte-identical to the released ones (see the verification record).

### Article and ESM figures and tables

```bash
export APE_DATA=/path/to/ape-derived-data
bash figures/run_all.sh          # about 20 s; see figures/README.md
```

The figures were drawn with Calibri on Windows; elsewhere matplotlib falls back to DejaVu Sans (same data and
layout, different text metrics). The qualitative figures (Fig. 5, Fig. S3) need 18 original frames, which are not
redistributed: `figures/extract_qual_frames.py` extracts them from your own dataset copy and checks the recorded
hashes (pin `opencv-python-headless==4.11.0.86` for bit-identical ACCIDENT frames; later OpenCV versions select the
same frames but decode slightly different pixels).

### Map from paper results to code and outputs

Numbering of the manuscript version of 2026-09-28 (article: Figs. 1–6, Tables 1–7, Algorithm 1; ESM: Figs. S1–S3,
Tables S1–S8). "Needs" names what a full recomputation needs beyond the data package: "raw" = raw videos or frames,
"GPU" = a GPU, "stride" = the optional coarse-step answer matrices. All article and ESM tables and quantitative
figures are regenerated from the data package by `figures/run_all.sh`; the table numbers were checked against the
manuscript (1,136 numbers in 14 tables, no mismatch).

| Paper result | Figure/table script (`figures/`) | Upstream computation | Data (package path under `accident/` or `mmau/`) | Needs |
|---|---|---|---|---|
| Fig. 1, Fig. 2 | schematics drawn in PowerPoint; photographs are original dataset frames | – | – | – |
| Algorithm 1 | – | `ape/metrics.py` (`evaluate_system`, `stable_correct`, `rmscd_from_curve`) | – | – |
| Table 1 (cohorts) | `make_tables.py` → `tab_cohorts.tex` | `make manifest`, `data/clusters.py` | `manifest/manifest_real.csv` | raw (manifest only) |
| Fig. 3 (eligibility, accounting) | `fig_accounting.py` → `fig3_accounting.*` | `scripts/k_c_all.py` (G3 v5) | `manifest/manifest_real.csv`, `outputs/r2c/gates/g3_v5.json` | stride |
| Fig. 4 (window-end-tied pairs) | `build_system_table.py`, `fig_tied_pairs.py` → `fig2_tied_pairs.*` | `scripts/k_b_all.py` (eval, g4) | `outputs/r2b/metrics/<hash>/*.json`, `outputs/r2b/gates/g4.json` | stride |
| Table 2 (tied-pair decomposition) | `tied_pairs_decomposition.py`, `g4_composition.py`, `make_tables.py` → `tab_tied.tex` | `scripts/k_b_all.py` | `outputs/r2b/gates/g4.json` | stride |
| Fig. 5 (ACCIDENT trajectories) | `fig_qualitative_cases.py` (frames via `extract_qual_frames.py`) | stored answers | `outputs/answers/clip__r18mean__seed20260903`, `figures/qualitative/*.json` | raw |
| Table 3 (E/R components, 27 base classifiers) | `paper_numbers.py` → `paper_base_process*` | `scripts/method_experiments.py` | `outputs/run002/systems_H10.csv` | – |
| Table 4 (per-class components) | `paper_numbers.py` → `paper_per_class*` | `scripts/m3_reanalysis.py` | `outputs/m3_20260928_r2/D_*.csv` | – |
| Table 5 (selection) | `paper_numbers.py` → `paper_selection*` | `scripts/m3_reanalysis.py` | `outputs/m3_20260928_r2/C1_test_selection.csv`, `C2_split_selection.csv`, `C3_dev_selection.csv` | – |
| Table 6 (characterization) | `knob_decomposition.py`, `make_tables.py` → `tab_characterization.tex`; check: `check_reproduce.py` | `scripts/k_b_all.py` (calibrate) | `outputs/r2b/calib_plaus/<hash>/calibration.json`, `R_*.csv`, `outputs/r2b/r2/scan_all.csv` | stride |
| Fig. 6 (protocol knobs) | `fig_knobs.py` → `fig5_knobs.*` | `scripts/k_b_all.py` | `outputs/r2b/r2/scan_all.csv`, `outputs/r2b/calib*/`, `outputs/r2b/mechanisms/p_c.json` | stride |
| Table 7 (reading protocols) | `paper_numbers.py` → `paper_process_protocols*` | `scripts/method_experiments.py` | `outputs/run002/robust_H{4,10}_{macro,micro}.csv` | – |
| Tied/heterogeneous pair counts in the text (5427/7815/9021; 1283/3019/3952 at H = 4/10/21.5 s), M3 text numbers | – | `scripts/k_b_all.py` (g4); `scripts/m3_reanalysis.py` | `outputs/r2b/gates/g4.json`; `outputs/m3_20260928_r2/A_tied_noncommit.json`, `SUMMARY.json`, `B_*.csv` | – |
| VLM secondary pool | – | `systems/vlm_full.py run`, `scripts/k_c_validate_vlm.py`, `scripts/k_c_all.py` | `outputs/answers/r2__qwen25vl7b__readonly/`, `outputs/r2c/tables/table1_audit_H10p00.csv` | raw, GPU |
| Table S1 (counterexample) | `paper_numbers.py` → `paper_counterexample*` | `ape.method_analysis.trajectory_components` | – | – |
| Table S2 (gauge blocks) | `make_tables.py` → `tab_blocks.tex` | `python -m ape.cli blocks`; `scripts/k_b_all.py` | `outputs/r2b/metrics` (blocks) | – |
| Table S3 (protocol revisions) | written by hand | – | `prereg/S1_freeze_*.yaml`; two values from `outputs/r2b/first_round_baseline.json` | – |
| Table S4 (G3 rules) | `make_tables.py` → `tab_g3.tex` | `scripts/k_c_all.py` | `outputs/r2c/gates/g3_v5.json` | stride |
| Table S5 (contrasts), Fig. S1 (comparability) | `make_tables.py` → `tab_contrasts.tex`; `fig_comparability.py` → `fig4_comparability.*` | `scripts/k_b_all.py` | `outputs/r2b/calib_plaus`, `outputs/r2b/calib`, `outputs/r2b/r2/scan_all.csv` | stride |
| Table S6 (coverage calibration) | `paper_numbers.py` → `paper_process_calibration*` | `scripts/stage2_calibration.py`, `stage2_final_calibration.py` | `outputs/stage2_calibration_v2`, `outputs/stage2_final_calibration`, `outputs/stage2_exact_bootstrap` | – |
| Fig. S2, Table S7 (MM-AU) | `fig_mmau.py` → `fig6_mmau.*`; `make_tables.py` → `tab_mmau.tex` | `mmau/study/*` runs (raw, GPU), `evaluate_*.py`, `protocol_sensitivity.py`, `summarize_*.py` | `mmau/deliverables/*/RESULTS.json`, `*/ape_evaluation.json`, `overlap_dense_20260920/sensitivity/summary.json` | raw, GPU to retrain |
| Fig. S3 (MM-AU trajectories) | `fig_qualitative_cases.py` | stored predictions | `mmau/deliverables/pixel_pilot_20260920/model_run`, `figures/qualitative/*.json` | raw |
| Table S8 (training objectives) | `paper_numbers.py` → `paper_process_training*` | `scripts/stage2_train.py` (GPU), `scripts/stage2_readout.py` | `outputs/stage2_training_v2/primary_summary.csv` | GPU and CLIP features to retrain |

`figures/README.md` gives the exact table and figure files each script writes.

## Steps that need raw data or a GPU

| Step | Command | Input | Hardware | Time (original run) |
|---|---|---|---|---|
| Manifest v2 (md5, decode probe, leak repair), clusters, statistics | `make manifest clusters stats` | ACCIDENT videos | CPU | minutes |
| Prefix lists, gauge blocks | `python -m ape.cli make-prefixes ...`, `python -m ape.cli blocks ...` | manifest | CPU | minutes |
| First-round systems (R18 features, prefix-mean and GRU classifiers, arms, commitment rules, trivial) | `scripts/b_02_*` to `b_11_*.sh` | videos, torchvision ResNet-18 | GPU | features 47 min; training to answers about 6 min |
| Second-round systems (CLIP-B/16 and R18, 120 fine-step systems, 204 coarse-step answer sets) | `python scripts/b_r2_run.py features|train|select|generate|verify|final|checks` | videos, CLIP | GPU | features 6 min, training 35 min |
| VLM applicant | `python systems/vlm_pilot/download.py`, `prepare.py`, `run.py`; `python systems/vlm_full.py run` | videos, Qwen2.5-VL-7B | 1 GPU (48 GB) | 10.3 h |
| Stage-two training comparison (250 fits) | `python scripts/stage2_train.py --source-root . --out outputs/stage2_training_v2` | CLIP-B/16 features (`outputs/features/clip_b16`) | 1 GPU | 6 min |
| MM-AU pilots, controls, anchor shift, input stride | `mmau/study/*/` (see `mmau/README.md`) | MM-AU frames, ResNet-18 | 1 GPU | 0.5–7 min per run after frame decoding and feature extraction |

## Seeds

All randomness is seeded. Manifest split and audit bootstrap: 20260903 (`protocol/pi0.yaml`, 1,000 source-cluster
replicates); system training seeds 20260903/04/05; VLM pilot selection 20260904; first re-analysis 20260927
(2,000 replicates; coverage simulation 20260928); M3 20260928 (2,000 replicates, 500 split-halves);
stage-two finite-template calibration 20260929, empirical population 20260930, final calibration and exact
bootstrap 20260931; training cross-fitting split 20260929 and RNG seeds 20260930–20260934; MM-AU model seeds
20260920/21/22.

## Citation

To be added after publication.
