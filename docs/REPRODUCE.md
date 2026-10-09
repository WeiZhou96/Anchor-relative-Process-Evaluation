# Replication guide

Replication material for the paper *Stable correctness after impact: Evaluating collision-type recognition beyond final accuracy*.

> **Scope.**
>
> - **These instructions** describe the larger release package, the APE derived-data package. It will be provided with a DOI archive (DOI: [TO BE ASSIGNED]); upon acceptance, the complete package will also be deposited on ETS-Data.
> - **This private pre-release snapshot** contains only the key derived-data subset in `data/` ([data/README.md](../data/README.md)). With the subset, `install_data.py verify`, the `figures/` entry points and `figures/make_media.py` run.
> - **Not supported by this subset alone:** `install_data.py install`, full audit recomputation and training.

Contents:

- [Using the key derived-data subset](#using-the-key-derived-data-subset)
- [Installation](#installation)
- [Data](#data)
- [Reproducing the results](#reproducing-the-results)
- [Map from paper results to code and outputs](#map-from-paper-results-to-code-and-outputs)
- [Steps that need raw data or a GPU](#steps-that-need-raw-data-or-a-gpu)
- [Seeds](#seeds)
- [README media](#readme-media)

## Using the key derived-data subset

From the repository root:

```bash
python scripts/release/install_data.py verify --data data
export APE_DATA="$PWD/data"
bash figures/run_all.sh
```

On PowerShell, set `$env:APE_DATA = (Resolve-Path data).Path` before invoking individual Python figure/table scripts. The shell entry point requires Bash. Set `APE_DATA` explicitly: the original `run_all.sh` default expects a separate sibling data package. No data installation is required for the `figures/` entry points.

**Do not run `install_data.py install` against this subset.** That command requires the omitted `accident/outputs/answers/` directory and its complete answer index; it may copy partial outputs before failing. `verify` supports this subset without changes.

## Installation

The computations were run with Python 3.11.14 on Linux.

```bash
conda env create -f environment.yml      # CPU+GPU environment as used (PyTorch 2.5.1, CUDA 12.1)
conda activate ape
# or: python -m pip install -r requirements.txt
```

All evaluation, audit, re-analysis, calibration and figure steps are CPU-only. A CUDA GPU is needed only for feature extraction, system training, the VLM run and the stage-two training comparison.

The MM-AU model runs used a separate environment, listed in `mmau/requirements-training.txt`:

- Python 3.10.21;
- `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`;
- `numpy==2.2.6`, `Pillow==12.3.0`.

### Paths and environment variables

No absolute path is hard-coded. Scripts resolve the repository root from their own location; external locations are set with environment variables (shell scripts read the defaults from `scripts/env.sh`):

| Variable | Meaning | Default |
|---|---|---|
| `APE_ROOT` | repository root | parent of `scripts/` |
| `APE_DATA` | root of the derived-data package | none (required by `install_data.py` and `figures/`; `figures/make_media.py` defaults to `data/`) |
| `APE_ACCIDENT` | ACCIDENT dataset root (contains `metadata-real.csv`, `real_videos/`) | `$APE_ROOT/external/ACCIDENT_2026` |
| `APE_MMAU` | MM-AU release root (contains `extracted/`, `official_metadata/`) | none; MM-AU scripts take `--root` |
| `APE_TMP` | scratch space for logs, pid files and run records | `$APE_ROOT/tmp` |
| `APE_PY` | interpreter used by the shell scripts | `python` |
| `HF_HOME` | Hugging Face cache (CLIP, Qwen2.5-VL) | `~/.cache/huggingface` |
| `TORCH_HOME` | torchvision weight cache (ResNet) | `~/.cache/torch` |
| `APE_FIG_OUT` | output directory of `figures/` | `figures/out` |

Some scripts pin GPUs exactly as in the original runs; adjust them to your machine. Examples:

- `CUDA_VISIBLE_DEVICES=5` in `scripts/b_04_train.sh`;
- physical GPU 1 in `systems/vlm_full.py`.

Several entry points record `git rev-parse HEAD` in their run records, and `scripts/stage2_train.py` refuses to start from a dirty working tree. Run them from a git checkout of this repository.

## Data

### 1. Derived-data package (required for everything except the raw-data steps)

The package is called `ape-derived-data`. It is deposited with the article (DOI to be added; upon acceptance, also on ETS-Data). It contains:

- the manifests;
- the stored answers of the 185 library systems;
- the audit results and the re-analysis outputs;
- the MM-AU predictions and evaluations.

Its `README.md` describes every file. Install it into this repository:

```bash
export APE_DATA=/path/to/ape-derived-data
python scripts/release/install_data.py verify  --data "$APE_DATA"   # SHA-256 of every file
python scripts/release/install_data.py install --data "$APE_DATA"   # copies into data/manifest/ and outputs/
```

`install` decompresses the 185 answer matrices (`answers.csv.gz`, 1.2 GB) to `outputs/answers/<system>/answers.csv` (4.7 GB). It checks each restored file against the SHA-256 of the file used in the original analyses, so restored files are byte-identical to the originals. The frozen protocol and pre-registration files of the package are checked against `protocol/` and `prereg/` of this repository.

### 2. ACCIDENT (raw videos; needed only to rebuild manifests, features, systems or the VLM run)

Download the ACCIDENT release from Kaggle (`picekl/accident`), for example with `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("picekl/accident")
```

Point `APE_ACCIDENT` to the directory that holds `metadata-real.csv` and `real_videos/` (2,027 mp4 files, 4.6 GB). The copy used for the paper was downloaded on 2026-05-14. Its `metadata-real.csv` has SHA-256 `a3d06073353a7fedff140b652d025135f017860a240cad693990f5288c6c0ec9`.

Every row of `data/manifest/manifest_real.csv` stores the MD5 of the video file in `decode_hash`. Check your copy with:

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

MM-AU is available from Hugging Face:

- repository `JeffreyChou/MM-AU`;
- revision `540cb1277cb70e91a7022abe852decb3ee9adb0a`;
- content: CAP-DATA and DADA-2000 image sequences plus `official_metadata/`.

The data package's `mmau/deliverables/` holds the released native development manifest, decode ledger and planning ledger. They record relative image directories and SHA-256 digests of the frames that were read.

### 4. Pretrained models (not redistributed)

| Model | Source and revision | Used for |
|---|---|---|
| ResNet-18 | torchvision `IMAGENET1K_V1` | first-round frame features; MM-AU pilots |
| ResNet-50 | torchvision `IMAGENET1K_V2` | feature check (`r50`) |
| CLIP ViT-B/16 | Hugging Face `openai/clip-vit-base-patch16`, revision `57c216476eefef5ab752ec549e440a49ae4ae5f3` | second-round features; stage-two training |
| Qwen2.5-VL-7B-Instruct | Hugging Face `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` | read-only VLM applicant (`systems/vlm_pilot/download.py` fetches and verifies it) |

## Reproducing the results

Unless stated otherwise, the commands start from the stored answers of the data package and run on a CPU. Runtimes were measured on a 2×Intel Xeon Gold 6530 server (4 threads unless noted).

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

`scripts/stage2_calibration.py` at this version is the one that produced `outputs/stage2_calibration_v2`. The first calibration run, `stage2_calibration`, used the version before the unconditional micro bootstrap was added; see `STAGE2_CALIBRATION_AMENDMENT.md`.

### Audit (start from the released audit results)

```bash
python -X utf8 report/make_tables.py && python -X utf8 report/make_figs.py   # audit tables/figures in outputs/r2c/, about 30 s
python scripts/k_c_verify.py                                                   # 5,677 consistency assertions, about 10 s
```

Recomputing the audit itself from the answer matrices also needs the coarse-step answer matrices of the 141 stateful systems:

- **What they are.** `outputs/answers/<system>__stride2|4`, 284 directories, 2.2 GB.
- **Why they are needed.** The per-system evaluation and the protocol scans read them for the 0.5-s and 1.0-s steps.
- **How to obtain them.** They are an optional extra of the data package. Alternatively, regenerate them with the system library: `scripts/b_06_postproc.sh`, `b_07_commit.sh`, `b_08b_trivial_stride.sh` and `b_r2_run.py generate`.

With them, run the commands below in a checkout whose `outputs/` holds only `answers/`, `prefix_lists/`, `r2_S/` and `report_index.json` of the data package. (`k_c_all.py prepare` refuses to overwrite an existing K-c index.)

```bash
python scripts/k_b_all.py prepare && python scripts/k_b_all.py eval && python scripts/k_b_all.py remaining   # K-b -> outputs/r2b, about 14 min, 1 thread
python scripts/k_c_validate_vlm.py        # checks the VLM run records (optional extra) -> outputs/r2c/vlm_acceptance.json
python scripts/k_c_all.py prepare && python scripts/k_c_all.py all                                           # K-c -> outputs/r2c, about 7 min
python -X utf8 report/make_tables.py && python -X utf8 report/make_figs.py
python scripts/k_c_verify.py
```

`scripts/k_c_report.py` renders the internal K-c run report and the final-conclusions table (`outputs/r2c/final_conclusions.json`, `tables/final_conclusions.csv`). It reads the K-c run records (`$APE_TMP/r2/Kc/`), which are not distributed, so the released `final_conclusions.json` is provided as produced.

The release verification recomputed K-b and K-c this way. All metric records, calibrations, gates, ablations, mechanisms and tables were byte-identical to the released ones (see the verification record).

### Article and ESM figures and tables

```bash
export APE_DATA=/path/to/ape-derived-data
bash figures/run_all.sh          # about 20 s; see figures/README.md
```

With the key derived-data subset of this repository, use `export APE_DATA="$PWD/data"` instead.

The figures were drawn with Calibri on Windows; elsewhere matplotlib falls back to DejaVu Sans (same data and layout, different text metrics).

The qualitative figures (Fig. 5, Fig. S3) need 18 original frames, which are not redistributed. `figures/extract_qual_frames.py` extracts them from your own dataset copy and checks the recorded hashes. Pin `opencv-python-headless==4.11.0.86` for bit-identical ACCIDENT frames; later OpenCV versions select the same frames but decode slightly different pixels.

## Map from paper results to code and outputs

**Numbering.** Numbering follows the manuscript version of 2026-09-28:

- article: Figs. 1–6, Tables 1–7, Algorithm 1;
- ESM: Figs. S1–S3, Tables S1–S8.

**The "Needs" column** names what a full recomputation needs beyond the data package: "raw" = raw videos or frames, "GPU" = a GPU, "stride" = the optional coarse-step answer matrices.

**Regeneration.** All article and ESM tables and quantitative figures are regenerated from the data package by `figures/run_all.sh`. The table numbers were checked against the manuscript (1,136 numbers in 14 tables, no mismatch).

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

[figures/README.md](../figures/README.md) gives the exact table and figure files each script writes.

## Steps that need raw data or a GPU

| Step | Command | Input | Hardware | Time (original run) |
|---|---|---|---|---|
| Manifest v2 (md5, decode probe, leak repair), clusters, statistics | `make manifest clusters stats` | ACCIDENT videos | CPU | minutes |
| Prefix lists, gauge blocks | `python -m ape.cli make-prefixes ...`, `python -m ape.cli blocks ...` | manifest | CPU | minutes |
| First-round systems (R18 features, prefix-mean and GRU classifiers, arms, commitment rules, trivial) | `scripts/b_02_*` to `b_11_*.sh` | videos, torchvision ResNet-18 | GPU | features 47 min; training to answers about 6 min |
| Second-round systems (CLIP-B/16 and R18, 120 fine-step systems, 204 coarse-step answer sets) | `python scripts/b_r2_run.py features\|train\|select\|generate\|verify\|final\|checks` | videos, CLIP | GPU | features 6 min, training 35 min |
| VLM applicant | `python systems/vlm_pilot/download.py`, `prepare.py`, `run.py`; `python systems/vlm_full.py run` | videos, Qwen2.5-VL-7B | 1 GPU (48 GB) | 10.3 h |
| Stage-two training comparison (250 fits) | `python scripts/stage2_train.py --source-root . --out outputs/stage2_training_v2` | CLIP-B/16 features (`outputs/features/clip_b16`) | 1 GPU | 6 min |
| MM-AU pilots, controls, anchor shift, input stride | `mmau/study/*/` (see `mmau/README.md`) | MM-AU frames, ResNet-18 | 1 GPU | 0.5–7 min per run after frame decoding and feature extraction |

## Seeds

All randomness is seeded.

| Component | Seeds and replicates |
|---|---|
| Manifest split and audit bootstrap | 20260903 (`protocol/pi0.yaml`, 1,000 source-cluster replicates) |
| System training | 20260903/04/05 |
| VLM pilot selection | 20260904 |
| First re-analysis | 20260927 (2,000 replicates); coverage simulation 20260928 |
| M3 | 20260928 (2,000 replicates, 500 split-halves) |
| Stage-two calibration | finite-template calibration 20260929; empirical population 20260930; final calibration and exact bootstrap 20260931 |
| Training | cross-fitting split 20260929; RNG seeds 20260930–20260934 |
| MM-AU models | 20260920/21/22 |

## README media

The animations and images shown in the root README are written by one command. It reads the same stored records as the figure scripts and uses no dataset imagery:

```bash
python figures/make_media.py            # GIF + MP4 (MP4 only when ffmpeg is on PATH)
python figures/make_media.py --no-mp4   # GIF and PNG only
```

Run `figures/run_all.sh` first to include copies of the paper's quantitative figures. [docs/media/MEDIA.md](media/MEDIA.md) gives each file's content, sources, checks and limits.
