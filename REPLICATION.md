# Replication explanatory file

**Article.** *Stable correctness after impact: Evaluating collision-type recognition beyond final accuracy.*
Submitted to Communications in Transportation Research.

**Package.** Two parts: (1) this code repository; (2) the APE derived-data package (deposited with the article;
DOI to be added). Raw videos, frames and pretrained weights are obtained from their sources (Section 2).

**Licences (drafts, to be confirmed by the authors).** Code: Apache License 2.0 (`LICENSE`). Derived data:
CC BY-NC-SA 4.0 (`LICENSE-DATA` in the data package), because the ACCIDENT Kaggle listing states
CC BY-NC-SA 4.0 (the ACCIDENT paper states CC BY 4.0 for its annotations; the conflict is unresolved) and MM-AU is
CC BY-NC 4.0.

## 1. Overview

The article defines anchor-relative process evaluation (APE) and applies it in three parts:

1. **ACCIDENT audit.** 185 answer sets (12 gauge blocks with analytic expectations, 4 trivial systems, 27 base
   classifiers, 36 post-processing arms, 105 commitment rules, 1 read-only vision-language model) are scored on
   the ACCIDENT real-video test split (1,514 clips, fixed cohorts of 1,328/1,113/742 clips at H = 4/10/21.5 s).
   The metric family is characterized over a frozen perturbation grid; gates G2–G5 and the window-end-tied pairs
   are evaluated under a frozen, versioned pre-registration.
2. **Re-analyses of the stored answers.** E/R components and paired decompositions of the 27 base classifiers,
   sparse-output bounds, nine reading protocols, coverage calibration of the simultaneous intervals, conditional
   delay versus RMSCD, selection by window-end accuracy versus RMSCD and per-class components (M3); plus a
   training-objective comparison on the development data.
3. **MM-AU development study.** The frame-axis adapter on the native nine-class MM-AU task: pixel and control
   pilots, output-reading sensitivity, input start shift and input cadence.

Every number of the article and its ESM is computed by code in this repository. All evaluation, audit,
re-analysis, calibration and figure steps start from the derived-data package and run on a CPU in minutes;
steps that create answers (features, model training, VLM inference, MM-AU model runs) need the raw data and a GPU.

## 2. Data availability and provenance

| Data | Source | Licence | In this package | Obtain |
|---|---|---|---|---|
| ACCIDENT real videos and metadata (2,027 clips, 4.6 GB) | Kaggle `picekl/accident` (downloaded 2026-05-14; `metadata-real.csv` SHA-256 `a3d06073…c0ec9`) | Kaggle listing: CC BY-NC-SA 4.0; ACCIDENT paper: CC BY 4.0 for annotations/metadata, upstream terms for videos | No | `kagglehub.dataset_download("picekl/accident")`; verify each file against `decode_hash` (MD5) in `manifest_real.csv` (README, Data §2) |
| MM-AU (CAP-DATA and DADA-2000 image sequences, metadata) | Hugging Face `JeffreyChou/MM-AU`, revision `540cb1277cb70e91a7022abe852decb3ee9adb0a` | CC BY-NC 4.0 | No | Hugging Face download at that revision |
| ResNet-18/50 ImageNet weights | torchvision `IMAGENET1K_V1` / `IMAGENET1K_V2` | torchvision/model terms | No | torchvision |
| CLIP ViT-B/16 | Hugging Face `openai/clip-vit-base-patch16` @ `57c216476eefef5ab752ec549e440a49ae4ae5f3` | model licence | No | Hugging Face |
| Qwen2.5-VL-7B-Instruct | Hugging Face `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5` | Qwen licence | No | `systems/vlm_pilot/download.py` (verifies digests) |
| Derived data (manifests, cohorts, stored answers, audit results, re-analysis outputs, MM-AU ledgers/predictions/evaluations) | produced by this code | draft CC BY-NC-SA 4.0 | Yes (2.2 GB; answers gzip-compressed) | deposit with the article; `scripts/release/install_data.py` |

The derived-data package's `README.md` lists every file with its origin (repository, commit, script) and
`SHA256SUMS`. Absolute server paths recorded inside some JSON/CSV/YAML files were rewritten to relative paths or
placeholders; `REWRITTEN_FILES.tsv` gives the original and released SHA-256 of each rewritten file. No personal
data are included; clip identifiers are the public dataset identifiers.

Not included, available from the authors on request (each needs its own licence decision): coarse-step answer
matrices of the 141 stateful systems (284 directories, 2.2 GB; needed to recompute the step axis of the
characterization and gates), frozen features (ResNet-18 186 MB, CLIP-B/16 182 MB, ResNet-50 26 MB; MM-AU feature
caches 116 MB), model checkpoints (27 first-round, 64 MB; 250 stage-two, 183 MB; 24 MM-AU, 12 MB; 12 stopping models,
2.1 MB), the VLM per-clip response records (321 MB), the pre-correction re-analysis `run001` (12 MB) and the K-a
audit outputs (40 MB).

## 3. Computational requirements

* **Software.** Linux, Python 3.11.14; package versions in `environment.yml` / `requirements.txt`. The MM-AU model
  runs used Python 3.10.21 with PyTorch 2.6.0+cu124 (`mmau/requirements-training.txt`).
* **Hardware used.** 2 × Intel Xeon Gold 6530 (128 cores), 1 TiB RAM, 8 × NVIDIA RTX 5880 Ada (48 GB); one GPU at a
  time. CPU steps pin 1–4 threads.
* **Disk.** Data package 2.2 GB (3,822 files); installed into the repository 5.2 GB (answers decompressed to 4.7 GB).
* **Runtimes** (measured on the release, CPU unless noted): tests 45 s; data installation under 1 min;
  `run002` 1.5 min; M3 2.5 min; stage-two calibrations 1–8 min each; audit tables and figures 30 s;
  `k_c_verify` 10 s; K-b audit from answers 14 min (1 thread; eval 1.5 min, remaining stages 12.5 min, needs the
  coarse-step answers); K-c 7 min; MM-AU APE evaluations 1 min. GPU steps (original runs): first-round features
  47 min, first-round training to answers about 6 min; second-round features 6 min and training 35 min; VLM 10.3 h;
  stage-two training (250 fits) 6 min; MM-AU runs 0.5–7 min each after feature extraction.

## 4. Description of programs

| Program | Purpose |
|---|---|
| `ape/` | APE library: protocol vectors, cohorts, scoring (`metrics.py`, Algorithm 1), calibration and rank preservation, gauge blocks, statistics, CLI; audit stages (`r2.py` K-a, `r2b.py` K-b, `r2c.py` K-c); re-analysis modules (`method_analysis.py`, `m3_analysis.py`); frame axis and task vocabularies for MM-AU |
| `data/*.py` | ACCIDENT manifest (MD5, decode probe, source-video leak repair), clusters, statistics, G0 length shift |
| `data/mmau/` | MM-AU preflight, decoding ledger, overlap audit, split constraints, native development manifest |
| `systems/`, `configs/` | system library (features, training, inference, post-processing, commitment rules, trivial systems, VLM applicant, stage-two learning recipes) |
| `scripts/b_*.sh`, `b_r2_run.py` | first- and second-round system library runs |
| `scripts/a_*.sh`, `c_*.sh` | first audit pass and data track (historical run order) |
| `scripts/k_*.py` | audit stages K-a (`k_all.py`), K-b (`k_b_all.py`), K-c (`k_c_all.py`), checks (`k_b_verify.py`, `k_c_verify.py`), final conclusions (`k_c_report.py`) |
| `scripts/method_experiments.py` | first re-analysis (`run002`) |
| `scripts/m3_reanalysis.py`, `m3_compare_runs.py` | M3 re-analysis and rerun check |
| `scripts/stage2_*.py` | coverage calibration (finite template, empirical population, final, exact bootstrap), training comparison, readouts, refit check |
| `report/` | audit tables and internal audit figures |
| `figures/` | article and ESM figures and tables |
| `mmau/study/` | MM-AU model runs, APE evaluations, summaries and checks |
| `tests/`, `data/mmau/test_*.py` | unit and integration tests |
| `scripts/release/install_data.py` | verify and install the derived-data package |

## 5. Instructions to replicators

```bash
conda env create -f environment.yml && conda activate ape
export APE_DATA=/path/to/ape-derived-data
make install-data            # verifies SHA256SUMS, installs manifests and outputs, restores answers byte-identically
make test                    # 628 passed, 12 skipped with a GPU; 618 passed, 22 skipped on CPU only
make reanalysis              # run002, M3 (+ comparison with the first run), stage-two calibrations -> outputs/*_rerun
make report audit-check      # audit tables/figures in outputs/r2c, 5,677 consistency assertions
make paper                   # article and ESM figures and tables -> figures/out
```

Compare the reruns with the released outputs (for example `outputs/run002` with `outputs/run002_rerun`). Run from
a git checkout: several scripts record `git rev-parse HEAD`. The complete audit from the answers (K-b, K-c) and the
steps that start from raw data are listed in `README.md` ("Audit", "Steps that need raw data or a GPU").

Expected agreement (release verification, see below): `run002` and all stage-two calibration outputs are identical
to the released copies (differences only in run records such as start time and code version); M3 agrees to within
2.1e-13; the audit tables regenerated by `report/make_tables.py` are byte-identical; the K-b audit recomputed from
the answers reproduces all 555 metric records, 132 calibration files, gates, ablations and mechanisms of
`outputs/r2b`; the MM-AU APE evaluations and the native development manifest are reproduced from the released
predictions and ledgers.

## 6. List of tables and figures

Numbering of the manuscript version of 2026-09-28. `figures/run_all.sh` runs all figure/table scripts (about 20 s);
the numbers of the 14 data tables were checked against the manuscript (1,136 numbers, no mismatch).

| Result | Program(s) | Data used (package path) |
|---|---|---|
| Figs. 1, 2 | schematics (PowerPoint), not code-generated | – |
| Algorithm 1 | `ape/metrics.py` | – |
| Table 1 | `data/build_manifest.py`, `data/clusters.py`; `figures/make_tables.py` | `accident/manifest/manifest_real.csv` |
| Fig. 3 | `scripts/k_c_all.py`; `figures/fig_accounting.py` | `accident/outputs/r2c/gates/g3_v5.json`, manifest |
| Fig. 4, Table 2 | `scripts/k_b_all.py`; `figures/build_system_table.py`, `tied_pairs_decomposition.py`, `g4_composition.py`, `fig_tied_pairs.py`, `make_tables.py` | `accident/outputs/r2b/metrics`, `r2b/gates/g4.json` |
| Fig. 5 | `figures/fig_qualitative_cases.py`, `extract_qual_frames.py` | stored answers; frames from the user's dataset copy |
| Table 3 | `scripts/method_experiments.py`; `figures/paper_numbers.py` | `accident/outputs/run002/systems_H10.csv` |
| Tables 4, 5 | `scripts/m3_reanalysis.py`; `figures/paper_numbers.py` | `accident/outputs/m3_20260928_r2/D_*`, `C1–C3_*` |
| Table 6, Fig. 6 | `scripts/k_b_all.py`; `figures/knob_decomposition.py`, `make_tables.py`, `fig_knobs.py`, `check_reproduce.py` | `accident/outputs/r2b/calib_plaus`, `r2b/calib`, `r2b/r2/scan_all.csv`, `r2b/mechanisms/p_c.json` |
| Table 7 | `scripts/method_experiments.py`; `figures/paper_numbers.py` | `accident/outputs/run002/robust_H{4,10}_{macro,micro}.csv` |
| Text: tied and heterogeneous pairs, G3, VLM, M3 | `scripts/k_b_all.py`, `k_c_all.py`, `m3_reanalysis.py` | `r2b/gates/g4.json`, `r2c/gates/g3_v5.json`, `r2c/tables/table1_audit_H10p00.csv`, `m3_20260928_r2/*` |
| Table S1 | `figures/paper_numbers.py` | synthetic example |
| Table S2 | `python -m ape.cli blocks`; `figures/make_tables.py` | `accident/outputs/r2b/metrics` (gauge blocks) |
| Table S3 | written by hand | `accident/prereg/S1_freeze_*.yaml`, `r2b/first_round_baseline.json` |
| Table S4 | `scripts/k_c_all.py`; `figures/make_tables.py` | `r2c/gates/g3_v5.json` |
| Table S5, Fig. S1 | `scripts/k_b_all.py`; `figures/make_tables.py`, `fig_comparability.py` | `r2b/calib_plaus`, `r2b/calib`, `r2b/r2/scan_all.csv` |
| Table S6 | `scripts/stage2_calibration.py`, `stage2_final_calibration.py`; `figures/paper_numbers.py` | `stage2_calibration_v2`, `stage2_final_calibration`, `stage2_exact_bootstrap` |
| Table S7, Fig. S2 | `mmau/study/*`; `figures/make_tables.py`, `fig_mmau.py` | `mmau/deliverables/*/RESULTS.json`, `*/ape_evaluation.json`, `overlap_dense_20260920/sensitivity/summary.json` |
| Fig. S3 | `figures/fig_qualitative_cases.py`, `extract_qual_frames.py` | stored predictions; frames from the user's dataset copy |
| Table S8 | `scripts/stage2_train.py`, `stage2_readout.py`; `figures/paper_numbers.py` | `stage2_training_v2/primary_summary.csv` |

`figures/README.md` lists the file each figure/table script writes.

## 7. Seeds and randomness

Every stochastic step is seeded; reruns on the same software reproduce the released outputs exactly or to
floating-point summation order (the largest difference observed, 2.1e-13 in M3). Seeds: manifest split and audit
bootstrap 20260903 (1,000 source-cluster replicates, `protocol/pi0.yaml`); system training 20260903/04/05; VLM pilot
selection 20260904; `run002` 20260927 (2,000 replicates) and its coverage simulation 20260928; M3 20260928 (2,000
replicates, 500 split-halves); stage-two finite-template calibration 20260929, empirical population 20260930,
final calibration and exact bootstrap 20260931; training cross-fitting split 20260929 and RNG seeds
20260930–20260934 (deterministic cuDNN/cuBLAS settings, `CUBLAS_WORKSPACE_CONFIG=:4096:8`); MM-AU model seeds
20260920/21/22. The VLM applicant decodes greedily and has no seed. GPU training can differ in the last digits
across hardware and driver versions; `scripts/stage2_reproduce.py` refits one fit to check this.

## 8. Steps that cannot be rerun from the package alone

| Step | Missing input | Hardware |
|---|---|---|
| Rebuilding the manifest (MD5, decode probe), G0 length shift | ACCIDENT videos | CPU |
| Frame features, first- and second-round system training, answer generation | ACCIDENT videos, torchvision/CLIP weights | GPU |
| VLM applicant run and its validation (`k_c_validate_vlm.py`) | ACCIDENT videos, Qwen2.5-VL-7B, VLM run records | GPU (48 GB), 10.3 h |
| Step axis of the characterization and gates (`k_b_all.py eval/remaining`, `k_c_all.py all`) | coarse-step answers (optional extra) or regeneration of them | CPU |
| Stage-two training | CLIP-B/16 features (optional extra) or re-extraction from the videos | GPU |
| MM-AU model runs, overlap audits, qualitative MM-AU frames | MM-AU frames, ResNet-18 weights | GPU |
| Qualitative figures (Fig. 5, Fig. S3) and the photographs in Figs. 1–2 | original frames (`figures/extract_qual_frames.py` extracts them from the user's copy) | CPU |

## 9. Contact

To be completed by the authors.
