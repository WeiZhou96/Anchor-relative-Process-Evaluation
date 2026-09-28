# MM-AU ingestion, overlap guards and the native development manifest

This module prepares MM-AU (Hugging Face `JeffreyChou/MM-AU`, revision
`540cb1277cb70e91a7022abe852decb3ee9adb0a`) for APE on its native frame axis. It does not modify the ACCIDENT
protocol, and the MM-AU work is a development study on reused development data, not a confirmatory test.
`$APE_MMAU` below is the root of your MM-AU copy (it contains `extracted/` and `official_metadata/`); outputs of the
original runs are in the derived-data package under `mmau/deliverables/`.

## Preflight

```bash
python data/mmau/mmau_preflight.py --root "$APE_MMAU" --annotations ANNOTATION_INDEX.json --out NEW_DIR
python data/mmau/test_mmau_preflight.py
```

The annotation index is extracted from the original CAP `annotation file` and DADA `Sheet1` worksheets and its
SHA-256 is recorded. Metadata `id` is a global row number; the original video number is the suffix of
`video_name`. It is joined with class, timing fields and the original annotation sheet, which establishes the
CAP/DADA provenance; ambiguous joins are rejected rather than resolved by taking a first match. The preflight
checks image file names and sizes, not decoding. Time stays in native frame indices.

`class_definitions.json` transcribes the release's CAP `Sheet1!A2:B59` (58 native classes).
`merge_completed_workbooks.py` validates and merges two independently completed human mapping forms; the paper
does not use a 58-to-5 mapping and the forms are not distributed.

## Decoding, candidates and content overlap

* `mmau_decode.py` decodes every image of the 11,730 sequences (2,195,613 image files; zero decode failures in the
  original run) with bounded workers and a recoverable ledger (`decode_20260920/decode.jsonl` in the data package).
  25 sequences with frame-index anomalies (9 longer, 15 shorter, 1 with an offset numbering origin) are excluded.
* `mmau_candidates.py` records input hashes and all exclusion reasons; its multi-index 64-bit dHash search recalls
  probe pairs up to Hamming radius 8. Probe similarity does not establish source-video identity. The optional
  `--proposal` input (an automatically generated 58-to-5 mapping proposal) is not part of the release and is not needed
  for the native task.
* `mmau_overlap_audit.py` hashes every frame of the byte-probe candidate clips. The original run read 1,616 frames
  in 18 clips: eight of nine pairs have identical complete frame-byte sequences, the remaining pair shares one
  frame; five identical pairs cross the released ArA partitions, one pair disagrees on the native class and six
  pairs disagree by one frame in the annotated anchor. These are evidence of content overlap and annotation
  disagreement, not permission to repair labels.
* `mmau_split_constraints.py` turns the overlap components into explicit split constraints
  (`continuation_20260920/outputs/content_constraints.json` in the data package). Passing these limited constraints
  does not certify global source independence.

## Frame axis, G0 floor and G6 diagnostics

`ape/frame_axis.py` evaluates on explicit integer frames and fixed windows, with input, cache, grid and
development-split validation; it does not infer FPS. `ape/g0_floor.py` gives conditional IID sample-planning
diagnostics, not a formal G0 sufficiency test. `ape/g6.py` and `ape/g6_pipeline.py` provide guarded configuration,
direction helpers and a diagnostic pipeline (per-video RMSCD/H, paired cluster resampling with video weights,
median paired-seed differences, track-A Holm selection, track-B direction and coverage). Only synthetic or
development inputs are accepted and the executable always reports `formal_verdict: not_evaluable`; G6 is not
registered. Four reproducible synthetic cases:

```bash
python -m data.mmau.g6_pipeline_smoke --config prereg/G6_draft_2026-09-20.yaml --out NEW_G6_SMOKE_DIRECTORY
```

## Native nine-class development task

`ape.vocabulary.TaskVocabulary` binds an ordered native-code mapping to a task identifier, version and SHA-256;
frame manifests, answer tables and protocol hashes carry that hash. Legacy five-class defaults and protocol hashes
are unchanged.

`data/mmau/native_development.py` builds the provisional CAP train/dev manifests. Before any class or cohort
filter it connects all exact-dHash probe candidates and known byte-identical components, then quarantines
components that cross released partitions or CAP/DADA. These are conservative overlap guards, not recovered
source-video identities. With H = 67 frames the train/dev support is 2,255/269 clips (native classes 12 and 14
have 8 and 9 development clips); with H = 39 it is 3,465/408.

```bash
M=$APE_DATA/mmau/deliverables
python -m data.mmau.native_development \
  --ledger $M/route_feasibility_20260920/outputs/planning_ledger.jsonl \
  --decode $M/decode_20260920/decode.jsonl \
  --constraints $M/continuation_20260920/outputs/content_constraints.json \
  --pilot $M/route_feasibility_20260920/outputs/pilot_candidate.json \
  --output NEW_DEVELOPMENT_DIR          # reproduces native_task_20260920/outputs/development_verified
python -m data.mmau.native_development_smoke --directory NEW_DEVELOPMENT_DIR --output NEW_JSON
```

The smoke tool refuses audit/test rows, fixes a common span of at least 67 frames across H = 39/67, reports no
confidence intervals and trains no model; its train-label majority baseline does not establish label learnability
or source independence. The model runs of the study are in `mmau/study/` (see `mmau/README.md`).

## Tests

`python -m pytest -q tests data/mmau` runs the MM-AU tests together with the APE tests.
