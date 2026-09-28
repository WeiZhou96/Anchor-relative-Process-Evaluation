# Qwen local pilot

This pilot is separate from the APE system library. It does not create an answer
matrix or a system card, and does not modify preregistration, the real manifest,
the fixed development set, or any evaluation setting.

The configuration is in `config.json`. The model is
`Qwen/Qwen2.5-VL-7B-Instruct`, revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. Only physical GPU 1 is exposed;
the process sees it as logical CUDA device 0. The model is read from the Hugging Face cache
(`HF_HOME`). Shared package versions are preserved.

`prepare.py` selects 20 clips from the 52 development clips with at least 10 seconds
of post-anchor support by sorting SHA256 hashes of `20260904:video_id`. It creates
11 requests per clip at offsets 0 through 10 seconds. Selection does not use labels
or predictions. The selected clips cover 19 source clusters; no claim of class
balance is made. The selected set has no sideswipe clip.

Each request receives eight temporally ordered frames sampled within the observed
prefix. Frame inclusion uses decoded OpenCV presentation timestamps, not the
manifest's average frame rate. All selected times satisfy `time <= anchor + offset`.
The selected video tensor has exactly eight frames, so the temporal patch size of
two needs no repeated-frame padding. The processor does not resample the video.
The model receives only the sampled tensor and the fixed closed-set prompt. Neither
the class, filename, original video duration/frame count, anchor, nor endpoint is
included in its input. Its video metadata count is eight, the size of the input
tensor. Video timing is encoded at an effective constant frame rate over the selected
frames; exact decoded timestamps and the approximation error are separately retained.

`validate_causality.py` checks all 220 requests and verifies that changing or removing
future timestamps cannot change a preceding request. `prefix_audit.json` also records
frame indices and tensor SHA256 values. Frames are cached only under the temporary
directory, not in git or under the read-only dataset.

Run `prepare.py`, then `validate_causality.py`, then `run.py`. Run downloads and
inference through `nohup` with logs under `$APE_TMP/r2/D/`.
The scripts resolve their repository root from their own locations. The inference
script requires a completed `model_download.json` in that temporary directory and
refuses to mix a new run with an existing `results.jsonl`.

The fixed settings are bfloat16, SDPA, batch size one, eight frames, a maximum pixel
budget of 224 × 392 per frame, greedy generation and at most 16 generated tokens.
No prompt search, constrained class-token decoding, output repair, or content-based
retry is performed. Format compliance means that the output, after removing only
leading/trailing whitespace, exactly equals one of the five allowed labels.

`summary.json` separates model load time, generation time, cached request time and
amortized video decoding. Full-run estimates multiply mean service time by explicit
request counts; they are extrapolations, not runs on the test set. The 21.5-second
estimates retain the eight-frame budget and assume similar per-request costs beyond
the measured 0–10-second range. They exclude download, model loading, compressed-cache
creation and external contention. No monetary price is assumed. GPU memory is
reported from both PyTorch peak counters and one-second `nvidia-smi` samples.

`model_inventory.json` records official file sizes, Git blob IDs and LFS SHA256 values;
`download.py` fetches the pinned revision and validates each weight shard against the official
SHA256 and smaller files against their Git blob SHA1. `runtime_audit.json` records the transport
events of the original download separately from the single successful inference run.
