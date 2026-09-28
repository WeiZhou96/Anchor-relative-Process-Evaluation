# Read-only VLM as an exam-taker: what is fixed and what is not run

Status this round: **interface only**. No large model is downloaded, no answer
matrix is produced, and no VLM row appears in any result. `systems/vlm_adapter.py`
exists so that the parts which must not be tuned after seeing the audit split are
written down now.

## Frozen now

- **Prompt.** `CLOSED_SET_PROMPT` in `vlm_adapter.py`. One prompt, five allowed
  labels, no chain of thought, no explanation requested. The prompt is tuned on
  `split=dev` only and is frozen before the audit split is read. Prompt search is
  not a contribution and is not reported as one.
- **Parsing.** `LABEL_ALIASES` plus `parse_label`. Text that matches nothing maps
  to the bot code, which the protocol scores as **wrong**, not as an abstention:
  a system may not raise its curve by declining to answer.
- **Causality.** A backend receives `PrefixRequest`, which carries only frame
  timestamps already filtered to `t_s <= end_s`. It is never given clip duration,
  distance to the clip end, total frame count, or any frame past the prefix.
- **Frame budget.** `max_frames_per_prefix` is a declared cost knob, uniform
  across prefixes, recorded in the system card.

## What a backend must provide

Subclass `VLMBackend`:

- `name` -- model identifier as published, including revision.
- `train_data_unknown` -- almost always `True` for a public VLM. The card carries
  this flag and the A track reports such systems separately from clean ones, since
  ACCIDENT clips come from public video and pretraining contamination cannot be
  ruled out.
- `answer(req) -> str` -- one generation per prefix.
- `logprobs(req) -> np.ndarray | None` -- optional 5-vector over the label tokens.
  Returning `None` is fine; the answer matrix then carries `NaN` probabilities,
  which the shared format allows. Without probabilities the system cannot take
  part in the commitment or post-processing arms.

## Cost, and why this is deferred

A VLM is the only family in the library whose cost scales with the number of
prefixes rather than with the number of clips: 2027 clips times 41 grid points is
about 83k generations per model. Every other system reuses one cached causal
feature stream. Wiring a VLM is therefore a separate budgeted run, not part of
the smoke pass.

## Candidate backends (none downloaded)

Any public video-language model with a documented revision and a permissive
research licence. Whichever is chosen, its weights are pinned by revision hash in
the card, and the ACCIDENT videos are not redistributed to any hosted endpoint --
a local checkpoint only.
