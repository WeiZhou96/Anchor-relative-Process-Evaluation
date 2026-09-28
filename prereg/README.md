# Pre-registration and freeze records

The files in this directory are the frozen records of the ACCIDENT audit, as committed, except that in this
edition the operator identifiers in the attribution fields (`frozen_by`, `drafted_by`) and in comments naming the
operator were replaced by `authors` (SHA-256 of the originals in `docs/PROVENANCE.md`, section 6); no registered
value changed. Comments are partly in Chinese. `protocol/pi0.yaml` is the run-time configuration read by the code; its values must agree
with the freeze records, and `protocol/pi0.frozen.sha256` guards it (the CLI refuses to start if it has drifted).

| File | Frozen | Content |
|---|---|---|
| `S1_freeze_template.yaml` | – | empty template (`frozen: false`) |
| `S1_freeze_2026-09-03.yaml` (v1) | 2026-09-03 | reference protocol `pi0`, horizons 4/10/21.5 s, metric family and report points, rank-preservation floor 0.9, bootstrap (1,000 source-cluster replicates, seed 20260903), perturbation grid and plausible sub-grid, gauge blocks, system library, manifest and development-subset hashes |
| `S1_freeze_2026-09-03_v2.yaml` (v2) | 2026-09-03 | A1 post-processing arms registered by rule; A2 per-metric rulers; A3 floating-point tolerance of the eligibility test; A4 coarse-step re-generation of the random trivial system; A5 scan size, p-value method and resolution guard |
| `S1_freeze_2026-09-04_v3.yaml` (v3) | 2026-09-08 | additive registration of the second-round systems (R2); protocol unchanged |
| `S1_freeze_2026-09-09_v4.yaml` (v4) | 2026-09-09 | A4-1 reading of gate G2; A4-2 within-cohort G3 rule; A4-3 read-only VLM applicant |
| `S1_freeze_2026-09-09_v5.yaml` (v5) | 2026-09-09 | A5-1 visible-end cap of the per-clip-end metric in G3 (22.0 s); A5-2 run-time discipline of the VLM run |
| `S1_freeze_2026-09-09_v6.yaml` (v6) | 2026-09-09 | frame-selector fix of the VLM applicant |
| `G6_draft_2026-09-20.yaml` | not registered | draft of the cross-dataset gate G6 for MM-AU; it abstains by construction |

Rules that the records impose on themselves: a frozen file is never edited in place; a change creates a new
version that names what it supersedes and why; permitted reasons are implementation errors, data errors and
technical infeasibility not foreseen at the freeze, never a result that looks unfavourable. All downstream
outputs are recomputed after a revision. Table S3 of the ESM lists the revisions.
