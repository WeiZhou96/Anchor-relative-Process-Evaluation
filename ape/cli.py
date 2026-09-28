"""Command line entry points for the APE protocol library.

    python -m ape.cli make-prefixes --pi protocol/pi0.yaml --manifest M --out D
    python -m ape.cli blocks        --pi protocol/pi0.yaml --manifest M --out D
    python -m ape.cli eval          --pi P --manifest M --answers A --out D
    python -m ape.cli scan          --pi P --manifest M --answers A --out D
    python -m ape.cli phenomena     --pi P --manifest M --answers A --out D
    python -m ape.cli report-json   --metrics D --out F

Every subcommand is offline, CPU only, and reads nothing but the manifest and
the cached answer matrices. No subcommand runs a model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import blocks as blocks_mod
from . import calib as calib_mod
from . import phenomena as phen_mod
from . import stats as stats_mod
from .cohort import cluster_codes, cohort_table, select_split, split_summary
from .systems import arm_rule, arm_value, family_of, parent_of, seed_of
from .metrics import (
    AnswerTable,
    alt_cohort_report,
    discover_answer_dirs,
    evaluate_system,
)
from .protocol import (
    ProtocolConfig,
    ProtocolVector,
    load_manifest,
    load_protocol,
    load_protocol_checked,
    make_prefixes,
    verify_frozen,
    write_freeze,
    write_prefix_list,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _jsonable(obj):
    """Convert numpy scalars/arrays and pandas frames into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def _dump_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2, sort_keys=True, ensure_ascii=False)


def _load_answer_tables(spec: str) -> List[AnswerTable]:
    """Resolve a comma-separated list of directories or globs into answer tables."""
    dirs: List[str] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        for d in discover_answer_dirs(part):
            if d not in dirs:
                dirs.append(d)
    if not dirs:
        raise SystemExit(f"no answer directories with answers.csv under {spec!r}")
    tables = [AnswerTable.from_dir(d) for d in sorted(dirs)]
    seen = {}
    for t in tables:
        if t.system_id in seen:
            raise SystemExit(
                f"duplicate system_id {t.system_id!r} in {seen[t.system_id]} and {t}"
            )
        seen[t.system_id] = t
    return tables


def _resolve_bootstrap_n(cfg, override) -> int:
    """Honour a CLI override but never let it pass unnoticed on a frozen protocol."""
    if override is None:
        return int(cfg.bootstrap_n)
    n = int(override)
    if cfg.frozen and n != int(cfg.bootstrap_n):
        print(
            f"[warn] protocol {cfg.protocol_version} is FROZEN with "
            f"bootstrap_n={cfg.bootstrap_n}; this run overrides it to {n}. "
            "Intervals from this run are not the frozen protocol's intervals and "
            "must not be reported as such."
        )
    return n


def _load_audit_manifest(args, cfg):
    """Load the manifest and restrict it to the split this run may score.

    The audit set is the frozen test split. Anything else is diagnostic and is
    announced loudly, because a dev-split number looks exactly like an audit
    number once it is in a json file -- which is why every output also carries
    ``audit_split``.
    """
    full = load_manifest(args.manifest)
    split = getattr(args, "split", None) or str(cfg.raw.get("audit_split", "test"))
    sub = select_split(full, split)
    print(f"[split] audit_split={split} -> {len(sub)} of {len(full)} clips "
          f"(manifest splits: {split_summary(full)})")
    if split != "test":
        print(
            f"[split] WARNING: {split!r} is NOT the audit split. The protocol was "
            "tuned on train/dev, so these numbers are diagnostic only and must "
            "not be reported as audit results."
        )
    return sub, split


def _track_of(manifest: pd.DataFrame) -> str:
    """The evaluation track (roadside / dashcam); tracks are never averaged."""
    if "track_id" not in manifest.columns:
        return "unknown"
    vals = sorted(set(manifest["track_id"].astype(str)))
    return vals[0] if len(vals) == 1 else "+".join(vals)


def _phenomena_block(res, manifest, table, cfg) -> Dict[str, object]:
    """Table 5's numbers, flattened next to the full report. Report only."""
    ph_cfg = cfg.phenomena_cfg
    rep = phen_mod.phenomena_report(
        res,
        manifest=manifest,
        answers=table,
        conf_threshold=float(ph_cfg.get("conf_threshold", phen_mod.DEFAULT_CONF_THRESHOLD)),
        short_prefix_s=float(ph_cfg.get("short_prefix_s", phen_mod.DEFAULT_SHORT_PREFIX_S)),
        pre_anchor_s=float(ph_cfg.get("pre_anchor_s", phen_mod.DEFAULT_PRE_ANCHOR_S)),
    )
    p, k = rep["phenomena"], rep["K"]
    return {
        "flip_rate": p["flip_rate_videos_with_any_flip"],
        "last_flip_median_s": p["last_flip_time_s"]["median"],
        "correct_wrong_correct": p["correct_wrong_correct_rate"],
        "pre_anchor_overconfidence": p["pre_anchor_confident_and_finally_wrong"],
        "K1": k["K1_prefix_equals_terminal_vs_prefix_equals_truth_agreement"],
        "K2": k["K2_full_clip_macro_acc"],
        "K3": k["K3_terminal_wrong_but_had_correct_prefix"],
        # table 5's K5 is the prefix-cell rate; the video-level rate is a
        # companion, never a substitute. NaN means the confidence threshold was
        # never reached, i.e. unmeasured -- see K5_confident_cells.
        "K5": k["K5_short_prefix_confident_inconsistent_rate"],
        "K5_video_rate": k["K5_video_rate"],
        "K5_confident_cells": k["K5_confident_cells"],
        "K5_definition": k["K5_definition"],
        "K6": k["K6_paired_difference_s"]["median"],
        "pre_anchor_confident_cells": p["pre_anchor_confident_cells"],
        "interpretation": rep["interpretation"],
        "full": rep,
    }


def _family_samples(results, cfg, replicates):
    """Frozen-family point values and bootstrap samples, one pass per system.

    Returns ``(point[sid][metric], samples[sid][metric] -> array)``. Every system
    is evaluated on the same ordered replicates, which is what makes the pairwise
    differences a genuine paired bootstrap.
    """
    point: Dict[str, Dict[str, float]] = {}
    samples: Dict[str, Dict[str, np.ndarray]] = {}
    for sid, res in results.items():
        p = res.frozen_family(cfg.s_report_delta_s)
        fams = [res.frozen_family(cfg.s_report_delta_s, idx) for idx in replicates]
        point[sid] = p
        samples[sid] = {m: np.array([f[m] for f in fams], dtype=float) for m in p}
    return point, samples


def _intervals_from_samples(
    point: Dict[str, float],
    samples: Dict[str, np.ndarray],
    alpha: float,
    n_boot: int,
) -> Dict[str, Dict[str, float]]:
    """Percentile intervals for the frozen family from already-computed samples."""
    out: Dict[str, Dict[str, float]] = {}
    for name, p in point.items():
        s = np.asarray(samples[name], dtype=float)
        s = s[np.isfinite(s)]
        out[name] = {
            "point": float(p),
            "ci_lo": float(np.percentile(s, 100.0 * alpha / 2.0)) if len(s) else float("nan"),
            "ci_hi": float(np.percentile(s, 100.0 * (1.0 - alpha / 2.0)))
            if len(s)
            else float("nan"),
            "n_boot": int(n_boot),
        }
    return out


def _split_stride_tables(tables, suffix: str):
    """Separate per-step re-runs from the systems they belong to.

    Contract 5.4 lets coarse steps be obtained by sub-sampling the finest-step
    matrix, but only for systems with no state dependence on the prefix end
    point. A system whose card sets ``subsampling_equivalent: false`` must be
    re-run at each step; track B supplies those runs as
    ``<system_id><suffix><factor>`` directories. Returns
    ``(base_tables, {system_id: {factor: table}})``.
    """
    by_id = {t.system_id: t for t in tables}
    strides: Dict[str, Dict[int, AnswerTable]] = {}
    consumed = set()
    for sid, t in by_id.items():
        if suffix not in sid:
            continue
        base, _, tail = sid.rpartition(suffix)
        if not base or base not in by_id or not tail.isdigit():
            continue
        strides.setdefault(base, {})[int(tail)] = t
        consumed.add(sid)
    base_tables = [t for t in tables if t.system_id not in consumed]
    return base_tables, strides


def _needs_rerun_per_step(table: AnswerTable) -> bool:
    """Whether this system's card denies the sub-sampling equivalence."""
    v = table.card.get("subsampling_equivalent")
    if v is None:
        return False
    return str(v).strip().lower() in ("false", "0", "no")


def _table_for_step(table, strides, delta_s, delta_fine_s, notes):
    """Pick the answer matrix to use for one system at one prefix step.

    For a system that is sub-sampling equivalent (or at the finest step) the
    cached matrix is used directly. Otherwise the matching per-step re-run is
    required; if it is absent we fall back to sub-sampling but record a loud
    note, because silently sub-sampling a system that declared it invalid would
    put an unsupported number in the step axis of E4.
    """
    factor = int(round(float(delta_s) / float(delta_fine_s)))
    if factor <= 1 or not _needs_rerun_per_step(table):
        return table, factor
    available = strides.get(table.system_id, {})
    if factor in available:
        # scored AS the base system: a re-run directory is a different answer
        # matrix for the same system, never a separate entrant in the audit
        return available[factor].with_identity(table.system_id, table.card), factor
    notes.append(
        f"{table.system_id}: card says subsampling_equivalent=false but no "
        f"stride-{factor} re-run was supplied; fell back to sub-sampling the "
        f"finest-step matrix. The delta_s={delta_s:g} row for this system is "
        "NOT supported by a re-run."
    )
    return table, factor


def _is_block(table: AnswerTable) -> bool:
    fam = str(table.card.get("family", "")).lower()
    return fam == "block" or table.system_id.startswith("block__")


# --------------------------------------------------------------------------
# make-prefixes
# --------------------------------------------------------------------------
def cmd_make_prefixes(args) -> int:
    cfg = load_protocol_checked(args.pi)
    manifest, audit_split = _load_audit_manifest(args, cfg)
    pvs = (
        cfg.scan_grid(plaus=args.plaus, mode=args.mode)
        if args.all_pi
        else [cfg.pi0(h) for h in cfg.h_list_s]
    )
    os.makedirs(args.out, exist_ok=True)
    written = []
    for pv in pvs:
        pref = make_prefixes(
            manifest, pv, cfg.grid_max_s, cfg.h_list_s, cfg.delta_fine_s
        )
        path = write_prefix_list(pref, args.out, pv)
        written.append({"pi_hash": pv.pi_hash, "pi": pv.as_dict(), "path": path,
                        "n_rows": int(len(pref))})
        print(f"[make-prefixes] {pv.pi_hash} {pv.label()} rows={len(pref)} -> {path}")
    _dump_json(
        {
            "protocol_file": cfg.path,
            "protocol_version": cfg.protocol_version,
            "frozen": cfg.frozen,
            "audit_split": audit_split,
            "n_clips": int(len(manifest)),
            "delta_fine_s": cfg.delta_fine_s,
            "grid_max_s": cfg.grid_max_s,
            "h_list_s": cfg.h_list_s,
            "cohorts": cohort_table(manifest, cfg.h_list_s),
            "written": written,
        },
        os.path.join(args.out, "index.json"),
    )
    return 0


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------
def cmd_blocks(args) -> int:
    cfg = load_protocol_checked(args.pi)
    # Gauge blocks are generated for EVERY clip, not just the audit split: a
    # block is an instrument, not a result, and it must be available whichever
    # split is later scored. Restricting the scoring is the CLI's job at eval
    # time, not something baked into the gauge.
    manifest = load_manifest(args.manifest)
    specs = cfg.block_factors or blocks_mod.default_block_specs()
    os.makedirs(args.out, exist_ok=True)
    made = []
    for spec in specs:
        family = str(spec["family"])
        params = dict(spec.get("params") or {})
        commit_after_s = spec.get("commit_after_s")
        path = blocks_mod.write_block(
            args.out,
            manifest,
            family,
            params,
            delta_fine_s=cfg.delta_fine_s,
            span_s=cfg.grid_max_s,
            pad_s=cfg.answers_pad_s,
            commit_after_s=None if commit_after_s is None else float(commit_after_s),
        )
        sid = blocks_mod.block_id(family, params, commit_after_s)
        made.append({"system_id": sid, "path": path, "family": family,
                     "params": params, "commit_after_s": commit_after_s})
        print(f"[blocks] {sid} -> {path}")
    _dump_json(
        {
            "delta_fine_s": cfg.delta_fine_s,
            "span_s": cfg.grid_max_s,
            "pad_s": cfg.answers_pad_s,
            "blocks": made,
            "note": (
                "gauge blocks only; answers are generated by a rule on the real "
                "video time axis and are not model outputs"
            ),
        },
        os.path.join(args.out, "blocks_index.json"),
    )
    return 0


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------
def _load_tables_with_strides(spec: str, cfg):
    """Load answer directories and set aside any per-step re-runs."""
    tables = _load_answer_tables(spec)
    suffix = str(cfg.answers_cfg.get("stride_suffix", "__stride"))
    return _split_stride_tables(tables, suffix)


def _evaluate_all(
    cfg: ProtocolConfig,
    manifest: pd.DataFrame,
    tables: Sequence[AnswerTable],
    pv: ProtocolVector,
):
    return {t.system_id: evaluate_system(manifest, t, pv, cfg.grid_max_s) for t in tables}


def cmd_eval(args) -> int:
    cfg = load_protocol_checked(args.pi)
    manifest, audit_split = _load_audit_manifest(args, cfg)
    tables, strides = _load_tables_with_strides(args.answers, cfg)
    stride_notes: List[str] = []
    h_list = [float(args.h)] if args.h is not None else cfg.h_list_s
    n_boot = _resolve_bootstrap_n(cfg, args.bootstrap_n)

    for h in h_list:
        pv = cfg.pi0(h)
        use_tables = [
            _table_for_step(t, strides, pv.delta_s, cfg.delta_fine_s, stride_notes)[0]
            for t in tables
        ]
        results = _evaluate_all(cfg, manifest, use_tables, pv)
        out_dir = os.path.join(args.out, pv.pi_hash)
        os.makedirs(out_dir, exist_ok=True)

        any_res = next(iter(results.values()))
        codes = cluster_codes(pd.DataFrame({"source_cluster_id": any_res.source_cluster_id}))
        reps = stats_mod.cluster_bootstrap_indices(codes, n_boot, cfg.seed)
        # one frozen-family pass per (system, replicate), shared by the per-system
        # intervals below and by the pair tests further down
        fam_point, fam_reps = _family_samples(results, cfg, reps)

        end_ci: Dict[str, Tuple[float, float]] = {}
        for sid, res in results.items():
            if not np.array_equal(res.video_ids, any_res.video_ids):
                raise SystemExit(
                    f"system {sid} was scored on a different cohort; the paired "
                    "comparison requires identical rows"
                )
            payload = res.to_json_dict(cfg.s_report_delta_s)
            payload["bootstrap"] = _intervals_from_samples(
                fam_point[sid], fam_reps[sid], cfg.alpha_pairs, len(reps)
            )
            payload["n_clusters"] = int(len(np.unique(codes)))
            payload["track"] = _track_of(manifest)
            payload["audit_split"] = audit_split
            payload["arm_rule"] = arm_rule(sid)
            payload["arm_value"] = arm_value(sid)
            payload["seed"] = seed_of(sid)
            payload["parent_system_id"] = parent_of(sid)
            payload["family"] = str(res.card.get("family", "unknown"))
            payload["commit"]["tau_c"] = payload["commit"]["tau_c_mean"]
            table = next(t for t in use_tables if t.system_id == sid)
            payload["alt_cohorts"] = alt_cohort_report(
                manifest, table, pv, cfg.grid_max_s, cfg.s_report_delta_s[0]
            )
            payload["phenomena"] = _phenomena_block(res, manifest, table, cfg)
            end_ci[sid] = (
                payload["bootstrap"]["end_window_macro_acc"]["ci_lo"],
                payload["bootstrap"]["end_window_macro_acc"]["ci_hi"],
            )
            _dump_json(payload, os.path.join(out_dir, f"{sid}.json"))
            print(
                f"[eval] H={h:g} {sid}: RMSCD={payload['RMSCD']:.4f} "
                f"endwin={payload['end_window_macro_acc']:.4f} "
                f"flips_med={payload['flips_median']:.2f} N_H={payload['N_H']}"
            )

        # the pair set P and the window-end-tied pairs, on real systems only
        real = {sid: r for sid, r in results.items()
                if not _is_block(next(t for t in use_tables if t.system_id == sid))}
        pairs_payload: Dict[str, object] = {
            "pi_hash": pv.pi_hash,
            "pi": pv.as_dict(),
            "alpha": cfg.alpha_pairs,
            "correction": "holm",
            "n_boot": n_boot,
            "systems_real": sorted(real),
            "systems_block": sorted(set(results) - set(real)),
            "audit_split": audit_split,
        }
        for metric in ["RMSCD@H", "end_window_macro_acc", "median_flips"] + [
            f"S_H@{k:g}" for k in cfg.s_report_delta_s
        ]:
            pairs, tests = stats_mod.significant_pairs_from_samples(
                {s: fam_point[s][metric] for s in real},
                {s: fam_reps[s][metric] for s in real},
                metric,
                cfg.alpha_pairs,
                "holm",
            )
            pairs_payload[metric] = {
                "significant_pairs": [list(p) for p in pairs],
                "tests": [t.__dict__ for t in tests],
            }
        n_pairs = len(real) * (len(real) - 1) // 2
        res_guard = stats_mod.correction_resolution_ok(
            n_boot, n_pairs, cfg.alpha_pairs, "holm"
        )
        pairs_payload["p_method"] = "normal"
        pairs_payload["stride_notes"] = sorted(set(stride_notes))
        pairs_payload["correction_resolution"] = res_guard
        if not res_guard["percentile_ok"]:
            print(
                f"[eval] note: with {n_pairs} pairs and n_boot={n_boot}, a percentile "
                f"p cannot survive Holm (floor {res_guard['percentile_p_floor']:.4g} x "
                f"{n_pairs} > alpha); using the normal-approximation p. A percentile "
                f"test would need n_boot >= {res_guard['n_boot_needed_for_percentile']}."
            )
        tied = stats_mod.tied_at_window_end({s: end_ci[s] for s in real})
        pairs_payload["window_end_tied_pairs"] = [list(p) for p in tied]
        rulers = _rulers_by_metric({k: fam_point[k] for k in real}, tied)
        pairs_payload["rulers_by_metric"] = rulers
        pairs_payload["ruler_definition"] = RULER_DEFINITION
        pairs_payload["ruler_rmscd_legacy"] = _progress_ruler(real, tied, cfg)
        _dump_json(pairs_payload, os.path.join(out_dir, "_pairs.json"))
        print(f"[eval] H={h:g} pairs written; window-end-tied pairs: {len(tied)}")
    return 0


def _rulers_by_metric(fam_point, tied_pairs) -> Dict[str, Optional[float]]:
    """One ruler per metric, in that metric's own units.

    "A difference worth reporting" is defined by what separates systems that are
    indistinguishable at the window end: for each metric M, the median
    |M(a) - M(b)| over the same set of window-end-tied pairs. It has to be per
    metric because the family is not commensurable -- RMSCD is seconds, the
    window-end accuracy is dimensionless, the flip count is an integer. Judging
    MRD for all of them against a single ruler expressed in seconds compares
    unlike quantities, and did produce a spurious "normalize" verdict for
    median_flips (MRD = 2 flips against a 0.37 s ruler).
    """
    out: Dict[str, Optional[float]] = {}
    if not fam_point:
        return out
    metrics = sorted(next(iter(fam_point.values())).keys())
    for m in metrics:
        gaps = []
        for a, b in tied_pairs:
            if a in fam_point and b in fam_point:
                va, vb = fam_point[a].get(m), fam_point[b].get(m)
                if (
                    va is not None
                    and vb is not None
                    and np.isfinite(va)
                    and np.isfinite(vb)
                ):
                    gaps.append(abs(float(va) - float(vb)))
        out[m] = float(np.median(gaps)) if gaps else None
    return out


RULER_DEFINITION = (
    "median |M(a) - M(b)| over the window-end-tied system pairs at pi0, in this "
    "metric's own units; the smallest difference this study treats as worth "
    "reporting for M"
)


def _progress_ruler(results, tied_pairs, cfg) -> Optional[float]:
    """Median |RMSCD| gap among the window-end-tied pairs (idea 5.1).

    Kept as the legacy single ruler and reported for comparison only; verdicts
    use :func:`_rulers_by_metric`.
    """
    gaps = []
    for a, b in tied_pairs:
        if a in results and b in results:
            va = results[a].metrics_on(None)["RMSCD@H"]
            vb = results[b].metrics_on(None)["RMSCD@H"]
            if np.isfinite(va) and np.isfinite(vb):
                gaps.append(abs(va - vb))
    return float(np.median(gaps)) if gaps else None


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------
def cmd_scan(args) -> int:
    cfg = load_protocol_checked(args.pi)
    manifest, audit_split = _load_audit_manifest(args, cfg)
    tables = _load_answer_tables(args.answers)  # split below
    pi0 = cfg.pi0()
    grid = cfg.scan_grid(plaus=bool(getattr(args, "plaus", False)), mode=args.mode)
    plaus_hashes = {pv.pi_hash for pv in cfg.scan_grid(plaus=True, mode=args.mode)}
    print(f"[scan] grid: {len(grid)} protocol points "
          f"(mode={args.mode}, plaus={bool(getattr(args, 'plaus', False))})")
    n_boot = _resolve_bootstrap_n(cfg, args.bootstrap_n)

    suffix = str(cfg.answers_cfg.get("stride_suffix", "__stride"))
    tables, strides = _split_stride_tables(tables, suffix)
    stride_notes: List[str] = []
    if strides:
        print(f"[scan] per-step re-runs supplied for: "
              f"{ {k: sorted(v) for k, v in strides.items()} }")

    rows: List[Dict[str, object]] = []
    flips_by_delta: Dict[str, Dict[float, float]] = {}
    for pv in grid:
        for t in tables:
            use, _factor = _table_for_step(
                t, strides, pv.delta_s, cfg.delta_fine_s, stride_notes
            )
            res = evaluate_system(manifest, use, pv, cfg.grid_max_s)
            fam = res.frozen_family(cfg.s_report_delta_s)
            for metric, value in fam.items():
                rows.append(
                    {
                        "system_id": t.system_id,
                        "is_block": _is_block(t),
                        "pi_hash": pv.pi_hash,
                        "eps_sys_s": pv.eps_sys_s,
                        "eps_jit_sd_s": pv.eps_jit_sd_s,
                        "delta_s": pv.delta_s,
                        "h_s": pv.h_s,
                        "in_plaus": pv.pi_hash in plaus_hashes,
                        "metric": metric,
                        "value": value,
                    }
                )
            # the step axis of E4: F(Delta) at otherwise reference settings
            if (
                abs(pv.eps_sys_s) < 1e-12
                and abs(pv.eps_jit_sd_s) < 1e-12
                and abs(pv.h_s - pi0.h_s) < 1e-12
            ):
                flips_by_delta.setdefault(t.system_id, {})[float(pv.delta_s)] = float(
                    fam["median_flips"]
                )
        print(f"[scan] {pv.pi_hash} {pv.label()} done ({len(tables)} systems)")

    scan = pd.DataFrame(rows, columns=calib_mod.SCAN_COLUMNS)
    out_dir = os.path.join(args.out, pi0.pi_hash)
    os.makedirs(out_dir, exist_ok=True)
    scan.to_csv(os.path.join(out_dir, "scan_table.csv"), index=False)

    # P and the ruler come from the eval stage; recompute here so scan is standalone
    real_tables = [t for t in tables if not _is_block(t)]
    ruler = None
    rulers: Dict[str, Optional[float]] = {}
    pairs_by_metric: Dict[str, List[Tuple[str, str]]] = {}
    if real_tables:
        reference_tables = [
            _table_for_step(t, strides, pi0.delta_s, cfg.delta_fine_s, stride_notes)[0]
            for t in real_tables
        ]
        results = _evaluate_all(cfg, manifest, reference_tables, pi0)
        any_res = next(iter(results.values()))
        codes = cluster_codes(pd.DataFrame({"source_cluster_id": any_res.source_cluster_id}))
        reps = stats_mod.cluster_bootstrap_indices(codes, n_boot, cfg.seed)
        fam_point, fam_reps = _family_samples(results, cfg, reps)
        end_ci = {}
        for sid in fam_reps:
            s = fam_reps[sid]["end_window_macro_acc"]
            s = s[np.isfinite(s)]
            end_ci[sid] = (
                (float(np.percentile(s, 100.0 * cfg.alpha_pairs / 2.0)),
                 float(np.percentile(s, 100.0 * (1.0 - cfg.alpha_pairs / 2.0))))
                if len(s)
                else (float("nan"), float("nan"))
            )
        tied = stats_mod.tied_at_window_end(end_ci)
        ruler = _progress_ruler(results, tied, cfg)   # legacy, comparison only
        rulers = _rulers_by_metric(fam_point, tied)
        for metric in sorted(scan["metric"].unique()):
            if metric not in fam_point[next(iter(fam_point))]:
                continue
            pairs, _ = stats_mod.significant_pairs_from_samples(
                {s: fam_point[s][metric] for s in fam_point},
                {s: fam_reps[s][metric] for s in fam_reps},
                metric,
                cfg.alpha_pairs,
                "holm",
            )
            pairs_by_metric[metric] = pairs

    calibration: Dict[str, object] = {
        "pi0_hash": pi0.pi_hash,
        "pi0": pi0.as_dict(),
        "audit_split": audit_split,
        "n_clips_scored": int(len(manifest)),
        "r0": cfg.r0,
        "scan_mode": args.mode,
        "n_pi": int(scan["pi_hash"].nunique()),
        "n_systems": int(scan["system_id"].nunique()),
        "ruler_rmscd_legacy": ruler,
        "rulers_by_metric": rulers,
        "ruler_definition": RULER_DEFINITION,
        "p_method": "normal",
        "correction_resolution": stats_mod.correction_resolution_ok(
            n_boot, len(real_tables) * (len(real_tables) - 1) // 2, cfg.alpha_pairs, "holm"
        ),
        "stride_reruns": {k: sorted(v) for k, v in strides.items()},
        "stride_notes": sorted(set(stride_notes)),
        "metrics": {},
        "delta_star": {},
        "note": (
            "main results exclude gauge blocks; the block-inclusive version is "
            "reported alongside and never merged (contract section 8)"
        ),
    }
    for metric in sorted(scan["metric"].unique()):
        pairs = pairs_by_metric.get(metric, [])
        for with_blocks in (False, True):
            key = f"{metric}{'' if not with_blocks else ' [with blocks]'}"
            try:
                c = calib_mod.calibrate_metric(
                    scan, pi0.pi_hash, metric, pairs, cfg.r0,
                    rulers.get(metric), blocks=with_blocks
                )
            except KeyError as exc:  # pragma: no cover - defensive
                calibration["metrics"][key] = {"error": str(exc)}
                continue
            c.pop("b_s_table", None)
            r_table = c.pop("R_table")
            r_table.to_csv(
                os.path.join(
                    out_dir,
                    f"R_{metric.replace('@','at').replace('/','_')}"
                    f"{'_withblocks' if with_blocks else ''}.csv",
                ),
                index=False,
            )
            calibration["metrics"][key] = c
    for sid, fbd in flips_by_delta.items():
        calibration["delta_star"][sid] = calib_mod.delta_star(fbd)
    _dump_json(calibration, os.path.join(out_dir, "calibration.json"))
    print(f"[scan] wrote {out_dir}")
    return 0


# --------------------------------------------------------------------------
# phenomena
# --------------------------------------------------------------------------
def cmd_phenomena(args) -> int:
    cfg = load_protocol_checked(args.pi)
    manifest, audit_split = _load_audit_manifest(args, cfg)
    tables, strides = _load_tables_with_strides(args.answers, cfg)
    stride_notes: List[str] = []
    ph_cfg = cfg.phenomena_cfg
    h = float(args.h) if args.h is not None else cfg.h_ref_s
    pv = cfg.pi0(h)
    out_dir = os.path.join(args.out, pv.pi_hash)
    os.makedirs(out_dir, exist_ok=True)

    reports = []
    for t in tables:
        t, _ = _table_for_step(t, strides, pv.delta_s, cfg.delta_fine_s, stride_notes)
        res = evaluate_system(manifest, t, pv, cfg.grid_max_s)
        rep = phen_mod.phenomena_report(
            res,
            manifest=manifest,
            answers=t,
            conf_threshold=float(ph_cfg.get("conf_threshold", phen_mod.DEFAULT_CONF_THRESHOLD)),
            short_prefix_s=float(ph_cfg.get("short_prefix_s", phen_mod.DEFAULT_SHORT_PREFIX_S)),
            pre_anchor_s=float(ph_cfg.get("pre_anchor_s", phen_mod.DEFAULT_PRE_ANCHOR_S)),
        )
        rep["audit_split"] = audit_split
        reports.append(rep)
        _dump_json(rep, os.path.join(out_dir, f"{t.system_id}.json"))
        print(f"[phenomena] {t.system_id}: flip_rate="
              f"{rep['phenomena']['flip_rate_videos_with_any_flip']:.3f}")
    table = phen_mod.phenomena_table(reports)
    table.to_csv(os.path.join(out_dir, "table5.csv"), index=False)
    _dump_json(
        {"pi_hash": pv.pi_hash, "pi": pv.as_dict(), "n_systems": len(reports),
         "audit_split": audit_split,
         "interpretation": "report only; no pass mark, no ranking"},
        os.path.join(out_dir, "_index.json"),
    )
    return 0


# --------------------------------------------------------------------------
# report-json
# --------------------------------------------------------------------------
def cmd_freeze(args) -> int:
    """Record (or deliberately re-record) the frozen protocol digest."""
    cfg = load_protocol(args.pi)
    existing = None
    try:
        from .protocol import read_frozen_hash

        existing = read_frozen_hash(args.pi)
    except Exception:  # pragma: no cover - defensive
        existing = None
    if existing is not None and not args.force:
        from .protocol import content_sha256, diff_against_snapshot

        if content_sha256(args.pi) == existing:
            print(f"[freeze] already frozen and unchanged: {existing}")
            return 0
        print(f"[freeze] {args.pi} already has a recorded digest and differs from it.")
        print("\n".join(diff_against_snapshot(args.pi)))
        print(
            "\nRefusing to overwrite without --force. Re-freezing changes every "
            "pi_hash and invalidates every cached artefact under outputs/."
        )
        return 1
    if not cfg.frozen:
        print(f"[freeze] warning: {args.pi} has frozen: false; recording the digest "
              "anyway, but the guard stays inactive until frozen is set to true")
    info = write_freeze(args.pi)
    print(f"[freeze] protocol_version : {cfg.protocol_version}")
    print(f"[freeze] sha256           : {info['sha256']}")
    print(f"[freeze] wrote            : {info['hash_file']}")
    print(f"[freeze] wrote            : {info['snapshot_file']}")
    for h in cfg.h_list_s:
        pv = cfg.pi0(h)
        print(f"[freeze] pi_hash H={h:<6g} : {pv.pi_hash}")
    return 0


def cmd_report_json(args) -> int:
    rows = []
    for root, _dirs, files in os.walk(args.metrics):
        for f in sorted(files):
            if not f.endswith(".json") or f.startswith("_"):
                continue
            with open(os.path.join(root, f), "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if "system_id" not in d or "RMSCD" not in d:
                continue
            rows.append(
                {
                    "pi_hash": d.get("pi_hash"),
                    "h_s": d.get("h_s"),
                    "delta_s": d.get("delta_s"),
                    "system_id": d.get("system_id"),
                    "arm_rule": arm_rule(d.get("system_id") or ""),
                    "arm_value": arm_value(d.get("system_id") or ""),
                    "system_family": family_of(d.get("system_id") or ""),
                    "seed": seed_of(d.get("system_id") or ""),
                    "parent_system_id": parent_of(d.get("system_id") or ""),
                    "audit_split": d.get("audit_split"),
                    "N_H": d.get("N_H"),
                    "full_clip_macro_acc": d.get("full_clip_macro_acc"),
                    "end_window_macro_acc": d.get("end_window_macro_acc"),
                    "RMSCD": d.get("RMSCD"),
                    "flips_median": d.get("flips_median"),
                    "commit_rho": (d.get("commit") or {}).get("rho"),
                    "commit_tau_c_mean": (d.get("commit") or {}).get("tau_c_mean"),
                    "commit_e_c": (d.get("commit") or {}).get("e_c"),
                    "missing_lookup_rate": d.get("missing_lookup_rate"),
                }
            )
    df = pd.DataFrame(rows).sort_values(["h_s", "RMSCD"], na_position="last")
    _dump_json({"n_rows": int(len(df)), "rows": df}, args.out)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"[report-json] {len(df)} rows -> {args.out} and {csv_path}")
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ape", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, answers=True):
        sp.add_argument("--pi", required=True, help="protocol yaml, e.g. protocol/pi0.yaml")
        sp.add_argument("--manifest", required=True, help="contract 5.1 manifest csv")
        sp.add_argument(
            "--split", default="test", choices=["test", "dev", "train", "all"],
            help="split to score; the audit set is 'test'. Anything else is "
                 "diagnostic only and is stamped into audit_split.",
        )
        if answers:
            sp.add_argument(
                "--answers",
                required=True,
                help="answers dir, glob, or comma-separated list of either",
            )
        sp.add_argument("--out", required=True, help="output directory")

    sp = sub.add_parser("make-prefixes", help="generate prefix lists")
    common(sp, answers=False)
    sp.add_argument("--all-pi", action="store_true", help="emit the whole scan grid")
    sp.add_argument("--plaus", action="store_true", help="restrict to the plausible grid")
    sp.add_argument("--mode", default="axis", choices=["axis", "full"])
    sp.set_defaults(func=cmd_make_prefixes)

    sp = sub.add_parser("blocks", help="generate synthetic gauge block answer matrices")
    common(sp, answers=False)
    sp.set_defaults(func=cmd_blocks)

    sp = sub.add_parser("eval", help="score systems at the reference protocol")
    common(sp)
    sp.add_argument("--h", type=float, default=None, help="single horizon instead of h_list")
    sp.add_argument("--bootstrap-n", type=int, default=None)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("scan", help="perturbation scan and characterisation")
    common(sp)
    sp.add_argument("--mode", default="axis", choices=["axis", "full"])
    sp.add_argument("--plaus", action="store_true",
                    help="restrict the grid to the plausible subgrid")
    sp.add_argument("--bootstrap-n", type=int, default=None)
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("phenomena", help="phenomenon statistics and K1-K6")
    common(sp)
    sp.add_argument("--h", type=float, default=None)
    sp.set_defaults(func=cmd_phenomena)

    sp = sub.add_parser("freeze", help="record the frozen protocol digest (S1)")
    sp.add_argument("--pi", required=True)
    sp.add_argument("--force", action="store_true",
                    help="re-freeze a changed file; invalidates every pi_hash")
    sp.set_defaults(func=cmd_freeze)

    sp = sub.add_parser("report-json", help="collect metric jsons into one index")
    sp.add_argument("--metrics", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_report_json)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .protocol import ProtocolFrozenError

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except ProtocolFrozenError as exc:
        print("", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
