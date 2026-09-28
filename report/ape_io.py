#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Readers for track A's real output layout. Shared by make_tables.py and make_figs.py.

Layout as produced by A (probed 2026-09-03, scripts/c_10_probe_a_outputs.py):

  outputs/report_index.json          {"n_rows": N, "rows": [{pi_hash, h_s, system_id, ...}]}
  outputs/metrics/<pi_hash>/<system_id>.json
        one file per (protocol, system); H lives at the TOP level (pi_hash encodes H),
        there is no per_H nesting. Keys used here:
          grid_offsets_s, S_H, S_H_macro, RMSCD, RMSCD_macro, N_H, n_clusters,
          h_s, effective_h_s, delta_s, track, family, system_id, pi_hash,
          end_window_macro_acc, end_window_micro_acc, full_clip_macro_acc,
          flips_median, flips_mean, flip_rate,
          class_counts, class_fractions, commit{rho,tau_c,e_c,...},
          bootstrap{<M>{point,ci_lo,ci_hi,n_boot}}, frozen_family{<M>: scalar},
          alt_cohorts{per_clip_end,dynamic_denominator,fixed_H_cohort,
                      length_stratified_change{per_clip_end,fixed_H_cohort},
                      conclusion_stable (always null), s_ref_delta_s},
          phenomena{K1,K2,K3,K5,K6,flip_rate,last_flip_median_s,
                    correct_wrong_correct,pre_anchor_overconfidence,full{...}},
          card{system_id, family, description, train_data_unknown, ...}
  outputs/metrics/<pi_hash>/_pairs.json
        {<M>: {significant_pairs, tests}, systems_real, systems_block,
         window_end_tied_pairs, ruler_median_progress_diff_RMSCD, alpha, correction, n_boot}
  outputs/calib/<pi0_hash>/calibration.json
        {pi0_hash, pi0, r0, scan_mode, delta_star{<system>:...},
         metrics{<M> and "<M> [with blocks]": {reference_range[2], max_b, max_s,
                 min_R, min_R_M_plaus, min_R_M_full, MRD, MRD_plaus, MRD_full_grid,
                 eps_max{...}, verdict, r0, n_pairs, blocks_included, ...}}}
  outputs/calib/<pi0_hash>/R_<mangled M>.csv  (and _withblocks siblings)
        pi_hash, eps_sys_s, eps_jit_sd_s, delta_s, h_s, in_plaus, metric, n_pairs,
        R_M, in_region
  outputs/calib/<pi0_hash>/scan_table.csv
        system_id, is_block, pi_hash, eps_sys_s, eps_jit_sd_s, delta_s, h_s,
        in_plaus, metric, value

Nothing here writes; track A's directories are read-only to C.
"""

import csv
import json
import os

KNOBS = ("eps_sys_s", "eps_jit_sd_s", "delta_s", "h_s")
BLOCKS_SUFFIX = " [with blocks]"


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def list_pi_hashes(metrics_root):
    if not os.path.isdir(metrics_root):
        return []
    return [d for d in sorted(os.listdir(metrics_root))
            if os.path.isdir(os.path.join(metrics_root, d))]


def load_index(outputs_root):
    """report_index.json -> list of rows; empty list if absent."""
    p = os.path.join(outputs_root, "report_index.json")
    if not os.path.isfile(p):
        return []
    obj = read_json(p)
    return obj.get("rows", []) if isinstance(obj, dict) else list(obj)


def reference_protocols(outputs_root, metrics_root):
    """Return [(pi_hash, h_s)] for the reference protocols, sorted by H.

    Taken from report_index.json when present (it names h_s per pi_hash); otherwise the
    metrics subdirectories are read and h_s recovered from any metrics file inside.
    """
    out = {}
    for row in load_index(outputs_root):
        ph, h = row.get("pi_hash"), to_float(row.get("h_s"))
        if ph and h is not None:
            out[ph] = h
    if not out:
        for ph in list_pi_hashes(metrics_root):
            recs = load_metrics(metrics_root, ph)
            if recs:
                out[ph] = to_float(recs[0].get("h_s"))
    # keep only protocols that actually have a metrics directory
    out = {k: v for k, v in out.items() if os.path.isdir(os.path.join(metrics_root, k))}
    return sorted(out.items(), key=lambda kv: (kv[1] is None, kv[1]))


def load_metrics(metrics_root, pi_hash):
    """All per-system metric records under one pi_hash. Files starting with '_' are
    bookkeeping (notably _pairs.json) and are not systems."""
    d = os.path.join(metrics_root, pi_hash)
    if not os.path.isdir(d):
        return []
    recs = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json") or name.startswith("_"):
            continue
        try:
            rec = read_json(os.path.join(d, name))
        except Exception:
            continue
        rec.setdefault("system_id", os.path.splitext(name)[0])
        rec["_file"] = name
        recs.append(rec)
    return recs


def load_pairs(metrics_root, pi_hash):
    p = os.path.join(metrics_root, pi_hash, "_pairs.json")
    return read_json(p) if os.path.isfile(p) else {}


def family_of(rec):
    fam = rec.get("family") or (rec.get("card") or {}).get("family") or ""
    return str(fam)


def is_block(rec):
    return family_of(rec) == "block" or str(rec.get("system_id", "")).startswith("block__")


def is_placeholder(rec):
    """A stand-in answer matrix exists only to exercise the pipeline before track B's
    systems land. Its numbers must never support a conclusion, so every table and figure
    labels it."""
    if family_of(rec) == "standin":
        return True
    desc = ((rec.get("card") or {}).get("description") or "").lower()
    return "placeholder" in desc


def is_trivial(rec):
    """majority / random / oracle / constant-BOT. They are legitimate library members
    (idea 4.3: they supply the extreme anchors of R_M and expose degenerate behaviour),
    so they stay in the ranking, but they are labelled: an oracle at rank 1 is a
    property of the ruler, not an audit finding."""
    return family_of(rec) == "trivial" or str(rec.get("system_id", "")).startswith("trivial__")


def system_class(rec):
    if is_block(rec):
        return "gauge block"
    if is_placeholder(rec):
        return "PLACEHOLDER"
    if is_trivial(rec):
        return "trivial anchor"
    return "system"


def load_calibration(calib_root, pi0_hash=None):
    """Return (pi0_hash, calibration dict) for one calibration directory.

    Tolerates two shapes: the flat document A writes (pi0_hash as a field), and a
    dict keyed by pi0_hash, in case that layout appears later.
    """
    if not os.path.isdir(calib_root):
        return None, {}
    subs = [d for d in sorted(os.listdir(calib_root)) if os.path.isdir(os.path.join(calib_root, d))]
    if pi0_hash:
        # An explicit protocol that has no calibration directory must come back empty,
        # never silently substituted by another protocol's calibration: the perturbation
        # scan is only run at the reference H, and the other horizons must say so.
        if pi0_hash not in subs:
            return pi0_hash, {}
        chosen = pi0_hash
    elif len(subs) == 1:
        chosen = subs[0]
    elif subs:
        chosen = subs[0]
    else:
        return None, {}
    p = os.path.join(calib_root, chosen, "calibration.json")
    if not os.path.isfile(p):
        return chosen, {}
    obj = read_json(p)
    if "metrics" not in obj and chosen in obj and isinstance(obj[chosen], dict):
        obj = obj[chosen]          # keyed-by-hash variant
    obj["_dir"] = os.path.join(calib_root, chosen)
    obj.setdefault("pi0_hash", chosen)
    return chosen, obj


def main_metrics(calibration):
    """Metric entries excluding the '[with blocks]' parallel series (contract section 8:
    main results use real systems only; the block-inclusive version is reported
    alongside and never merged)."""
    m = calibration.get("metrics") or {}
    return {k: v for k, v in m.items() if not k.endswith(BLOCKS_SUFFIX)}


def block_metrics(calibration):
    m = calibration.get("metrics") or {}
    return {k[: -len(BLOCKS_SUFFIX)]: v for k, v in m.items() if k.endswith(BLOCKS_SUFFIX)}


def load_R_tables(calibration, with_blocks=False):
    """Return {metric_name: [rows]} from the R_*.csv files of a calibration directory."""
    d = calibration.get("_dir")
    if not d or not os.path.isdir(d):
        return {}
    out = {}
    for name in sorted(os.listdir(d)):
        if not (name.startswith("R_") and name.endswith(".csv")):
            continue
        has_blocks = name.endswith("_withblocks.csv")
        if has_blocks != bool(with_blocks):
            continue
        rows = read_csv_rows(os.path.join(d, name))
        if not rows:
            continue
        metric = rows[0].get("metric") or os.path.splitext(name)[0][2:]
        for r in rows:
            for k in KNOBS + ("R_M",):
                r[k] = to_float(r.get(k))
            r["in_plaus"] = to_bool(r.get("in_plaus"))
            r["in_region"] = to_bool(r.get("in_region"))
        out[metric] = rows
    return out


def axis_profiles(rows, pi0):
    """Split an R table into one profile per knob, holding the other knobs at pi0.

    A's scan_mode is 'axis': one knob is moved at a time, so the scan is a star around
    pi0, not a filled grid. Returns {knob: [(x, R_M, in_plaus), ...]} for knobs that
    actually vary.
    """
    out = {}
    for knob in KNOBS:
        pts = []
        for r in rows:
            if r.get(knob) is None:
                continue
            others_at_pi0 = True
            for other in KNOBS:
                if other == knob:
                    continue
                a, b = r.get(other), to_float(pi0.get(other))
                if a is None or b is None or abs(a - b) > 1e-9:
                    others_at_pi0 = False
                    break
            if others_at_pi0:
                pts.append((r[knob], r["R_M"], r["in_plaus"]))
        pts = sorted(set(pts))
        if len(pts) >= 2:
            out[knob] = pts
    return out


def grid_slice(rows, xkey, ykey, pi0):
    """A filled 2-D slice, if the scan ever becomes a real grid. Returns (xs, ys, Z) or
    None when fewer than 2x2 cells are populated."""
    others = [k for k in KNOBS if k not in (xkey, ykey)]
    sel = []
    for r in rows:
        ok = True
        for k in others:
            a, b = r.get(k), to_float(pi0.get(k))
            if a is None or b is None or abs(a - b) > 1e-9:
                ok = False
                break
        if ok and r.get("R_M") is not None and r.get(xkey) is not None and r.get(ykey) is not None:
            sel.append(r)
    xs = sorted(set(r[xkey] for r in sel))
    ys = sorted(set(r[ykey] for r in sel))
    if len(xs) < 2 or len(ys) < 2:
        return None
    Z = [[None] * len(xs) for _ in ys]
    filled = 0
    for r in sel:
        i, j = ys.index(r[ykey]), xs.index(r[xkey])
        if Z[i][j] is None:
            filled += 1
        Z[i][j] = r["R_M"]
    if filled < len(xs) * len(ys):
        return None                # a star, not a grid
    return xs, ys, Z


def rulers_of(pairs):
    """Per-metric ruler at this protocol: the median |M(a) - M(b)| over the
    window-end-tied pairs, in the metric's own units. Present in _pairs.json for every
    horizon, whereas calibration.json exists only at the reference horizon."""
    return (pairs or {}).get("rulers_by_metric") or {}


def seed_of(rec):
    s = rec.get("seed")
    if s is None:
        s = (rec.get("card") or {}).get("seed")
    return None if s is None else str(s)


def arm_rule_of(rec):
    """Arm identity is the RULE name (ema / majority / hysteresis / patience / ...).
    The parameter is chosen per seed on dev, so it is not part of the identity."""
    return rec.get("arm_rule") or (rec.get("card") or {}).get("arm_rule") or ""


def arm_value_of(rec):
    v = rec.get("arm_value")
    if v is None:
        v = (rec.get("card") or {}).get("arm_value")
    return "" if v is None else str(v)


KNOB_LABEL = {
    "eps_sys_s": r"anchor common-mode shift $\varepsilon^{sys}$ (s)",
    "eps_jit_sd_s": r"anchor jitter s.d. $\varepsilon^{jit}$ (s)",
    "delta_s": r"prefix step $\Delta$ (s)",
    "h_s": r"horizon $H$ (s)",
}


def active_outputs_root(root):
    """Follow the explicitly published index, never mix K-b with the legacy trees."""
    path=os.path.join(root,'report_index.json')
    if os.path.isfile(path):
        selected=read_json(path).get('outputs_root')
        if selected and os.path.abspath(root)!=os.path.abspath(os.path.join(root,selected)):
            candidate=os.path.join(root,selected)
            if os.path.isdir(candidate): return candidate
    return root
