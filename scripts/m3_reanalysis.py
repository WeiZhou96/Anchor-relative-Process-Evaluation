"""Run the M3 re-analysis of stored predictions specified in M3_SCOPE_20260928.md."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ape.m3_analysis import (  # noqa: E402
    class_support,
    identity_check,
    kendall,
    mode_weights,
    own_end_delay,
    pair_disagreement,
    readout,
    select,
    selection_regret,
    stable_onset_index,
    stratified_halves,
    top_k_overlap,
)
from ape.method_analysis import trajectory_components  # noqa: E402
from ape.metrics import AnswerTable  # noqa: E402
from ape.r2 import root_classifier, seed_consistency  # noqa: E402
from ape.r2c import Context  # noqa: E402
from method_experiments import sha256, write_json  # noqa: E402

LOGGER = logging.getLogger(__name__)
STEP, OFFSET, LAST = 0.25, 2, 88  # cached grid columns j = -2..88 (22 s after the anchor)
SEED = 20260928
HORIZONS = [10.0, 4.0, 21.5]
MODES = ["macro", "micro"]
FRACTIONS = np.round(np.arange(1, 11) * 0.1, 1)
N_SPLITS = 500
MIN_VALID = 0.95
FIRST_ROUND_MEAN = "R1|r18|mean|"  # frozen group_key format of the first-round prefix-mean classifier
ANY_MEAN = "|mean|"
PAPER_TABLE2 = {  # tied, heterogeneous, cross-base, groups, delay-dominant, noncommit pairs, noncommit groups
    4.0: (5427, 1283, 845, 28, 358, 325, 14),
    10.0: (7815, 3019, 2141, 171, 2216, 408, 19),
    21.5: (9021, 3952, 2869, 279, 3028, 516, 36),
}
PAPER_TEXT_H10 = {"same": 2899, "opposite": 719, "med": (1.03, 0.79, 0.29), "nc_cross": 330, "nc_same": 395,
                  "nc_dom": 223, "nc_med": 0.80, "nc_groups_first_mean": 14}


# ----------------------------------------------------------------------------- helpers
def load_matrices(ctx: Context, frames: dict[str, pd.DataFrame], ids: list[str], keep_pred: set[str]):
    columns = np.arange(-OFFSET, LAST + 1, dtype=int)
    correct = {n: {} for n in frames}
    pred_out = {n: {} for n in frames}
    provenance = []
    for k, sid in enumerate(ids):
        path = ctx.answers / sid / "answers.csv"
        frame = pd.read_csv(path, usecols=["video_id", "j", "delta_s", "pred"])
        if frame.duplicated(["video_id", "j"]).any():
            raise ValueError(f"Duplicate cached prediction keys: {sid}")
        table = AnswerTable.from_frame(frame, sid, ctx.cards[sid])
        if table.delta_s != STEP:
            raise ValueError(f"Unexpected finest grid for {sid}")
        for name, split in frames.items():
            grid = np.broadcast_to(columns, (len(split), len(columns)))
            pred, present, _, _ = table.lookup_2d(split.video_id.tolist(), grid)
            supported = (split.anchor_s.to_numpy()[:, None] + columns * STEP >= -1e-9) & (
                split.post_anchor_length_s.to_numpy()[:, None] >= columns * STEP - 1e-9)
            if np.any(supported & ~present):
                raise ValueError(f"Unobserved cached cells in video support: {sid} ({name})")
            correct[name][sid] = pred == split.class_code.to_numpy()[:, None]
            if sid in keep_pred:
                pred_out[name][sid] = np.asarray(pred)
        provenance.append({"system_id": sid, "answers": str(path), "sha256": sha256(path)})
        if (k + 1) % 25 == 0:
            LOGGER.info("Read %d/%d systems", k + 1, len(ids))
    return correct, pred_out, provenance


def bootstrap_counts(clusters, labels, masks, n_boot, seed):
    _, inv = np.unique(clusters, return_inverse=True)
    n_clusters = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    rows, rejected = [], 0
    while len(rows) < n_boot:
        row = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters))[inv].astype(float)
        if all(class_support(row[m][None, :], labels[m])[0] for m in masks.values()):
            rows.append(row)
        else:
            rejected += 1
            if rejected > 100 * n_boot:
                raise ValueError("Too many redraws")
    return np.stack(rows), rejected


def interval(samples):
    s = np.asarray(samples, dtype=float)
    ok = np.isfinite(s)
    n_valid = int(ok.sum())
    if n_valid < MIN_VALID * len(s):
        return float("nan"), float("nan"), n_valid
    lo, hi = np.quantile(s[ok], [0.025, 0.975])
    return float(lo), float(hi), n_valid


class Scores:
    """Point values and bootstrap replicates of per-clip quantities for many systems at once."""

    def __init__(self, arrays: dict[str, np.ndarray], labels: np.ndarray, raw: np.ndarray, mode: str):
        ones = np.ones((1, len(labels)))
        self.point_w = mode_weights(ones, labels, mode)[0]
        self.boot_w = mode_weights(raw, labels, mode)
        self.support = class_support(raw, labels) if mode == "macro" else np.ones(len(raw), bool)
        self.arrays = arrays

    def point(self, name):
        return self.point_w @ self.arrays[name]

    def boot(self, name):
        out = self.boot_w @ self.arrays[name]
        out[~self.support] = np.nan
        return out

    def ratio_point(self, num, den):
        n, d = self.point(num), self.point(den)
        return np.where(d > 1e-9, n / np.where(d > 1e-9, d, 1), np.nan)

    def ratio_boot(self, num, den):
        n, d = self.boot(num), self.boot(den)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 1e-9, n / np.where(d > 1e-9, d, 1), np.nan)


# ----------------------------------------------------------------------------- A
def audit_tied(ctx: Context, source: Path, out: Path) -> dict[str, Any]:
    g4 = json.loads((source / "outputs/r2b/gates/g4.json").read_text(encoding="utf-8"))
    fam = {s: ctx.cards[s]["family"] for s in ctx.nontrivial}
    mean_root = {s: "mean" in root_classifier(s, ctx.cards) for s in ctx.nontrivial}
    result, group_rows = {}, []

    def is_commit_variant(v: str) -> bool:
        return "commit" in v or "ringel" in v or "teaser" in v

    for bh in g4["by_h"]:
        h = float(bh["h_s"])
        level, delay = {}, {}
        for sid in ctx.nontrivial:
            d = json.loads((source / "outputs/r2b/metrics" / bh["pi_hash"] / f"{sid}.json").read_text(encoding="utf-8"))
            s_curve = np.asarray(d["S_H"], dtype=float)
            dx, heff = float(d["delta_s"]), float(d["effective_h_s"])
            level[sid] = heff * (1.0 - s_curve[-1])
            v = s_curve[-1] - s_curve
            delay[sid] = dx * (v.sum() - 0.5 * (v[0] + v[-1]))
        pairs = pd.DataFrame(bh["pairs"])
        # Frozen diff = RMSCD(a) - RMSCD(b); components share its orientation, so the sign criteria are invariant.
        pairs["lvl"] = pairs.system_a.map(level) - pairs.system_b.map(level)
        pairs["dly"] = pairs.system_a.map(delay) - pairs.system_b.map(delay)
        if (pairs.lvl + pairs.dly - pairs["diff"]).abs().max() > 1e-9:
            raise AssertionError("Endpoint/delay split does not add up to the frozen RMSCD difference")
        pairs["commit_pair"] = (pairs.system_a.map(fam) == "commit") | (pairs.system_b.map(fam) == "commit")
        het_all = pairs[pairs.significant]
        zero = pairs["diff"].abs() < 1e-12
        groups = [g for g in bh["seed_groups"] if g["pass"]]

        def summary(frame: pd.DataFrame) -> dict[str, Any]:
            if len(frame) == 0:
                return {"n": 0}
            return {
                "n": int(len(frame)),
                "cross_base": int(frame.cross_base.sum()),
                "delay_same_sign": int((np.sign(frame.dly) == np.sign(frame["diff"])).sum()),
                "delay_dominant": int((frame.dly.abs() > frame.lvl.abs() + 1e-9).sum()),
                "components_below_1e-9": int(((frame.dly.abs() < 1e-9) | (frame.lvl.abs() < 1e-9)).sum()),
                "endpoint_opposite": int((np.sign(frame.lvl) != np.sign(frame["diff"])).sum()),
                "median_abs_diff": float(frame["diff"].abs().median()),
                "median_abs_delay": float(frame.dly.abs().median()),
                "median_abs_endpoint": float(frame.lvl.abs().median()),
            }

        nc_filtered = [g for g in groups if not is_commit_variant(g["variant_a"]) and not is_commit_variant(g["variant_b"])]
        nc_rows = [dict(r) for r in bh["pairs"] if fam[r["system_a"]] != "commit" and fam[r["system_b"]] != "commit"]
        nc_recomputed = [g for g in seed_consistency(nc_rows, ctx.cards) if g["pass"]]
        if sorted((g["variant_a"], g["variant_b"]) for g in nc_recomputed) != sorted(
                (g["variant_a"], g["variant_b"]) for g in nc_filtered):
            raise AssertionError("Recomputed noncommitment groups differ from the filtered frozen groups")
        got = (len(pairs), len(het_all), int(het_all.cross_base.sum()), len(groups), summary(het_all)["delay_dominant"],
               int((~het_all.commit_pair).sum()), len(nc_filtered))
        if got != PAPER_TABLE2[h]:
            raise AssertionError(f"Table 2 not reproduced at H={h}: {got} vs {PAPER_TABLE2[h]}")
        n_zero = int(zero.sum())
        n_tied_raw, n_tied_nc_raw = int(len(pairs)), int((~pairs.commit_pair).sum())
        pairs = pairs[~zero].copy()
        het = pairs[pairs.significant]
        nc = het[~het.commit_pair].copy()
        if h == 10.0:
            s_all, s_nc = summary(het), summary(nc)
            first_mean = sum(FIRST_ROUND_MEAN in g["variant_a"] or FIRST_ROUND_MEAN in g["variant_b"] for g in nc_filtered)
            checks = [
                s_all["delay_same_sign"] == PAPER_TEXT_H10["same"],
                s_all["endpoint_opposite"] == PAPER_TEXT_H10["opposite"],
                tuple(round(s_all[k], 2) for k in ["median_abs_diff", "median_abs_delay", "median_abs_endpoint"]) == PAPER_TEXT_H10["med"],
                s_nc["cross_base"] == PAPER_TEXT_H10["nc_cross"],
                s_nc["delay_same_sign"] == PAPER_TEXT_H10["nc_same"],
                s_nc["delay_dominant"] == PAPER_TEXT_H10["nc_dom"],
                round(s_nc["median_abs_diff"], 2) == PAPER_TEXT_H10["nc_med"],
                first_mean == PAPER_TEXT_H10["nc_groups_first_mean"],
            ]
            if not all(checks):
                raise AssertionError(f"H=10 text figures not reproduced: {checks}")

        def kind(sid: str) -> str:
            return "arm" if fam[sid] == "postproc" else "base"

        nc["pair_type"] = ["-".join(sorted([kind(a), kind(b)])) for a, b in zip(nc.system_a, nc.system_b)]
        nc["involves_prefix_mean"] = [mean_root[a] or mean_root[b] for a, b in zip(nc.system_a, nc.system_b)]
        tied_nc = pairs[~pairs.commit_pair]
        base_mask = pairs.system_a.map(fam).isin(["clip", "prefix"]) & pairs.system_b.map(fam).isin(["clip", "prefix"])
        result[str(h)] = {
            "all": {"tied": int(len(pairs)), "heterogeneous": summary(het), "replicated_groups": len(groups)},
            "noncommit": {
                "tied": int(len(tied_nc)),
                "heterogeneous": summary(nc),
                "heterogeneous_by_pair_type": {k: int(v) for k, v in nc.pair_type.value_counts().items()},
                "heterogeneous_involving_prefix_mean": int(nc.involves_prefix_mean.sum()),
                "replicated_groups": len(nc_filtered),
                "replicated_groups_involving_first_round_prefix_mean": sum(
                    FIRST_ROUND_MEAN in g["variant_a"] or FIRST_ROUND_MEAN in g["variant_b"] for g in nc_filtered),
                "replicated_groups_involving_any_prefix_mean": sum(
                    ANY_MEAN in g["variant_a"] or ANY_MEAN in g["variant_b"] for g in nc_filtered),
            },
            "base_only": {"tied": int(base_mask.sum()), "heterogeneous": summary(het[base_mask[het.index]])},
            "accuracy_auc_column": "not reproduced: frozen source not among the inputs",
            "zero_difference_pairs_excluded": n_zero,
            "tied_including_zero_difference": n_tied_raw,
            "noncommit_tied_including_zero_difference": n_tied_nc_raw,
        }
        for g in nc_filtered:
            group_rows.append({"h_s": h, "variant_a": g["variant_a"], "variant_b": g["variant_b"],
                               "n_same_direction": g["n_same_direction"]})
        keep = ["system_a", "system_b", "diff", "significant", "cross_base", "same_seed", "lvl", "dly", "pair_type",
                "involves_prefix_mean"]
        nc[[c for c in keep if c in nc.columns]].to_csv(out / f"A_noncommit_heterogeneous_H{h:g}.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "A_noncommit_replicated_groups.csv", index=False)
    write_json(out / "A_tied_noncommit.json", result)
    return result


# ----------------------------------------------------------------------------- comparisons
def compare(members, idx, bad_x, bad_y, boot_x, boot_y, pairs_all, pairs_same):
    """Kendall tau-b (bootstrap interval) and sign disagreement between two 'smaller is better' scores."""
    sel = np.array([idx[s] for s in members])
    x, y = bad_x[sel], bad_y[sel]
    ok = np.isfinite(x) & np.isfinite(y)
    reps = np.array([kendall(boot_x[r, sel], boot_y[r, sel]) for r in range(len(boot_x))])
    lo, hi, n_valid = interval(reps)
    local = {s: i for i, s in enumerate(members)}
    out = {"n_systems": len(members), "n_undefined_systems": int((~ok).sum()), "kendall_tau": kendall(x, y),
           "tau_lo": lo, "tau_hi": hi, "tau_valid_replicates": n_valid}
    for label, pairs in [("all", pairs_all), ("same_seed", pairs_same)]:
        p = np.array([[local[a], local[b]] for a, b in pairs]) if pairs else np.zeros((0, 2), int)
        res = pair_disagreement(x, y, p) if len(p) else {"pairs": 0}
        out.update({f"{label}_{k}": v for k, v in res.items()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run002", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.source_root
    frozen = ["data/manifest/manifest_real.csv", "protocol/pi0.yaml", "protocol/pi0.frozen.sha256",
              "outputs/r2b/gates/g4.json"]
    before = {f: sha256(source / f) for f in frozen}
    run002_files = {f.name: sha256(f) for f in sorted(args.run002.glob("systems_H*.csv"))}
    status = {
        "started": datetime.now(timezone.utc).isoformat(),
        "code_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True,
                                    text=True).stdout.strip(),
        "plan_sha256": sha256(ROOT / "M3_SCOPE_20260928.md"),
        "inputs_before": before, "run002_files": run002_files, "bootstrap": args.bootstrap, "seed": SEED,
        "pid": os.getpid(),
        "claim_status": "retrospective descriptive re-analysis of audited test predictions; no training or inference",
    }
    write_json(args.out / "run_state.json", status)
    ctx = Context(source, False)

    LOGGER.info("A: frozen tied pairs by commitment status")
    a_result = audit_tied(ctx, source, args.out)

    test, dev = ctx.test.reset_index(drop=True), ctx.dev.reset_index(drop=True)
    ids = list(ctx.nontrivial)
    fam = {s: ctx.cards[s]["family"] for s in ids}
    trivial = [s for s in ["trivial__majority", "trivial__random"] if s in ctx.ids]  # references fixed in the scope
    if len(trivial) != 2:
        raise AssertionError("Planned trivial references not found")
    sets = {"B": [s for s in ids if fam[s] in ("clip", "prefix")], "N": [s for s in ids if fam[s] != "commit"], "A": ids}
    if tuple(len(v) for v in sets.values()) != (27, 63, 168):
        raise AssertionError("System sets differ from the plan")
    idx = {s: i for i, s in enumerate(ids)}
    seed_of = {s: ctx.meta[s].get("seed") for s in ids}
    pairs_of = {k: list(itertools.combinations(sorted(v), 2)) for k, v in sets.items()}
    same_of = {k: [(a, b) for a, b in p if seed_of[a] is not None and seed_of[a] == seed_of[b]] for k, p in pairs_of.items()}

    correct, preds, provenance = load_matrices(ctx, {"test": test, "dev": dev}, ids + trivial, set(sets["B"] + trivial))
    write_json(args.out / "prediction_provenance.json", provenance)
    cache = correct["test"]
    lplus = test.post_anchor_length_s.to_numpy()
    labels_all = test.class_code.to_numpy()
    masks = {f"E{h:g}": lplus >= h - 1e-9 for h in HORIZONS}
    masks.update(own=lplus >= -1e-9, C4=(lplus >= 4.0 - 1e-9) & (lplus <= 22.0 + 1e-9))
    raw, rejected = bootstrap_counts(test.source_cluster_id.to_numpy(), labels_all, masks, args.bootstrap, SEED)
    LOGGER.info("Bootstrap: %d replicates, %d redraws", len(raw), rejected)

    # ---- per-horizon clip arrays for every system (columns follow ids)
    per_h = {}
    for h in HORIZONS:
        m = masks[f"E{h:g}"]
        n_col = int(round(h / STEP)) + 1
        times = np.arange(n_col) * STEP
        arr = {k: np.empty((m.sum(), len(ids))) for k in ["delay", "error", "retracted", "end", "end_delay", "end_onset"]}
        for s in ids:
            c = cache[s][m][:, OFFSET:OFFSET + n_col]
            comp = trajectory_components(c, times)
            onset = stable_onset_index(c)
            j = idx[s]
            arr["delay"][:, j], arr["error"][:, j], arr["retracted"][:, j] = comp["delay"], comp["error"], comp["retracted"]
            arr["end"][:, j] = comp["endpoint"]
            arr["end_delay"][:, j] = comp["endpoint"] * comp["delay"]
            arr["end_onset"][:, j] = np.where(onset >= 0, onset * STEP, 0.0)
        per_h[h] = dict(mask=m, labels=labels_all[m], times=times, arr=arr)

    # ---- own-end (cache-capped) arrays
    own_m = masks["own"]
    own_last = np.floor(np.minimum(lplus[own_m], 22.0) / STEP + 1e-9).astype(int)
    own = {k: np.empty((own_m.sum(), len(ids))) for k in ["end", "end_delay", "end_onset"]}
    for s in ids:
        d_own, e_own, o_own = own_end_delay(cache[s][own_m][:, OFFSET:], own_last, STEP)
        own["end"][:, idx[s]] = e_own
        own["end_delay"][:, idx[s]] = e_own * d_own
        own["end_onset"][:, idx[s]] = np.where(e_own, np.nan_to_num(o_own), 0.0)
    capped_share = float((lplus[own_m] > 22.0 + 1e-9).mean())

    # ---- C4 arrays (clip-relative versus absolute time axis)
    c4 = masks["C4"]
    t_rel = lplus[c4][:, None] * FRACTIONS[None, :]
    t_abs = np.broadcast_to(4.0 * FRACTIONS[None, :], t_rel.shape)
    times4 = np.arange(17) * STEP
    c4_arr = {k: np.empty((c4.sum(), len(ids))) for k in ["afa", "aaa4", "delay4"]}
    for s in ids:
        c4_arr["afa"][:, idx[s]] = readout(cache[s][c4], t_rel, STEP, OFFSET).mean(axis=1)
        c4_arr["aaa4"][:, idx[s]] = readout(cache[s][c4], t_abs, STEP, OFFSET).mean(axis=1)
        c4_arr["delay4"][:, idx[s]] = trajectory_components(cache[s][c4][:, OFFSET:OFFSET + 17], times4)["delay"]

    system_rows, agreement_rows, selection_rows, identity_violations = [], [], [], {}
    example = None
    for h in HORIZONS:
        LOGGER.info("B/C at H=%s", h)
        ph = per_h[h]
        heff = ph["times"][-1]
        for mode in MODES:
            sc = Scores(ph["arr"], ph["labels"], raw[:, ph["mask"]], mode)
            so = Scores(own, labels_all[own_m], raw[:, own_m], mode)
            own_in_h = ph["mask"][own_m]
            som = Scores({k: v[own_in_h] for k, v in own.items()}, labels_all[own_m][own_in_h],
                         raw[:, own_m][:, own_in_h], mode)
            rm, acc = sc.point("delay"), sc.point("end")
            ctf, cto = sc.ratio_point("end_delay", "end"), sc.ratio_point("end_onset", "end")
            own_ct, own_on = so.ratio_point("end_delay", "end"), so.ratio_point("end_onset", "end")
            own_ct_m = som.ratio_point("end_delay", "end")
            b_rm, b_acc = sc.boot("delay"), sc.boot("end")
            b_ctf, b_cto = sc.ratio_boot("end_delay", "end"), sc.ratio_boot("end_onset", "end")
            b_own, b_own_m = so.ratio_boot("end_delay", "end"), som.ratio_boot("end_delay", "end")
            ok = np.isfinite(ctf)
            if not np.allclose(rm[ok], heff * (1 - acc[ok]) + acc[ok] * ctf[ok], atol=1e-10):
                raise AssertionError("CT_fixed identity failed")
            for s in ids:
                j = idx[s]
                system_rows.append({
                    "system_id": s, "family": fam[s], "horizon": h, "weighting": mode, "n": int(ph["mask"].sum()),
                    "rmscd": rm[j], "window_end_accuracy": acc[j], "error_area": sc.point("error")[j],
                    "retracted_area": sc.point("retracted")[j], "ct_fixed": ctf[j], "ct_onset_discrete": cto[j],
                    "ct_own": own_ct[j], "ct_own_onset_discrete": own_on[j], "own_end_accuracy": so.point("end")[j],
                    "ct_own_matched": own_ct_m[j]})
            for set_name, members in sets.items():
                sel = np.array([idx[s] for s in members])
                local_pairs = np.array([[members.index(a), members.index(b)] for a, b in pairs_of[set_name]])
                identity_violations[f"{set_name}_H{h:g}_{mode}"] = identity_check(rm[sel], ctf[sel], acc[sel], local_pairs)
                for label, y, by in [("ct_fixed", ctf, b_ctf), ("ct_onset_discrete", cto, b_cto), ("ct_own", own_ct, b_own),
                                     ("ct_own_matched", own_ct_m, b_own_m), ("neg_window_end_accuracy", -acc, -b_acc)]:
                    row = compare(members, idx, rm, y, b_rm, by, pairs_of[set_name], same_of[set_name])
                    agreement_rows.append({"set": set_name, "horizon": h, "weighting": mode, "comparison": f"rmscd_vs_{label}", **row})
                # C1: whole test split
                point = selection_regret(members, acc[sel], rm[sel])
                reps = []
                for r in range(len(raw)):
                    ra, rd = b_acc[r, sel], b_rm[r, sel]
                    if not (np.isfinite(ra).all() and np.isfinite(rd).all()):
                        continue
                    sr = selection_regret(members, ra, rd)
                    reps.append((sr["same_choice"], sr["rmscd_regret"], sr["accuracy_change"], top_k_overlap(members, ra, rd)))
                reps = np.array(reps, dtype=float)
                selection_rows.append({
                    "set": set_name, "horizon": h, "weighting": mode, **point,
                    "n_accuracy_ties_at_max": int((np.abs(acc[sel] - acc[sel].max()) < 1e-9).sum()),
                    "top5_overlap": top_k_overlap(members, acc[sel], rm[sel]), "boot_valid": len(reps),
                    "boot_frequency_different": float(1 - reps[:, 0].mean()),
                    "boot_rmscd_regret_median": float(np.median(reps[:, 1])),
                    "boot_rmscd_regret_lo": float(np.quantile(reps[:, 1], 0.025)),
                    "boot_rmscd_regret_hi": float(np.quantile(reps[:, 1], 0.975)),
                    "boot_accuracy_change_median": float(np.median(reps[:, 2])),
                    "boot_accuracy_change_lo": float(np.quantile(reps[:, 2], 0.025)),
                    "boot_accuracy_change_hi": float(np.quantile(reps[:, 2], 0.975)),
                    "boot_top5_overlap_mean": float(reps[:, 3].mean())})
            if h == 10.0 and mode == "macro" and example is None:
                base = sorted(sets["B"])
                for a, b in itertools.combinations(base, 2):
                    if seed_of[a] is None or seed_of[a] != seed_of[b]:
                        continue
                    ia, ib = idx[a], idx[b]
                    d_r, d_c = rm[ia] - rm[ib], ctf[ia] - ctf[ib]
                    if abs(d_r) >= 1e-9 and abs(d_c) >= 1e-9 and np.sign(d_r) != np.sign(d_c):
                        ea, eb = ph["arr"]["end"][:, ia].astype(bool), ph["arr"]["end"][:, ib].astype(bool)
                        strata = {k: float(sc.point_w @ v) for k, v in
                                  {"p11": ea & eb, "p10": ea & ~eb, "p01": ~ea & eb, "p00": ~ea & ~eb}.items()}
                        example = {
                            "rule": "first same-seed base pair in sorted identifier order with opposite CT_fixed and RMSCD differences; H=10 s, class-macro",
                            "system_a": a, "system_b": b, "rmscd": [rm[ia], rm[ib]], "window_end_accuracy": [acc[ia], acc[ib]],
                            "ct_fixed": [ctf[ia], ctf[ib]],
                            "diff_rmscd": [d_r, *interval(b_rm[:, ia] - b_rm[:, ib])[:2]],
                            "diff_accuracy": [acc[ia] - acc[ib], *interval(b_acc[:, ia] - b_acc[:, ib])[:2]],
                            "diff_ct_fixed": [d_c, *interval(b_ctf[:, ia] - b_ctf[:, ib])[:2]],
                            "strata": strata}
                        break
        if h == 4.0:
            for mode in MODES:
                sc4 = Scores(c4_arr, labels_all[c4], raw[:, c4], mode)
                afa, aaa, d4 = sc4.point("afa"), sc4.point("aaa4"), sc4.point("delay4")
                b_afa, b_aaa, b_d4 = sc4.boot("afa"), sc4.boot("aaa4"), sc4.boot("delay4")
                for set_name, members in sets.items():
                    for label, x, y, bx, by in [("aaa4_vs_afa", -aaa, -afa, -b_aaa, -b_afa),
                                                ("rmscd4_vs_afa", d4, -afa, b_d4, -b_afa),
                                                ("rmscd4_vs_aaa4", d4, -aaa, b_d4, -b_aaa)]:
                        row = compare(members, idx, x, y, bx, by, pairs_of[set_name], same_of[set_name])
                        agreement_rows.append({"set": set_name, "horizon": "C4", "weighting": mode, "comparison": label, **row})
                for s in ids:
                    system_rows.append({"system_id": s, "family": fam[s], "horizon": "C4", "weighting": mode,
                                        "n": int(c4.sum()), "afa": afa[idx[s]], "aaa4": aaa[idx[s]], "rmscd4_C4": d4[idx[s]]})
    if any(identity_violations.values()):
        raise AssertionError(f"CT identity check violated: {identity_violations}")

    # ---- C2 split-sample selection on test
    LOGGER.info("C2: split-sample selection")
    clusters = test.source_cluster_id.to_numpy()
    uniq, first_pos, inv = np.unique(clusters, return_index=True, return_inverse=True)
    first_class = labels_all[first_pos]
    rng = np.random.default_rng(SEED)
    split_rows, split_records = [], []
    halves = [stratified_halves(first_class, rng)[inv] for _ in range(N_SPLITS)]
    for h in HORIZONS:
        ph = per_h[h]
        lab = ph["labels"]
        for mode in MODES:
            for set_name, members in sets.items():
                sel = np.array([idx[s] for s in members])
                d_mat, e_mat = ph["arr"]["delay"][:, sel], ph["arr"]["end"][:, sel]
                recs, undefined = [], 0
                for k_split, half in enumerate(halves):
                    hm = half[ph["mask"]]
                    for direction, (fit, ev) in enumerate([(hm, ~hm), (~hm, hm)]):
                        w_fit = np.atleast_2d(fit.astype(float))
                        w_ev = np.atleast_2d(ev.astype(float))
                        if mode == "macro" and not (class_support(w_fit, lab)[0] and class_support(w_ev, lab)[0]):
                            undefined += 1
                            continue
                        wf, we = mode_weights(w_fit, lab, mode)[0], mode_weights(w_ev, lab, mode)[0]
                        fa, fd, ea, ed = wf @ e_mat, wf @ d_mat, we @ e_mat, we @ d_mat
                        a_sel, r_sel = members.index(select(members, fa, True)), members.index(select(members, fd, False))
                        recs.append((a_sel != r_sel, ed[a_sel] - ed[r_sel], ea[r_sel] - ea[a_sel]))
                        split_records.append({
                            "set": set_name, "horizon": h, "weighting": mode, "split": k_split, "direction": direction,
                            "selected_by_accuracy": members[a_sel], "selected_by_rmscd": members[r_sel],
                            "fit_accuracy_of_acc_choice": fa[a_sel], "fit_accuracy_of_rmscd_choice": fa[r_sel],
                            "eval_accuracy_of_acc_choice": ea[a_sel], "eval_accuracy_of_rmscd_choice": ea[r_sel],
                            "eval_rmscd_of_acc_choice": ed[a_sel], "eval_rmscd_of_rmscd_choice": ed[r_sel],
                            "delta_rmscd": ed[a_sel] - ed[r_sel], "delta_accuracy": ea[r_sel] - ea[a_sel]})
                recs = np.array(recs, dtype=float)
                split_rows.append({
                    "set": set_name, "horizon": h, "weighting": mode, "evaluations": len(recs), "undefined": undefined,
                    "frequency_different": float(recs[:, 0].mean()),
                    "delta_rmscd_median": float(np.median(recs[:, 1])), "delta_rmscd_mean": float(recs[:, 1].mean()),
                    "delta_rmscd_lo": float(np.quantile(recs[:, 1], 0.025)), "delta_rmscd_hi": float(np.quantile(recs[:, 1], 0.975)),
                    "delta_accuracy_median": float(np.median(recs[:, 2])), "delta_accuracy_mean": float(recs[:, 2].mean()),
                    "delta_accuracy_lo": float(np.quantile(recs[:, 2], 0.025)), "delta_accuracy_hi": float(np.quantile(recs[:, 2], 0.975))})

    # ---- C3 development-set selection evaluated on test (exploratory)
    LOGGER.info("C3: development-set selection")
    dev_rows = []
    for h in [10.0, 4.0]:
        dmask = dev.post_anchor_length_s.to_numpy() >= h - 1e-9
        dlab = dev.class_code.to_numpy()[dmask]
        n_col = int(round(h / STEP)) + 1
        times = np.arange(n_col) * STEP
        ph = per_h[h]
        for mode in MODES:
            if mode == "macro" and len(set(dlab.tolist())) < 5:
                dev_rows.append({"horizon": h, "weighting": mode, "skipped": "class missing in development cohort"})
                continue
            dw = mode_weights(np.ones((1, dmask.sum())), dlab, mode)[0]
            dev_acc = np.array([dw @ correct["dev"][s][dmask][:, OFFSET + n_col - 1] for s in ids])
            dev_rm = np.array([dw @ trajectory_components(correct["dev"][s][dmask][:, OFFSET:OFFSET + n_col], times)["delay"] for s in ids])
            sc = Scores(ph["arr"], ph["labels"], raw[:, ph["mask"]], mode)
            rm, acc, b_rm, b_acc = sc.point("delay"), sc.point("end"), sc.boot("delay"), sc.boot("end")
            for set_name, members in sets.items():
                sel = np.array([idx[s] for s in members])
                choice = selection_regret(members, dev_acc[sel], dev_rm[sel])
                ia, ir = idx[choice["selected_by_accuracy"]], idx[choice["selected_by_rmscd"]]
                dev_rows.append({
                    "horizon": h, "weighting": mode, "set": set_name, "n_dev": int(dmask.sum()),
                    "dev_clusters": int(dev.source_cluster_id[dmask].nunique()),
                    "dev_class_counts": json.dumps({int(k): int(v) for k, v in zip(*np.unique(dlab, return_counts=True))}),
                    **choice,
                    "test_rmscd_of_acc_choice": rm[ia], "test_rmscd_of_rmscd_choice": rm[ir],
                    "test_acc_of_acc_choice": acc[ia], "test_acc_of_rmscd_choice": acc[ir],
                    "test_delta_rmscd": float(rm[ia] - rm[ir]),
                    "test_delta_rmscd_lo": interval(b_rm[:, ia] - b_rm[:, ir])[0],
                    "test_delta_rmscd_hi": interval(b_rm[:, ia] - b_rm[:, ir])[1],
                    "test_delta_accuracy": float(acc[ir] - acc[ia]),
                    "test_delta_accuracy_lo": interval(b_acc[:, ir] - b_acc[:, ia])[0],
                    "test_delta_accuracy_hi": interval(b_acc[:, ir] - b_acc[:, ia])[1]})

    # ---- D per-class components (base classifiers and trivial references)
    LOGGER.info("D: per-class components")
    manifest = pd.read_csv(source / "data/manifest/manifest_real.csv")
    train_share = manifest[manifest.split == "train"].class_code.value_counts(normalize=True).sort_index().to_dict()
    d_rows, d_agg = [], []
    for h in [10.0, 4.0]:
        m = masks[f"E{h:g}"]
        lab = labels_all[m]
        n_col = int(round(h / STEP)) + 1
        times = np.arange(n_col) * STEP
        deltas = [d for d in [0.0, 1.0, 3.0, 5.0] if d < h - 1e-9]
        share_at = [d for d in [0.0, 5.0, h] if d <= h + 1e-9]
        per_sys = {}
        for s in sets["B"] + trivial:
            c = correct["test"][s][m][:, OFFSET:OFFSET + n_col]
            p = preds["test"][s][m][:, OFFSET:OFFSET + n_col]
            comp = trajectory_components(c, times)
            stable = np.logical_and.accumulate(c[:, ::-1], axis=1)[:, ::-1]
            per_sys[s] = comp
            for k in range(5):
                ink = lab == k
                row = {"system_id": s, "family": ctx.cards[s].get("family"), "round": ctx.meta.get(s, {}).get("library_round"),
                       "backbone": ctx.meta.get(s, {}).get("backbone"), "model_kind": ctx.meta.get(s, {}).get("model_kind"),
                       "horizon": h, "class_code": k, "n": int(ink.sum()), "train_share": train_share.get(k, 0.0),
                       "accuracy": float(comp["endpoint"][ink].mean()), "error_area": float(comp["error"][ink].mean()),
                       "retracted_area": float(comp["retracted"][ink].mean()), "rmscd": float(comp["delay"][ink].mean())}
                for d in deltas:
                    row[f"S_{d:g}"] = float(stable[ink, int(round(d / STEP))].mean())
                for d in share_at:
                    row[f"pred_share_{d:g}"] = float((p[:, int(round(d / STEP))] == k).mean())
                for j in range(5):
                    row[f"confusion_to_{j}"] = float((p[ink, -1] == j).mean())
                d_rows.append(row)
        # aggregates with bootstrap intervals
        base = sets["B"]
        groups = {"all27": base, "round1": [s for s in base if ctx.meta[s]["library_round"] == "R1"],
                  "round2": [s for s in base if ctx.meta[s]["library_round"] != "R1"]}
        rw = raw[:, m]
        for gname, members in groups.items():
            for k in range(5):
                ink = lab == k
                wk = rw[:, ink]
                rec = {"horizon": h, "group": gname, "n_systems": len(members), "class_code": k, "n": int(ink.sum()),
                       "train_share": train_share.get(k, 0.0)}
                for name, key in [("accuracy", "endpoint"), ("error_area", "error"), ("retracted_area", "retracted"), ("rmscd", "delay")]:
                    vals = np.stack([per_sys[s][key][ink].astype(float) for s in members], axis=1)
                    rec[name] = float(vals.mean(axis=0).mean())
                    reps = (wk @ vals) / wk.sum(axis=1, keepdims=True)
                    rec[f"{name}_lo"], rec[f"{name}_hi"], _ = interval(reps.mean(axis=1))
                d_agg.append(rec)
    d_frame = pd.DataFrame(d_rows)
    d_frame.to_csv(args.out / "D_per_class_systems.csv", index=False)
    base_rows = d_frame[d_frame.family.isin(["clip", "prefix"])]
    base_rows.groupby(["horizon", "round", "backbone", "model_kind", "class_code"]).mean(numeric_only=True).reset_index().to_csv(
        args.out / "D_per_class_configurations.csv", index=False)
    pd.DataFrame(d_agg).to_csv(args.out / "D_per_class_aggregates.csv", index=False)

    pd.DataFrame(system_rows).to_csv(args.out / "B_system_values.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(args.out / "B_rank_agreement.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(args.out / "C1_test_selection.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.out / "C2_split_selection.csv", index=False)
    pd.DataFrame(split_records).to_csv(args.out / "C2_split_records.csv", index=False)
    pd.DataFrame(dev_rows).to_csv(args.out / "C3_dev_selection.csv", index=False)
    write_json(args.out / "B_example_pair.json", example)

    # ---- reproduction of run002 system tables
    repro = {}
    sysdf = pd.DataFrame(system_rows)
    for h in HORIZONS:
        ref = pd.read_csv(args.run002 / f"systems_H{h:g}.csv")
        mine = sysdf[sysdf.horizon == h].set_index(["system_id", "weighting"])
        worst = 0.0
        for _, r in ref.iterrows():
            q = mine.loc[(r.system_id, r.weighting)]
            for ours, theirs in [("rmscd", "delay"), ("error_area", "error"), ("retracted_area", "retracted"),
                                 ("window_end_accuracy", "endpoint")]:
                worst = max(worst, abs(float(q[ours]) - float(r[theirs])))
        if worst > 1e-10:
            raise AssertionError(f"run002 system table not reproduced at H={h}: {worst}")
        repro[str(h)] = worst

    after = {f: sha256(source / f) for f in frozen}
    if after != before or any(sha256(Path(p["answers"])) != p["sha256"] for p in provenance) or \
            {f.name: sha256(f) for f in sorted(args.run002.glob("systems_H*.csv"))} != run002_files:
        raise AssertionError("An input changed during the run")
    write_json(args.out / "SUMMARY.json", {
        "bootstrap_redraws": rejected, "cohort_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "own_end_capped_share": capped_share, "run002_reproduction_max_abs_diff": repro,
        "identity_violations": identity_violations, "tied_pairs": a_result, "example_pair": example,
        "trivial_references": trivial})
    status.update(finished=datetime.now(timezone.utc).isoformat(), complete=True, inputs_unchanged=True,
                  prediction_hashes_rechecked=len(provenance))
    write_json(args.out / "run_state.json", status)
    LOGGER.info("Finished")


if __name__ == "__main__":
    main()
