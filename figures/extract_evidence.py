"""Collect every number used in the manuscript into internal/evidence_values.json.

Each entry: value, unit, source (file), field, identity (pool / split / version), status.
Sources are the frozen records of the derived-data package ($APE_DATA/accident/outputs, manifest)
and the MM-AU development records ($APE_DATA/mmau/deliverables); see paths.py.
Values are copied, not re-estimated, except where the entry says `derived` (then the
derivation is stated in `field`).
"""
import json

import numpy as np
import pandas as pd

from paths import *

EV = {}


def put(key, value, unit, source, field, identity, status="record"):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    src = placeholder(source)  # release: <APE_FIG_OUT>, <APE_DATA>, <APE_ROOT> instead of local roots
    EV[key] = dict(value=value, unit=unit, source=src, field=field, identity=identity, status=status)


def J(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


ACC = "ACCIDENT real, manifest v2, test audit split"
PRIM = "K-b primary pool (184 answer sets, no VLM), frozen pi0 1.0-S1"

# ---------------------------------------------------------------- data
man = pd.read_csv(MANIFEST, dtype={"video_id": str})
put("acc.n_clips", len(man), "clips", MANIFEST, "rows", "ACCIDENT real manifest v2")
put("acc.n_clusters", man.source_cluster_id.nunique(), "source-video clusters", MANIFEST, "nunique(source_cluster_id)", "manifest v2")
for s in ["train", "dev", "test"]:
    put(f"acc.split.{s}", int((man.split == s).sum()), "clips", MANIFEST, f"split=={s}", "manifest v2")
    put(f"acc.split_clusters.{s}", int(man[man.split == s].source_cluster_id.nunique()), "clusters", MANIFEST, f"split=={s}", "manifest v2")
put("acc.fps_min", float(man.fps.min()), "fps", MANIFEST, "min(fps)", "manifest v2")
put("acc.fps_max", float(man.fps.max()), "fps", MANIFEST, "max(fps)", "manifest v2")
put("acc.dur_median", float(man.duration_s.median()), "s", MANIFEST, "median(duration_s)", "manifest v2")
put("acc.dur_min", float(man.duration_s.min()), "s", MANIFEST, "min(duration_s)", "manifest v2")
put("acc.dur_max", float(man.duration_s.max()), "s", MANIFEST, "max(duration_s)", "manifest v2")
dev = man[man.split == "dev"]
q = np.quantile(dev.post_anchor_length_s, [0.25, 0.5, 0.75])
put("acc.dev_Lplus_quantiles", [round(float(x), 2) for x in q], "s", MANIFEST, "quantile(dev L+, .25/.5/.75), numpy linear", "dev 102 clips", "derived")
test = man[man.split == "test"]
put("acc.test_Lplus_median", float(test.post_anchor_length_s.median()), "s", MANIFEST, "median(test L+)", ACC, "derived")
put("acc.class_names", sorted(man.class_name.unique().tolist()), "", MANIFEST, "class_name", "manifest v2")
for H in [4.0, 10.0, 21.5]:
    e = test[test.post_anchor_length_s >= H - 1e-9]
    put(f"acc.N_H.{H}", len(e), "clips", MANIFEST, "test & L+>=H-1e-9", ACC, "derived")
    put(f"acc.N_H_clusters.{H}", e.source_cluster_id.nunique(), "clusters", MANIFEST, "nunique cluster in E_H", ACC, "derived")
    put(f"acc.class_counts.{H}", e.class_name.value_counts().to_dict(), "clips", MANIFEST, "class counts in E_H", ACC, "derived")
put("acc.class_counts.test", test.class_name.value_counts().to_dict(), "clips", MANIFEST, "class counts test", ACC, "derived")
put("acc.class_counts.train", man[man.split == "train"].class_name.value_counts().to_dict(), "clips", MANIFEST, "class counts train", "manifest v2", "derived")
g0 = J(REPO / "report/text/g0_length_shift_stats.json")
put("acc.official_Lplus_stats", g0, "s/fraction", REPO / "report/text/g0_length_shift_stats.json", "all", "official IID 507/1520 split (before leak repair)")

# ---------------------------------------------------------------- library
st = pd.read_csv(INTERNAL / "system_table_reference.csv")
t10 = st[st.h_s == 10.0]
fam = t10.groupby(["library_round", "family"]).size().to_dict()
put("lib.family_counts", {f"{k[0]}|{k[1]}": int(v) for k, v in fam.items()}, "answer sets", REMOTE04 / "r2b/metrics/8ac32aae418b", "card family x library_round", PRIM)
nt = t10[~t10.family.isin(["block", "trivial"])]
put("lib.n_nontrivial", len(nt), "systems", REMOTE04 / "r2b/metrics/8ac32aae418b", "family not in block/trivial", PRIM)
base = nt[nt.family.isin(["clip", "prefix"])]
for rnd in ["R1", "R2"]:
    b = base[base.library_round == rnd]
    put(f"lib.base_end_macro_range.{rnd}", [float(b.end_macro.min()), float(b.end_macro.max())], "macro-Acc",
        REMOTE04 / "r2b/metrics/8ac32aae418b", "end_window_macro_acc of clip/prefix families @H=10", PRIM)
put("lib.nontrivial_end_macro_range", [float(nt.end_macro.min()), float(nt.end_macro.max())], "macro-Acc",
    REMOTE04 / "r2b/metrics/8ac32aae418b", "end_window_macro_acc, 168 systems @H=10", PRIM)
put("lib.nontrivial_RMSCD_range", [float(nt.RMSCD.min()), float(nt.RMSCD.max())], "s",
    REMOTE04 / "r2b/metrics/8ac32aae418b", "RMSCD, 168 systems @H=10", PRIM)
put("lib.max_missing_lookup_rate_ref", float(st.missing_lookup_rate.max()), "fraction", REMOTE04 / "r2b/metrics",
    "max missing_lookup_rate over 552 records", PRIM)
put("lib.delay_term_median_iqr", [float(nt.delay_term.median()), float(nt.delay_term.quantile(.25)), float(nt.delay_term.quantile(.75))], "s",
    INTERNAL / "system_table_reference.csv", "D@10 = trapezoid(S_H(H)-S_H(delta)); median, q25, q75 over 168", PRIM, "derived")
put("lib.SH_rise_median", float((nt.SH - nt.S0).median()), "fraction", INTERNAL / "system_table_reference.csv",
    "median S_H(10)-S_H(0) over 168", PRIM, "derived")
noncommit = nt[nt.family != "commit"]
put("lib.noncommit_delay_median", float(noncommit.delay_term.median()), "s", INTERNAL / "system_table_reference.csv",
    "median D@10 over non-commit nontrivial", PRIM, "derived")
put("lib.commit_delay_median", float(nt[nt.family == "commit"].delay_term.median()), "s", INTERNAL / "system_table_reference.csv",
    "median D@10 over commit-rule systems", PRIM, "derived")
tp = t10.set_index("system_id").loc[["clip__r18mean__seed20260903", "postproc__ema0p7__prefix__gru512__seed20260903"]]
cols = ["end_macro", "end_macro_lo", "end_macro_hi", "end_micro", "RMSCD", "RMSCD_lo", "RMSCD_hi", "S0", "SH", "level_term",
        "delay_term", "median_flips", "mean_flips"]
put("teaser.pair", tp[cols].round(6).to_dict(orient="index"), "mixed", REMOTE04 / "r2b/metrics/8ac32aae418b", "per-system records", PRIM)
blk = t10[t10.family == "block"].set_index("system_id")[["RMSCD", "end_macro", "median_flips", "S0", "SH"]]
put("blocks.H10", blk.round(6).to_dict(orient="index"), "mixed", REMOTE04 / "r2b/metrics/8ac32aae418b", "block records", PRIM)
triv = t10[t10.family == "trivial"].set_index("system_id")[["RMSCD", "end_macro", "median_flips"]]
put("trivial.H10", triv.round(6).to_dict(orient="index"), "mixed", REMOTE04 / "r2b/metrics/8ac32aae418b", "trivial records", PRIM)

# ---------------------------------------------------------------- table 3 (primary) and secondary
for H, ph in HASH.items():
    cal = J(REMOTE04 / f"r2b/calib_plaus/{ph}/calibration.json")
    for m in METRICS:
        c = cal["metrics"][m]
        put(f"t3.{H}.{m}", dict(min_R=c["min_R"], MRD=c["MRD_plaus"], ruler=c["ruler"], P=c["n_pairs"], verdict=c["verdict"],
                                max_b=c["max_b"], max_s=c["max_s"], min_R_full=c["min_R_M_full"], MRD_full=c["MRD_full_grid"],
                                eps_max=c["eps_max"]["eps_max"], eps_saturated=c["eps_max"]["saturated"]),
            "mixed", REMOTE04 / f"r2b/calib_plaus/{ph}/calibration.json", f"metrics['{m}']", PRIM + f"; H={H}; 54-point plausible product")
        cb = cal["metrics"][m + " [with blocks]"]
        put(f"t3blocks.{H}.{m}", dict(min_R=cb["min_R"], MRD=cb["MRD_plaus"]), "mixed", REMOTE04 / f"r2b/calib_plaus/{ph}/calibration.json",
            f"metrics['{m} [with blocks]']", PRIM + "; blocks included")
    ax = J(REMOTE04 / f"r2b/calib/{ph}/calibration.json")
    for m in METRICS:
        c = ax["metrics"][m]
        put(f"axis.{H}.{m}", dict(min_R=c["min_R_M_full"], MRD_full=c["MRD_full_grid"], eps_max=c["eps_max"]["eps_max"],
                                  eps_saturated=c["eps_max"]["saturated"], scanned=c["eps_max"]["scanned_max_abs_eps"]),
            "mixed", REMOTE04 / f"r2b/calib/{ph}/calibration.json", f"metrics['{m}'] (axis scan, full grid 14 points)", PRIM)
sec = pd.read_csv(R2C_TABLES / "table3_characterization.csv")
put("t3.secondary_H10", sec[["Metric", "Min R [secondary +VLM]", "MRD plaus [secondary +VLM]", "Ruler [secondary +VLM]",
                             "Verdict [secondary +VLM]"]].to_dict(orient="records"),
    "mixed", R2C_TABLES / "table3_characterization.csv", "secondary columns", "K-c secondary pool (185, with VLM), H=10")
put("t3.knob_decomposition", J(INTERNAL / "knob_decomposition.json"), "mixed", INTERNAL / "knob_decomposition.json",
    "same-H vs all plausible points", PRIM, "derived (post hoc)")
kdt = pd.read_csv(TAB / "knob_decomposition.csv")
put("t3.knob_decomposition_groups", kdt.round(6).to_dict(orient="records"), "mixed", TAB / "knob_decomposition.csv",
    "min R and q95 shift by knob group", PRIM, "derived (post hoc)")

# ---------------------------------------------------------------- gates
g2 = J(REMOTE04 / "r2b/gates/g2.json")
put("g2.pass", g2.get("pass_gate"), "bool", REMOTE04 / "r2b/gates/g2.json", "pass_gate", PRIM)
g3 = J(REMOTE04 / "r2c/gates/g3_v5.json")
for rule in ["original_rule", "v4_23p5", "v5_22p0"]:
    for bh in g3["primary"][rule]["by_h"]:
        H = float(bh["h_s"])
        rec = dict(groups=bh.get("groups"), pass_gate=bh.get("pass_gate"), status=bh.get("status"),
                   cuts=bh.get("dev_tercile_cuts_s"), n_dev=bh.get("n_dev_eligible"), n_test=bh.get("n_test_eligible"))
        if rule == "v5_22p0":
            rows = {r["system_id"]: r for r in bh["rows"]}
            for sid in ["block__rand__eta-0.5", "block__rand__eta-0.9"]:
                r = rows[sid]
                rec[sid] = dict(own=r["own"]["gap_long_minus_short"], own_ci=[r["own"]["ci_lo"], r["own"]["ci_hi"]],
                                fixed=r["fixed"]["gap_long_minus_short"], fixed_ci=[r["fixed"]["ci_lo"], r["fixed"]["ci_hi"]])
            rec["oracle_controls"] = bh.get("oracle_negative_controls")
            rec["strata_counts"] = rows["block__rand__eta-0.5"]["own"]["counts"]
            real = [r for r in bh["rows"] if r["role"] == "real"]
            rec["real_own_gap_median"] = float(np.median([r["own"]["gap_long_minus_short"] for r in real]))
            rec["real_fixed_gap_median"] = float(np.median([r["fixed"]["gap_long_minus_short"] for r in real]))
            rec["real_abs_fixed_gap_median"] = float(np.median([abs(r["fixed"]["gap_long_minus_short"]) for r in real]))
            rec["real_max_missing_visible"] = int(max(r["coverage"]["n_missing_visible"] for r in real))
        put(f"g3.{rule}.{H}", rec, "mixed", REMOTE04 / "r2c/gates/g3_v5.json", f"primary.{rule}.by_h[h={H}]", PRIM)
put("g3.record_rule", g3.get("record_rule"), "", REMOTE04 / "r2c/gates/g3_v5.json", "record_rule", PRIM)

g4 = J(REMOTE04 / "r2b/gates/g4.json")
for bh in g4["by_h"]:
    H = float(bh["h_s"])
    put(f"g4.{H}", dict(tied=bh["n_tied"], heterogeneous=bh["n_heterogeneous"], cross_base=bh["n_cross_base_heterogeneous"],
                        replicated_groups=bh["n_replicated_cross_base_groups"], seed_groups=len(bh["seed_groups"])),
        "pairs/groups", REMOTE04 / "r2b/gates/g4.json", f"by_h[h={H}]", PRIM + "; 168 non-trivial systems")
    if H == 10.0:
        pair = {"clip__r18mean__seed20260903", "postproc__ema0p7__prefix__gru512__seed20260903"}
        ex = [p for p in bh["pairs"] if {p["system_a"], p["system_b"]} == pair]
        put("g4.teaser_pair_record", ex, "mixed", REMOTE04 / "r2b/gates/g4.json", "pairs entry for teaser", PRIM)
put("g4.tied_decomposition", J(INTERNAL / "tied_pairs_decomposition.json"), "mixed", INTERNAL / "tied_pairs_decomposition.json",
    "level/delay split", PRIM, "derived (post hoc)")
put("g4.composition", J(INTERNAL / "g4_composition.json"), "mixed", INTERNAL / "g4_composition.json", "by family", PRIM, "derived (post hoc)")


def tab(name):
    return pd.read_csv(R2C_TABLES / "primary" / f"{name}.csv")


for name in ["tableS_A1_reversals", "tableS_A3_prefix_accuracy", "tableS_A4_region", "tableS_A5_region", "tableS_Pb_jitter",
             "tableS_G2_two_arms", "tableS_G3_two_rules"]:
    put(f"sup.{name}", tab(name).to_dict(orient="records"), "mixed", R2C_TABLES / "primary" / f"{name}.csv", "all rows", PRIM)
pc = tab("tableS_Pc_stride")
put("sup.Pc_random_blocks", pc[pc.System.str.startswith("block__rand")].to_dict(orient="records"), "flips",
    R2C_TABLES / "primary/tableS_Pc_stride.csv", "block__rand rows", PRIM)
g5 = tab("tableS_G5_response_shape")
g5s = g5[(g5.H == 10.0) & (g5.Metric == "RMSCD@H") & (g5["Block family"] == "all_blocks")]
put("g5.H10_RMSCD_all_blocks", g5s.to_dict(orient="records"), "corr", R2C_TABLES / "primary/tableS_G5_response_shape.csv",
    "H=10, RMSCD, all_blocks", PRIM, "descriptive (no threshold)")
t1 = pd.read_csv(R2C_TABLES / "table1_audit_H10p00.csv")
v = t1[t1.System.str.contains("qwen")]
put("vlm.H10", v.to_dict(orient="records"), "mixed", R2C_TABLES / "table1_audit_H10p00.csv", "row r2__qwen25vl7b__readonly",
    "K-c secondary pool")

# ---------------------------------------------------------------- MM-AU development
dm = MMAU
inp = J(dm / "input_stride_20260921/RESULTS.json")
anc = J(dm / "anchor_shift_20260921/RESULTS.json")
ctl = J(dm / "control_pilot_20260920/RESULTS.json")
sen = J(dm / "overlap_dense_20260920/sensitivity/summary.json")
MM = "MM-AU native nine-class development cohort (269 dev clips, frame units, dev reused for selection)"
put("mm.stride.cells", [dict(arm=c["arm"], cond=c["condition"], frames=c.get("input_frames"), macro=c["metrics"]["end_window_macro_acc"],
                             micro=c["metrics"]["end_window_micro_acc"], rmscd_macro=c["metrics"]["RMSCD@H_norm_macro"],
                             rmscd=c["metrics"]["RMSCD@H_norm"], flips=c["metrics"]["mean_flips"]) for c in inp["cells"]],
    "mixed", dm / "input_stride_20260921/RESULTS.json", "cells", MM + "; H64, output step 8")
put("mm.stride.paired", inp["paired"], "mixed", dm / "input_stride_20260921/RESULTS.json", "paired", MM)
put("mm.anchor.cells", [dict(arm=c["arm"], shift=c["shift"], macro=c["metrics"]["end_window_macro_acc"],
                             rmscd_macro=c["metrics"]["RMSCD@H_norm_macro"], micro=c["metrics"]["end_window_micro_acc"]) for c in anc["cells"]],
    "mixed", dm / "anchor_shift_20260921/RESULTS.json", "cells", MM + "; H60")
put("mm.anchor.paired", anc["paired"], "mixed", dm / "anchor_shift_20260921/RESULTS.json", "paired", MM)
put("mm.controls", {k: {kk: vv for kk, vv in v.items() if kk != "class_recalls"} for k, v in ctl["arms"].items()}, "mixed",
    dm / "control_pilot_20260920/RESULTS.json", "arms", MM + "; H67")
put("mm.readgrid.delta8_vs1", sen["delta8_vs1"], "mixed", dm / "overlap_dense_20260920/sensitivity/summary.json", "delta8_vs1",
    MM + "; output reading grid only")
put("mm.readgrid.reversals", dict(n_cells=len(sen["macro_rmscd_pair_reversal_cells"]),
                                  pairs=sorted({tuple(sorted(c["arms"])) for c in sen["macro_rmscd_pair_reversal_cells"]}),
                                  h_reversals=len(sen["horizon32_to64_pair_reversals"])), "count",
    dm / "overlap_dense_20260920/sensitivity/summary.json", "macro_rmscd_pair_reversal_cells", MM)

(INTERNAL / "evidence_values.json").write_text(json.dumps(EV, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
print("entries", len(EV))
for k in ["acc.dev_Lplus_quantiles", "acc.N_H.10.0", "acc.N_H_clusters.10.0", "acc.fps_min", "acc.fps_max", "acc.dur_median",
          "lib.family_counts", "lib.base_end_macro_range.R1", "lib.base_end_macro_range.R2", "lib.max_missing_lookup_rate_ref",
          "lib.delay_term_median_iqr", "lib.noncommit_delay_median", "lib.commit_delay_median", "teaser.pair", "g3.v5_22p0.10.0",
          "g4.teaser_pair_record", "vlm.H10", "acc.class_counts.10.0", "acc.test_Lplus_median", "blocks.H10"]:
    print(k, json.dumps(EV[k]["value"], default=str)[:1200])
