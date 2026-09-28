"""Regenerate the numbers of the re-analysis tables from the released derived data (no new analysis).

The values are read from the stored outputs of the method re-analysis (run002), the stage-2 simulations
and training comparison, and the M3 re-analysis, and are formatted exactly as printed in the paper.
The selection of rows and the rounding follow the numeric checks used when the tables were filled.

  manuscript label           source under $APE_DATA/accident/outputs
  tab:base-process           run002/systems_H10.csv (macro weights, families clip/prefix, mean over seeds)
  tab:per-class              m3_20260928_r2/D_per_class_aggregates.csv, D_per_class_systems.csv
  tab:selection              m3_20260928_r2/B_rank_agreement.csv (rmscd_vs_ct_fixed), C2_split_selection.csv
  tab:process-protocols      run002/robust_H{10,4}_{macro,micro}.csv, run002/robust_summary.json
  tab:counterexample         ape.method_analysis.trajectory_components on the two 2x5 example matrices
  tab:process-calibration    stage2_exact_bootstrap/, stage2_final_calibration/, stage2_calibration_v2/ SUMMARY.json
  tab:process-training       stage2_training_v2/primary_summary.csv

Outputs in $APE_FIG_OUT/tables/: paper_<name>.csv (raw and formatted values), paper_<name>_rows.tex
(table body rows as printed) and paper_numbers.json (everything).
"""
import json
import sys

import numpy as np
import pandas as pd

from paths import REMOTE04, ROOT, TAB, placeholder

sys.path.insert(0, str(ROOT))
from ape.method_analysis import trajectory_components  # noqa: E402

RUN002 = REMOTE04 / "run002"
M3 = REMOTE04 / "m3_20260928_r2"
RESULT = {}


def J(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def emit(name, df, rows, sources):
    df.to_csv(TAB / f"paper_{name}.csv", index=False)
    (TAB / f"paper_{name}_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    RESULT[name] = dict(sources=[placeholder(s) for s in sources], rows_tex=rows,
                     records=json.loads(df.to_json(orient="records")))
    print(f"{name}: {len(rows)} rows")


f1 = lambda x: f"{x:.1f}"  # noqa: E731
f2 = lambda x: f"{x:.2f}"  # noqa: E731

# ----------------------------------------------------------------- tab:base-process (27 base classifiers)
src = RUN002 / "systems_H10.csv"
data = pd.read_csv(src)
primary = data[(data.weighting == "macro") & data.family.isin(["clip", "prefix"])]
assert len(primary) == 27
g = primary.groupby(["library_round", "backbone", "model_kind"])[["endpoint", "error", "retracted", "delay"]].mean()
LAB = {"mean": "Prefix mean", "gru128": "GRU-128", "gru512": "GRU-512"}
recs, rows = [], []
for (rnd, bb) in [("R1", "r18"), ("R2", "r18"), ("R2", "clipb16")]:
    for kind in ["mean", "gru128", "gru512"]:
        r = g.loc[(rnd, bb, kind)]
        cells = [f1(100 * r.endpoint), f2(r.error), f2(r.retracted), f2(r.delay)]
        recs.append(dict(library_round=rnd, backbone=bb, model_kind=kind, endpoint=r.endpoint, error=r.error,
                         retracted=r.retracted, delay=r.delay, A_H_pct=cells[0], E_H=cells[1], R_H=cells[2], RMSCD=cells[3]))
        rows.append(" & ".join([LAB[kind]] + cells) + r" \\")
emit("base_process", pd.DataFrame(recs), rows, [src])

# ----------------------------------------------------------------- tab:per-class
agg_src, sys_src = M3 / "D_per_class_aggregates.csv", M3 / "D_per_class_systems.csv"
agg = pd.read_csv(agg_src)
syst = pd.read_csv(sys_src)
CLS = ["head-on", "rear-end", "t-bone", "sideswipe", "single"]
recs, rows = [], []
for H in [10.0, 4.0]:
    a = agg[(agg.horizon == H) & (agg.group == "all27")].set_index("class_code")
    r1 = agg[(agg.horizon == H) & (agg.group == "round1")].set_index("class_code")
    r2 = agg[(agg.horizon == H) & (agg.group == "round2")].set_index("class_code")
    s = syst[(syst.horizon == H) & syst.family.isin(["clip", "prefix"])]  # the 27 base classifiers
    assert s.system_id.nunique() == 27
    pred_col = f"pred_share_{H:g}"
    n_h = int(a.n.sum())
    for k, name in enumerate(CLS):
        x = a.loc[k]
        pred = float(s[s.class_code == k][pred_col].mean())
        cells = [f1(100 * x.n / n_h), f1(100 * pred),
                 f"{f1(100 * x.accuracy)} ({f1(100 * x.accuracy_lo)}--{f1(100 * x.accuracy_hi)})",
                 f2(x.error_area), f2(x.retracted_area), f"{f2(x.rmscd)} ({f2(x.rmscd_lo)}--{f2(x.rmscd_hi)})",
                 f1(100 * r1.loc[k].accuracy), f1(100 * r2.loc[k].accuracy)]
        recs.append(dict(horizon=H, N_H=n_h, class_code=k, class_name=name, n=int(x.n), share=x.n / n_h,
                         predicted_share=pred, accuracy=x.accuracy, accuracy_lo=x.accuracy_lo, accuracy_hi=x.accuracy_hi,
                         error_area=x.error_area, retracted_area=x.retracted_area, rmscd=x.rmscd, rmscd_lo=x.rmscd_lo,
                         rmscd_hi=x.rmscd_hi, accuracy_round1=r1.loc[k].accuracy, accuracy_round2=r2.loc[k].accuracy,
                         train_share=x.train_share, formatted=" | ".join(cells)))
        rows.append(" & ".join([name] + cells) + r" \\")
emit("per_class", pd.DataFrame(recs), rows, [agg_src, sys_src])
RESULT["per_class"]["train_share_pct"] = [round(100 * v, 1) for v in agg[(agg.horizon == 10.0) & (agg.group == "all27")]
                                       .sort_values("class_code").train_share]

# ----------------------------------------------------------------- tab:selection
rk_src, c2_src = M3 / "B_rank_agreement.csv", M3 / "C2_split_selection.csv"
rk = pd.read_csv(rk_src)
rk = rk[rk.comparison == "rmscd_vs_ct_fixed"].copy()  # the file also holds rows with horizon "C4"
rk["horizon"] = rk["horizon"].astype(float)
c2 = pd.read_csv(c2_src)
WLAB = {"macro": "class-macro", "micro": "clip-weighted"}
recs, rows = [], []
for H in [4.0, 10.0, 21.5]:
    for w in ["macro", "micro"]:
        for i, st in enumerate(["B", "N", "A"]):
            a = rk[(rk.horizon == H) & (rk.weighting == w) & (rk["set"] == st)].iloc[0]
            b = c2[(c2.horizon == H) & (c2.weighting == w) & (c2["set"] == st)].iloc[0]
            da = 100 * b.delta_accuracy_mean
            cells = [f"{a.kendall_tau:.2f} ({a.tau_lo:.2f}--{a.tau_hi:.2f})", f1(100 * a.all_share_opposite),
                     f2(a.all_median_abs_dx_opposite), f1(100 * b.frequency_different),
                     f"{f2(b.delta_rmscd_median)} ({f2(b.delta_rmscd_mean)})", f"${da:+.2f}$"]
            recs.append(dict(horizon=H, weighting=w, set=st, n_systems=int(a.n_systems),
                             n_undefined_systems=int(a.n_undefined_systems), kendall_tau=a.kendall_tau, tau_lo=a.tau_lo,
                             tau_hi=a.tau_hi, share_opposite=a.all_share_opposite,
                             median_abs_drmscd_opposite=a.all_median_abs_dx_opposite, evaluations=int(b.evaluations),
                             frequency_different=b.frequency_different, delta_rmscd_median=b.delta_rmscd_median,
                             delta_rmscd_mean=b.delta_rmscd_mean, delta_accuracy_mean=b.delta_accuracy_mean,
                             formatted=" | ".join(cells)))
            lead = [f"{H:g}" if (w == "macro" and i == 0) else "", WLAB[w] if i == 0 else "", str(int(a.n_systems))]
            rows.append(" & ".join(lead + cells) + r" \\")
emit("selection", pd.DataFrame(recs), rows, [rk_src, c2_src])

# ----------------------------------------------------------------- tab:process-protocols
rs_src = RUN002 / "robust_summary.json"
rs = {float(r["horizon"]): r for r in J(rs_src)}
recs, rows, srcs = [], [], [rs_src]
for H in [10, 4]:
    for w in ["macro", "micro"]:
        p = RUN002 / f"robust_H{H}_{w}.csv"
        srcs.append(p)
        df = pd.read_csv(p)
        for j, eps in enumerate([0.0, 0.125, 0.25]):
            counts = df[df.epsilon == eps].status.value_counts().to_dict()
            assert not counts.get("reversal", 0), counts
            robust = counts.get("robust_a", 0) + counts.get("robust_b", 0)
            spec, insuf = counts.get("protocol_specific", 0), counts.get("inconclusive", 0)
            recs.append(dict(horizon=H, weighting=w, margin=eps, robust=robust, protocol_specific=spec,
                             insufficient=insuf, reversal=counts.get("reversal", 0), cohort_n=rs[H]["n"],
                             cohort_clusters=rs[H]["n_clusters"], critical=rs[H]["weightings"][w]["critical"]))
            lead = [str(H) if (w == "macro" and j == 0) else "", WLAB[w] if j == 0 else "", f"{eps:g}"]
            rows.append(" & ".join(lead + [str(robust), str(spec), str(insuf)]) + r" \\")
emit("process_protocols", pd.DataFrame(recs), rows, srcs)
RESULT["process_protocols"]["notes"] = {f"H{H:g}": dict(clips=rs[H]["n"], clusters=rs[H]["n_clusters"],
                                                     critical_macro=round(rs[H]["weightings"]["macro"]["critical"], 2),
                                                     critical_micro=round(rs[H]["weightings"]["micro"]["critical"], 2))
                                     for H in (10.0, 4.0)}

# ----------------------------------------------------------------- tab:counterexample
counter_a = np.array([[0, 0, 0, 1, 1], [0, 1, 1, 0, 1]], dtype=bool)
counter_b = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 0, 1]], dtype=bool)
np.testing.assert_array_equal(counter_a.mean(0), counter_b.mean(0))
np.testing.assert_array_equal((counter_a[:, 1:] != counter_a[:, :-1]).sum(1), (counter_b[:, 1:] != counter_b[:, :-1]).sum(1))
recs, rows, means = [], [], []
for name, x in [("A", counter_a), ("B", counter_b)]:
    comp = trajectory_components(x, np.arange(5))
    for i in range(2):
        cells = [f1(comp["error"][i]), f1(comp["retracted"][i]), f1(comp["delay"][i])]
        recs.append(dict(system=name, clip=str(i + 1), c=" ".join(str(int(v)) for v in x[i]), e=comp["error"][i],
                         r=comp["retracted"][i], d=comp["delay"][i]))
        rows.append(" & ".join([name, str(i + 1), r"\,".join(str(int(v)) for v in x[i])] + cells) + r" \\")
    means.append((name, comp))
for name, comp in means:
    recs.append(dict(system=name, clip="mean", c="", e=comp["error"].mean(), r=comp["retracted"].mean(), d=comp["delay"].mean()))
    rows.append(" & ".join([name, "mean", "", f1(comp["error"].mean()), f1(comp["retracted"].mean()),
                            f1(comp["delay"].mean())]) + r" \\")
assert [float(c["delay"].mean()) for _, c in means] == [3.0, 2.5]
emit("counterexample", pd.DataFrame(recs), rows, [ROOT / "ape/method_analysis.py"])

# ----------------------------------------------------------------- tab:process-calibration
eb_src = REMOTE04 / "stage2_exact_bootstrap/SUMMARY.json"
fc_src = REMOTE04 / "stage2_final_calibration/SUMMARY.json"
fin_src = REMOTE04 / "stage2_calibration_v2/SUMMARY.json"
emp = [J(eb_src), J(fc_src)]
FAM = [(0, "decomposition_H10", r"Components, $H=10$\,s"), (1, "decomposition_H10", r"Components, $H=10$\,s"),
       (1, "decomposition_H4", r"Components, $H=4$\,s"), (1, "protocol_H10", r"Nine protocols, $H=10$\,s"),
       (1, "protocol_H4", r"Nine protocols, $H=4$\,s")]
recs, rows = [], []
for i, family, lab in FAM:
    s = emp[i]
    mm = [next(x for x in s["summaries"] if x["family"] == family and x["weighting"] == w) for w in ["macro", "micro"]]
    cells = [str(mm[0]["coordinates"]), str(mm[0]["clusters"]), str(s["bootstrap"]), str(s["trials"])]
    cells += [f"{f1(100 * m['covered'])} ({f2(100 * m['covered_mcse'])})" for m in mm]
    cells.append(" / ".join(f1(100 * m["wrong_direction_fraction"]) for m in mm))
    recs.append(dict(run=["stage2_exact_bootstrap", "stage2_final_calibration"][i], family=family,
                     coordinates=mm[0]["coordinates"], clusters=mm[0]["clusters"], bootstrap=s["bootstrap"],
                     samples=s["trials"], covered_macro=mm[0]["covered"], mcse_macro=mm[0]["covered_mcse"],
                     covered_micro=mm[1]["covered"], mcse_micro=mm[1]["covered_mcse"],
                     wrong_macro=mm[0]["wrong_direction_fraction"], wrong_micro=mm[1]["wrong_direction_fraction"]))
    rows.append(" & ".join([lab] + cells) + r" \\")
fin = J(fin_src)
# Cluster counts are design parameters of the finite-state scenarios (code/scripts/stage2_calibration.py);
# they are not stored in SUMMARY.json and are taken from the scenario definition, as in the table check.
SCEN = [("ordinary", "Ordinary", 300), ("small_support", "Small support", 60), ("imbalanced", "Imbalanced classes", 120),
        ("rare_late_error", "Rare late error", 300)]
notes = {}
for scen, lab, ncl in SCEN:
    rr = {w: next(x for x in fin["summaries"] if x["scenario"] == scen and x["weighting"] == w) for w in ["macro", "micro"]}
    cells = [str(fin["coordinates"]), str(ncl), str(fin["bootstrap"]), str(rr["macro"]["trials"])]
    for w in ["macro", "micro"]:
        x = rr[w]
        if x["estimable"] < x["trials"] and x["covered_fraction_conditional"] == 1.0:
            cells.append(f1(100 * x["covered_fraction_conditional"]) + r"\tnote{" + ("a" if scen == "small_support" else "b") + "}")
            notes[scen] = dict(estimable=x["estimable"], trials=x["trials"], mean_width=round(x["mean_width"], 1))
        else:
            cells.append(f"{f1(100 * x['covered_fraction_conditional'])} ({f2(100 * x['covered_mcse'])})")
    cells.append("--")
    recs.append(dict(run="stage2_calibration_v2", family=scen, coordinates=fin["coordinates"], clusters=ncl,
                     bootstrap=fin["bootstrap"], samples=rr["macro"]["trials"],
                     covered_macro=rr["macro"]["covered_fraction_conditional"], mcse_macro=rr["macro"]["covered_mcse"],
                     covered_micro=rr["micro"]["covered_fraction_conditional"], mcse_micro=rr["micro"]["covered_mcse"],
                     estimable_macro=rr["macro"]["estimable"], mean_width_macro=rr["macro"]["mean_width"]))
    rows.append(" & ".join([lab] + cells) + r" \\")
emit("process_calibration", pd.DataFrame(recs), rows, [eb_src, fc_src, fin_src])
RESULT["process_calibration"]["notes"] = notes

# ----------------------------------------------------------------- tab:process-training
tr_src = REMOTE04 / "stage2_training_v2/primary_summary.csv"
tr = pd.read_csv(tr_src)
ARMS = [("mean_ce", "Prefix mean, CE"), ("gru_ce", "GRU-128, CE"), ("gru_ema0.5", r"GRU-128, CE + EMA ($\alpha=0.5$)"),
        ("gru_ema0.25", r"GRU-128, CE + EMA ($\alpha=0.25$)"), ("gru_ema0.1", r"GRU-128, CE + EMA ($\alpha=0.1$)"),
        ("gru_ema0.05", r"GRU-128, CE + EMA ($\alpha=0.05$)"), ("gru_time", "GRU-128, time-weighted CE"),
        ("gru_suffix", "GRU-128, suffix-max CE"), ("transformer_ce", "Causal Transformer, CE")]
recs, rows = [], []
for lr in [0.0003, 0.001]:
    for arm, lab in ARMS:
        r = tr[(tr.lr == lr) & (tr.arm == arm)].iloc[0]
        cells = []
        for field in ["endpoint", "error", "retracted", "delay"]:
            sc, fmt = (100, f1) if field == "endpoint" else (1, f2)
            cells.append(f"{fmt(sc * r[field + '_mean'])} ({fmt(sc * r[field + '_min'])}--{fmt(sc * r[field + '_max'])})")
        recs.append(dict(lr=lr, arm=arm, **{c: r[c] for c in tr.columns if c not in ("lr", "arm")}, formatted=" | ".join(cells)))
        rows.append(" & ".join([lab] + cells) + r" \\")
emit("process_training", pd.DataFrame(recs), rows, [tr_src])

(TAB / "paper_numbers.json").write_text(json.dumps(RESULT, indent=1, default=float), encoding="utf-8")
print("wrote", placeholder(TAB / "paper_numbers.json"))
