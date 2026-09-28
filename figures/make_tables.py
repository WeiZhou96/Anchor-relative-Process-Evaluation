"""Generate the numeric LaTeX tables of the manuscript from the frozen records.

Every number is read from files (manifest, K-b/K-c records, R2c tables, MM-AU development
records, and the derived internal JSON files); nothing is typed by hand.
Output: $APE_FIG_OUT/tables/*.tex (tabular bodies wrapped in CAS table environments).
"""
import json

import numpy as np
import pandas as pd

from paths import HASH, INTERNAL, MANIFEST, METRICS, MMAU, R2C_TABLES, REMOTE04, TAB


def J(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


E = J(INTERNAL / "evidence_values.json")
V = lambda k: E[k]["value"]  # noqa: E731
CLASSES = ["head-on", "rear-end", "t-bone", "sideswipe", "single"]
MLAB = {"RMSCD@H": r"RMSCD@$H$ (s)", "S_H@1": r"$S_H(1\,\mathrm{s})$", "S_H@3": r"$S_H(3\,\mathrm{s})$",
        "end_window_macro_acc": "window-end macro-Acc", "median_flips": "median flips"}


def f3(x):
    return f"{x:.3f}"


def write(name, text):
    (TAB / f"{name}.tex").write_text(text, encoding="utf-8")
    print("wrote", name)


# ------------------------------------------------------------------ Table: data and cohorts
man = pd.read_csv(MANIFEST)
rows = []
for lab, sub in [("train", man[man.split == "train"]), ("development", man[man.split == "dev"]), ("test (audit)", man[man.split == "test"])]:
    rows.append((lab, sub))
test = man[man.split == "test"]
for H in [4.0, 10.0, 21.5]:
    rows.append((rf"$\mathcal{{E}}_{{{H:g}}}$ (test, $L^+\geq {H:g}$\,s)", test[test.post_anchor_length_s >= H - 1e-9]))
body = []
for lab, sub in rows:
    cc = sub.class_name.value_counts()
    body.append(" & ".join([lab, str(len(sub)), str(sub.source_cluster_id.nunique())] + [str(int(cc.get(c, 0))) for c in CLASSES]) + r" \\")
    if lab == "test (audit)":
        body.append(r"\midrule")
assert V("acc.N_H.10.0") == len(test[test.post_anchor_length_s >= 10 - 1e-9])
write("tab_cohorts", r"""\begin{table*}[pos=t]
\caption{ACCIDENT real clips after source-video leak repair, and the fixed eligibility cohorts of the audit split. Clusters are source videos (the resampling unit). Class columns count clips.}
\label{tab:cohorts}
\footnotesize
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}
\toprule
Subset & Clips & Clusters & head-on & rear-end & t-bone & sideswipe & single \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")

# ------------------------------------------------------------------ Table: characterization
kd = J(INTERNAL / "knob_decomposition.json")
body = []
for H, ph in HASH.items():
    cal = J(REMOTE04 / f"r2b/calib_plaus/{ph}/calibration.json")
    first = True
    for m in METRICS:
        c = cal["metrics"][m]
        k = kd[f"H{H}|{m}"]
        ruler = c["ruler"]
        if ruler > 0:
            ratio, ratio_h = f3(c["MRD_plaus"] / ruler), f3(k["MRD_sameH"] / ruler)
        else:
            ratio, ratio_h = r"--", r"--"
        verdict = c["verdict"].split(" (")[0]
        hcell = rf"\multirow{{5}}{{*}}{{{H:g}}}" if first else ""
        first = False
        mrd = f"{c['MRD_plaus']:.3f}"
        rul = f"{ruler:.3f}"
        body.append(" & ".join([hcell, MLAB[m], f3(c["min_R"]), mrd, rul, ratio, verdict, f3(k["minR_sameH"]), ratio_h]) + r" \\")
    if H != 21.5:
        body.append(r"\midrule")
write("tab_characterization", r"""\begin{table*}[pos=t]
\caption{Characterization of the frozen metric family on the 54-point plausible grid (primary pool: 168 non-trivial systems and 4 trivial anchors; gauge blocks excluded). $\min R_M$: lowest rank preservation over the 53 non-reference points; MRD: 95th percentile of $|M(a;\pi)-M(a;\pi_0)|$ (RMSCD: same-$H$ points only); ruler: median $|M(a)-M(b)|$ over window-end-tied pairs at $\pi_0$. Verdict: not comparable if $\min R_M<0.9$, poolable if MRD $<$ ruler, otherwise normalize. The last two columns restrict the same quantities to the 17 points that keep $H$ at its reference value (post hoc decomposition).}
\label{tab:characterization}
\footnotesize
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}clccccccc@{}}
\toprule
 & & \multicolumn{5}{c}{Frozen analysis (all plausible points)} & \multicolumn{2}{c}{Same-$H$ points (post hoc)} \\
\cmidrule(lr){3-7}\cmidrule(l){8-9}
$H$ (s) & Metric $M$ & $\min R_M$ & MRD & Ruler & MRD/ruler & Verdict & $\min R_M$ & MRD/ruler \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")

# ------------------------------------------------------------------ Table: tied pairs
td = J(INTERNAL / "tied_pairs_decomposition.json")
comp = J(INTERNAL / "g4_composition.json")
a3 = {r["H"]: r for r in V("sup.tableS_A3_prefix_accuracy")}
body = []
for H in [4.0, 10.0, 21.5]:
    g = V(f"g4.{H}")
    het = td[str(H)]["heterogeneous"]
    c = comp[str(H)]
    body.append(" & ".join([f"{H:g}", str(g["tied"]), str(g["heterogeneous"]), str(g["cross_base"]), str(g["replicated_groups"]),
                            f"{het['delay_dominant']}", f"{c['heterogeneous_no_commit']}", f"{c['replicated_groups_no_commit']}",
                            f"{int(a3[H]['AUC detected'])}"]) + r" \\")
write("tab_tied", r"""\begin{table*}[pos=t]
\caption{System pairs whose window-end macro-accuracy intervals overlap (``tied'') and their RMSCD differences, by horizon. Tied pairs are not equivalent systems: their window-end macro-accuracies differ, but the 95\% cluster-bootstrap intervals overlap. Heterogeneous: RMSCD difference significant after Holm correction over the tied pairs of the horizon. Cross-base: the two systems differ in round, backbone or model class. Replicated groups: cross-base variant pairs heterogeneous in the same direction in at least two of three seeds. Delay-dominant: the delay term of Eq.~(\ref{eq:decomp}) exceeds the endpoint-error term in magnitude. Accuracy-AUC: heterogeneous pairs also separated by the area under the dynamic-denominator accuracy curve.}
\label{tab:tied}
\footnotesize
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}crrrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{Heterogeneous in RMSCD} & & \multicolumn{2}{c}{Without commitment rule} & \\
\cmidrule(lr){3-5}\cmidrule(lr){7-8}
$H$ (s) & Tied pairs & Pairs & Cross-base & Replicated groups & Delay-dominant & Pairs & Groups & Accuracy-AUC \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")

# ------------------------------------------------------------------ Table: accounting rules (appendix)
body = []
names = {"original_rule": "tertiles of all dev clips; own end", "v4_23p5": r"within-cohort tertiles; own end $\leq$ 23.5\,s",
         "v5_22p0": r"within-cohort tertiles; own end $\leq$ 22\,s"}
status = {"original_rule": "undefined / failed", "v4_23p5": "invalid (cache gap)", "v5_22p0": "record"}
for rule in ["original_rule", "v4_23p5", "v5_22p0"]:
    for H in [4.0, 10.0, 21.5]:
        r = V(f"g3.{rule}.{H}")
        gr = r["groups"]
        und = gr["real"]["n_undefined"]
        real = "undef." if und == gr["real"]["n"] else f"{gr['real']['n_consequence']}/168"
        rand = "undef." if gr["random_block"]["n_undefined"] == 2 else f"{gr['random_block']['n_consequence']}/2"
        gate = {True: "pass", False: "fail", None: "undefined"}[r["pass_gate"]]
        if rule == "v4_23p5":
            gate = "not evidence"
        body.append(" & ".join([names[rule] if H == 4.0 else "", f"{H:g}", rand, real, gate]) + r" \\")
    if rule != "v5_22p0":
        body.append(r"\midrule")
write("tab_g3", r"""\begin{table*}[pos=t]
\caption{Consequence of the accounting rule under the three registered versions of the length-stratified comparison (primary pool). A system shows a consequence when its per-clip-end long-minus-short gap has a 95\% interval excluding 0 and exceeding the RMSCD MRD, while its fixed-cohort gap has an interval including 0 or is smaller than the RMSCD ruler. The second version is kept for traceability only: the answer cache ends at 22\,s, so cells between 22 and 23.5\,s were scored as errors and produced an artefactual effect. The third version is the result of record.}
\label{tab:g3}
\footnotesize
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
Rule version & $H$ (s) & Random blocks & Real systems & Outcome \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")

# ------------------------------------------------------------------ Table: obligatory contrasts (appendix)
a1 = {r["Metric"]: r for r in V("sup.tableS_A1_reversals") if r["H"] == 10.0}
a4 = {r["Metric"]: r for r in V("sup.tableS_A4_region") if r["H"] == 10.0}
a5 = {r["Metric"]: r for r in V("sup.tableS_A5_region") if r["H"] == 10.0}
sec = {r["Metric"]: r for r in V("t3.secondary_H10")}
body = []
for m in METRICS:
    body.append(" & ".join([MLAB[m], str(a1[m]["P reversal events"]), str(a1[m]["P unique pairs"]),
                            f"{a4[m]['Min alternative']:.3f}", f"{a4[m]['Alternative domain']}/{a4[m]['R domain']}",
                            f"{a5[m]['Min alternative']:.3f}", f"{a5[m]['Alternative domain']}",
                            f"{sec[m]['Min R [secondary +VLM]']:.3f}", sec[m]["Verdict [secondary +VLM]"]]) + r" \\")
write("tab_contrasts", r"""\begin{table*}[pos=t]
\caption{Pre-specified contrasts at $H=10$\,s (primary pool unless stated). Reversals: sign reversals of reference-significant pairs at the directly neighbouring points of the full grid (events counted over point--pair combinations; distinct pairs in the next column). Spearman: rank preservation replaced by the Spearman correlation of system values; region size counts plausible points reaching 0.9, relative to the $R_M$ region. Unscreened: $R_M$ computed over all reference-nontied pairs instead of the significant set $\mathcal{P}$ (region size out of 54). Last two columns: $\min R_M$ and verdict when the read-only vision--language model is added (secondary pool).}
\label{tab:contrasts}
\footnotesize
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrcccccc@{}}
\toprule
 & \multicolumn{2}{c}{Neighbour reversals} & \multicolumn{2}{c}{Spearman instead of $R_M$} & \multicolumn{2}{c}{Unscreened pairs} & \multicolumn{2}{c}{Secondary pool} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}
Metric & Events & Pairs & Min & Region & $\min R_M$ & Region & $\min R_M$ & Verdict \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")

# ------------------------------------------------------------------ Table: gauge blocks (appendix)
blk = V("blocks.H10")
expect = [("block__oracle__default", "oracle", "0.0", "0.00"),
          ("block__lock__d0-1", r"lock($d_0{=}1$)", "1.0", "0.75"),
          ("block__lock__d0-3", r"lock($d_0{=}3$)", "3.0", "2.75"),
          ("block__lock__commit-3_d0-3", r"lock($d_0{=}3$), commits at 3\,s", "3.0", "2.75"),
          ("block__lock__d0-8", r"lock($d_0{=}8$)", "8.0", "7.75"),
          ("block__lock__d0-18", r"lock($d_0{=}18$)", r"$>H$", "10.00"),
          ("block__osc__d0-4_p-1", r"osc($p{=}1, d_0{=}4$)", "3.0", "2.75"),
          ("block__osc__d0-6_p-0.5", r"osc($p{=}0.5, d_0{=}6$)", "5.5", "5.25"),
          ("block__osc__d0-15_p-2", r"osc($p{=}2, d_0{=}15$)", r"10.0$^{a}$", "9.75")]
body = []
for sid, lab, tau, rm in expect:
    b = blk[sid]
    assert abs(b["RMSCD"] - float(rm)) < 1e-9, (sid, b["RMSCD"], rm)
    body.append(" & ".join([lab, tau, rm, f"{b['RMSCD']:.2f}", f"{b['end_macro']:.3f}", f"{b['median_flips']:.0f}"]) + r" \\")
for sid, lab in [("block__frac__c-0.6", r"frac($c{=}0.6$)"), ("block__rand__eta-0.5", r"rand($\eta{=}0.5$)"), ("block__rand__eta-0.9", r"rand($\eta{=}0.9$)")]:
    b = blk[sid]
    body.append(" & ".join([lab, "per clip", "--", f"{b['RMSCD']:.2f}", f"{b['end_macro']:.3f}", f"{b['median_flips']:.0f}"]) + r" \\")
write("tab_blocks", r"""\begin{table*}[pos=t]
\caption{Gauge blocks at the reference protocol ($H=10$\,s, $\Delta=0.5$\,s, $N_H=1113$). $\tau$: stable-correct onset implied by the block definition; expected RMSCD is $\tau-\Delta/2$ under the trapezoid rule on the grid ($H$ if $\tau>H$). $^{a}$The block becomes stably correct inside the window at $\delta=10$\,s because its phase covering $[10,12)$\,s is correct. frac and rand blocks depend on clip length or on random draws and have no single closed-form value.}
\label{tab:blocks}
\footnotesize
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
Block & $\tau$ (s) & Expected RMSCD (s) & Computed RMSCD (s) & End macro-Acc & Median flips \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")

# ------------------------------------------------------------------ Table: MM-AU development summary (appendix)
sc = {(c["arm"], c["cond"]): c for c in V("mm.stride.cells")}
ac = {(c["arm"], c["shift"]): c for c in V("mm.anchor.cells")}
ctl = V("mm.controls")
arms = [("anchor_weighted", "anchor_weighted", "Static (anchor frame)", "w"), ("anchor_unweighted", "anchor_unweighted", "Static (anchor frame)", "u"),
        ("prefix_mean", "mean_weighted", "Prefix mean", "w"), ("mean_unweighted", "mean_unweighted", "Prefix mean", "u"),
        ("gru", "gru_weighted", "GRU", "w"), ("gru_unweighted", "gru_unweighted", "GRU", "u")]
body = []
pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
for a, actl, lab, w in arms:
    c67 = ctl[actl]
    body.append(" & ".join([lab, w, pct(c67["macro_67"]), pct(c67["micro67"]),
                            pct(sc[(a, "stride2")]["macro"]), pct(sc[(a, "stride4")]["macro"]), pct(sc[(a, "stride8")]["macro"]),
                            pct(ac[(a, -4)]["macro"]), pct(ac[(a, 0)]["macro"]), pct(ac[(a, 4)]["macro"])]) + r" \\")
write("tab_mmau", r"""\begin{table*}[pos=t]
\caption{MM-AU development cohort (269 development clips, nine native classes, frames as time unit): window-end accuracy (\%) of frozen checkpoints, mean of three seeds. w/u: class-weighted/unweighted training loss. The $H{=}67$ columns come from the original input schedule; input-step columns fix $H{=}64$ and the output step at 8 frames and change the actual inputs (33/17/9 frames); start-shift columns fix $H{=}60$ and move the actual input start and end. Numbers are development readings used repeatedly for design and are not test estimates.}
\label{tab:mmau}
\footnotesize
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lcrrrrrrrr@{}}
\toprule
 & & \multicolumn{2}{c}{$H{=}67$} & \multicolumn{3}{c}{Input step (frames), macro} & \multicolumn{3}{c}{Start shift (frames), macro} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}
Model & Loss & macro & micro & 2 & 4 & 8 & $-4$ & 0 & $+4$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")
