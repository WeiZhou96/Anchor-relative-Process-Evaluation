"""Figure: three sampling operations in the MM-AU development cohort (269 dev clips, frame units).

Data (MM-AU development records under $APE_DATA/mmau/deliverables, frozen checkpoints, no retraining):
  overlap_dense_20260920/sensitivity/summary.json  -- output reading step only (H = 64, seed means)
  input_stride_20260921/input/ape_evaluation.json  -- actual input cadence 2/4/8 frames (H = 64, output step 8)
  anchor_shift_20260921/anchor/ape_evaluation.json -- actual input start -4/0/+4 frames (H = 60)
Metric: class-macro RMSCD/H (lower = earlier stable correctness), as recorded by the frame-axis evaluator.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figstyle import FULL_W, INK, MUTED, OKABE_ITO, export, panel_tag, style_ax
from paths import MMAU


def J(p):
    return json.loads(open(p, encoding="utf-8").read())


ARMS = [("anchor_weighted", "static", True), ("anchor_unweighted", "static", False),
        ("prefix_mean", "prefix mean", True), ("mean_unweighted", "prefix mean", False),
        ("gru", "GRU", True), ("gru_unweighted", "GRU", False)]
COL = {"static": "#555555", "prefix mean": OKABE_ITO["blue"], "GRU": OKABE_ITO["vermillion"]}
MET = "RMSCD@H_norm_macro"

fig = plt.figure(figsize=(FULL_W, 2.55))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.38, left=0.075, right=0.99, bottom=0.2, top=0.88)

# (a) output reading step only
sen = J(MMAU / "overlap_dense_20260920/sensitivity/summary.json")
cells = {(c["arm"], c["horizon"], c["delta"]): c["metrics"] for c in sen["cells"]}
ax = fig.add_subplot(gs[0])
order = [("anchor_weighted", "static, w"), ("anchor_unweighted", "static, u"), ("mean_weighted", "prefix mean, w"),
         ("mean_unweighted", "prefix mean, u"), ("gru_weighted", "GRU, w"), ("gru_unweighted", "GRU, u"),
         ("shuffle_weighted", "shuffled GRU, w"), ("shuffle_unweighted", "shuffled GRU, u")]
for i, (arm, lab) in enumerate(order):
    a, b = cells[(arm, 64, 1)], cells[(arm, 64, 8)]
    c = COL["static"] if arm.startswith("anchor") else COL["prefix mean"] if arm.startswith("mean") else COL["GRU"]
    ax.plot([a[MET], b[MET]], [i, i], color=c, lw=1.0, alpha=0.8)
    ax.plot(a[MET], i, "o", ms=4, mfc="white", mec=c, mew=0.9)
    ax.plot(b[MET], i, "o", ms=4, color=c, mec="white", mew=0.4)
    ax.text(0.795, i, f"{a['mean_flips']:.2f}$\\rightarrow${b['mean_flips']:.2f}", fontsize=6.5, va="center", color=MUTED)
    assert abs(a["end_window_macro_acc"] - b["end_window_macro_acc"]) < 1e-12
ax.set_yticks(range(len(order)))
ax.set_yticklabels([o[1] for o in order], fontsize=6.5)
ax.set_ylim(len(order) - 0.4, -0.9)
ax.set_xlim(0.5, 0.88)
ax.text(0.795, -0.75, "mean flips", fontsize=6.5, color=MUTED, va="center")
ax.set_xlabel("macro RMSCD/$H$ ($H$ = 64 frames)")
ax.set_title("Reading step 1 (open) vs. 8 (filled)", fontsize=8.5)
style_ax(ax)
panel_tag(ax, "(a)", dx=-0.4)


def slope_panel(ax, runs, key, conds, xt, title, xlabel):
    for arm, typ, w in ARMS:
        c = COL[typ]
        ls = "-" if w else (0, (3, 1.6))
        ys = []
        for seed in (20260920, 20260921, 20260922):
            y = [next(r["metrics"][MET] for r in runs if r["arm"] == arm and r["seed"] == seed and r[key] == k) for k in conds]
            ys.append(y)
            ax.plot(range(len(conds)), y, ls="", marker="o", ms=1.8, color=c, alpha=0.55)
        m = np.mean(ys, axis=0)
        ax.plot(range(len(conds)), m, ls=ls, color=c, lw=1.2, marker="o", ms=3.2, mfc="white" if not w else c, mec=c, mew=0.8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(xt)
    ax.set_xlim(-0.3, len(conds) - 0.7)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel(xlabel)


stride = J(MMAU / "input_stride_20260921/input/ape_evaluation.json")["runs"]
ax = fig.add_subplot(gs[1])
slope_panel(ax, stride, "condition", ["stride2", "stride4", "stride8"], ["2 (33)", "4 (17)", "8 (9)"],
            "Actual input cadence", "input step in frames (inputs)")
ax.set_ylabel("macro RMSCD/$H$ ($H$ = 64 frames)")
style_ax(ax)
panel_tag(ax, "(b)", dx=-0.3)

shift = J(MMAU / "anchor_shift_20260921/anchor/ape_evaluation.json")["runs"]
ax = fig.add_subplot(gs[2])
slope_panel(ax, shift, "shift", [-4, 0, 4], ["$-$4", "0", "+4"], "Actual input start", "start relative to anchor (frames)")
ax.set_ylabel("macro RMSCD/$H$ ($H$ = 60 frames)")
style_ax(ax)
panel_tag(ax, "(c)", dx=-0.3)
handles = [Line2D([], [], color=COL[t], lw=1.2, label=t) for t in ["static", "prefix mean", "GRU"]] + \
          [Line2D([], [], color="#555555", lw=1.2, ls="-", marker="o", ms=3, label="class-weighted (mean of 3 seeds)"),
           Line2D([], [], color="#555555", lw=1.2, ls=(0, (3, 1.6)), marker="o", ms=3, mfc="white", label="unweighted"),
           Line2D([], [], ls="", marker="o", ms=2, color="#555555", label="single seed")]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.62, -0.005), handlelength=1.6,
           columnspacing=1.0)
export(fig, "fig6_mmau")
