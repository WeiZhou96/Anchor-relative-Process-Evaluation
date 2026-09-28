"""Figure: comparability verdicts of the frozen metric family (ACCIDENT, primary pool).

Data: $APE_DATA/accident/outputs/r2b/calib_plaus/<pi0>/calibration.json and R_*.csv (frozen 54-point
plausible product grid); $APE_FIG_OUT/internal/knob_decomposition.json (post hoc regrouping of the same points).
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figstyle import FULL_W, INK, MUTED, OKABE_ITO, export, panel_tag, style_ax
from paths import HASH, INTERNAL, METRIC_FILE, REMOTE04

MET = ["RMSCD@H", "S_H@1", "S_H@3", "end_window_macro_acc"]
LAB = {"RMSCD@H": "RMSCD@H", "S_H@1": r"$S_H(1\,\mathrm{s})$", "S_H@3": r"$S_H(3\,\mathrm{s})$",
       "end_window_macro_acc": "window-end macro-Acc", "median_flips": "median flips"}
COL = {"RMSCD@H": OKABE_ITO["vermillion"], "S_H@1": OKABE_ITO["blue"], "S_H@3": OKABE_ITO["sky"],
       "end_window_macro_acc": OKABE_ITO["purple"]}
MK = {4.0: "o", 10.0: "s", 21.5: "^"}
kd = json.loads((INTERNAL / "knob_decomposition.json").read_text(encoding="utf-8"))

fig = plt.figure(figsize=(FULL_W, 2.75))
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.3, left=0.075, right=0.99, bottom=0.2, top=0.9)

# (a) verdict map
ax = fig.add_subplot(gs[0])
xmin, xmax, ymin, ymax = 0.15, 6.5, 0.80, 1.005
ax.fill_between([xmin, xmax], ymin, 0.9, color="#FBE3D6", lw=0, zorder=0)
ax.fill_between([xmin, 1.0], 0.9, ymax, color="#E3F1E8", lw=0, zorder=0)
ax.fill_between([1.0, xmax], 0.9, ymax, color="#E3ECF5", lw=0, zorder=0)
ax.text(0.17, 0.803, "not comparable ($\\min R_M < r_0$)", fontsize=6.5, color="#A5451A", va="bottom")
ax.text(0.17, 0.9985, "poolable", fontsize=6.5, color="#2F7A4C", va="top")
ax.text(6.3, 0.9985, "normalize", fontsize=6.5, color="#2B5C8A", va="top", ha="right")
ax.axhline(0.9, color="#555555", lw=0.6)
ax.axvline(1.0, color="#555555", lw=0.6)
rows = []
for H, ph in HASH.items():
    cal = json.loads((REMOTE04 / f"r2b/calib_plaus/{ph}/calibration.json").read_text(encoding="utf-8"))
    for m in MET:
        c = cal["metrics"][m]
        x0, y0 = c["MRD_plaus"] / c["ruler"], c["min_R"]
        k = kd[f"H{H}|{m}"]
        x1, y1 = k["MRD_sameH"] / c["ruler"], k["minR_sameH"]
        ax.plot([x0, x1], [y0, y1], color=COL[m], lw=0.6, alpha=0.7, zorder=2)
        ax.plot(x0, y0, MK[H], ms=5.0, color=COL[m], mec="white", mew=0.5, zorder=4)
        ax.plot(x1, y1, MK[H], ms=4.6, mfc="white", mec=COL[m], mew=0.9, zorder=3)
        rows.append((H, m, x0, y0, x1, y1, c["verdict"]))
ax.set_xscale("log")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xticks([0.2, 0.5, 1, 2, 5])
ax.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
ax.set_xlabel(r"$\mathrm{MRD}_M$ / ruler$_M$")
ax.set_ylabel(r"$\min R_M$ over plausible points")
ax.set_title("Frozen verdict (filled) vs. same-$H$ part (open)", fontsize=8.5)
h1 = [Line2D([], [], ls="", marker="o", ms=4.5, color=COL[m], label=LAB[m]) for m in MET]
h2 = [Line2D([], [], ls="", marker=MK[H], ms=4.5, color="#555555", label=f"$H$={H:g} s") for H in HASH]
h3 = [Line2D([], [], ls="", marker="s", ms=4.5, color="#555555", label="all 53 plausible points (frozen)"),
      Line2D([], [], ls="", marker="s", ms=4.5, mfc="white", mec="#555555", label="same-$H$ points only (post hoc)")]
leg1 = ax.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.0, 0.07), fontsize=6.5, handletextpad=0.1, borderaxespad=0.2)
ax.add_artist(leg1)
leg2 = ax.legend(handles=h2, loc="lower center", bbox_to_anchor=(0.53, 0.07), fontsize=6.5, handletextpad=0.1)
ax.add_artist(leg2)
ax.text(0.99, 0.02, "median flips: ruler = 0,\n$\\min R$ = 0.12-0.50 (off scale)", fontsize=6.5, color=MUTED, ha="right", va="bottom", transform=ax.transAxes)
style_ax(ax)
panel_tag(ax, "(a)", dx=-0.19)

# (b) R_M at the 53 non-reference plausible points, H = 10 s, grouped by knob
ax = fig.add_subplot(gs[1])
ph = HASH[10.0]
GROUPS = [("anchor only", OKABE_ITO["green"]), ("step only", OKABE_ITO["sky"]), ("anchor+step", OKABE_ITO["blue"]),
          ("H changed", OKABE_ITO["vermillion"])]
rng = np.random.default_rng(3)
allm = MET + ["median_flips"]
for i, m in enumerate(allm):
    R = pd.read_csv(REMOTE04 / f"r2b/calib_plaus/{ph}/R_{METRIC_FILE[m]}.csv")
    R = R[R.pi_hash != ph]
    anc = (R.eps_sys_s.abs() > 1e-9) | (R.eps_jit_sd_s.abs() > 1e-9)
    stp = (R.delta_s - 0.5).abs() > 1e-9
    hch = (R.h_s - 10.0).abs() > 1e-9
    grp = np.where(hch, "H changed", np.where(anc & stp, "anchor+step", np.where(anc, "anchor only", "step only")))
    for j, (g, c) in enumerate(GROUPS):
        v = R.R_M.values[grp == g]
        xs = i + (j - 1.5) * 0.18 + rng.uniform(-0.05, 0.05, len(v))
        low = v < 0.745
        ax.scatter(xs[~low], v[~low], s=7, color=c, lw=0, alpha=0.85, zorder=3)
        if low.any():
            ax.scatter(xs[low], np.full(low.sum(), 0.752), s=12, marker="v", color=c, lw=0, zorder=3)
            ax.text(i - 0.02, 0.768, f"{int(low.sum())} points at {v[low].min():.2f}-{v[low].max():.2f}", fontsize=6.5,
                    color=c, va="bottom", ha="center")
ax.axhline(0.9, color="#555555", lw=0.6)
ax.text(-0.45, 0.903, "$r_0$ = 0.9", fontsize=6.5, color=INK, ha="left", va="bottom")
ax.set_xticks(range(len(allm)))
ax.set_xlim(-0.55, 4.95)
ax.set_xticklabels(["RMSCD", r"$S_H(1)$", r"$S_H(3)$", "end\nmacro-Acc", "median\nflips"], fontsize=6.8)
ax.set_ylim(0.74, 1.008)
ax.set_ylabel(r"rank preservation $R_M(\pi)$")
ax.set_title(r"Which knob reorders systems ($H_0$ = 10 s)", fontsize=8.5)
ax.legend(handles=[Patch(color=c, label=g) for g, c in GROUPS], loc="lower left", fontsize=6.5, ncol=2, handlelength=0.9,
          columnspacing=0.8, borderaxespad=0.2)
style_ax(ax, grid="y")
panel_tag(ax, "(b)", dx=-0.17)
for r in rows:
    print(r)
export(fig, "fig4_comparability")
