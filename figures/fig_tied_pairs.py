"""Figure: window-end-tied systems with different post-anchor courses (ACCIDENT, H = 10 s).

Data: $APE_FIG_OUT/internal/system_table_reference.csv (from frozen per-system records, K-b primary pool)
      $APE_DATA/accident/outputs/r2b/gates/g4.json (tied pairs, Holm flags)
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figstyle import FULL_W, FAMILY_COLOR, FAMILY_LABEL, EMPH, INK, MUTED, OKABE_ITO, export, panel_tag, style_ax
from paths import INTERNAL, REMOTE04

st = pd.read_csv(INTERNAL / "system_table_reference.csv")
t = st[st.h_s == 10.0].copy()
nt = t[~t.family.isin(["block", "trivial"])]
A = "clip__r18mean__seed20260903"
B = "postproc__ema0p7__prefix__gru512__seed20260903"
ta = t.set_index("system_id")

fig = plt.figure(figsize=(FULL_W, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.42, left=0.06, right=0.99, bottom=0.2, top=0.9)

# (a) S_H curves
ax = fig.add_subplot(gs[0])
grid = np.arange(0, 10.01, 0.5)
for _, r in nt.iterrows():
    c = "#F2C98B" if r.family == "commit" else "#B8BEC6"
    ax.plot(grid, json.loads(r.S_H), color=c, lw=0.45, alpha=0.6, zorder=1)
sa, sb = np.array(json.loads(ta.loc[A, "S_H"])), np.array(json.loads(ta.loc[B, "S_H"]))
ax.plot(grid, sa, color=OKABE_ITO["blue"], lw=1.5, zorder=3)
ax.plot(grid, sb, color=EMPH, lw=1.5, zorder=3)
ax.text(10.1, sa[-1] + 0.005, "A", color=OKABE_ITO["blue"], fontsize=7.5, va="bottom", fontweight="bold")
ax.text(10.1, sb[-1] - 0.005, "B", color=EMPH, fontsize=7.5, va="top", fontweight="bold")
ax.set_xlim(0, 10.9)
ax.set_ylim(0, 0.75)
ax.set_xlabel(r"time after anchor $\delta$ (s)")
ax.set_ylabel(r"stable-correct fraction $S_H(\delta)$")
ax.set_title("Same end, different course", fontsize=8.5)
ax.text(0.3, 0.705, "light orange: commitment rules", color=MUTED, fontsize=6.5)
ax.text(0.3, 0.655, "A: end macro-Acc 0.327, RMSCD 5.73 s", color=OKABE_ITO["blue"], fontsize=6.6)
ax.text(0.3, 0.605, "B: end macro-Acc 0.302, RMSCD 6.98 s", color=EMPH, fontsize=6.6)
style_ax(ax)
panel_tag(ax, "(a)", dx=-0.2)

# (b) window-end macro-Acc vs RMSCD
ax = fig.add_subplot(gs[1])
for fam in ["commit", "postproc", "prefix", "clip"]:
    s = nt[nt.family == fam]
    ax.scatter(s.end_macro, s.RMSCD, s=9, color=FAMILY_COLOR[fam], alpha=0.85, lw=0, label=FAMILY_LABEL[fam], zorder=2)
for sid, c, lab in [(A, OKABE_ITO["blue"], "A"), (B, EMPH, "B")]:
    r = ta.loc[sid]
    ax.errorbar(r.end_macro, r.RMSCD, xerr=[[r.end_macro - r.end_macro_lo], [r.end_macro_hi - r.end_macro]],
                yerr=[[r.RMSCD - r.RMSCD_lo], [r.RMSCD_hi - r.RMSCD]], fmt="o", ms=4.2, mfc="white", mec=c, ecolor=c,
                elinewidth=0.9, capsize=1.6, zorder=4)
    ax.text(r.end_macro + 0.008, r.RMSCD + 0.12, lab, color=c, fontsize=7.5, fontweight="bold", zorder=5)
ax.set_xlim(-0.01, 0.38)
ax.set_ylim(5.4, 10.15)
ax.axhline(10.0, color="#9A9A9A", lw=0.5, ls=(0, (2, 2)))
ax.text(0.005, 9.93, "RMSCD = H (never stable)", fontsize=6.5, color=MUTED, va="top")
ax.set_xlabel("window-end macro-Acc")
ax.set_ylabel("RMSCD@10 s (s)")
ax.set_title("End accuracy vs. delay", fontsize=8.5)
ax.legend(loc="lower left", fontsize=6.5, handletextpad=0.2, borderaxespad=0.1, markerscale=1.1)
style_ax(ax)
panel_tag(ax, "(b)", dx=-0.2)

# (c) decomposition of RMSCD differences among window-end-tied pairs
g4 = json.loads((REMOTE04 / "r2b/gates/g4.json").read_text(encoding="utf-8"))
P = pd.DataFrame(g4["by_h"][1]["pairs"])
lvl = ta.loc[P.system_a, "level_term"].values - ta.loc[P.system_b, "level_term"].values
dly = ta.loc[P.system_a, "delay_term"].values - ta.loc[P.system_b, "delay_term"].values
sgn = np.sign(P["diff"].values)
sgn[sgn == 0] = 1
lvl, dly = lvl * sgn, dly * sgn  # orient each pair so that the RMSCD difference is positive
fa = ta.loc[P.system_a, "family"].values
fb = ta.loc[P.system_b, "family"].values
commit = (fa == "commit") | (fb == "commit")
sig = P.significant.values
ax = fig.add_subplot(gs[2])
ax.scatter(lvl[~sig], dly[~sig], s=2.2, color="#C9CDD2", lw=0, alpha=0.6, rasterized=True, zorder=1)
ax.scatter(lvl[sig & commit], dly[sig & commit], s=3.0, color=OKABE_ITO["orange"], lw=0, alpha=0.75, rasterized=True, zorder=2)
ax.scatter(lvl[sig & ~commit], dly[sig & ~commit], s=4.0, color=OKABE_ITO["blue"], lw=0, alpha=0.9, rasterized=True, zorder=3)
lim = (-2.2, 4.2)
ax.plot([-2.2, 4.2], [2.2, -4.2], color="#9A9A9A", lw=0.5, ls=(0, (2, 2)), zorder=0)
ax.plot([0, 4.2], [0, 4.2], color="#6B7280", lw=0.5, zorder=0)
ax.plot([0, 4.2], [0, -4.2], color="#6B7280", lw=0.5, zorder=0)
ax.axhline(0, color="#333333", lw=0.5, zorder=0)
ax.axvline(0, ymax=(4.05 + 1.6) / 6.9, color="#333333", lw=0.5, zorder=0)
ax.set_xlim(-1.6, 3.6)
ax.set_xticks([-1, 0, 1, 2, 3])
ax.set_ylim(-1.6, 5.3)
ax.set_xlabel(r"endpoint-error term difference (s)")
ax.set_ylabel(r"delay term difference (s)")
ax.set_title("Tied-pair gaps decomposed", fontsize=8.5)
n_het = int(sig.sum())
dom = int(((np.abs(dly) > np.abs(lvl)) & sig).sum())
same = int(((dly > 0) & sig).sum())
ax.text(3.55, 5.25, f"{len(P)} tied pairs; {n_het} Holm-heterogeneous\ndelay term > 0 in {same}; dominant in {dom}",
        ha="right", va="top", fontsize=6.5, color=INK, linespacing=1.3)
ax.text(-1.5, 4.3, "with a commitment rule", color="#B8741A", fontsize=6.5, ha="left", va="center")
ax.text(3.55, 0.55, "no commitment\nrule", color=OKABE_ITO["blue"], fontsize=6.5, ha="right", va="center")
ax.text(2.0, -0.5, "not separated\nafter Holm", color="#8A9099", fontsize=6.5, ha="left", va="center")
style_ax(ax)
panel_tag(ax, "(c)", dx=-0.2)
print("het", n_het, "dominant", dom, "delay positive", same, "no-commit het", int((sig & ~commit).sum()))
export(fig, "fig2_tied_pairs")
