"""Figure: eligibility cohorts and the consequence of per-clip-end accounting (ACCIDENT).

Data: $APE_DATA/accident/manifest/manifest_real.csv (manifest v2; post-anchor lengths)
      $APE_DATA/accident/outputs/r2c/gates/g3_v5.json, primary pool, rule v5_22p0 (record rule):
      long-minus-short stratum gaps of the per-clip-end ("own") and fixed-cohort delays.
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figstyle import FULL_W, EMPH, INK, MUTED, OKABE_ITO, export, panel_tag, style_ax
from paths import MANIFEST, REMOTE04

man = pd.read_csv(MANIFEST)
g3 = json.loads((REMOTE04 / "r2c/gates/g3_v5.json").read_text(encoding="utf-8"))["primary"]["v5_22p0"]["by_h"]

fig = plt.figure(figsize=(FULL_W, 2.7))
gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.9], wspace=0.28, left=0.075, right=0.985, bottom=0.17, top=0.9)

# (a) eligibility-cohort size as a function of the horizon
ax = fig.add_subplot(gs[0])
h = np.linspace(0, 30, 301)
for split, c, ls, lab in [("test", INK, "-", "test (audit)"), ("dev", MUTED, (0, (3, 2)), "dev (protocol design)")]:
    L = man.loc[man.split == split, "post_anchor_length_s"].to_numpy()
    frac = [(L >= x - 1e-9).mean() for x in h]
    ax.plot(h, frac, color=c, ls=ls, lw=1.1, label=lab)
Lt = man.loc[man.split == "test", "post_anchor_length_s"].to_numpy()
for H, col in [(4.0, OKABE_ITO["sky"]), (10.0, EMPH), (21.5, OKABE_ITO["blue"])]:
    n = int((Lt >= H - 1e-9).sum())
    ax.plot([H, H], [0, n / len(Lt)], color=col, lw=0.8, ls=(0, (1, 1.2)))
    ax.plot(H, n / len(Lt), "o", ms=3.6, color=col, mec="white", mew=0.5, zorder=4)
    ax.text(H + 0.6, n / len(Lt) + 0.03, f"H={H:g} s\n$N_H$={n}", fontsize=6.5, color=col, va="bottom")
ax.axvline(22.0, color="#9A9A9A", lw=0.5)
ax.text(21.6, 1.02, "answer cache\nends at 22 s", fontsize=6.5, color=MUTED, va="top", ha="right")
ax.set_xlim(0, 30)
ax.set_ylim(0, 1.05)
ax.set_xlabel("horizon $H$ (s)")
ax.set_ylabel(r"fraction of clips with $L^{+} \geq H$")
ax.set_title("Eligibility cohort per horizon", fontsize=8.5)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.18), fontsize=6.5)
style_ax(ax)
panel_tag(ax, "(a)", dx=-0.21)

# (b) long-minus-short gaps: per-clip end vs fixed cohort
ax = fig.add_subplot(gs[1])
rng = np.random.default_rng(20260903)
ylab, yt = [], []
y = 0
summary = {}
for bh in g3:
    H = float(bh["h_s"])
    rows = bh["rows"]
    real = [r for r in rows if r["role"] == "real"]
    rand = [r for r in rows if r["role"] == "random_block"]
    orc = [r for r in rows if r["system_id"] == "block__oracle__default"]
    for kind, name in [("fixed", "fixed cohort"), ("own", "per-clip end")]:
        g = np.array([r[kind]["gap_long_minus_short"] for r in real])
        ax.scatter(g, y + rng.uniform(-0.17, 0.17, len(g)), s=6, color=OKABE_ITO["blue"] if kind == "fixed" else OKABE_ITO["orange"],
                   alpha=0.55, lw=0, zorder=2)
        for r, mk in zip(rand, ["D", "s"]):
            ax.errorbar(r[kind]["gap_long_minus_short"], y, xerr=[[r[kind]["gap_long_minus_short"] - r[kind]["ci_lo"]],
                        [r[kind]["ci_hi"] - r[kind]["gap_long_minus_short"]]], fmt=mk, ms=3.6, color="black", mfc="white",
                        elinewidth=0.8, capsize=1.5, zorder=4)
        ax.plot(orc[0][kind]["gap_long_minus_short"], y, marker="*", ms=6, color=OKABE_ITO["green"], mec="white", mew=0.3, zorder=5)
        ylab.append(f"$H$={H:g} s, {name}")
        yt.append(y)
        y += 1
    n_cons = bh["groups"]["real"]["n_consequence"]
    rb = bh["groups"]["random_block"]["n_consequence"]
    ruler = bh["ruler"]
    ax.fill_betweenx([y - 2.45, y - 0.55], -ruler, ruler, color="#E3ECF5", zorder=0, lw=0)
    ax.text(21.2, y - 1.5, f"consequence:\nreal {n_cons}/168, random {rb}/2", fontsize=6.5, color=INK, va="center", ha="right")
    summary[H] = (n_cons, rb, ruler)
    y += 0.3
ax.axvline(0, color="#333333", lw=0.5, zorder=1)
ax.set_yticks(yt)
ax.set_yticklabels(ylab, fontsize=6.8)
ax.set_ylim(y - 0.2, -0.6)
ax.set_xlim(-3.5, 21.5)
ax.set_xlabel("long-stratum minus short-stratum mean delay (s)")
ax.set_title("Per-clip-end accounting absorbs remaining length", fontsize=8.5)
style_ax(ax)
panel_tag(ax, "(b)", dx=-0.265)
print(summary)
export(fig, "fig3_accounting")
