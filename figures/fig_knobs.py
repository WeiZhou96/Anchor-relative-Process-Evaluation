"""Figure: propagation along single protocol knobs (ACCIDENT, H = 10 s, axis scan of the full frozen grid).

Data: $APE_DATA/accident/outputs/r2b/r2/scan_all.csv (per-system readings at every scanned protocol point)
      $APE_DATA/accident/outputs/r2b/calib/8ac32aae418b/R_*.csv (rank preservation on the axis scan)
      $APE_DATA/accident/outputs/r2b/calib_plaus/8ac32aae418b/calibration.json (rulers)
      $APE_DATA/accident/outputs/r2b/mechanisms/p_c.json (flip counts at the three steps)
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figstyle import FULL_W, EMPH, INK, MUTED, OKABE_ITO, export, panel_tag, style_ax
from paths import INTERNAL, METRIC_FILE, REMOTE04

ph = "8ac32aae418b"
scan = pd.read_csv(REMOTE04 / "r2b/r2/scan_all.csv")
st = pd.read_csv(INTERNAL / "system_table_reference.csv")
fam = st[st.h_s == 10.0].set_index("system_id").family
nontriv = set(fam[~fam.isin(["block", "trivial"])].index)
cal = json.loads((REMOTE04 / f"r2b/calib_plaus/{ph}/calibration.json").read_text(encoding="utf-8"))
ruler = cal["rulers_by_metric"]

fig = plt.figure(figsize=(FULL_W, 2.35))
gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.065, right=0.99, bottom=0.2, top=0.88)


def axis_rows(metric, knob):
    s = scan[(scan.metric == metric) & (scan.h_s == 10.0)]
    others = {"eps_sys_s": 0.0, "eps_jit_sd_s": 0.0, "delta_s": 0.5}
    for k, v in others.items():
        if k != knob:
            s = s[np.isclose(s[k], v)]
    return s


# (a) common anchor shift
ax = fig.add_subplot(gs[0])
s = axis_rows("RMSCD@H", "eps_sys_s")
piv = s.pivot_table(index="system_id", columns="eps_sys_s", values="value")
eps = np.array(sorted(piv.columns))
d = piv[eps].sub(piv[0.0], axis=0)
real = d.loc[[i for i in d.index if i in nontriv]]
for sid, row in real.iterrows():
    ax.plot(eps, row.values, color="#B8BEC6", lw=0.4, alpha=0.5, zorder=1)
ax.plot(eps, real.mean().values, color=INK, lw=1.3, zorder=3, label="non-trivial systems (mean)")
for bid, c, lab in [("block__lock__d0-3", OKABE_ITO["green"], "gauge block lock(3 s)"),
                    ("block__rand__eta-0.5", "#555555", "gauge block rand(0.5)")]:
    ax.plot(eps, d.loc[bid].values, color=c, lw=1.1, ls=(0, (3, 1.5)), zorder=2, label=lab)
R = pd.read_csv(REMOTE04 / f"r2b/calib/{ph}/R_RMSCDatH.csv")
Re = R[np.isclose(R.eps_jit_sd_s, 0) & np.isclose(R.delta_s, 0.5) & np.isclose(R.h_s, 10.0)]
ax.text(0.98, 0.97, f"min $R_M$ on this axis: {Re.R_M.min():.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK)
ax.axhline(0, color="#333333", lw=0.5)
ax.set_xlabel(r"common anchor shift $\varepsilon_{\mathrm{sys}}$ (s)")
ax.set_ylabel(r"RMSCD shift vs. $\pi_0$ (s)")
ax.set_xlim(-1.05, 1.05)
ax.set_title("Common anchor shift", fontsize=8.5)
ax.legend(loc="lower left", fontsize=6.5, handlelength=1.8, borderaxespad=0.1)
style_ax(ax)
panel_tag(ax, "(a)", dx=-0.22)

# (b) independent jitter: spread across systems, in units of each metric's ruler
ax = fig.add_subplot(gs[1])
cols = {"RMSCD@H": OKABE_ITO["vermillion"], "S_H@1": OKABE_ITO["blue"], "S_H@3": OKABE_ITO["sky"],
        "end_window_macro_acc": OKABE_ITO["purple"]}
labs = {"RMSCD@H": "RMSCD", "S_H@1": r"$S_H(1)$", "S_H@3": r"$S_H(3)$", "end_window_macro_acc": "end macro-Acc"}
for m, c in cols.items():
    s = axis_rows(m, "eps_jit_sd_s")
    piv = s.pivot_table(index="system_id", columns="eps_jit_sd_s", values="value")
    jit = np.array(sorted(piv.columns))
    dd = piv[jit].sub(piv[0.0], axis=0)
    dd = dd.loc[[i for i in dd.index if i in nontriv]]
    sd = dd.std(ddof=1).values / ruler[m]
    ax.plot(jit, sd, "-o", color=c, ms=3.2, lw=1.1, mec="white", mew=0.4, label=labs[m], zorder=4 if m == "RMSCD@H" else 3)
ax.axhline(1.0, color="#9A9A9A", lw=0.5, ls=(0, (2, 2)))
ax.text(0.5, 1.02, "spread = ruler", fontsize=6.5, color=MUTED, va="bottom", ha="right")
ax.set_xlabel(r"independent jitter SD $\varepsilon_{\mathrm{jit}}$ (s)")
ax.set_ylabel(r"$s_M$ / ruler$_M$")
ax.set_ylim(0, 1.25)
ax.set_title("Independent anchor jitter", fontsize=8.5)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.76), fontsize=6.5, borderaxespad=0.1)
style_ax(ax)
panel_tag(ax, "(b)", dx=-0.22)

# (c) flip counts vs prefix step
ax = fig.add_subplot(gs[2])
pc = json.loads((REMOTE04 / "r2b/mechanisms/p_c.json").read_text(encoding="utf-8"))
rows = [r for r in pc["rows"] if float(r["h_s"]) == 10.0]
steps = [0.25, 0.5, 1.0]
for r in rows:
    if r["system_id"] in nontriv:
        y = [r["points"][str(k)]["mean"] for k in steps]
        ax.plot(steps, y, color="#B8BEC6", lw=0.4, alpha=0.6, zorder=1)
mean_real = np.mean([[r["points"][str(k)]["mean"] for k in steps] for r in rows if r["system_id"] in nontriv], axis=0)
ax.plot(steps, mean_real, "-o", color=INK, lw=1.3, ms=3.4, mec="white", mew=0.4, zorder=3, label="non-trivial systems (mean)")
for r in rows:
    if r["system_id"] == "block__rand__eta-0.5":
        y = [r["points"][str(k)]["median"] for k in steps]
        ax.plot(steps, y, "-s", color=EMPH, lw=1.1, ms=3.2, mec="white", mew=0.4, zorder=3, label="rand(0.5) block (median)")
        for k, v in zip(steps, y):
            ax.text(k * 1.07, v, f"{v:.0f}", fontsize=6.5, color=EMPH, va="center")
ax.set_xscale("log", base=2)
ax.set_xticks(steps)
ax.set_xticklabels(["0.25", "0.5", "1"])
ax.set_yscale("log")
ax.set_xlabel(r"prefix step $\Delta$ (s)")
ax.set_ylabel("flips within the window")
ax.set_title("Prefix step and flip counts", fontsize=8.5)
ax.legend(loc="lower left", fontsize=6.5, borderaxespad=0.1)
style_ax(ax)
panel_tag(ax, "(c)", dx=-0.22)
print("mean real flips", mean_real, "n real", len([r for r in rows if r["system_id"] in nontriv]))
export(fig, "fig5_knobs")
