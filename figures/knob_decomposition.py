"""Post hoc decomposition of the frozen 54-point plausible-grid characterization by knob.

Uses only the frozen scan ($APE_DATA/accident/outputs/r2b/r2/scan_all.csv) and the stored per-point R tables
($APE_DATA/accident/outputs/r2b/calib_plaus/<pi0>/R_*.csv) of the primary pool (no VLM). It does not add any
protocol point, system or test-set evaluation; it regroups the existing 53 non-reference points.
Output: $APE_FIG_OUT/internal/knob_decomposition.json and $APE_FIG_OUT/tables/knob_decomposition.csv
"""
import json
import numpy as np
import pandas as pd
from paths import *

scan = pd.read_csv(REMOTE04 / "r2b/r2/scan_all.csv")
real = scan[~scan.is_block.astype(bool)]
out = {}
rows = []
for h, ph in HASH.items():
    for m in METRICS:
        R = pd.read_csv(REMOTE04 / f"r2b/calib_plaus/{ph}/R_{METRIC_FILE[m]}.csv")
        R = R[R.pi_hash != ph].copy()
        piv = real[real.metric == m].pivot_table(index="system_id", columns="pi_hash", values="value", aggfunc="first")
        base = piv[ph]
        def grp(r):
            anchor = (abs(r.eps_sys_s) > 1e-9) or (abs(r.eps_jit_sd_s) > 1e-9)
            step = abs(r.delta_s - 0.5) > 1e-9
            hor = abs(r.h_s - h) > 1e-9
            if hor:
                return "H changed"
            if anchor and step:
                return "anchor+step"
            if anchor:
                return "anchor only"
            return "step only"
        R["group"] = R.apply(grp, axis=1)
        for g, sub in R.groupby("group"):
            cols = [c for c in sub.pi_hash if c in piv.columns]
            d = np.abs(piv[cols].values - base.values[:, None]); d = d[np.isfinite(d)]
            rec = dict(h_s=h, metric=m, group=g, n_points=len(sub), min_R=float(sub.R_M.min()),
                       q95_abs_shift=float(np.percentile(d, 95)) if len(d) else float("nan"))
            rows.append(rec)
        # same-H plausible MRD for every metric (RMSCD already same-H by definition)
        same = R[abs(R.h_s - h) < 1e-9]
        cols = list(same.pi_hash)
        d = np.abs(piv[cols].values - base.values[:, None]); d = d[np.isfinite(d)]
        out[f"H{h}|{m}"] = dict(minR_sameH=float(same.R_M.min()), MRD_sameH=float(np.percentile(d, 95)),
                                minR_all=float(R.R_M.min()), n_sameH=len(same))
pd.DataFrame(rows).to_csv(TAB / "knob_decomposition.csv", index=False)
(INTERNAL / "knob_decomposition.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
df = pd.DataFrame(rows)
print(df.pivot_table(index=["h_s", "metric"], columns="group", values=["min_R", "q95_abs_shift"]).round(4).to_string())
print(json.dumps(out, indent=0))
