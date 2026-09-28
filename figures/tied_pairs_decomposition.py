"""Descriptive decomposition of RMSCD differences among window-end-tied pairs (post hoc).

Pairs and Holm flags: $APE_DATA/accident/outputs/r2b/gates/g4.json (K-b primary pool).
Per-system level/delay terms: internal/system_table_reference.csv (built from frozen records).
No new test-set evaluation; only regrouping of stored point values.
"""
import json
import numpy as np
import pandas as pd
from paths import *

sys_t = pd.read_csv(INTERNAL / "system_table_reference.csv")
g4 = json.loads((REMOTE04 / "r2b/gates/g4.json").read_text(encoding="utf-8"))
out = {}
for bh in g4["by_h"]:
    h = float(bh["h_s"])
    t = sys_t[sys_t.h_s == h].set_index("system_id")
    P = pd.DataFrame(bh["pairs"])
    systems = set(P.system_a) | set(P.system_b)
    P["lvl"] = t.loc[P.system_a, "level_term"].values - t.loc[P.system_b, "level_term"].values
    P["dly"] = t.loc[P.system_a, "delay_term"].values - t.loc[P.system_b, "delay_term"].values
    P["chk"] = P.lvl + P.dly - P["diff"]
    assert P.chk.abs().max() < 1e-9
    het = P[P.significant]
    xb = het[het.cross_base]
    def share(df):
        same_sign = np.sign(df.dly) == np.sign(df["diff"])
        dominant = df.dly.abs() > df.lvl.abs()
        opposite_level = np.sign(df.lvl) != np.sign(df["diff"])
        return dict(n=int(len(df)), delay_same_sign=int(same_sign.sum()), delay_dominant=int(dominant.sum()),
                    delay_dominant_same_sign=int((dominant & same_sign).sum()), level_opposite_sign=int(opposite_level.sum()),
                    median_abs_diff=float(df["diff"].abs().median()), median_abs_delay=float(df.dly.abs().median()),
                    median_abs_level=float(df.lvl.abs().median()))
    out[str(h)] = dict(n_systems_in_pairs=len(systems), n_tied=len(P), heterogeneous=share(het), cross_base=share(xb),
                       all_tied=share(P))
    print(h, json.dumps(out[str(h)], indent=0))
(INTERNAL / "tied_pairs_decomposition.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
