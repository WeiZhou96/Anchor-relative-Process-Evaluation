"""Per-system table at the three reference protocols, from the frozen per-system metric records.

Source: $APE_DATA/accident/outputs/r2b/metrics/<pi_hash>/<system_id>.json (K-b primary pool,
184 systems). Files whose name starts with '_' (e.g. the aggregate _pairs.json shipped in the
same folder of the data package) are not per-system records and are skipped.
Adds the exact decomposition RMSCD@H = H*(1 - S_H(H)) + D@H with
D@H = trapezoid of [S_H(H) - S_H(delta)] over the window (the stabilization-delay term).
S_H(H) equals the window-end micro accuracy because I_{i,J_H} = 1[correct at J_H].
"""
import json
import numpy as np
import pandas as pd
from paths import *


def trap(v, dx):
    v = np.asarray(v, float)
    return dx * (v.sum() - 0.5 * (v[0] + v[-1]))


rows = []
for h, ph in HASH.items():
    for f in sorted((REMOTE04 / f"r2b/metrics/{ph}").glob("*.json")):
        if f.name.startswith("_"):  # release layout: aggregate files next to the 184 records
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        S = np.asarray(d["S_H"], float)
        dx = float(d["delta_s"])
        H = float(d["effective_h_s"])
        level = H * (1.0 - S[-1])
        D = trap(S[-1] - S, dx)
        b = d.get("bootstrap", {})
        card = d.get("card", {})
        rows.append(dict(
            h_s=h, system_id=d["system_id"], family=d.get("family"), group_key=d.get("group_key"),
            library_round=d.get("library_round"), backbone=d.get("backbone"), model_kind=d.get("model_kind"),
            arm_rule=d.get("arm_rule"), arm_value=d.get("arm_value"), commit_threshold=d.get("commit_threshold"),
            seed=d.get("seed"), train_data_unknown=d.get("train_data_unknown"), N_H=d["N_H"], n_clusters=d.get("n_clusters"),
            end_macro=d["end_window_macro_acc"], end_macro_lo=b.get("end_window_macro_acc", {}).get("ci_lo"),
            end_macro_hi=b.get("end_window_macro_acc", {}).get("ci_hi"), end_micro=d["end_window_micro_acc"],
            RMSCD=d["RMSCD"], RMSCD_lo=b.get("RMSCD@H", {}).get("ci_lo"), RMSCD_hi=b.get("RMSCD@H", {}).get("ci_hi"),
            RMSCD_macro=d.get("RMSCD_macro"), S0=S[0], SH=S[-1], S1=d["frozen_family"].get("S_H@1"),
            S3=d["frozen_family"].get("S_H@3"), level_term=level, delay_term=D, check=level + D - d["RMSCD"],
            median_flips=d["flips_median"], mean_flips=d["flips_mean"], flip_rate=d["flip_rate"],
            full_clip_macro=d["full_clip_macro_acc"], rho=d["commit"]["rho"], tau_c=d["commit"]["tau_c_mean"],
            e_c=d["commit"]["e_c"], missing_lookup_rate=d["missing_lookup_rate"], S_H=json.dumps([round(x, 6) for x in S])))
df = pd.DataFrame(rows)
assert df.check.abs().max() < 1e-9, df.check.abs().max()
df.drop(columns="check").to_csv(INTERNAL / "system_table_reference.csv", index=False)
print(df.groupby("h_s").size())
print("max |identity residual|", df.check.abs().max())
t = df[(df.h_s == 10.0)]
print(t.family.value_counts())
nt = t[~t.family.isin(["block", "trivial"])]
print("nontrivial", len(nt))
print("end macro range", nt.end_macro.min(), nt.end_macro.max())
print("RMSCD range", nt.RMSCD.min(), nt.RMSCD.max())
print("delay term median/IQR", nt.delay_term.median(), nt.delay_term.quantile([.25, .75]).tolist(), "max", nt.delay_term.max())
print("SH-S0 median", (nt.SH - nt.S0).median(), (nt.SH - nt.S0).quantile([.1, .9]).tolist())
print("corr RMSCD vs level", np.corrcoef(nt.RMSCD, nt.level_term)[0, 1])
