"""Pull the numbers the S1 rebuild report needs out of the summary tables. Read-only."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402

te = pd.read_csv(os.path.join(C.OUTPUTS, "b_summary_test.csv"))
dv = pd.read_csv(os.path.join(C.OUTPUTS, "b_summary_dev.csv"))

mine = te[~te["system_id"].str.startswith("block__")]
print("=== system directory census (mine only) ===")
print("total:", len(mine))
print(mine.groupby("family", observed=True).size().to_string())
print("with __stride suffix:", int(mine["system_id"].str.contains("__stride").sum()))
print("delta_s values:", sorted(mine["delta_s"].unique().tolist()))
print()

print("=== trained systems, test H=10.0, N_H =",
      int(mine.loc[mine["family"].isin(["clip", "prefix"]), "H10_N"].iloc[0]), "===")
tr = mine[mine["family"].isin(["clip", "prefix"])].copy()
tr["seed"] = tr["system_id"].str.extract(r"seed(\d+)")
tr["model"] = tr["system_id"].str.replace(r"__seed\d+", "", regex=True)
piv = tr.pivot(index="model", columns="seed", values="H10_end_macro_acc")
print(piv.round(4).to_string())
print()
print("per model mean over seeds:")
print(piv.mean(axis=1).round(4).to_string())
print()
print("RMSCD@10 (test):")
print(tr.pivot(index="model", columns="seed", values="H10_rmscd").round(3).to_string())
print()

print("=== trivial (test H=10) ===")
tv = mine[mine["family"] == "trivial"][["system_id", "H10_end_macro_acc", "H10_rmscd",
                                        "H10_median_flips"]]
print(tv.round(4).to_string(index=False))
print()

print("=== stride consistency: does a coarse build differ from the fine one? ===")
pp = mine[mine["family"] == "postproc"].copy()
pp["stride"] = pp["system_id"].str.extract(r"__stride(\d+)").fillna("1")
pp["stem"] = pp["system_id"].str.replace(r"__stride\d+", "", regex=True)
tab = pp.pivot(index="stem", columns="stride", values="H10_end_macro_acc")
print(tab.round(4).to_string())
print()
print("=== commit triplet at the two pre-registered rho levels (test, H=10) ===")
cm = mine[(mine["family"] == "commit") & (~mine["system_id"].str.contains("__stride"))]
cols = [c for c in ["system_id", "commit_rho", "commit_tau_c", "commit_e_c"] if c in cm.columns]
sel = cm[cm["system_id"].str.contains("msp0p7|msp0p9")][cols]
print(sel.round(4).to_string(index=False))
print()
print("monotonicity violations across all commit systems:",
      int(mine.get("committed_monotone_violations", pd.Series([0])).fillna(0).sum()))
