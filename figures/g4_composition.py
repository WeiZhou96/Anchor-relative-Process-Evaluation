"""Composition of H=10 heterogeneous tied pairs and replicated groups by system family (post hoc, descriptive)."""
import json
import numpy as np
import pandas as pd
from paths import *

t = pd.read_csv(INTERNAL / "system_table_reference.csv")
g4 = json.loads((REMOTE04 / "r2b/gates/g4.json").read_text(encoding="utf-8"))
res = {}
for bh in g4["by_h"]:
    h = float(bh["h_s"]); tt = t[t.h_s == h].set_index("system_id")
    P = pd.DataFrame(bh["pairs"])
    P["fa"] = tt.loc[P.system_a, "family"].values; P["fb"] = tt.loc[P.system_b, "family"].values
    P["lvl"] = tt.loc[P.system_a, "level_term"].values - tt.loc[P.system_b, "level_term"].values
    P["dly"] = tt.loc[P.system_a, "delay_term"].values - tt.loc[P.system_b, "delay_term"].values
    het = P[P.significant]
    nc = het[(het.fa != "commit") & (het.fb != "commit")]
    sg = [g for g in bh["seed_groups"] if g["pass"]]
    def fam_of_variant(v):
        return "commit" if ("commit" in v or "ringel" in v or "teaser" in v) else "noncommit"
    nc_groups = [g for g in sg if fam_of_variant(g["variant_a"]) == "noncommit" and fam_of_variant(g["variant_b"]) == "noncommit"]
    res[str(h)] = dict(
        heterogeneous=int(len(het)), heterogeneous_no_commit=int(len(nc)), heterogeneous_no_commit_cross_base=int(nc.cross_base.sum()),
        no_commit_delay_dominant=int((nc.dly.abs() > nc.lvl.abs()).sum()),
        no_commit_delay_same_sign=int((np.sign(nc.dly) == np.sign(nc["diff"])).sum()),
        no_commit_median_abs_diff=float(nc["diff"].abs().median()) if len(nc) else None,
        replicated_groups=len(sg), replicated_groups_no_commit=len(nc_groups),
        replicated_no_commit_examples=[(g["variant_a"], g["variant_b"]) for g in nc_groups[:12]])
    print(h, json.dumps(res[str(h)], indent=0))
(INTERNAL / "g4_composition.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
