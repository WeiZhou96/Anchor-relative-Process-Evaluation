"""Which source clusters straddle the official IID split? Read-only diagnostic for A and C."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402

man = C.load_manifest(C.TMP_MANIFEST_B)
raw = pd.read_csv(C.METADATA_REAL)
raw["video_id"] = raw["path"].astype(str).str.rsplit("/", n=1).str[-1].str.replace(
    r"\.mp4$", "", regex=True)
raw["cluster"] = [C.source_cluster_id(v) for v in raw["video_id"]]

print("== straddling in the OFFICIAL split_in_distribution (before any dev carve) ==")
g = raw.groupby("cluster", observed=True)["split_in_distribution"].nunique()
bad = g[g > 1].index.tolist()
print(f"clusters straddling official IID train/test: {len(bad)} -> {bad}")
for cl in bad:
    sub = raw[raw["cluster"] == cl][["video_id", "type", "split_in_distribution"]]
    print(sub.to_string(index=False))

print()
print("== straddling in split_geo_aware ==")
g2 = raw.groupby("cluster", observed=True)["split_geo_aware"].nunique()
print(f"clusters straddling geo-aware split: {int((g2 > 1).sum())}")

print()
print("== cluster size distribution ==")
sz = raw.groupby("cluster", observed=True).size()
print(sz.value_counts().sort_index().to_string())
print(f"clusters total: {len(sz)}  clips total: {len(raw)}")

print()
print("== label agreement inside a cluster ==")
lab = raw.groupby("cluster", observed=True)["type"].nunique()
print(f"clusters with more than one type label: {int((lab > 1).sum())}")

print()
print("== after the B dev carve ==")
g3 = man.groupby("source_cluster_id", observed=True)["split"].nunique()
print(f"clusters straddling any two of train/dev/test: {int((g3 > 1).sum())}")
print(man.groupby("split", observed=True).size().to_string())
