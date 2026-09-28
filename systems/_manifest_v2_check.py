"""Inspect the frozen v2 manifest before retraining. Read-only."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402

df = pd.read_csv(C.MANIFEST_C)
print("shape:", df.shape)
print("columns:", list(df.columns))
print()
print("split counts:")
print(df["split"].value_counts().to_string())
if "split_official" in df.columns:
    print()
    print("split vs split_official:")
    print(pd.crosstab(df["split"], df["split_official"]).to_string())
print()
print("clusters per split:")
print(df.groupby("split", observed=True)["source_cluster_id"].nunique().to_string())
strad = df.groupby("source_cluster_id", observed=True)["split"].nunique()
print("clusters straddling splits:", int((strad > 1).sum()))
print()
print("class_code x split:")
print(pd.crosstab(df["split"], df["class_code"]).to_string())
print()
print("train class prior order:")
tc = df.loc[df["split"] == "train", "class_code"].value_counts().sort_index()
print(tc.to_string())
print("majority class:", int(tc.idxmax()), C.CLASS_NAMES[int(tc.idxmax())])
print()
post = df["post_anchor_length_s"]
for h in (3.0, 6.0, 10.0):
    print(f"eligible@H={h}: all={int((post >= h).sum())} "
          f"test={int(((post >= h) & df['split'].eq('test')).sum())} "
          f"dev={int(((post >= h) & df['split'].eq('dev')).sum())} "
          f"train={int(((post >= h) & df['split'].eq('train')).sum())}")
print()
print("feature cache present:", len(C.available_feature_ids()))
have = set(C.available_feature_ids())
print("manifest rows without features:", int((~df["video_id"].astype(str).isin(have)).sum()))
