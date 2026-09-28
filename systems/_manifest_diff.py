"""Compare the C track's manifest with B's temporary one. Read-only."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402

c = pd.read_csv(C.MANIFEST_C)
b = pd.read_csv(C.TMP_MANIFEST_B)
print("C manifest shape:", c.shape)
print("C columns:", list(c.columns))
print()
print("B manifest shape:", b.shape)
print("B columns:", list(b.columns))
print()
print("columns in C not in B:", [x for x in c.columns if x not in b.columns])
print("columns in B not in C:", [x for x in b.columns if x not in c.columns])
print()
for name, df in (("C", c), ("B", b)):
    if "split" in df.columns:
        print(f"{name} split counts:")
        print(df["split"].value_counts().to_string())
print()
if "video_id" in c.columns and "video_id" in b.columns:
    m = c[["video_id", "split"]].merge(b[["video_id", "split"]], on="video_id",
                                       suffixes=("_c", "_b"))
    print("joined rows:", len(m))
    print(pd.crosstab(m["split_c"], m["split_b"]).to_string())
    print()
    print("dev sets identical:",
          set(m.loc[m["split_c"] == "dev", "video_id"]) == set(m.loc[m["split_b"] == "dev", "video_id"]))
for col in ("anchor_s", "post_anchor_length_s", "class_code", "fps", "duration_s"):
    if col in c.columns and col in b.columns:
        mm = c[["video_id", col]].merge(b[["video_id", col]], on="video_id", suffixes=("_c", "_b"))
        if mm[f"{col}_c"].dtype.kind in "if":
            d = (mm[f"{col}_c"] - mm[f"{col}_b"]).abs().max()
            print(f"max abs diff {col}: {d}")
        else:
            print(f"{col} mismatches: {(mm[f'{col}_c'] != mm[f'{col}_b']).sum()}")
if "decode_ok" in c.columns:
    print("C decode_ok False:", int((~c['decode_ok'].astype(str).str.lower().isin(['true','1','yes'])).sum()))
