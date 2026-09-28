"""Recover per-stage wall-clock of the S1 rebuild from artefact mtimes. Read-only."""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402


def mt(p):
    return os.path.getmtime(p)


def fmt(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S")


groups = {"ckpt": [], "trained": [], "postproc": [], "postproc_stride": [],
          "commit": [], "commit_stride": [], "trivial": []}

for f in os.listdir(C.CKPT_DIR):
    if f.endswith(".pt"):
        groups["ckpt"].append(mt(os.path.join(C.CKPT_DIR, f)))

for d in os.listdir(C.ANSWERS_DIR):
    p = os.path.join(C.ANSWERS_DIR, d, "answers.csv")
    if not os.path.exists(p):
        continue
    t = mt(p)
    if d.startswith("block__"):
        continue
    stride = "__stride" in d
    if d.startswith(("clip__", "prefix__")):
        groups["trained"].append(t)
    elif d.startswith("postproc__"):
        groups["postproc_stride" if stride else "postproc"].append(t)
    elif d.startswith("commit__"):
        groups["commit_stride" if stride else "commit"].append(t)
    elif d.startswith("trivial__"):
        groups["trivial"].append(t)

allt = []
for k, v in groups.items():
    if not v:
        continue
    allt += v
    print(f"{k:18s} n={len(v):3d}  first={fmt(min(v))}  last={fmt(max(v))}  "
          f"span={max(v) - min(v):7.1f}s")

print()
print(f"whole rebuild: {fmt(min(allt))} -> {fmt(max(allt))} = {(max(allt) - min(allt)) / 60:.1f} min")

for name in ("b_summary_test.csv", "b_summary_dev.csv"):
    p = os.path.join(C.OUTPUTS, name)
    if os.path.exists(p):
        print(f"{name}: {fmt(mt(p))}")

tot = 0
n = 0
for d in os.listdir(C.ANSWERS_DIR):
    p = os.path.join(C.ANSWERS_DIR, d, "answers.csv")
    if os.path.exists(p) and not d.startswith("block__"):
        tot += os.path.getsize(p)
        n += 1
print(f"\nanswer matrices owned: {n}, total {tot / 1e9:.2f} GB")
fs = sum(os.path.getsize(os.path.join(C.FEATURES_DIR, f))
         for f in os.listdir(C.FEATURES_DIR))
print(f"feature cache: {fs / 1e9:.2f} GB")
