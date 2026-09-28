# -*- coding: utf-8 -*-
"""Print the shape of track A's real outputs: key names, nesting, dtypes."""
import json, os, csv

ROOT = os.path.join(os.environ.get("APE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")

def shape(o, depth=0, maxd=3, prefix=""):
    pad = "  " * depth
    if isinstance(o, dict):
        for k in list(o.keys())[:40]:
            v = o[k]
            if isinstance(v, dict):
                print("%s%s: dict(%d) keys=%s" % (pad, k, len(v), list(v.keys())[:12]))
                if depth < maxd:
                    shape(v, depth + 1, maxd)
            elif isinstance(v, list):
                print("%s%s: list(%d) head=%s" % (pad, k, len(v), v[:3]))
                if v and isinstance(v[0], dict) and depth < maxd:
                    print("%s  [0] keys=%s" % (pad, list(v[0].keys())))
            else:
                print("%s%s: %r" % (pad, k, v))

p = os.path.join(ROOT, "metrics", "0dcc423a8192", "standin_early.json")
print("=" * 20, p)
shape(json.load(open(p, encoding="utf-8")))

p = os.path.join(ROOT, "metrics", "0dcc423a8192", "block__lock__d0-3.json")
print("=" * 20, "block file top-level keys")
print(sorted(json.load(open(p, encoding="utf-8")).keys()))

p = os.path.join(ROOT, "metrics", "0dcc423a8192", "_pairs.json")
print("=" * 20, p)
shape(json.load(open(p, encoding="utf-8")), maxd=1)

p = os.path.join(ROOT, "calib", "0dcc423a8192", "calibration.json")
print("=" * 20, p)
shape(json.load(open(p, encoding="utf-8")), maxd=3)

p = os.path.join(ROOT, "report_index.json")
print("=" * 20, p)
shape(json.load(open(p, encoding="utf-8")), maxd=2)

d = os.path.join(ROOT, "phenomena", "0dcc423a8192")
print("=" * 20, d, os.listdir(d)[:20])
f = sorted(os.listdir(d))[0]
print("--- ", f)
shape(json.load(open(os.path.join(d, f), encoding="utf-8")), maxd=2)

p = os.path.join(ROOT, "calib", "0dcc423a8192", "scan_table.csv")
print("=" * 20, p)
with open(p, encoding="utf-8") as fh:
    r = csv.reader(fh)
    for i, row in enumerate(r):
        print(row)
        if i >= 3:
            break

p = os.path.join(ROOT, "calib", "0dcc423a8192", "R_RMSCDatH.csv")
print("=" * 20, p)
with open(p, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        print(line.rstrip())
        if i >= 4:
            break
