#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-off probe: how are ACCIDENT real clip file names structured?

Compares the conservative cluster rule (strip one trailing numeric segment) against the
greedy rule (strip every trailing numeric segment) and prints the disagreement so the
cluster definition can be chosen on evidence rather than on a guess.
"""
import csv, os, sys
from collections import Counter, defaultdict

CSV = os.path.join(os.environ.get("APE_ACCIDENT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "ACCIDENT_2026")), "metadata-real.csv")

def strip_one(v):
    if "_" in v:
        h, t = v.rsplit("_", 1)
        if t.isdigit() and h:
            return h
    return v

def strip_all(v):
    cur = v
    while True:
        nxt = strip_one(cur)
        if nxt == cur:
            return cur
        cur = nxt

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
vids = [os.path.splitext(os.path.basename(r["path"]))[0] for r in rows]
print("clips:", len(vids))

seg = Counter(v.count("_") for v in vids)
print("underscore count histogram:", dict(sorted(seg.items())))

print("\nexamples with >=2 underscores:")
for v in [v for v in vids if v.count("_") >= 2][:15]:
    print("  ", v, "| strip_one ->", strip_one(v), "| strip_all ->", strip_all(v))

print("\nexamples with 1 underscore:")
for v in [v for v in vids if v.count("_") == 1][:10]:
    print("  ", v, "| strip_one ->", strip_one(v), "| strip_all ->", strip_all(v))

c1 = defaultdict(list)
c2 = defaultdict(list)
for v in vids:
    c1[strip_one(v)].append(v)
    c2[strip_all(v)].append(v)
print("\nclusters strip_one:", len(c1), " clusters strip_all:", len(c2))

# clusters that greedy stripping merges together but the conservative rule keeps apart
merged = {k: v for k, v in c2.items() if len(set(strip_one(x) for x in v)) > 1}
print("greedy merges", len(merged), "cluster groups; showing 10:")
for k, v in list(sorted(merged.items()))[:10]:
    print("  ", k, "<-", sorted(set(strip_one(x) for x in v)))

# risk of over-merging: greedy clusters whose members' full ids look unrelated
sizes = Counter(len(v) for v in c2.values())
print("\ngreedy cluster size histogram:", dict(sorted(sizes.items())))
