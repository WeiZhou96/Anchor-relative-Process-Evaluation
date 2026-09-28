# -*- coding: utf-8 -*-
"""How many source clusters carry more than one collision type, under each scope?
The freeze note must quote a number whose scope is stated, so all of them are printed."""
import csv, collections, os
M = os.path.join(os.environ.get("APE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/manifest/manifest_real.csv")
rows = list(csv.DictReader(open(M, encoding="utf-8")))

def count(sub, label):
    d = collections.defaultdict(set)
    for r in sub:
        d[r["source_cluster_id"]].add(r["class_name"])
    multi = [c for c, v in d.items() if len(v) > 1]
    sizes = collections.Counter(len(v) for v in d.values())
    print("%-46s clusters=%5d  multi-type=%4d  clips=%5d  type-count hist=%s"
          % (label, len(d), len(multi), len(sub), dict(sorted(sizes.items()))))
    return multi

count(rows, "all clips (v2)")
for sp in ("train", "dev", "test"):
    count([r for r in rows if r["split"] == sp], "split=%s" % sp)
for h in (4.0, 10.0, 21.5, 4.11, 10.03, 21.82):
    sub = [r for r in rows if float(r["post_anchor_length_s"]) >= h]
    count(sub, "eligible cohort L+ >= %.2f" % h)
    count([r for r in sub if r["split"] == "test"], "  ...and split=test, H=%.2f" % h)
