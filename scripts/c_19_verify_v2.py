# -*- coding: utf-8 -*-
"""Verify the numbers the v2 amendment asserts, rather than transcribing them."""
import csv, yaml, itertools, os

ROOT = os.environ.get("APE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M = os.path.join(ROOT, "data/manifest/manifest_real.csv")
rows = list(csv.DictReader(open(M, encoding="utf-8")))

print("== A3: the boundary clip at H=10.0")
tgt = [r for r in rows if r["video_id"] == "KUBbn-T3XYI_00"]
for r in tgt:
    lp = float(r["post_anchor_length_s"])
    print("   %s split=%s L+=%.17g  L+>=10.0 -> %s | with 1e-9 tol -> %s"
          % (r["video_id"], r["split"], lp, lp >= 10.0, lp >= 10.0 - 1e-9))
    print("   duration=%s anchor=%s" % (r["duration_s"], r["anchor_s"]))
for h in (4.0, 10.0, 21.5):
    strict = sum(1 for r in rows if r["split"] == "test" and float(r["post_anchor_length_s"]) >= h)
    tol = sum(1 for r in rows if r["split"] == "test" and float(r["post_anchor_length_s"]) >= h - 1e-9)
    print("   H=%-5s test cohort: strict=%d  with tol=%d" % (h, strict, tol))

print("== A5: grid arithmetic")
p = yaml.safe_load(open(os.path.join(ROOT, "protocol/pi0.yaml"), encoding="utf-8"))
pl = p["perturb"]["plaus"]
pi0 = {"eps_sys_s": p["delta_s"] * 0 + 0.0, "eps_jit_sd_s": 0.0,
       "delta_s": p["delta_s"], "h_s": 10.0}
axes = {}
for k in ("eps_sys_s", "eps_jit_sd_s", "delta_s", "h_s"):
    vals = sorted(set(list(pl[k]) + [pi0[k]]))
    axes[k] = vals
    print("   plaus axis %-14s -> %s (%d)" % (k, vals, len(vals)))
prod = 1
for v in axes.values():
    prod *= len(v)
print("   plaus product =", prod)
full = {k: sorted(set(p["perturb"][k])) for k in ("eps_sys_s", "eps_jit_sd_s", "delta_s", "h_s")}
full["h_s"] = [4.0, 10.0, 21.5]
n_axis = sum(len(v) for v in full.values()) - (len(full) - 1)
print("   full axes sizes:", {k: len(v) for k, v in full.items()}, "-> axis points =", n_axis)
