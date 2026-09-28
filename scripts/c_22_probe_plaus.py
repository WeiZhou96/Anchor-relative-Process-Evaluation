# -*- coding: utf-8 -*-
"""Is the plaus scan a filled 4-D product grid? Figure 3 depends on the answer."""
import json, csv, os, itertools, collections
R = os.path.join(os.environ.get("APE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
D = os.path.join(R, "calib_plaus", "8ac32aae418b")

c = json.load(open(os.path.join(D, "calibration.json"), encoding="utf-8"))
print("top keys:", sorted(c.keys()))
for k in ("pi0_hash", "scan_mode", "n_pi", "n_systems", "r0", "audit_split",
          "p_method", "ruler_definition"):
    if k in c:
        print("  %-18s %r" % (k, c[k]))
print("  pi0:", c.get("pi0"))
print("  rulers_by_metric:", c.get("rulers_by_metric"))
m = c.get("metrics", {})
print("metrics keys:", sorted(m.keys()))
for name in sorted(k for k in m if not k.endswith("[with blocks]")):
    e = m[name]
    print("  %-22s min_R=%s MRD_plaus=%s MRD=%s verdict=%s ruler=%s degen=%s n_pairs=%s"
          % (name, e.get("min_R"), e.get("MRD_plaus"), e.get("MRD"), e.get("verdict"),
             e.get("ruler"), e.get("ruler_degenerate"), e.get("n_pairs")))

rows = list(csv.DictReader(open(os.path.join(D, "R_RMSCDatH.csv"), encoding="utf-8")))
print("R_RMSCDatH rows:", len(rows), "cols:", rows[0].keys())
KN = ("eps_sys_s", "eps_jit_sd_s", "delta_s", "h_s")
axes = {k: sorted(set(float(r[k]) for r in rows)) for k in KN}
for k, v in axes.items():
    print("  axis %-14s %s" % (k, v))
prod = 1
for v in axes.values():
    prod *= len(v)
combos = set(tuple(float(r[k]) for k in KN) for r in rows)
print("  product=%d distinct combos present=%d  FILLED=%s" % (prod, len(combos), prod == len(combos)))

pi0 = c.get("pi0") or {}
for xk, yk in (("eps_sys_s", "delta_s"), ("eps_sys_s", "h_s"), ("delta_s", "h_s")):
    others = [k for k in KN if k not in (xk, yk)]
    sel = [r for r in rows if all(abs(float(r[k]) - float(pi0[k])) < 1e-9 for k in others)]
    xs = sorted(set(float(r[xk]) for r in sel)); ys = sorted(set(float(r[yk]) for r in sel))
    print("  slice (%s,%s): %d rows, grid %dx%d, filled=%s"
          % (xk, yk, len(sel), len(xs), len(ys), len(sel) == len(xs) * len(ys)))

st = os.path.join(D, "scan_table.csv")
srows = list(csv.DictReader(open(st, encoding="utf-8")))
print("scan_table rows:", len(srows), "cols:", list(srows[0].keys()))
print("  metrics:", collections.Counter(r["metric"] for r in srows))
