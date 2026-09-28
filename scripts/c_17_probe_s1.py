# -*- coding: utf-8 -*-
"""Probe A's S1-final layout: new per-row fields and the per-metric ruler keys."""
import json, os, csv, collections
R = os.path.join(os.environ.get("APE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")

print("== metrics dirs:", sorted(os.listdir(os.path.join(R, "metrics"))))
print("== calib dirs:", sorted(os.listdir(os.path.join(R, "calib"))))

idx = json.load(open(os.path.join(R, "report_index.json"), encoding="utf-8"))
rows = idx.get("rows", idx)
print("== report_index rows:", len(rows), "keys:", sorted(rows[0].keys()))
print("   pi_hash set:", sorted(set(r.get("pi_hash") for r in rows)))
print("   h_s set:", sorted(set(r.get("h_s") for r in rows)))
print("   system_family:", collections.Counter(r.get("system_family") for r in rows))
print("   arm_rule:", collections.Counter(r.get("arm_rule") for r in rows))
print("   seed:", collections.Counter(r.get("seed") for r in rows))
for r in rows[:3]:
    print("   sample:", {k: r[k] for k in sorted(r)})

ph = "8ac32aae418b"
d = os.path.join(R, "metrics", ph)
names = sorted(n for n in os.listdir(d) if n.endswith(".json"))
print("== metrics files under", ph, ":", len(names))
one = [n for n in names if n.startswith("postproc")] or names
rec = json.load(open(os.path.join(d, one[0]), encoding="utf-8"))
print("== sample file:", one[0])
print("   top keys:", sorted(rec.keys()))
for k in ("arm_rule", "arm_value", "system_family", "seed", "N_H", "h_s", "audit_split",
          "RMSCD", "end_window_macro_acc", "flips_median"):
    if k in rec:
        print("   %-22s %r" % (k, rec[k]))
print("   card:", json.dumps(rec.get("card", {}), ensure_ascii=False)[:400])

cal = json.load(open(os.path.join(R, "calib", ph, "calibration.json"), encoding="utf-8"))
print("== calibration top keys:", sorted(cal.keys()))
m = cal.get("metrics", {})
print("== metric entries:", sorted(m.keys()))
k0 = "RMSCD@H"
print("== %s keys: %s" % (k0, sorted(m[k0].keys())))
for kk in ("ruler", "ruler_definition", "ruler_degenerate", "verdict", "min_R", "MRD",
           "MRD_plaus", "max_b", "max_s", "reference_range", "n_pairs", "r0"):
    if kk in m[k0]:
        print("   %-20s %r" % (kk, m[k0][kk]))
print("== ruler per metric, all three H:")
for h in sorted(os.listdir(os.path.join(R, "calib"))):
    c = json.load(open(os.path.join(R, "calib", h, "calibration.json"), encoding="utf-8"))
    print("  ", h, "pi0.h_s=", (c.get("pi0") or {}).get("h_s"))
    for mn, mv in sorted((c.get("metrics") or {}).items()):
        if mn.endswith("[with blocks]"):
            continue
        print("      %-24s ruler=%s degenerate=%s verdict=%s"
              % (mn, mv.get("ruler"), mv.get("ruler_degenerate"), mv.get("verdict")))
    print("      ruler_rmscd_legacy:", c.get("ruler_rmscd_legacy"))
