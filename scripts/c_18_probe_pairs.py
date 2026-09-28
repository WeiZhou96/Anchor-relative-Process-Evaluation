# -*- coding: utf-8 -*-
"""Where do the per-H rulers live? calib exists for one H only, so check _pairs.json."""
import json, os
R = os.path.join(os.environ.get("APE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
for ph in sorted(os.listdir(os.path.join(R, "metrics"))):
    p = os.path.join(R, "metrics", ph, "_pairs.json")
    if not os.path.isfile(p):
        print(ph, "NO _pairs.json"); continue
    o = json.load(open(p, encoding="utf-8"))
    print("==", ph, "h_s=", (o.get("pi") or {}).get("h_s"))
    print("   top keys:", sorted(o.keys()))
    for k in ("ruler_median_progress_diff_RMSCD", "rulers_by_metric", "ruler_definition",
              "n_boot", "alpha", "correction", "p_method", "correction_resolution",
              "audit_split", "n_systems"):
        if k in o:
            print("   %-34s %r" % (k, o[k]))
    for k in ("systems_real", "systems_block", "window_end_tied_pairs"):
        v = o.get(k)
        if isinstance(v, list):
            print("   %-34s len=%d head=%s" % (k, len(v), v[:2]))
    for m in ("RMSCD@H", "median_flips"):
        if m in o and isinstance(o[m], dict):
            print("   %s: keys=%s n_sig=%s n_tests=%s"
                  % (m, sorted(o[m].keys()), len(o[m].get("significant_pairs", [])),
                     len(o[m].get("tests", []))))
c = json.load(open(os.path.join(R, "calib", "8ac32aae418b", "calibration.json"), encoding="utf-8"))
print("== calibration.rulers_by_metric:", json.dumps(c.get("rulers_by_metric"), ensure_ascii=False))
print("== ruler_definition:", c.get("ruler_definition"))
print("== p_method:", c.get("p_method"), "| correction_resolution:", c.get("correction_resolution"))
print("== audit_split:", c.get("audit_split"), "| n_clips_scored:", c.get("n_clips_scored"),
      "| n_pi:", c.get("n_pi"), "| n_systems:", c.get("n_systems"), "| scan_mode:", c.get("scan_mode"))
print("== stride_notes:", json.dumps(c.get("stride_notes"), ensure_ascii=False)[:600])
print("== stride_reruns:", json.dumps(c.get("stride_reruns"), ensure_ascii=False)[:400])
