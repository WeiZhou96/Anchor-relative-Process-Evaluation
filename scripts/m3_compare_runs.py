"""Check that a rerun reproduces every numeric value shared with the first M3 run (DEVIATIONS.md #3)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

first, second = Path(sys.argv[1]), Path(sys.argv[2])
report = {"files": {}, "max_abs_diff": 0.0, "text_mismatches": 0}
for f in sorted(first.glob("*.csv")):
    g = second / f.name
    if not g.exists():
        report["files"][f.name] = "missing in rerun"
        continue
    a, b = pd.read_csv(f), pd.read_csv(g)
    if len(a) != len(b):
        report["files"][f.name] = f"row count {len(a)} vs {len(b)}"
        continue
    worst, text_bad, compared = 0.0, 0, 0
    for col in a.columns:
        if col not in b.columns:
            continue
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            x, y = a[col].to_numpy(float), b[col].to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            if np.any(np.isnan(x) != np.isnan(y)):
                text_bad += 1
            d = np.abs(np.where(both_nan, 0.0, x - y))
            worst = max(worst, float(np.nanmax(d)) if len(d) else 0.0)
        elif a[col].dtype == object and b[col].dtype == object:
            text_bad += int((a[col].astype(str) != b[col].astype(str)).sum())
        compared += 1
    report["files"][f.name] = {"columns_compared": compared, "max_abs_diff": worst, "text_mismatches": text_bad}
    report["max_abs_diff"] = max(report["max_abs_diff"], worst)
    report["text_mismatches"] += text_bad


def walk(x, y, path=""):
    worst, bad = 0.0, 0
    if isinstance(x, dict) and isinstance(y, dict):
        for k in x:
            if k in y:
                w, b = walk(x[k], y[k], f"{path}/{k}")
                worst, bad = max(worst, w), bad + b
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            w, b = walk(u, v, f"{path}[{i}]")
            worst, bad = max(worst, w), bad + b
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        worst = abs(float(x) - float(y))
    elif x != y:
        bad = 1
    return worst, bad


for name in ["SUMMARY.json", "A_tied_noncommit.json", "B_example_pair.json"]:
    w, b = walk(json.loads((first / name).read_text()), json.loads((second / name).read_text()))
    report["files"][name] = {"max_abs_diff": w, "mismatches": b}
    report["max_abs_diff"] = max(report["max_abs_diff"], w)
    report["text_mismatches"] += b
report["pass"] = report["max_abs_diff"] <= 1e-12 and report["text_mismatches"] == 0
(second / "COMPARE_WITH_FIRST_RUN.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "files"}), flush=True)
for k, v in report["files"].items():
    print(k, v)
