"""Reproducible CLI smoke with synthetic predictions, uneven clusters and three seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ape.g6_pipeline import rmscd_contributions


def fixture(delays: dict[str, int]) -> dict[str, Any]:
    """Create scores via prediction integration, never invent real model readings."""
    cohort = [
        {"video_id": f"synthetic_{i:02}", "source_cluster_id": f"source_{min(i, 2)}", "class_code": str(i % 2)}
        for i in range(12)
    ]
    result = {
        "name": "synthetic_pipeline_smoke",
        "split_role": "synthetic",
        "metric": "RMSCD@H_norm",
        "provenance": {
            "cohort_sha256": hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest(),
            "protocol_id": "synthetic_uniform_grid_h10_delta1",
        },
        "systems": {},
        "observations": [],
    }
    labels = np.arange(12) % 2
    for family, delay in delays.items():
        for seed in range(3):
            sid = f"{family}_{seed}"
            pred = np.where(
                np.arange(11)[None, :] >= delay + (np.arange(12)[:, None] + seed) % 2,
                labels[:, None],
                1 - labels[:, None],
            )
            values = rmscd_contributions(pred, labels)
            result["systems"][sid] = {
                "backbone": "synthetic",
                "model_kind": "step",
                "arm_rule": family,
                "commit_threshold": None,
                "seed": str(seed),
                "library_round": "synthetic",
            }
            result["observations"].extend(
                {**row, "system_id": sid, "value": float(value)} for row, value in zip(cohort, values)
            )
    return result


def main() -> None:
    """Run the public CLI for agreement, reversal, ties and missing families."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    a_path = args.out / "track_a.json"
    a_path.write_text(json.dumps(fixture({"fast": 2, "middle": 5, "slow": 8}), indent=2), encoding="utf-8")
    cases = {
        "agree": ({"fast": 2, "middle": 5, "slow": 8}, {"agree": 3}),
        "reverse": ({"fast": 8, "middle": 5, "slow": 2}, {"reverse": 3}),
        "tie": ({"fast": 5, "middle": 5, "slow": 5}, {"tie": 3}),
        "missing": ({"fast": 2, "middle": 5}, {"agree": 1, "uncertain": 2}),
    }
    receipts = {}
    selected_hashes = set()
    for name, (delays, expected) in cases.items():
        b_path = args.out / f"track_b_{name}.json"
        b_path.write_text(json.dumps(fixture(delays), indent=2), encoding="utf-8")
        report_path = args.out / f"report_{name}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ape.g6_pipeline",
                "--config",
                str(args.config),
                "--track-a",
                str(a_path),
                "--track-b",
                str(b_path),
                "--output",
                str(report_path),
            ],
            check=True,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        counts = {key: value for key, value in report["rates"]["counts"].items() if value}
        if counts != expected or report["formal_verdict"] != "not_evaluable":
            raise AssertionError((name, counts, expected))
        selected_hashes.add(report["selected_pairs_sha256"])
        receipts[name] = {
            "expected": expected,
            "observed": counts,
            "passed": True,
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
    if len(selected_hashes) != 1:
        raise AssertionError("track B altered track A selection")
    receipt = {
        "is_experiment": False,
        "uses_real_model_outputs": False,
        "cases": receipts,
        "selection_identical_across_b_variants": True,
        "all_passed": True,
    }
    (args.out / "smoke_verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
