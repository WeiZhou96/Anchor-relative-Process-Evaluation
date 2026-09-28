"""Calibrate the actual primary comparison family under its empirical cluster population."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ape.method_analysis import paired_components, simultaneous_band, trajectory_components  # noqa: E402
from ape.r2c import Context  # noqa: E402
from scripts.method_experiments import load_inputs, pairs_for, select_correct  # noqa: E402


def run(source: Path, out: Path, trials: int, bootstrap: int) -> None:
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    context = Context(source, False)
    context.nontrivial = [sid for sid in context.nontrivial if context.cards[sid]["family"] in ["clip", "prefix"]]
    assert len(context.nontrivial) == 27
    cache, _ = load_inputs(context, out)
    mask = context.test.post_anchor_length_s.to_numpy() >= 10 - 1e-9
    cohort = context.test.loc[mask]
    components = {}
    for sid in context.nontrivial:
        correct, times = select_correct(cache[sid], mask, 10)
        components[sid] = trajectory_components(correct, times)
    pairs = pairs_for(context.nontrivial, context.meta)
    assert len(pairs) == 108
    values = np.concatenate([paired_components(components[a], components[b], 10) for a, b in pairs], axis=1)
    labels = cohort.class_code.to_numpy()
    clusters, inverse = np.unique(cohort.source_cluster_id, return_inverse=True)
    counts = np.zeros((len(clusters), 5))
    sums = np.zeros((len(clusters), 5, values.shape[1]))
    for i, (cluster, category) in enumerate(zip(inverse, labels)):
        counts[cluster, category] += 1
        sums[cluster, category] += values[i]
    truth = {"micro": sums.sum((0, 1)) / counts.sum(), "macro": (sums.sum(0) / counts.sum(0)[:, None]).mean(0)}
    rng = np.random.default_rng(20260930)
    records = []
    for size in [len(clusters), 300]:
        for trial in range(trials):
            sampled = rng.integers(len(clusters), size=size)
            n, x = counts[sampled], sums[sampled]
            if (n.sum(0) == 0).any():
                raise RuntimeError("An outer sample lacks a target class; do not silently redefine macro")
            point = {"micro": x.sum((0, 1)) / n.sum(), "macro": (x.sum(0) / n.sum(0)[:, None]).mean(0)}
            multiplicities = rng.multinomial(size, np.full(size, 1 / size), size=bootstrap)
            denominator = multiplicities @ n
            invalid = (denominator == 0).any(1)
            redrawn = 0
            while invalid.any():
                redrawn += int(invalid.sum())
                multiplicities[invalid] = rng.multinomial(size, np.full(size, 1 / size), size=int(invalid.sum()))
                denominator = multiplicities @ n
                invalid = (denominator == 0).any(1)
            reps = {
                "micro": (multiplicities @ x.sum(1)) / denominator.sum(1)[:, None],
                "macro": sum((multiplicities @ x[:, c]) / denominator[:, c, None] for c in range(5)) / 5,
            }
            for mode in ["micro", "macro"]:
                df = size - 1 if mode == "micro" else int((n > 0).sum(0).min()) - 1
                band = simultaneous_band(point[mode], reps[mode], degrees_freedom=df)
                covered = (band["lower"] <= truth[mode] + 1e-10) & (truth[mode] <= band["upper"] + 1e-10)
                usable = ~band["zero_variance"]
                wrong_direction = usable & (
                    ((band["lower"] > 0) & (truth[mode] < -1e-10)) | ((band["upper"] < 0) & (truth[mode] > 1e-10))
                )
                records.append(
                    {
                        "clusters": size,
                        "trial": trial,
                        "weighting": mode,
                        "covered": bool(covered.all()),
                        "support_guard_covered": bool((covered | ~usable).all()),
                        "wrong_direction": bool(wrong_direction.any()),
                        "width": float(np.mean(band["upper"] - band["lower"])),
                        "zero_coordinates": int((~usable).sum()),
                        "redrawn": redrawn,
                    }
                )
            if (trial + 1) % 25 == 0:
                print("PLASMODE", size, trial + 1, "/", trials, flush=True)
        pd.DataFrame(records).to_csv(out / "trials.csv", index=False)
    frame = pd.DataFrame(records)
    summaries = []
    for (size, mode), part in frame.groupby(["clusters", "weighting"]):
        p = float(part.covered.mean())
        summaries.append(
            {
                "clusters": int(size),
                "weighting": mode,
                "trials": len(part),
                "coverage": p,
                "mcse": float(np.sqrt(p * (1 - p) / len(part))),
                "wrong_direction_fraction": float(part.wrong_direction.mean()),
                "mean_width": float(part.width.mean()),
                "support_guard_coverage": float(part.support_guard_covered.mean()),
                "mean_zero_coordinates": float(part.zero_coordinates.mean()),
            }
        )
    result = {
        "seed": 20260930,
        "code_head": code,
        "started_unix": started,
        "finished_unix": time.time(),
        "complete": True,
        "bootstrap": bootstrap,
        "pairs": len(pairs),
        "coordinates": values.shape[1],
        "truth": {k: v.tolist() for k, v in truth.items()},
        "summaries": summaries,
        "claim": (
            "conditional on empirical source-cluster population, "
            "not external validation or a universal coverage guarantee"
        ),
    }
    (out / "SUMMARY.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    run(args.source_root, args.out, args.trials, args.bootstrap)
