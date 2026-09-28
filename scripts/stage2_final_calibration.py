"""Empirical calibration of all predeclared primary and protocol comparison families."""

from __future__ import annotations

import argparse
import itertools
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


def calibrate(
    values: np.ndarray,
    cohort: pd.DataFrame,
    horizon: int,
    kind: str,
    trials: int,
    bootstrap: int,
    seed: int,
    out: Path,
) -> list[dict]:
    labels = cohort.class_code.to_numpy()
    clusters, inverse = np.unique(cohort.source_cluster_id, return_inverse=True)
    size = len(clusters)
    counts = np.zeros((size, 5))
    sums = np.zeros((size, 5, values.shape[1]))
    supports = np.zeros((size, values.shape[1]), dtype=bool)
    for i, (cluster, category) in enumerate(zip(inverse, labels)):
        counts[cluster, category] += 1
        sums[cluster, category] += values[i]
        supports[cluster] |= np.abs(values[i]) > 1e-12
    truth = {"micro": sums.sum((0, 1)) / counts.sum(), "macro": (sums.sum(0) / counts.sum(0)[:, None]).mean(0)}
    tag = f"{kind}_H{horizon}"
    (out / f"{tag}_population.json").write_text(
        json.dumps(
            {
                "n": len(cohort),
                "clusters": size,
                "nonzero_cluster_counts": supports.sum(0).tolist(),
                "truth": {k: v.tolist() for k, v in truth.items()},
            },
            indent=2,
        )
    )
    rng = np.random.default_rng(seed)
    records = []
    for trial in range(trials):
        sampled = rng.integers(size, size=size)
        n, x = counts[sampled], sums[sampled]
        assert np.all(n.sum(0) > 0)
        low_support = supports[sampled].sum(0) < 20
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
            uninformative = band["zero_variance"] | low_support
            wrong = (~uninformative) & (
                ((band["lower"] > 0) & (truth[mode] < -1e-10)) | ((band["upper"] < 0) & (truth[mode] > 1e-10))
            )
            records.append(
                {
                    "trial": trial,
                    "weighting": mode,
                    "covered": bool(covered.all()),
                    "information_guard_covered": bool((covered | uninformative).all()),
                    "wrong_direction": bool(wrong.any()),
                    "width": float(np.mean(band["upper"] - band["lower"])),
                    "information_guard_width": float(
                        np.mean(np.where(uninformative, 2 * horizon, band["upper"] - band["lower"]))
                    ),
                    "uninformative_coordinates": int(uninformative.sum()),
                    "redrawn": redrawn,
                }
            )
        if (trial + 1) % 100 == 0:
            print(tag, trial + 1, "/", trials, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / f"{tag}_trials.csv", index=False)
    summaries = []
    for mode, part in frame.groupby("weighting"):
        row = {
            "family": tag,
            "weighting": mode,
            "coordinates": values.shape[1],
            "n": len(cohort),
            "clusters": size,
            "trials": trials,
        }
        for name in ["covered", "information_guard_covered"]:
            p = float(part[name].mean())
            row[name] = p
            row[name + "_mcse"] = float(np.sqrt(p * (1 - p) / trials))
        row.update(
            mean_width=float(part.width.mean()),
            information_guard_width=float(part.information_guard_width.mean()),
            mean_uninformative_coordinates=float(part.uninformative_coordinates.mean()),
            wrong_direction_fraction=float(part.wrong_direction.mean()),
            population_low_support_coordinates=int((supports.sum(0) < 20).sum()),
        )
        summaries.append(row)
    return summaries


def run(source: Path, out: Path, trials: int, bootstrap: int, only_h10_decomposition: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=False)
    state = {
        "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_unix": time.time(),
        "seed": 20260931,
        "trials": trials,
        "bootstrap": bootstrap,
        "only_h10_decomposition": only_h10_decomposition,
    }
    context = Context(source, False)
    context.nontrivial = [sid for sid in context.nontrivial if context.cards[sid]["family"] in ["clip", "prefix"]]
    assert len(context.nontrivial) == 27
    cache, _ = load_inputs(context, out)
    pairs = pairs_for(context.nontrivial, context.meta)
    summaries = []
    configurations = [(10, "decomposition"), (10, "protocol"), (4, "decomposition"), (4, "protocol")]
    if only_h10_decomposition:
        configurations = configurations[:1]
    for offset, (horizon, kind) in enumerate(configurations):
        mask = context.test.post_anchor_length_s.to_numpy() >= horizon - 1e-9
        if kind == "protocol":
            mask = (context.test.anchor_s.to_numpy() >= 0.5 - 1e-9) & (
                context.test.post_anchor_length_s.to_numpy() >= horizon + 0.5 - 1e-9
            )
        cohort = context.test.loc[mask]
        blocks = []
        protocols = [(0, 0.25)] if kind == "decomposition" else list(itertools.product([-0.5, 0, 0.5], [0.25, 0.5, 1]))
        for shift, step in protocols:
            comp = {}
            for sid in context.nontrivial:
                c, t = select_correct(cache[sid], mask, horizon, shift, step)
                comp[sid] = trajectory_components(c, t)
            if kind == "decomposition":
                blocks.append(np.concatenate([paired_components(comp[a], comp[b], horizon) for a, b in pairs], axis=1))
            else:
                blocks.append(np.column_stack([comp[b]["delay"] - comp[a]["delay"] for a, b in pairs]))
        values = np.concatenate(blocks, axis=1)
        summaries.extend(calibrate(values, cohort, horizon, kind, trials, bootstrap, 20260931 + offset, out))
        (out / "SUMMARY.json").write_text(json.dumps({**state, "summaries": summaries, "complete": False}, indent=2))
    state.update(
        complete=True,
        finished_unix=time.time(),
        summaries=summaries,
        claim="Empirical population calibration only; no universal coverage guarantee",
    )
    (out / "SUMMARY.json").write_text(json.dumps(state, indent=2))
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--only-h10-decomposition", action="store_true")
    args = parser.parse_args()
    run(args.source_root, args.out, args.trials, args.bootstrap, args.only_h10_decomposition)
