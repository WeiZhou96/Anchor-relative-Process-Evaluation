"""Finite-population trajectory stress tests with analytically enumerated truth."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ape.method_analysis import (  # noqa: E402
    cluster_weights,
    paired_components,
    point_weights,
    simultaneous_band,
    trajectory_components,
)


def finite_population(rare: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return class-by-state contrast vectors and state probabilities."""
    times = np.linspace(0, 10, 21)
    patterns = np.ones((8, 21), dtype=bool)
    patterns[1] = False
    patterns[2, :4] = False
    patterns[3, :12] = False
    patterns[4, -1] = False
    patterns[5, 17] = False
    patterns[6, [8, 16]] = False
    patterns[7, 1::2] = False
    probabilities = np.array([0.35, 0.1, 0.1, 0.1, 0.08, 0.1, 0.07, 0.1])
    if rare:
        probabilities = np.array([0.998, 0.002])
    values = []
    for category in range(5):
        if rare:
            trajectories = [np.ones((2, 21), dtype=bool) for _ in range(3)]
            trajectories[1][1, 17] = False
            trajectories[2][1, -1] = False
        else:
            latent = np.arange(8)
            trajectories = [
                patterns[(latent + offset) % 8] for offset in [category, 2 * category + 1, 2 * category + 3]
            ]
        components = [trajectory_components(x, times) for x in trajectories]
        values.append(
            np.concatenate(
                [paired_components(components[a], components[b], 10) for a, b in itertools.combinations(range(3), 2)],
                axis=1,
            )
        )
    return np.stack(values), probabilities


def run_simulation(out: Path, trials: int, bootstrap: int) -> None:
    """Run fixed stress conditions; never select the most favorable condition."""
    out.mkdir(parents=True, exist_ok=False)
    ordinary = [0.06, 0.16, 0.35, 0.11, 0.32]
    scenarios = [
        ("ordinary", 300, ordinary, False),
        ("small_support", 60, ordinary, False),
        ("imbalanced", 120, [0.01, 0.04, 0.15, 0.30, 0.50], False),
        ("rare_late_error", 300, ordinary, True),
    ]
    rows = []
    truth_record = {}
    for sid, (name, n_clusters, class_p, rare) in enumerate(scenarios):
        table, state_p = finite_population(rare)
        class_means = np.einsum("ckd,k->cd", table, state_p)
        truths = {"macro": class_means.mean(0), "micro": np.array(class_p) @ class_means}
        truth_record[name] = {mode: vector.tolist() for mode, vector in truths.items()}
        rng = np.random.default_rng(20260929 + sid)
        for trial in range(trials):
            categories = rng.choice(5, size=n_clusters, p=class_p)
            states = rng.choice(len(state_p), size=n_clusters, p=state_p)
            labels = np.repeat(categories, 2)
            clusters = np.repeat(np.arange(n_clusters), 2)
            sample = np.repeat(table[categories, states], 2, axis=0)
            counts = np.bincount(categories, minlength=5)
            weights, redrawn = cluster_weights(clusters, labels, bootstrap, 2026092900 + sid * trials + trial)
            unconditional_rng = np.random.default_rng(2026092900 + sid * trials + trial)
            multiplicities = unconditional_rng.multinomial(
                n_clusters, np.full(n_clusters, 1 / n_clusters), size=bootstrap
            )
            weights["micro_unconditional"] = np.repeat(multiplicities, 2, axis=1) / (2 * n_clusters)
            for mode in ["micro", "macro", "micro_unconditional"]:
                target = "macro" if mode == "macro" else "micro"
                available = target == "micro" or counts.min() >= 2
                row = {
                    "scenario": name,
                    "trial": trial,
                    "weighting": mode,
                    "estimable": available,
                    "min_class_clusters": int(counts.min()),
                    "redrawn": redrawn if mode != "micro_unconditional" else 0,
                    "low_support_coordinates": int((np.count_nonzero(table[categories, states], axis=0) < 20).sum()),
                }
                if available:
                    estimate = point_weights(labels, target) @ sample
                    replicates = weights[mode] @ sample
                    df = n_clusters - 1 if target == "micro" else int(counts.min()) - 1
                    band = simultaneous_band(estimate, replicates, degrees_freedom=df)
                    truth = truths[target]
                    covered = (band["lower"] <= truth + 1e-10) & (truth <= band["upper"] + 1e-10)
                    guarded = np.where(band["zero_variance"], True, covered)
                    row.update(
                        covered=bool(covered.all()),
                        support_guard_covered=bool(guarded.all()),
                        zero_coordinates=int(band["zero_variance"].sum()),
                        zero_coordinates_wrong_truth=int(
                            (band["zero_variance"] & (np.abs(estimate - truth) > 1e-10)).sum()
                        ),
                        mean_width=float(np.mean(band["upper"] - band["lower"])),
                        support_guard_mean_width=float(
                            np.mean(np.where(band["zero_variance"], 20, band["upper"] - band["lower"]))
                        ),
                    )
                rows.append(row)
            if (trial + 1) % 100 == 0:
                print(name, trial + 1, "/", trials, flush=True)
        pd.DataFrame(rows).to_csv(out / "trials.csv", index=False)
    frame = pd.DataFrame(rows)
    summaries = []
    for (name, mode), part in frame.groupby(["scenario", "weighting"], sort=False):
        valid = part[part.estimable]
        row = {
            "scenario": name,
            "weighting": mode,
            "trials": len(part),
            "estimable": len(valid),
            "unestimable": int((~part.estimable).sum()),
            "unestimable_fraction": float((~part.estimable).mean()),
        }
        for column in ["covered", "support_guard_covered"]:
            p = float(valid[column].mean())
            row[column + "_fraction_conditional"] = p
            row[column + "_mcse"] = float(np.sqrt(p * (1 - p) / len(valid)))
        row.update(
            mean_width=float(valid.mean_width.mean()),
            support_guard_mean_width=float(valid.support_guard_mean_width.mean()),
            trials_with_zero_variance_missed_truth=int((valid.zero_coordinates_wrong_truth > 0).sum()),
            total_redrawn=int(part.redrawn.sum()),
            mean_low_support_coordinates=float(part.low_support_coordinates.mean()),
        )
        summaries.append(row)
    result = {
        "seed": 20260929,
        "bootstrap": bootstrap,
        "coordinates": 18,
        "clusters_repeated_identically": 2,
        "truths": truth_record,
        "summaries": summaries,
        "claim": "stress tests of an approximate procedure; no universal coverage claim",
    }
    (out / "SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=600)
    args = parser.parse_args()
    run_simulation(args.out, args.trials, args.bootstrap)
