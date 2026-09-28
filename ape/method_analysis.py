"""Finite-grid trajectory decomposition, partial identification and paired inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import t as student_t


def integration_weights(times: np.ndarray) -> np.ndarray:
    """Return trapezoidal weights on a strictly increasing finite grid."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("A finite grid with at least two timestamps is required")
    gaps = np.diff(times)
    if np.any(gaps <= 0):
        raise ValueError("Timestamps must strictly increase")
    weights = np.empty(len(times), dtype=float)
    weights[0], weights[-1] = gaps[0] / 2, gaps[-1] / 2
    weights[1:-1] = (gaps[:-1] + gaps[1:]) / 2
    return weights


def trajectory_components(correct: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    """Separate ordinary errors from correct predictions later invalidated."""
    correct = np.asarray(correct, dtype=bool)
    weights = integration_weights(times)
    if correct.ndim != 2 or correct.shape[1] != len(weights):
        raise ValueError("Expected a sample-by-time correctness matrix")
    stable = np.logical_and.accumulate(correct[:, ::-1], axis=1)[:, ::-1]
    delay = (~stable).astype(float) @ weights
    error = (~correct).astype(float) @ weights
    retracted = (correct & ~stable).astype(float) @ weights
    if not np.allclose(delay, error + retracted, atol=1e-10):
        raise AssertionError("Trajectory decomposition failed")
    return {"delay": delay, "error": error, "retracted": retracted, "endpoint": correct[:, -1]}


def delay_bounds(correct: np.ndarray, observed: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sharp bounds over arbitrary completions on the declared finite grid.

    Explicit abstentions are observed errors. Only unobserved outputs may vary.
    No distributional or continuous-time assertion is made.
    """
    correct = np.asarray(correct, dtype=bool)
    observed = np.broadcast_to(np.asarray(observed, dtype=bool), correct.shape)
    optimistic = trajectory_components(np.where(observed, correct, True), times)["delay"]
    pessimistic = trajectory_components(np.where(observed, correct, False), times)["delay"]
    return optimistic, pessimistic


CONTRAST_NAMES = (
    "advantage",
    "common_success",
    "success_set_mismatch",
    "error_area",
    "retracted_area",
    "endpoint_scaled",
)


def paired_components(a: dict[str, np.ndarray], b: dict[str, np.ndarray], horizon: float) -> np.ndarray:
    """Return six sample-level contrasts, positive when system A is better.

    The common-success component is weighted over the full cohort, not a new
    population estimand conditional on both models succeeding.
    """
    ca, cb = a["endpoint"].astype(bool), b["endpoint"].astype(bool)
    da, db = a["delay"], b["delay"]
    common = np.where(ca & cb, db - da, 0.0)
    mismatch = np.where(ca & ~cb, horizon - da, 0.0) + np.where(~ca & cb, db - horizon, 0.0)
    out = np.column_stack(
        [
            db - da,
            common,
            mismatch,
            b["error"] - a["error"],
            b["retracted"] - a["retracted"],
            horizon * (ca.astype(float) - cb),
        ]
    )
    if not np.allclose(out[:, 0], out[:, 1] + out[:, 2], atol=1e-10):
        raise AssertionError("Four-stratum decomposition failed")
    return out


def point_weights(labels: np.ndarray, mode: str) -> np.ndarray:
    """Clip weights or equal mass for every class present in the cohort."""
    labels = np.asarray(labels)
    if mode == "micro":
        return np.full(len(labels), 1.0 / len(labels))
    if mode != "macro":
        raise ValueError(mode)
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inv])


def cluster_weights(
    clusters: np.ndarray, labels: np.ndarray, n_boot: int, seed: int
) -> tuple[dict[str, np.ndarray], int]:
    """Use identical cluster draws for both estimands and every model.

    Replicates lacking a cohort class are redrawn, explicitly counted, and never
    silently assigned a different macro-accuracy denominator.
    """
    labels = np.asarray(labels)
    _, inv = np.unique(clusters, return_inverse=True)
    n_clusters = int(inv.max()) + 1
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    accepted = []
    rejected = 0
    while len(accepted) < n_boot:
        counts = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters))
        row = counts[inv].astype(float)
        if any(row[labels == c].sum() == 0 for c in classes):
            rejected += 1
            if rejected > 100 * n_boot:
                raise ValueError("Too few class-supporting clusters for macro bootstrap")
            continue
        accepted.append(row)
    raw = np.stack(accepted)
    micro = raw / raw.sum(axis=1, keepdims=True)
    macro = np.zeros_like(raw)
    for c in classes:
        mask = labels == c
        macro[:, mask] = raw[:, mask] / raw[:, mask].sum(axis=1, keepdims=True) / len(classes)
    return {"micro": micro, "macro": macro}, rejected


def simultaneous_band(
    points: np.ndarray, replicates: np.ndarray, alpha: float = 0.05, degrees_freedom: int | None = None
) -> dict[str, Any]:
    """Approximate bootstrap max-t bands over all trailing array dimensions.

    This is an empirical bootstrap approximation, not a finite-sample guarantee.
    Zero-variance coordinates are flagged and retain zero empirical radius.
    A supplied degrees_freedom enables a Bonferroni Student-t critical-value
    floor. Cluster ratio estimators still have approximate, not exact, coverage.
    """
    points = np.asarray(points, dtype=float)
    replicates = np.asarray(replicates, dtype=float)
    if replicates.shape[1:] != points.shape or len(replicates) < 2:
        raise ValueError("Bootstrap and point shapes do not match")
    if not np.isfinite(points).all() or not np.isfinite(replicates).all():
        raise ValueError("Nonfinite estimates cannot enter simultaneous bands")
    se = replicates.std(axis=0, ddof=1)
    zero = se < 1e-12
    scale = np.where(zero, 1.0, se)
    z = np.abs((replicates - points) / scale).reshape(len(replicates), -1)
    empirical_critical = float(np.quantile(z.max(axis=1), 1 - alpha))
    guard = 0.0
    if degrees_freedom is not None:
        if degrees_freedom < 1:
            raise ValueError("At least two supporting clusters are needed")
        guard = float(student_t.ppf(1 - alpha / (2 * points.size), degrees_freedom))
    critical = max(empirical_critical, guard)
    radius = critical * np.where(zero, 0.0, se)
    return {
        "lower": points - radius,
        "upper": points + radius,
        "se": se,
        "critical": critical,
        "zero_variance": zero,
        "bootstrap_critical": empirical_critical,
        "bonferroni_t_guard": guard,
    }
