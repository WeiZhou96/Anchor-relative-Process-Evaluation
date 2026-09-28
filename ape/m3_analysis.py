"""Pure functions for the M3 re-analysis of stored predictions (see M3_SCOPE_20260928.md)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import kendalltau

TIE = 1e-9


def mode_weights(raw: np.ndarray, labels: np.ndarray, mode: str) -> np.ndarray:
    """Normalise clip multiplicities (rows = replicates) to clip weights or equal class mass.

    A class whose multiplicities sum to zero in a row receives no mass; such rows are flagged by
    ``class_support`` and must be treated as undefined for class-macro quantities.
    """
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    if mode == "micro":
        return raw / raw.sum(axis=1, keepdims=True)
    if mode != "macro":
        raise ValueError(mode)
    classes = np.unique(labels)
    out = np.zeros_like(raw)
    for c in classes:
        m = labels == c
        tot = raw[:, m].sum(axis=1, keepdims=True)
        out[:, m] = np.divide(raw[:, m], tot * len(classes), out=np.zeros_like(raw[:, m]), where=tot > 0)
    return out


def class_support(raw: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """True for rows in which every class present in ``labels`` has positive multiplicity."""
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    return np.all([raw[:, labels == c].sum(axis=1) > 0 for c in np.unique(labels)], axis=0)


def weighted(values: np.ndarray, raw: np.ndarray, labels: np.ndarray, mode: str) -> np.ndarray:
    """Weighted mean of per-clip values for each row; NaN where class-macro support is incomplete."""
    v = mode_weights(raw, labels, mode) @ np.asarray(values, dtype=float)
    if mode == "macro":
        v = np.where(class_support(raw, labels), v, np.nan)
    return v


def ratio(numerator: np.ndarray, denominator: np.ndarray, raw: np.ndarray, labels: np.ndarray, mode: str) -> np.ndarray:
    """Ratio of weighted means with the same weights; NaN where the denominator vanishes."""
    num = weighted(numerator, raw, labels, mode)
    den = weighted(denominator, raw, labels, mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > TIE, num / np.where(den > TIE, den, 1.0), np.nan)


def stable_onset_index(correct: np.ndarray) -> np.ndarray:
    """First column from which every later answer is correct; -1 if the last answer is wrong."""
    correct = np.asarray(correct, dtype=bool)
    stable = np.logical_and.accumulate(correct[:, ::-1], axis=1)[:, ::-1]
    return np.where(stable[:, -1], np.argmax(stable, axis=1), -1)


def own_end_delay(correct: np.ndarray, last_index: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trapezoid stable-correct delay on each clip's own grid [0, last_index * step].

    Returns (delay, end_correct, discrete_onset_seconds). Columns after a clip's own end are ignored.
    A clip whose own grid has a single point has zero window length and zero delay.
    """
    correct = np.asarray(correct, dtype=bool)
    last_index = np.asarray(last_index, dtype=int)
    if np.any(last_index < 0) or np.any(last_index >= correct.shape[1]):
        raise ValueError("Own-end column outside the stored grid")
    cols = np.arange(correct.shape[1])[None, :]
    inside = cols <= last_index[:, None]
    padded = np.where(inside, correct, True)
    stable = np.logical_and.accumulate(padded[:, ::-1], axis=1)[:, ::-1]
    w = np.where(inside, step, 0.0)
    w[:, 0] = np.where(last_index > 0, step / 2, 0.0)
    rows = np.arange(len(correct))
    w[rows, last_index] = np.where(last_index > 0, step / 2, 0.0)
    delay = (((~stable) & inside) * w).sum(axis=1)
    end_correct = correct[rows, last_index]
    onset = np.where(end_correct, np.argmax(stable, axis=1) * step, np.nan)
    return delay, end_correct, onset


def readout(correct_full: np.ndarray, times: np.ndarray, step: float, offset: int) -> np.ndarray:
    """Correctness at the last grid point at or before each clip-specific time (clips x reads)."""
    times = np.asarray(times, dtype=float)
    idx = np.floor(times / step + TIE).astype(int) + offset
    if np.any(idx < offset) or np.any(idx >= correct_full.shape[1]):
        raise ValueError("Readout outside the stored grid")
    return np.take_along_axis(np.asarray(correct_full, dtype=bool), idx, axis=1)


def select(ids: Sequence[str], scores: np.ndarray, maximize: bool) -> str:
    """Best system by score; ties (|difference| < TIE) broken by sorted identifier."""
    scores = np.asarray(scores, dtype=float)
    best = np.nanmax(scores) if maximize else np.nanmin(scores)
    return sorted(i for i, s in zip(ids, scores) if np.isfinite(s) and abs(s - best) < TIE)[0]


def selection_regret(ids: Sequence[str], accuracy: np.ndarray, delay: np.ndarray) -> dict[str, object]:
    ids = list(ids)
    acc, dly = dict(zip(ids, accuracy)), dict(zip(ids, delay))
    by_acc, by_delay = select(ids, accuracy, True), select(ids, delay, False)
    return {
        "selected_by_accuracy": by_acc,
        "selected_by_rmscd": by_delay,
        "same_choice": by_acc == by_delay,
        "rmscd_regret": float(dly[by_acc] - dly[by_delay]),
        "accuracy_change": float(acc[by_delay] - acc[by_acc]),
    }


def top_k_overlap(ids: Sequence[str], accuracy: np.ndarray, delay: np.ndarray, k: int = 5) -> float:
    ids = list(ids)
    order_a = sorted(range(len(ids)), key=lambda i: (-round(accuracy[i], 9), ids[i]))[:k]
    order_d = sorted(range(len(ids)), key=lambda i: (round(delay[i], 9), ids[i]))[:k]
    return len({ids[i] for i in order_a} & {ids[i] for i in order_d}) / k


def pair_disagreement(bad_x: np.ndarray, bad_y: np.ndarray, pairs: np.ndarray) -> dict[str, float]:
    """Opposite orderings of two 'smaller is better' scores over the given index pairs (ties excluded)."""
    x, y = np.asarray(bad_x, dtype=float), np.asarray(bad_y, dtype=float)
    i, j = pairs[:, 0], pairs[:, 1]
    dx, dy = x[i] - x[j], y[i] - y[j]
    valid = np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) >= TIE) & (np.abs(dy) >= TIE)
    opposite = valid & (np.sign(dx) != np.sign(dy))
    return {
        "pairs": int(len(i)),
        "compared": int(valid.sum()),
        "opposite": int(opposite.sum()),
        "share_opposite": float(opposite.sum() / valid.sum()) if valid.sum() else float("nan"),
        "median_abs_dx_opposite": float(np.median(np.abs(dx[opposite]))) if opposite.any() else float("nan"),
    }


def identity_check(rmscd: np.ndarray, ct: np.ndarray, acc: np.ndarray, pairs: np.ndarray) -> int:
    """Count B1 disagreements violating 'the CT-favoured system has the lower accuracy' (must be 0)."""
    i, j = pairs[:, 0], pairs[:, 1]
    dr, dc, da = rmscd[i] - rmscd[j], ct[i] - ct[j], acc[i] - acc[j]
    valid = np.isfinite(dc) & (np.abs(dr) >= TIE) & (np.abs(dc) >= TIE)
    opposite = valid & (np.sign(dr) != np.sign(dc))
    # CT favours the system with the smaller CT; it must have the smaller accuracy.
    favoured_lower = np.sign(dc) == np.sign(da)
    return int((opposite & ~favoured_lower & (np.abs(da) >= 1e-12)).sum())


def kendall(bad_x: np.ndarray, bad_y: np.ndarray) -> float:
    """Kendall tau-b on values rounded to 1e-9, ignoring undefined entries; NaN if degenerate."""
    x, y = np.round(np.asarray(bad_x, dtype=float), 9), np.round(np.asarray(bad_y, dtype=float), 9)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(kendalltau(x, y, variant="b").statistic)


def stratified_halves(first_class: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Assign each cluster to half 0 or 1, balancing within each class stratum."""
    first_class = np.asarray(first_class)
    half = np.zeros(len(first_class), dtype=bool)
    for c in np.unique(first_class):
        idx = np.flatnonzero(first_class == c)
        perm = rng.permutation(idx)
        half[perm[: len(perm) // 2]] = True
    return half
