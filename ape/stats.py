"""Cluster-level paired bootstrap, the significant-pair set P, and Holm correction.

Implements idea section 3.8 and contract section 8.

The statistical unit is the **source video cluster**, never the prefix and never
the clip: clips cut from the same source video are one observation, and all of a
cluster's prefixes follow it into a resample as a block. Anything else would
treat shared footage as independent evidence and shrink every interval.

The unit is the source *video*, not the accident. Measured on manifest v2, 54 of
the 1789 clusters carry more than one collision type, so a cluster can hold clips
of different accidents that merely came from the same upload. Grouping by video
is still the right conservative choice -- it is what controls shared scene,
shared encoding and possible near-duplicate frames -- but it must not be
described as grouping by event.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------
def cluster_row_groups(cluster_codes: np.ndarray) -> List[np.ndarray]:
    """Row indices of each cluster, in a stable order."""
    codes = np.asarray(cluster_codes)
    uniq = np.unique(codes)
    return [np.nonzero(codes == c)[0] for c in uniq]


def cluster_bootstrap_indices(
    cluster_codes: np.ndarray, n_boot: int, seed: int
) -> List[np.ndarray]:
    """``n_boot`` replicate row-index arrays, resampling clusters with replacement."""
    groups = cluster_row_groups(cluster_codes)
    n_clusters = len(groups)
    rng = np.random.default_rng(int(seed))
    out: List[np.ndarray] = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        out.append(np.concatenate([groups[p] for p in pick]) if n_clusters else np.zeros(0, int))
    return out


def bootstrap_stat(
    cluster_codes: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
    replicates: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, object]:
    """Percentile interval of any statistic under the cluster bootstrap."""
    n = len(cluster_codes)
    point = float(stat_fn(np.arange(n)))
    reps = (
        list(replicates)
        if replicates is not None
        else cluster_bootstrap_indices(cluster_codes, n_boot, seed)
    )
    samples = np.array([float(stat_fn(idx)) for idx in reps], dtype=float)
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        lo = hi = float("nan")
    else:
        lo = float(np.percentile(finite, 100.0 * alpha / 2.0))
        hi = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_boot": int(len(samples)),
        "n_clusters": int(len(np.unique(cluster_codes))),
        "samples": samples,
    }


def paired_bootstrap_diff(
    cluster_codes: np.ndarray,
    diff_fn: Callable[[np.ndarray], float],
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
    replicates: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, object]:
    """Paired difference between two systems on the same resampled clusters.

    Paired means both systems are scored on the *same* rows in every replicate,
    which is the only way the comparison is about the systems rather than about
    which crashes were drawn.
    """
    res = bootstrap_stat(cluster_codes, diff_fn, n_boot, seed, alpha, replicates)
    s = res["samples"]
    s = s[np.isfinite(s)]
    if len(s) == 0:
        res["p_value"] = float("nan")
        return res
    p_le = float((s <= 0).mean())
    p_ge = float((s >= 0).mean())
    p = 2.0 * min(p_le, p_ge)
    res["p_value"] = float(min(1.0, max(p, 1.0 / len(s))))
    return res


# --------------------------------------------------------------------------
# multiple comparison
# --------------------------------------------------------------------------
def holm(p_values: Sequence[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Holm step-down correction; returns adjusted p-values and the reject mask."""
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    if n == 0:
        return {"adjusted": np.zeros(0), "reject": np.zeros(0, dtype=bool)}
    order = np.argsort(p, kind="stable")
    adj_sorted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj_sorted[rank] = min(1.0, running)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adj_sorted
    return {"adjusted": adjusted, "reject": adjusted <= float(alpha)}


# --------------------------------------------------------------------------
# the significant-pair set P
# --------------------------------------------------------------------------
@dataclass
class PairTest:
    system_a: str
    system_b: str
    metric: str
    diff: float
    ci_lo: float
    ci_hi: float
    p_value: float
    p_adjusted: float
    significant: bool
    p_percentile: float = float("nan")
    p_normal: float = float("nan")
    p_method: str = "normal"


def percentile_p_floor(n_boot: int) -> float:
    """Smallest p-value a percentile bootstrap can express: ``1/n_boot``."""
    return 1.0 / float(n_boot) if n_boot else float("nan")


def correction_resolution_ok(
    n_boot: int, n_pairs: int, alpha: float, correction: str = "holm"
) -> Dict[str, object]:
    """Can a percentile p survive the multiple-comparison correction at all?

    Holm multiplies the smallest p by up to ``n_pairs``, while a percentile
    bootstrap cannot report anything below ``1/n_boot``. When
    ``n_pairs / n_boot > alpha`` **no pair can ever be declared significant**,
    however large the true effect -- the significant-pair set ``P`` comes back
    empty and every ``R_M`` is undefined, which looks exactly like "the protocol
    destroyed the ordering". This check exists so that failure can never be
    silent again.
    """
    floor = percentile_p_floor(n_boot)
    worst = floor * float(n_pairs) if correction == "holm" else floor
    return {
        "n_boot": int(n_boot),
        "n_pairs": int(n_pairs),
        "alpha": float(alpha),
        "percentile_p_floor": float(floor),
        "smallest_adjusted_p_reachable": float(worst),
        "percentile_ok": bool(worst <= alpha),
        "n_boot_needed_for_percentile": int(np.ceil(float(n_pairs) / float(alpha)))
        if correction == "holm"
        else int(np.ceil(1.0 / float(alpha))),
    }


def _normal_p(samples: np.ndarray, point: float) -> float:
    """Two-sided p from the bootstrap standard error, with no resolution floor.

    The percentile p cannot go below ``1/n_boot``; this one can, which is what
    makes a corrected test usable on a library with many pairs. The bootstrap
    distribution of a difference of cluster means is close to normal, so the
    ratio of the observed difference to its bootstrap standard error is referred
    to the normal distribution. The percentile interval is still what gets
    reported as the interval.
    """
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 2:
        return float("nan")
    se = float(np.std(s, ddof=1))
    if not np.isfinite(se) or se <= 0.0:
        return 0.0 if abs(point) > 0 else 1.0
    from scipy.stats import norm

    return float(2.0 * norm.sf(abs(float(point)) / se))


def _finalise_pairs(
    tests: List["PairTest"], alpha: float, correction: str
) -> Tuple[List[Tuple[str, str]], List["PairTest"]]:
    if not tests:
        return [], []
    if correction == "holm":
        adj = holm([t.p_value for t in tests], alpha)
        for t, a_p, rej in zip(tests, adj["adjusted"], adj["reject"]):
            t.p_adjusted = float(a_p)
            t.significant = bool(rej)
    elif correction == "none":
        for t in tests:
            t.p_adjusted = t.p_value
            t.significant = bool(t.p_value <= alpha)
    else:
        raise ValueError(f"unknown correction {correction!r}")
    return [(t.system_a, t.system_b) for t in tests if t.significant], tests


def significant_pairs_from_samples(
    point_values: Dict[str, float],
    replicate_values: Dict[str, np.ndarray],
    metric_name: str,
    alpha: float = 0.05,
    correction: str = "holm",
    p_method: str = "normal",
) -> Tuple[List[Tuple[str, str]], List[PairTest]]:
    """``P`` from per-system bootstrap samples that were computed once.

    Every system must have been evaluated on the *same* ordered list of
    replicates, so that subtracting two sample vectors reproduces the paired
    bootstrap exactly. This is the form to use whenever more than a couple of
    systems are compared: evaluating the metric inside the pair loop repeats the
    same work once per pair, which is quadratic in the size of the system
    library for no gain.
    """
    systems = sorted(point_values)
    lengths = {len(np.asarray(replicate_values[s])) for s in systems}
    if len(lengths) > 1:
        raise ValueError("systems were bootstrapped over different replicate counts")
    if p_method not in ("normal", "percentile"):
        raise ValueError(f"unknown p_method {p_method!r}")
    tests: List[PairTest] = []
    for a, b in itertools.combinations(systems, 2):
        d = np.asarray(replicate_values[a], dtype=float) - np.asarray(
            replicate_values[b], dtype=float
        )
        d = d[np.isfinite(d)]
        point = float(point_values[a] - point_values[b])
        if len(d) == 0:
            lo = hi = p_pct = p_norm = float("nan")
        else:
            lo = float(np.percentile(d, 100.0 * alpha / 2.0))
            hi = float(np.percentile(d, 100.0 * (1.0 - alpha / 2.0)))
            p_pct = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
            p_pct = float(min(1.0, max(p_pct, 1.0 / len(d))))
            p_norm = _normal_p(d, point)
        tests.append(
            PairTest(
                system_a=a,
                system_b=b,
                metric=metric_name,
                diff=point,
                ci_lo=lo,
                ci_hi=hi,
                p_value=p_norm if p_method == "normal" else p_pct,
                p_adjusted=float("nan"),
                significant=False,
                p_percentile=p_pct,
                p_normal=p_norm,
                p_method=p_method,
            )
        )
    return _finalise_pairs(tests, alpha, correction)


def significant_pairs(
    metric_fns: Dict[str, Callable[[np.ndarray], float]],
    cluster_codes: np.ndarray,
    metric_name: str,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
    correction: str = "holm",
    p_method: str = "normal",
) -> Tuple[List[Tuple[str, str]], List[PairTest]]:
    """``P``: the system pairs separated at the reference protocol.

    ``metric_fns`` maps ``system_id -> f(row_index) -> metric value``; all
    systems must be aligned to the same cohort rows. Only these pairs enter the
    rank-preservation rate ``R_M``, and per contract section 8 the caller is
    responsible for having excluded the synthetic blocks from the main result.

    Convenience wrapper: it evaluates each system once per replicate and then
    defers to :func:`significant_pairs_from_samples`.
    """
    systems = sorted(metric_fns)
    reps = cluster_bootstrap_indices(cluster_codes, n_boot, seed)
    n_rows = len(cluster_codes)
    point = {s: float(metric_fns[s](np.arange(n_rows))) for s in systems}
    samples = {
        s: np.array([float(metric_fns[s](idx)) for idx in reps], dtype=float)
        for s in systems
    }
    return significant_pairs_from_samples(
        point, samples, metric_name, alpha, correction, p_method
    )


def overlapping_intervals(
    ci_a: Tuple[float, float], ci_b: Tuple[float, float]
) -> bool:
    """Whether two intervals overlap: the "tied at the window end" test of 5.1."""
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    if not all(np.isfinite([lo_a, hi_a, lo_b, hi_b])):
        return False
    return not (hi_a < lo_b or hi_b < lo_a)


def tied_at_window_end(
    end_window_ci: Dict[str, Tuple[float, float]]
) -> List[Tuple[str, str]]:
    """Pairs whose window-end macro-accuracy intervals overlap (idea 5.1).

    This is the operational definition of "tied at the window end" used to
    select the pairs that C3 is about; the whole-clip accuracy is only a
    comparison column and must not be used here.
    """
    systems = sorted(end_window_ci)
    return [
        (a, b)
        for a, b in itertools.combinations(systems, 2)
        if overlapping_intervals(end_window_ci[a], end_window_ci[b])
    ]
