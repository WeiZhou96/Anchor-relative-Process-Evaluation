"""G0 planning diagnostics, not a validated sample-sufficiency gate.

APE S_H is video-weighted. Cluster counts alone do not give its variance when
cluster sizes differ. The normal-approximation calculations here assume equal
weights and independent Bernoulli units; the unpaired calculation is a reference,
not a conservative bound for arbitrary paired systems (negative correlation can
increase paired variance). These diagnostics cannot close G0. A formal rule still
needs the estimand, cluster weights/dependence assumptions and a registered
precision or power target. Per-class support and bootstrap resolution are reported
separately and do not validate those assumptions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from .classes import CLASS_NAMES
from .stats import correction_resolution_ok


def _z(alpha: float) -> float:
    """Two-sided normal quantile; computed from the error function, no SciPy needed."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    # Inverse of the standard normal CDF at 1 - alpha/2, via bisection on erf.
    target = 1.0 - float(alpha) / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def s_curve_resolution(n_clusters: int, p: float = 0.5, alpha: float = 0.05) -> Dict[str, float]:
    """Half-width of the interval on ``S_H`` and on a difference of two ``S_H``.

    ``p = 0.5`` is the worst case for a proportion's variance and is the default
    because a floor should be derived at the point where the data are least
    informative, not at a convenient one.
    """
    if int(n_clusters) != n_clusters or n_clusters < 0:
        raise ValueError("n_clusters must be a nonnegative integer")
    if not math.isfinite(float(p)) or not 0 <= float(p) <= 1:
        raise ValueError("p must be finite and in [0, 1]")
    n = int(n_clusters)
    if n <= 0:
        return {
            "n_clusters": n,
            "half_width_single": float("nan"),
            "half_width_unpaired_difference": float("nan"),
        }
    z = _z(alpha)
    var = float(p) * (1.0 - float(p)) / n
    return {
        "n_clusters": n,
        "p_assumed": float(p),
        "alpha": float(alpha),
        "half_width_single": z * math.sqrt(var),
        "half_width_unpaired_difference": z * math.sqrt(2.0 * var),
    }


def n_clusters_needed(
    target_resolution: float,
    p: float = 0.5,
    alpha: float = 0.05,
    paired_correlation: Optional[float] = None,
) -> Dict[str, object]:
    """Clusters required to resolve ``target_resolution`` in a difference of ``S_H``.

    Returns the unpaired IID planning reference, which is not a gate,
    and the measured-correlation paired estimate beside it. Negative correlation
    increases the paired estimate. Guessing a correlation would make
    the floor look achievable for a reason nobody verified.
    """
    d = float(target_resolution)
    if not math.isfinite(d) or not 0.0 < d <= 1.0:
        raise ValueError(f"target_resolution must be > 0, got {target_resolution!r}")
    if not math.isfinite(float(p)) or not 0 <= float(p) <= 1:
        raise ValueError("p must be finite and in [0, 1]")
    z = _z(alpha)
    var_unit = float(p) * (1.0 - float(p))
    n_unpaired = math.ceil(2.0 * var_unit * (z / d) ** 2)
    out: Dict[str, object] = {
        "target_resolution": d,
        "p_assumed": float(p),
        "alpha": float(alpha),
        "n_clusters_needed_unpaired_reference": int(n_unpaired),
        "n_clusters_needed_paired": None,
        "paired_correlation_used": None,
        "assumptions_verified": False,
        "note": (
            "Equal-weight IID normal-approximation planning reference only. "
            "Negative paired correlation increases variance. "
            "This is not a universal conservative bound or a G0 verdict."
        ),
    }
    if paired_correlation is not None:
        rho = float(paired_correlation)
        if not -1.0 < rho < 1.0:
            raise ValueError(f"paired_correlation must be in (-1, 1), got {rho!r}")
        out["n_clusters_needed_paired"] = int(math.ceil(2.0 * var_unit * (1.0 - rho) * (z / d) ** 2))
        out["paired_correlation_used"] = rho
    return out


def macro_metric_support(
    class_counts: Dict[str, int], min_per_class: int, closed_set: Optional[Sequence[str]] = None
) -> Dict[str, object]:
    """Which classes fall below the registered per-class floor.

    An empty class is separated from a merely thin one: a class with no members
    contributes nothing to a macro average and silently reduces the number of
    classes being averaged, which is a different defect from a noisy class.
    """
    if int(min_per_class) != min_per_class or min_per_class < 1:
        raise ValueError("min_per_class must be a positive integer")
    classes = list(CLASS_NAMES if closed_set is None else closed_set)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("closed_set must be explicit, nonempty and unique")
    if set(class_counts) - set(classes):
        raise ValueError("class_counts contains labels outside the closed set")
    for count in class_counts.values():
        if not math.isfinite(float(count)) or int(count) != count or count < 0:
            raise ValueError("class counts must be nonnegative integers")
    class_counts = {name: int(class_counts.get(name, 0)) for name in classes}
    empty = sorted(k for k, v in class_counts.items() if int(v) == 0)
    thin = sorted(k for k, v in class_counts.items() if 0 < int(v) < int(min_per_class))
    return {
        "min_per_class": int(min_per_class),
        "empty_classes": empty,
        "thin_classes": thin,
        "n_classes_with_support": sum(1 for v in class_counts.values() if int(v) >= int(min_per_class)),
        "supported": not empty and not thin,
        "note": (
            "A macro average over fewer classes than the closed set is not the same metric as "
            "one over all of them; an empty class must be reported, never dropped quietly."
        ),
    }


class G0FloorNotRegisteredError(RuntimeError):
    """Raised when the floor is requested without its registered inputs."""


def g0_sample_floor_report(
    cohort_rows: Sequence[Dict[str, object]],
    target_resolution: Optional[float],
    min_per_class: Optional[int],
    n_boot: int,
    n_pairs: int,
    alpha: float = 0.05,
    correction: str = "holm",
    p_assumed: float = 0.5,
    cluster_count_key: str = "n_clusters",
) -> Dict[str, object]:
    """Report conditional planning estimates for every horizon, without a G0 verdict.

    ``cohort_rows`` is what :func:`ape.cohort.cohort_table` or
    :func:`ape.frame_axis.frame_cohort_table` produces, so the same rule applies
    to the seconds axis and the frame axis without translation.

    Raises when either registered input is missing. The floor is meant to be
    recomputable from the manifest *given* a registered target; supplying a
    default target here would make the gate self-fulfilling.
    """
    if target_resolution is None or min_per_class is None:
        raise G0FloorNotRegisteredError(
            "G0's sample floor needs two registered inputs: the smallest S_H difference the "
            "audit intends to resolve, and the minimum per-class count a macro average may "
            "rest on. Neither has a defensible default -- a floor chosen after seeing N_H is "
            "not a floor. Register both, then call this again."
        )
    need = n_clusters_needed(target_resolution, p=p_assumed, alpha=alpha)
    required = int(need["n_clusters_needed_unpaired_reference"])
    correction_check = correction_resolution_ok(n_boot, n_pairs, alpha, correction)

    per_horizon: List[Dict[str, object]] = []
    for row in cohort_rows:
        clusters = int(row.get(cluster_count_key) or 0)
        counts = dict(row.get("class_counts") or {})
        support = macro_metric_support(counts, min_per_class, row.get("closed_set"))
        resolution = s_curve_resolution(clusters, p=p_assumed, alpha=alpha)
        label = row.get("h_s", row.get("h_f"))
        per_horizon.append(
            {
                "horizon": label,
                "axis_unit": row.get("axis_unit", "second" if "h_s" in row else "frame"),
                "N_H": int(row.get("N_H") or 0),
                "n_clusters": clusters,
                "n_clusters_required": required,
                "cluster_floor_met": clusters >= required,
                "achieved_resolution_unpaired_difference": resolution["half_width_unpaired_difference"],
                "macro_support": support,
                "planning_iid_support_met": bool(clusters >= required and support["supported"]),
                "passes_sample_floor": None,
                "formal_status": "not_assessed_cluster_weight_and_dependence_model_unvalidated",
            }
        )

    passing = [h for h in per_horizon if h["passes_sample_floor"]]
    return {
        "rule_version": "g0-planning-v2-2026-09-20",
        "is_formal_sample_floor": False,
        "formal_status": "not_assessed",
        "registered_inputs": {
            "target_resolution": float(target_resolution),
            "min_per_class": int(min_per_class),
            "alpha": float(alpha),
            "p_assumed": float(p_assumed),
            "correction": correction,
        },
        "cluster_requirement": need,
        "correction_resolution": correction_check,
        "per_horizon": per_horizon,
        "n_horizons_passing": None,
        "horizons_passing": [h["horizon"] for h in passing],
        # G0 also has a non-quantitative part that no computation can close.
        "not_covered_by_this_rule": [
            "dataset licence confirmation (A5): a human correspondence task, still open",
            "whether the audit labels are visible on the intended split",
            "whether two or more horizons are scientifically meaningful, as opposed to merely populated",
        ],
        "verdict_scope": (
            "This planning diagnostic does not close G0; cluster-weighted precision/power is not established. "
            "the licence confirmation is a separate and currently open precondition."
        ),
    }
