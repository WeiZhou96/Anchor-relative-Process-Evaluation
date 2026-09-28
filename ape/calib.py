"""Characterisation quantities: b_M, s_M, R_M, the comparability region, MRD_M.

Implements idea sections 3.4, 3.5 and 5.1, and contract section 8. "Calibration"
here is the instrument sense (characterisation): measuring how the reading moves
when a known perturbation is applied to the protocol. It has nothing to do with
probability calibration, and nothing in this module touches a model.

The scan table
--------------
Every function consumes one long table with the columns

    system_id, is_block, pi_hash, eps_sys_s, eps_jit_sd_s, delta_s, h_s,
    in_plaus, metric, value

i.e. one row per (system, protocol vector, metric). ``pi0_hash`` names the
reference row group. Main results use ``is_block == False``; the block-inclusive
version is computed separately and reported beside it, never merged
(contract section 8).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCAN_COLUMNS = [
    "system_id",
    "is_block",
    "pi_hash",
    "eps_sys_s",
    "eps_jit_sd_s",
    "delta_s",
    "h_s",
    "in_plaus",
    "metric",
    "value",
]


def _pivot(scan: pd.DataFrame, metric: str) -> pd.DataFrame:
    """``system_id`` x ``pi_hash`` matrix of one metric's values."""
    sub = scan.loc[scan["metric"] == metric]
    return sub.pivot_table(
        index="system_id", columns="pi_hash", values="value", aggfunc="first"
    )


def _pi_meta(scan: pd.DataFrame) -> pd.DataFrame:
    cols = ["pi_hash", "eps_sys_s", "eps_jit_sd_s", "delta_s", "h_s", "in_plaus"]
    return scan[cols].drop_duplicates("pi_hash").set_index("pi_hash")


# --------------------------------------------------------------------------
# b and s: systematic offset and spread across systems
# --------------------------------------------------------------------------
def offsets_and_spread(
    scan: pd.DataFrame, pi0_hash: str, metric: str, blocks: bool = False
) -> pd.DataFrame:
    """``b_M(pi)`` and ``s_M(pi)``: mean and sd across systems of ``M(a;pi)-M(a;pi0)``.

    ``b`` says how far the whole reading slides (correctable in principle);
    ``s`` says how much systems slide by different amounts (not correctable by a
    single constant). They are kept apart because the remedies differ -- that is
    the reason idea section 3.5 refuses to merge them into one robustness score.
    """
    sub = scan if blocks else scan.loc[~scan["is_block"].astype(bool)]
    mat = _pivot(sub, metric)
    if pi0_hash not in mat.columns:
        raise KeyError(f"pi0 hash {pi0_hash} absent from the scan table")
    base = mat[pi0_hash]
    rows = []
    meta = _pi_meta(scan)
    for pi_hash in mat.columns:
        d = (mat[pi_hash] - base).to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        m = meta.loc[pi_hash]
        rows.append(
            {
                "pi_hash": pi_hash,
                "eps_sys_s": float(m["eps_sys_s"]),
                "eps_jit_sd_s": float(m["eps_jit_sd_s"]),
                "delta_s": float(m["delta_s"]),
                "h_s": float(m["h_s"]),
                "in_plaus": bool(m["in_plaus"]),
                "metric": metric,
                "n_systems": int(len(d)),
                "b_M": float(np.mean(d)) if len(d) else float("nan"),
                "s_M": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
                "max_abs_shift": float(np.max(np.abs(d))) if len(d) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# R: rank preservation
# --------------------------------------------------------------------------
def _sign(x: float) -> int:
    if not np.isfinite(x):
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def rank_preservation(
    scan: pd.DataFrame, pi0_hash: str, metric: str,
    pairs: Sequence[Tuple[str, str]], blocks: bool = False,
) -> pd.DataFrame:
    """Pairwise R_M with the exact historical tie/nonfinite rules, vectorized."""
    sub = scan if blocks else scan.loc[~scan["is_block"].astype(bool)]
    mat = _pivot(sub, metric); meta = _pi_meta(scan)
    if pi0_hash not in mat.columns:
        raise KeyError(f"pi0 hash {pi0_hash} absent from the scan table")
    pairs=list(pairs)
    ia=mat.index.get_indexer([a for a,b in pairs]);ib=mat.index.get_indexer([b for a,b in pairs])
    valid=(ia>=0)&(ib>=0);ia,ib=ia[valid],ib[valid]
    values=mat.to_numpy(dtype=float);base=values[:,mat.columns.get_loc(pi0_hash)]
    d0=base[ia]-base[ib]
    s0=np.where(np.isfinite(d0),np.sign(d0),0)
    selected=s0!=0;total=int(selected.sum());rows=[]
    for j,pi_hash in enumerate(mat.columns):
        diff=values[ia,j]-values[ib,j]
        signs=np.where(np.isfinite(diff),np.sign(diff),0)
        kept=int(np.sum(selected&(signs==s0)));m=meta.loc[pi_hash]
        rows.append(dict(pi_hash=pi_hash,eps_sys_s=float(m["eps_sys_s"]),
            eps_jit_sd_s=float(m["eps_jit_sd_s"]),delta_s=float(m["delta_s"]),
            h_s=float(m["h_s"]),in_plaus=bool(m["in_plaus"]),metric=metric,
            n_pairs=total,R_M=float(kept)/total if total else float("nan")))
    return pd.DataFrame(rows)


def comparability_region(r_table: pd.DataFrame, r0: float) -> pd.DataFrame:
    """``A_M(r0) = {pi : R_M(pi) >= r0}`` as a boolean column on the R table."""
    out = r_table.copy()
    out["in_region"] = out["R_M"].to_numpy(dtype=float) >= float(r0)
    return out


# --------------------------------------------------------------------------
# MRD: the smallest reportable difference
# --------------------------------------------------------------------------
#: Metrics whose numerical range is set by the horizon, so that a value at one
#: ``H`` is not commensurable with a value at another (idea 3.2 item 5:
#: "do not compare absolute values across H"). Differences across the ``H`` axis
#: are a change of scale, not a measurement error, and must not enter ``MRD``.
H_SCALED_METRICS = ("RMSCD@H", "RMSCD@H_macro")


def is_h_scaled(metric: str) -> bool:
    return metric in H_SCALED_METRICS


def mrd(
    scan: pd.DataFrame,
    pi0_hash: str,
    metric: str,
    quantile: float = 0.95,
    blocks: bool = False,
    plaus_only: bool = True,
    same_h_only: Optional[bool] = None,
) -> Dict[str, object]:
    """``MRD_M = Q_0.95(|M(a;pi) - M(a;pi0)|)`` over systems and plausible ``pi``.

    Reading: two papers reporting this metric under settings inside the
    plausible grid, and differing by less than ``MRD_M``, are not reporting a
    difference that can be attributed to the systems.

    ``same_h_only`` drops the ``H`` axis for metrics whose scale is set by ``H``.
    ``RMSCD@H`` is bounded above by ``H``, so a shift from ``H = 4.11`` to
    ``H = 21.82`` moves it by several seconds purely by redefining the quantity;
    folding that into ``MRD`` would inflate the smallest reportable difference to
    the point of declaring everything incomparable. Defaults to ``True`` exactly
    for those metrics, and the cross-``H`` spread is reported separately as
    ``h_axis_shift`` so it is visible rather than hidden.
    """
    if same_h_only is None:
        same_h_only = is_h_scaled(metric)
    sub = scan if blocks else scan.loc[~scan["is_block"].astype(bool)]
    mat = _pivot(sub, metric)
    if pi0_hash not in mat.columns:
        raise KeyError(f"pi0 hash {pi0_hash} absent from the scan table")
    meta = _pi_meta(scan)
    h0 = float(meta.loc[pi0_hash, "h_s"])
    use = [
        c
        for c in mat.columns
        if c != pi0_hash
        and (not plaus_only or bool(meta.loc[c, "in_plaus"]))
        and (not same_h_only or abs(float(meta.loc[c, "h_s"]) - h0) < 1e-9)
    ]
    h_axis = [
        c
        for c in mat.columns
        if c != pi0_hash and abs(float(meta.loc[c, "h_s"]) - h0) >= 1e-9
    ]
    base = mat[pi0_hash]
    vals: List[float] = []
    for c in use:
        d = np.abs((mat[c] - base).to_numpy(dtype=float))
        vals.extend([float(x) for x in d[np.isfinite(d)]])
    h_vals: List[float] = []
    for c in h_axis:
        d = np.abs((mat[c] - base).to_numpy(dtype=float))
        h_vals.extend([float(x) for x in d[np.isfinite(d)]])
    h_axis_shift = (
        float(np.percentile(h_vals, 100.0 * float(quantile))) if h_vals else float("nan")
    )
    if not vals:
        return {
            "metric": metric,
            "MRD": float("nan"),
            "n": 0,
            "quantile": quantile,
            "same_h_only": bool(same_h_only),
            "h_axis_shift": h_axis_shift,
        }
    return {
        "metric": metric,
        "MRD": float(np.percentile(vals, 100.0 * float(quantile))),
        "max_abs_shift": float(np.max(vals)),
        "n": int(len(vals)),
        "quantile": float(quantile),
        "n_pi": int(len(use)),
        "blocks_included": bool(blocks),
        "plaus_only": bool(plaus_only),
        "same_h_only": bool(same_h_only),
        "h_axis_shift": h_axis_shift,
        "h_axis_note": (
            "spread across the H axis, reported separately because for an "
            "H-scaled metric it is a change of scale rather than a measurement "
            "error; not included in MRD when same_h_only is true"
        ),
    }


# --------------------------------------------------------------------------
# eps_max and Delta*
# --------------------------------------------------------------------------
def eps_max(
    r_table: pd.DataFrame,
    r0: float,
    pi0_hash: Optional[str] = None,
    tol: float = 1e-9,
) -> Dict[str, object]:
    """``eps_max(M, r0)``: tolerance to a common-mode anchor shift.

    Walks outwards from zero along the ``eps_sys`` axis (all other knobs held at
    ``pi0``) and returns the largest ``|eps|`` reached before ``R_M`` first drops
    below ``r0``. ``saturated`` means the whole scanned range held, so the number
    is a lower bound set by the grid, not by the protocol.

    The reference step and horizon are read from ``pi0_hash``. They must not be
    guessed from whichever ``eps == 0`` row happens to come first: in an axis
    scan there is one such row per ``delta`` and per ``H``, so picking the wrong
    one filters the whole shift axis away and silently reports ``eps_max = 0``
    with nothing scanned. When ``pi0_hash`` is absent the axis is identified by
    the modal ``(delta, H)`` among the ``eps == 0`` rows instead, and the result
    carries ``pi0_hash_used = None`` so the weaker basis is visible.
    """
    empty = {
        "eps_max": float("nan"),
        "saturated": False,
        "n_points": 0,
        "scanned_max_abs_eps": float("nan"),
        "pi0_hash_used": pi0_hash,
    }
    ax = r_table.loc[
        np.isclose(r_table["eps_jit_sd_s"].to_numpy(dtype=float), 0.0)
    ].copy()
    if ax.empty:
        return empty

    d0 = h0 = None
    if pi0_hash is not None:
        row = r_table.loc[r_table["pi_hash"].astype(str) == str(pi0_hash)]
        if len(row):
            d0 = float(row["delta_s"].iloc[0])
            h0 = float(row["h_s"].iloc[0])
    if d0 is None:
        ref = ax.loc[np.isclose(ax["eps_sys_s"].to_numpy(dtype=float), 0.0)]
        if ref.empty:
            return empty
        modal = ref.groupby(["delta_s", "h_s"]).size().sort_values(ascending=False)
        d0, h0 = (float(x) for x in modal.index[0])

    ax = ax.loc[
        np.isclose(ax["delta_s"].to_numpy(dtype=float), d0)
        & np.isclose(ax["h_s"].to_numpy(dtype=float), h0)
    ]
    if ax.empty:
        return empty
    ax = ax.assign(abs_eps=np.abs(ax["eps_sys_s"].to_numpy(dtype=float)))
    grouped = ax.groupby("abs_eps")["R_M"].min().sort_index()
    best = 0.0
    saturated = True
    for a, r in grouped.items():
        if not np.isfinite(r) or float(r) < float(r0) - tol:
            saturated = False
            break
        best = float(a)
    return {
        "eps_max": float(best),
        "saturated": bool(saturated),
        "n_points": int(len(grouped)),
        "scanned_max_abs_eps": float(grouped.index.max()) if len(grouped) else float("nan"),
        "pi0_hash_used": pi0_hash,
        "reference_delta_s": float(d0),
        "reference_h_s": float(h0),
    }


def delta_star(
    flips_by_delta: Dict[float, float], rel_tol: float = 0.05
) -> Dict[str, object]:
    """``Delta*``: the step at which the flip count stops growing (P-c).

    ``F(Delta)`` is expected to rise as the step gets finer and then flatten. We
    walk from the coarsest step towards the finest and return the coarsest step
    from which every finer step agrees to within ``rel_tol``. If the finest
    scanned step is still growing, ``saturated`` is ``False`` and the caller must
    report a per-second flip rate instead of a flip count -- at that point ``F``
    is a property of the protocol, not of the system (idea section 3.5, P-c).
    """
    if not flips_by_delta:
        return {"delta_star": float("nan"), "saturated": False, "n_points": 0}
    deltas = sorted(flips_by_delta, reverse=True)  # coarse -> fine
    vals = [float(flips_by_delta[d]) for d in deltas]
    finest = vals[-1]
    scale = max(abs(finest), 1.0)
    star = float("nan")
    for i, d in enumerate(deltas):
        if all(abs(v - finest) <= rel_tol * scale for v in vals[i:]):
            star = float(d)
            break
    saturated = bool(np.isfinite(star)) and len(deltas) > 1 and star != deltas[-1]
    return {
        "delta_star": star,
        "saturated": saturated,
        "n_points": int(len(deltas)),
        "flips_by_delta": {float(d): float(flips_by_delta[d]) for d in deltas},
        "monotone_nonincreasing_with_coarser_step": bool(
            all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
        ),
    }


# --------------------------------------------------------------------------
# verdict column of table 3
# --------------------------------------------------------------------------
def verdict(
    min_R: float, mrd_value: float, r0: float, ruler: Optional[float]
) -> str:
    """Table 3's ``poolable / normalize / not comparable`` column.

    ``ruler`` is the median difference **in this metric's own units** among the
    pairs that tie at the window end (idea section 5.1); it is what "a difference
    worth reporting" means for this metric. It must not be shared across the
    family: RMSCD is in seconds, the window-end accuracy is dimensionless and the
    flip count is an integer, so a single ruler would compare unlike quantities.
    A missing ruler leaves the verdict undecided rather than guessing.
    """
    if not np.isfinite(min_R):
        return "undetermined"
    if min_R < float(r0):
        return "not comparable"
    if ruler is None or not np.isfinite(ruler):
        return "undetermined (no window-end-tied ruler available)"
    if float(ruler) == 0.0:
        # Tied systems do not differ at all on this metric, so every non-zero
        # protocol shift exceeds the between-system difference. The conclusion
        # ("do not pool without normalising") still holds, but it rests on the
        # metric being too coarse to resolve anything rather than on a magnitude
        # comparison, and the label has to say so.
        return "normalize (ruler is zero: metric cannot resolve tied systems)"
    if np.isfinite(mrd_value) and mrd_value < float(ruler):
        return "poolable"
    return "normalize"


def calibrate_metric(
    scan: pd.DataFrame,
    pi0_hash: str,
    metric: str,
    pairs: Sequence[Tuple[str, str]],
    r0: float,
    ruler: Optional[float] = None,
    blocks: bool = False,
) -> Dict[str, object]:
    """Everything table 3 needs for one metric of the frozen family."""
    bs = offsets_and_spread(scan, pi0_hash, metric, blocks=blocks)
    rt = comparability_region(
        rank_preservation(scan, pi0_hash, metric, pairs, blocks=blocks), r0
    )
    m = mrd(scan, pi0_hash, metric, blocks=blocks, plaus_only=True)
    m_all = mrd(scan, pi0_hash, metric, blocks=blocks, plaus_only=False)
    plaus = rt.loc[rt["in_plaus"].astype(bool)]
    min_R_plaus = float(plaus["R_M"].min()) if len(plaus) else float("nan")
    bs_plaus = bs.loc[bs["in_plaus"].astype(bool)]

    sub = scan if blocks else scan.loc[~scan["is_block"].astype(bool)]
    ref = sub.loc[(sub["metric"] == metric) & (sub["pi_hash"] == pi0_hash), "value"]
    ref = ref.to_numpy(dtype=float)
    ref = ref[np.isfinite(ref)]
    reference_range = [float(ref.min()), float(ref.max())] if len(ref) else None

    max_b = (
        float(np.nanmax(np.abs(bs_plaus["b_M"].to_numpy(float))))
        if len(bs_plaus)
        else float("nan")
    )
    max_s = (
        float(np.nanmax(bs_plaus["s_M"].to_numpy(float))) if len(bs_plaus) else float("nan")
    )
    return {
        # short aliases matching the reporting stage's table-3 column names
        "reference_range": reference_range,
        "max_b": max_b,
        "max_s": max_s,
        "min_R": min_R_plaus,
        "MRD": m.get("MRD", float("nan")),
        "metric": metric,
        "r0": float(r0),
        "blocks_included": bool(blocks),
        "max_abs_b_M_plaus": max_b,
        "max_s_M_plaus": max_s,
        "min_R_M_plaus": min_R_plaus,
        "min_R_M_full": float(rt["R_M"].min()) if len(rt) else float("nan"),
        "MRD_plaus": m.get("MRD", float("nan")),
        "MRD_full_grid": m_all.get("MRD", float("nan")),
        "n_pairs": int(rt["n_pairs"].max()) if len(rt) else 0,
        "eps_max": eps_max(rt, r0, pi0_hash),
        "ruler": ruler,
        "ruler_degenerate": bool(ruler is not None and np.isfinite(ruler)
                                 and float(ruler) == 0.0),
        "ruler_definition": (
            "median |M(a) - M(b)| over the window-end-tied system pairs at pi0, "
            "in this metric's own units"
        ),
        "ruler_window_end_tied_median_progress_diff": ruler,
        "verdict": verdict(min_R_plaus, m.get("MRD", float("nan")), r0, ruler),
        "b_s_table": bs,
        "R_table": rt,
    }
