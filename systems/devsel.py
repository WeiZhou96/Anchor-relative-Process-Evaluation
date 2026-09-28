"""Dev-side selection proxy for the B track.

The protocol metrics of record are the A track's (``ape/metrics.py``).  This
module re-implements just enough of them -- fixed observation window, fixed
cohort, stable-correct curve, its area, window-end macro accuracy and flip
count -- so that post-processing windows and commitment thresholds can be
chosen on ``split=dev`` without importing an unfinished package.  Numbers
printed by this module are selection aids and smoke values; every reported
figure in the paper comes from the A track.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from systems import common as C


def cohort_mask(post_anchor_length_s: np.ndarray, h_s: float) -> np.ndarray:
    """CONTRACT 8: eligible_H = post_anchor_length_s >= H, fixed across all systems and all j."""
    return post_anchor_length_s >= h_s


def stable_correct_curve(preds: np.ndarray, labels: np.ndarray, j_end: int) -> np.ndarray:
    """S_H(delta_j) for j = 0..j_end. preds (n, >=j_end+1), labels (n,)."""
    correct = preds[:, : j_end + 1] == labels[:, None]
    # suffix-AND: stable_{i,j} = all correct from j to j_end
    stable = np.flip(np.cumprod(np.flip(correct.astype(np.int8), axis=1), axis=1), axis=1)
    return stable.mean(axis=0)


def rmscd(curve: np.ndarray, delta_s: float) -> float:
    """Restricted mean stable-correct delay: trapezoidal integral of 1 - S_H over [0, H]."""
    return float(np.trapezoid(1.0 - curve, dx=delta_s))


def macro_acc(pred_at: np.ndarray, labels: np.ndarray, n_classes: int = C.N_CLASSES) -> float:
    accs = []
    for k in range(n_classes):
        m = labels == k
        if m.sum() == 0:
            continue
        accs.append(float((pred_at[m] == k).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def flips(preds: np.ndarray, j_end: int) -> np.ndarray:
    win = preds[:, : j_end + 1]
    return (win[:, 1:] != win[:, :-1]).sum(axis=1)


def window_summary(preds: np.ndarray, labels: np.ndarray, post_len: np.ndarray,
                   h_s: float, delta_s: float = C.FINEST_DELTA_S) -> Dict[str, float]:
    j_end = int(round(h_s / delta_s))
    m = cohort_mask(post_len, h_s)
    if m.sum() == 0 or preds.shape[1] <= j_end:
        return {"H": h_s, "N_H": int(m.sum()), "rmscd": float("nan"),
                "end_macro_acc": float("nan"), "end_acc": float("nan"),
                "median_flips": float("nan"), "s_at_0": float("nan")}
    p, y = preds[m], labels[m]
    curve = stable_correct_curve(p, y, j_end)
    f = flips(p, j_end)
    return {
        "H": h_s,
        "N_H": int(m.sum()),
        "rmscd": rmscd(curve, delta_s),
        "end_macro_acc": macro_acc(p[:, j_end], y),
        "end_acc": float((p[:, j_end] == y).mean()),
        "median_flips": float(np.median(f)),
        "s_at_0": float(curve[0]),
    }


def full_clip_macro_acc(preds: np.ndarray, labels: np.ndarray) -> float:
    """Macro accuracy at the last grid point available (proxy for a whole-clip read)."""
    return macro_acc(preds[:, -1], labels)


def summarize(preds: np.ndarray, labels: np.ndarray, post_len: np.ndarray,
              h_list=None, delta_s: float = C.FINEST_DELTA_S) -> Dict[float, Dict[str, float]]:
    h_list = h_list or C.H_PLACEHOLDERS
    return {h: window_summary(preds, labels, post_len, h, delta_s) for h in h_list}


def fmt(summary: Dict[float, Dict[str, float]], prefix: str = "") -> str:
    lines = []
    for h, s in summary.items():
        lines.append(
            f"{prefix}H={h:<5g} N_H={s['N_H']:<5d} RMSCD={s['rmscd']:.3f} "
            f"end_macro_acc={s['end_macro_acc']:.4f} end_acc={s['end_acc']:.4f} "
            f"median_flips={s['median_flips']:.1f} S_H(0)={s['s_at_0']:.4f}"
        )
    return "\n".join(lines)
