"""Phenomenon statistics and the K1-K6 terminal-consistency checks.

Implements idea section 3.7. **Everything here is reported, nothing is judged.**
There is no pass mark, no threshold, no ranking: these are deliverables that
describe how systems behave under the protocol, and a result such as "the
answers barely change after the anchor" is as useful an outcome as its
opposite. Nothing in this module endorses any stopping signal, and no quantity
here supports a claim that one system is better than another.

Confidence-based items need probabilities in the answer matrix. A system that
reports none yields ``NaN`` and an explicit availability count, rather than a
zero that would read as "no overconfidence".
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .classes import BOT, N_CLASSES
from .metrics import (
    AnswerTable,
    EvalResult,
    _row_max_skipnan,
    correctness,
    first_stable_index,
    flip_counts,
    macro_accuracy,
    micro_accuracy,
    stable_correct,
)
from .protocol import TOL, ProtocolVector, n_grid_points, perturb_manifest

DEFAULT_CONF_THRESHOLD = 0.9
DEFAULT_SHORT_PREFIX_S = 1.0
DEFAULT_PRE_ANCHOR_S = 2.0


def _quantiles(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": float("nan"), "p25": float("nan"),
                "median": float("nan"), "p75": float("nan")}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
    }


def _sustained_from(match: np.ndarray) -> np.ndarray:
    """First index from which ``match`` holds through the last column; ``-1`` if never."""
    sust = np.logical_and.accumulate(match[:, ::-1], axis=1)[:, ::-1]
    ever = sust.any(axis=1)
    return np.where(ever, np.argmax(sust, axis=1), -1)


def pre_anchor_answers(
    manifest: pd.DataFrame,
    answers: AnswerTable,
    pv: ProtocolVector,
    video_ids: Sequence[str],
    pre_anchor_s: float = DEFAULT_PRE_ANCHOR_S,
) -> Dict[str, object]:
    """Raw system outputs strictly before the anchor, for the overconfidence count.

    The protocol masks ``delta < 0`` to ``BOT`` in the score (contract 8); these
    raw readings are only a diagnostic, exactly as idea section 3.1 allows. If
    the system recorded nothing before the anchor, the availability count says
    so and the derived rates are ``NaN``.
    """
    eff = perturb_manifest(manifest, pv).set_index("video_id")
    sub = eff.loc[[str(v) for v in video_ids]]
    k = n_grid_points(pre_anchor_s, pv.delta_s)
    if k < 1:
        return {"available": 0, "pred": None, "msp": None, "offsets_s": np.zeros(0)}
    offsets = -np.arange(1, k + 1, dtype=float) * float(pv.delta_s)
    anchor_eff = sub["anchor_s_eff"].to_numpy(dtype=float)
    anchor_base = sub["anchor_s"].to_numpy(dtype=float)
    end_s = anchor_eff[:, None] + offsets[None, :]
    base_j = np.rint((end_s - anchor_base[:, None]) / answers.delta_s).astype(np.int64)
    n = len(sub)
    pred, present, _, probs = answers.lookup_2d(sub.index.to_numpy(), base_j)
    msp = None
    if probs is not None:
        msp = _row_max_skipnan(probs.reshape(-1, N_CLASSES)).reshape(n, k)
    return {
        "available": int(present.sum()),
        "pred": pred,
        "present": present,
        "msp": msp,
        "offsets_s": offsets,
    }


def phenomena_report(
    result: EvalResult,
    manifest: Optional[pd.DataFrame] = None,
    answers: Optional[AnswerTable] = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    short_prefix_s: float = DEFAULT_SHORT_PREFIX_S,
    pre_anchor_s: float = DEFAULT_PRE_ANCHOR_S,
) -> Dict[str, object]:
    """Table 5 of the idea: phenomenon statistics plus K1-K6, for one system.

    Pass ``manifest`` and ``answers`` to enable the pre-anchor items; without
    them those entries report ``NaN`` and an availability count of zero.
    """
    pred = result.pred
    y = result.y
    n, n_j = pred.shape
    delta = result.delta_s
    offsets = result.offsets_s
    corr = correctness(pred, y)
    ind = stable_correct(corr)
    flips = flip_counts(pred)
    terminal = pred[:, -1]

    # ---- flip behaviour ----
    if n_j >= 2:
        changed = pred[:, 1:] != pred[:, :-1]
        has_flip = changed.any(axis=1)
        last_idx = np.where(
            has_flip, n_j - 1 - np.argmax(changed[:, ::-1], axis=1), -1
        )
        last_flip_s = np.where(has_flip, last_idx.astype(float) * delta, np.nan)
    else:
        has_flip = np.zeros(n, dtype=bool)
        last_flip_s = np.full(n, np.nan)

    # ---- correct -> wrong -> correct ----
    cwc = np.zeros(n, dtype=bool)
    if n_j >= 3:
        seen_correct = np.zeros(n, dtype=bool)
        went_wrong = np.zeros(n, dtype=bool)
        for j in range(n_j):
            c = corr[:, j]
            cwc |= c & went_wrong
            went_wrong |= (~c) & seen_correct
            seen_correct |= c

    # ---- terminal consistency ----
    match_term = pred == terminal[:, None]
    first_term_idx = _sustained_from(match_term)
    first_stable_idx = first_stable_index(ind)
    h_s = result.h_s
    t_term = np.where(first_term_idx >= 0, first_term_idx.astype(float) * delta, h_s)
    t_stable = np.where(first_stable_idx >= 0, first_stable_idx.astype(float) * delta, h_s)

    # ---- confidence based ----
    # K5 is a CONDITIONAL rate: among short confident prefix cells, how many
    # disagree with where the system ends up. When no cell clears the threshold
    # the denominator is zero and the rate is undefined -- reported as NaN with
    # the cell counts, never as 0.0, which would read as "no inconsistency".
    short_mask = offsets <= float(short_prefix_s) + TOL
    n_conf = 0
    if result.msp is not None and short_mask.any():
        conf = result.msp[:, short_mask] >= float(conf_threshold)
        inconsist = pred[:, short_mask] != terminal[:, None]
        conf_available = int(np.isfinite(result.msp[:, short_mask]).sum())
        n_conf = int(conf.sum())
        if n_conf:
            k5_pair = float((conf & inconsist).sum()) / n_conf
            k5_video = float(((conf & inconsist).any(axis=1)).mean())
        else:
            k5_pair = float("nan")
            k5_video = float("nan")
    else:
        k5_pair = float("nan")
        k5_video = float("nan")
        conf_available = 0

    # ---- pre-anchor overconfidence ----
    pre_conf_wrong = float("nan")
    pre_conf_correct = float("nan")
    pre_answer_rate = float("nan")
    pre_available = 0
    pre_conf_cells = 0
    if manifest is not None and answers is not None:
        pre = pre_anchor_answers(manifest, answers, result.pi, result.video_ids, pre_anchor_s)
        pre_available = int(pre.get("available", 0))
        if pre_available > 0:
            p_pred = pre["pred"]
            p_present = pre["present"]
            answered = p_present & (p_pred != BOT)
            pre_answer_rate = float(answered.any(axis=1).mean())
            if pre.get("msp") is not None:
                hi = answered & (pre["msp"] >= float(conf_threshold))
                pre_conf_cells = int(hi.sum())
                # same rule as K5: with no cell over the threshold the quantity is
                # unmeasured, not zero. A 0.0 here would be read as "this system is
                # never overconfident before the anchor", which is a claim the data
                # does not support.
                if pre_conf_cells:
                    any_hi = hi.any(axis=1)
                    final_wrong = terminal != y
                    pre_conf_wrong = float((any_hi & final_wrong).mean())
                    pre_conf_correct = float((any_hi & ~final_wrong).mean())

    # ---- K block ----
    a_event = match_term  # prefix prediction equals the terminal prediction
    b_event = corr  # prefix prediction equals the truth
    k1_agreement = float((a_event == b_event).mean()) if a_event.size else float("nan")
    terminal_wrong = terminal != y
    k3 = (
        float(corr[terminal_wrong].any(axis=1).mean())
        if bool(terminal_wrong.any())
        else float("nan")
    )
    try:
        from scipy.stats import ks_2samp

        ks = ks_2samp(t_term, t_stable)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    except Exception:  # pragma: no cover - scipy always present in target env
        ks_stat, ks_p = float("nan"), float("nan")

    return {
        "system_id": result.system_id,
        "pi_hash": result.pi.pi_hash,
        "h_s": h_s,
        "delta_s": delta,
        "N_H": int(n),
        "interpretation": "report only; no pass mark, no ranking, no endorsement",
        "phenomena": {
            "flip_rate_videos_with_any_flip": float(has_flip.mean()) if n else float("nan"),
            "flips": _quantiles(flips.astype(float)),
            "last_flip_time_s": _quantiles(last_flip_s),
            "correct_wrong_correct_rate": float(cwc.mean()) if n else float("nan"),
            "pre_anchor_answer_rate": pre_answer_rate,
            "pre_anchor_confident_and_finally_wrong": pre_conf_wrong,
            "pre_anchor_confident_and_finally_correct": pre_conf_correct,
            "pre_anchor_cells_available": pre_available,
            "pre_anchor_confident_cells": pre_conf_cells,
            "short_prefix_confident_inconsistent_rate": k5_pair,
            "short_prefix_confident_inconsistent_video_rate": k5_video,
            "confidence_cells_available": conf_available,
            "short_prefix_confident_cells": n_conf,
            "conf_threshold": float(conf_threshold),
            "short_prefix_s": float(short_prefix_s),
            "pre_anchor_s": float(pre_anchor_s),
        },
        "K": {
            "K1_prefix_equals_terminal_vs_prefix_equals_truth_agreement": k1_agreement,
            "K1_rate_prefix_equals_terminal": float(a_event.mean()) if a_event.size else float("nan"),
            "K1_rate_prefix_equals_truth": float(b_event.mean()) if b_event.size else float("nan"),
            "K2_full_clip_macro_acc": macro_accuracy(result.full_clip_pred, y),
            "K2_full_clip_micro_acc": micro_accuracy(result.full_clip_pred, y),
            "K3_terminal_wrong_but_had_correct_prefix": k3,
            "K4_flips": _quantiles(flips.astype(float)),
            "K5_short_prefix_confident_inconsistent_rate": k5_pair,
            "K5_definition": (
                "prefix-cell granularity: among (video, grid point) cells with "
                f"delta <= {float(short_prefix_s)} s and MSP >= {float(conf_threshold)}, "
                "the fraction whose Top-1 differs from the system's terminal Top-1. "
                "This is the table-5 K5. The video-level companion is "
                "phenomena.short_prefix_confident_inconsistent_video_rate. "
                f"Cells clearing the threshold here: {int(n_conf)}. "
                "A NaN K5 means that count was zero, i.e. unmeasured, not zero"
            ),
            "K5_confident_cells": int(n_conf),
            "K5_video_rate": k5_video,
            "K6_first_terminal_consistent_s": _quantiles(t_term),
            "K6_first_stable_correct_s": _quantiles(t_stable),
            "K6_paired_difference_s": _quantiles(t_term - t_stable),
            "K6_ks_statistic": ks_stat,
            "K6_ks_pvalue": ks_p,
            "K6_note": (
                "first sustained agreement with the terminal prediction vs first "
                "sustained correctness; both censored at H"
            ),
        },
    }


def phenomena_table(reports: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """Flatten several reports into table 5's row layout."""
    rows = []
    for r in reports:
        ph = r["phenomena"]
        k = r["K"]
        rows.append(
            {
                "System": r["system_id"],
                "N_H": r["N_H"],
                "Flip rate": ph["flip_rate_videos_with_any_flip"],
                "Last-flip median (s)": ph["last_flip_time_s"]["median"],
                "Correct-wrong-correct": ph["correct_wrong_correct_rate"],
                "Pre-anchor overconfidence": ph["pre_anchor_confident_and_finally_wrong"],
                "K1": k["K1_prefix_equals_terminal_vs_prefix_equals_truth_agreement"],
                "K2": k["K2_full_clip_macro_acc"],
                "K3": k["K3_terminal_wrong_but_had_correct_prefix"],
                "K5": k["K5_short_prefix_confident_inconsistent_rate"],
                "K6 median diff (s)": k["K6_paired_difference_s"]["median"],
            }
        )
    return pd.DataFrame(rows)
