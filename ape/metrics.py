"""Process metrics: S_H, RMSCD@H, flips, window-end accuracy, commitment triple.

Implements the metric layer of idea sections 1.1, 3.2, 3.3 and contract
section 8. Everything here is pure numpy/pandas; nothing touches video or GPU.

Definitions, verbatim from the idea
-----------------------------------
``I^H_{i,j} = prod_{k=j..J_H} 1[yhat_{i,k} = y_i]``  (stable correctness)
``S_H(delta_j) = (1/N_H) sum_{i in E_H} I^H_{i,j}``  (the single protagonist)
``RMSCD@H = int_0^H (1 - S_H(u)) du``                (trapezoid over the grid)
``F^H_i = sum_{j=1..J_H} 1[yhat_{i,j} != yhat_{i,j-1}]``

Scoring rules that are easy to get wrong and are enforced here:

* ``pred = -1`` (``BOT``) at ``delta >= 0`` is **wrong**, never an abstention;
  the video stays in the denominator (contract section 8).
* An answer that cannot be found in the cached matrix is likewise scored as
  ``BOT``/wrong and is counted in ``missing_lookup_rate``, so a silent coverage
  hole shows up as a number instead of as a quiet improvement.
* Videos that never stabilise are kept in ``1 - S_H`` and contribute the full
  ``H`` to RMSCD; they are not deleted (this is the D6 failure mode).
* The commitment triple is reported as a triple, with ``tau_c = H`` for the
  never-committing videos (idea section 1.1).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .classes import BOT, N_CLASSES
from .cohort import cohort_summary
from .protocol import (
    TOL,
    ProtocolVector,
    n_grid_points,
    perturb_manifest,
)

PROB_COLS = [f"p{c}" for c in range(N_CLASSES)]


# --------------------------------------------------------------------------
# answer matrices (contract 5.4)
# --------------------------------------------------------------------------
class AnswerTable:
    """A cached answer matrix indexed by ``(video_id, j)`` on the finest step.

    ``j`` is relative to the **unperturbed** anchor, so ``j`` and ``delta_s``
    together pin an absolute end time. Negative ``j`` (raw pre-anchor outputs,
    kept for diagnostics per idea section 3.1) is allowed and is what lets a
    negative ``eps_sys`` be evaluated without re-running the model.
    """

    def __init__(
        self,
        system_id: str,
        delta_s: float,
        video_ids: Sequence[str],
        j_min: int,
        pred: np.ndarray,
        present: np.ndarray,
        committed: np.ndarray,
        probs: Optional[np.ndarray] = None,
        card: Optional[Dict[str, object]] = None,
        n_classes: int = N_CLASSES,
    ) -> None:
        if type(n_classes) is not int or n_classes < 2:
            raise ValueError("n_classes must be an integer >= 2")
        if probs is not None and probs.shape != (*pred.shape, n_classes):
            raise ValueError("probability shape does not match class dimension")
        self.n_classes = n_classes
        self.prob_cols = [f"p{c}" for c in range(n_classes)]
        self.system_id = str(system_id)
        self.delta_s = float(delta_s)
        self.video_ids = pd.Index([str(v) for v in video_ids])
        self.j_min = int(j_min)
        self.pred = pred
        self.present = present
        self.committed = committed
        self.probs = probs
        self.card = dict(card or {})

    # ---- construction ----
    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        system_id: str,
        card: Optional[Dict[str, object]] = None,
        n_classes: int = N_CLASSES,
    ) -> "AnswerTable":
        if type(n_classes) is not int or n_classes < 2:
            raise ValueError("n_classes must be an integer >= 2")
        prob_cols = [f"p{c}" for c in range(n_classes)]
        need = {"video_id", "j", "delta_s", "pred"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"answers for {system_id}: missing columns {sorted(missing)}")
        deltas = np.unique(np.round(df["delta_s"].to_numpy(dtype=float), 9))
        if len(deltas) != 1:
            raise ValueError(
                f"answers for {system_id}: expected one delta_s (the finest step), " f"got {deltas.tolist()}"
            )
        delta_s = float(deltas[0])
        vids = pd.Index(sorted(df["video_id"].astype(str).unique().tolist()))
        j = df["j"].to_numpy(dtype=np.int64)
        j_min, j_max = int(j.min()), int(j.max())
        n_j = j_max - j_min + 1
        rows = vids.get_indexer(df["video_id"].astype(str))
        cols = j - j_min
        if (rows < 0).any():
            raise ValueError(f"answers for {system_id}: internal video index failure")

        pred = np.full((len(vids), n_j), BOT, dtype=np.int64)
        present = np.zeros((len(vids), n_j), dtype=bool)
        committed = np.zeros((len(vids), n_j), dtype=bool)
        pred[rows, cols] = df["pred"].to_numpy(dtype=np.int64)
        present[rows, cols] = True
        if "committed" in df.columns:
            committed[rows, cols] = df["committed"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
        probs = None
        if all(c in df.columns for c in prob_cols):
            probs = np.full((len(vids), n_j, n_classes), np.nan, dtype=float)
            probs[rows, cols, :] = df[prob_cols].to_numpy(dtype=float)
        return cls(system_id, delta_s, vids, j_min, pred, present, committed, probs, card, n_classes)

    @classmethod
    def from_dir(cls, path: str) -> "AnswerTable":
        """Load ``<dir>/answers.csv`` plus the optional ``system_card.yaml``."""
        answers_csv = os.path.join(path, "answers.csv")
        if not os.path.exists(answers_csv):
            raise FileNotFoundError(answers_csv)
        df = pd.read_csv(answers_csv)
        card: Dict[str, object] = {}
        card_path = os.path.join(path, "system_card.yaml")
        if os.path.exists(card_path):
            try:
                import yaml as _yaml

                with open(card_path, "r", encoding="utf-8") as fh:
                    card = dict(_yaml.safe_load(fh) or {})
            except Exception:
                card = {}
        system_id = str(card.get("system_id") or os.path.basename(os.path.normpath(path)))
        return cls.from_frame(df, system_id, card)

    def to_frame(self) -> pd.DataFrame:
        """Long-format table in contract 5.4 order (only the present cells)."""
        rows, cols = np.nonzero(self.present)
        out = pd.DataFrame(
            {
                "video_id": self.video_ids.to_numpy()[rows],
                "j": cols + self.j_min,
                "delta_s": self.delta_s,
                "pred": self.pred[rows, cols],
            }
        )
        if self.probs is not None:
            for c, name in enumerate(self.prob_cols):
                out[name] = self.probs[rows, cols, c]
        else:
            for name in self.prob_cols:
                out[name] = np.nan
        out["committed"] = self.committed[rows, cols]
        return out.sort_values(["video_id", "j"]).reset_index(drop=True)

    # ---- lookup ----
    @property
    def j_max(self) -> int:
        return self.j_min + self.pred.shape[1] - 1

    def lookup(
        self, video_ids: Sequence[str], base_j: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Vectorised ``(pred, present, committed, probs)`` lookup by absolute index."""
        rows = self.video_ids.get_indexer(pd.Index([str(v) for v in video_ids]))
        cols = np.asarray(base_j, dtype=np.int64) - self.j_min
        ok = (rows >= 0) & (cols >= 0) & (cols < self.pred.shape[1])
        pred = np.full(len(rows), BOT, dtype=np.int64)
        present = np.zeros(len(rows), dtype=bool)
        committed = np.zeros(len(rows), dtype=bool)
        probs = None
        if ok.any():
            r, c = rows[ok], cols[ok]
            pred[ok] = self.pred[r, c]
            present[ok] = self.present[r, c]
            committed[ok] = self.committed[r, c]
            if self.probs is not None:
                probs = np.full((len(rows), self.n_classes), np.nan, dtype=float)
                probs[ok, :] = self.probs[r, c, :]
        # a cell that exists in the array but was never filled is still missing
        pred = np.where(present, pred, BOT)
        return pred, present, committed, probs

    def lookup_2d(
        self, video_ids: Sequence[str], base_j: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Grid-shaped lookup: one row per clip, one column per grid point.

        Prefer this over :meth:`lookup` on the ``(clip, grid point)`` grid. The
        flat form has to hash one string per *cell* -- 180k of them for a 2000
        clip manifest on a 90 point grid -- which dominated the runtime of every
        stage. Here the id lookup happens once per clip and the answers come out
        of a single vectorised gather.
        """
        rows = self.video_ids.get_indexer(pd.Index([str(v) for v in video_ids]))
        cols = np.asarray(base_j, dtype=np.int64) - self.j_min
        n_cols = self.pred.shape[1]
        ok = (rows[:, None] >= 0) & (cols >= 0) & (cols < n_cols)
        r = np.broadcast_to(np.where(rows >= 0, rows, 0)[:, None], cols.shape)
        c = np.clip(cols, 0, n_cols - 1)
        present = ok & self.present[r, c]
        pred = np.where(present, self.pred[r, c], BOT)
        committed = ok & self.committed[r, c]
        probs = None
        if self.probs is not None:
            probs = np.where(ok[..., None], self.probs[r, c, :], np.nan)
        return pred, present, committed, probs

    def with_identity(self, system_id: str, card: Optional[Dict[str, object]] = None) -> "AnswerTable":
        """Same answers, presented under another system's identity.

        Used for per-step re-runs: ``sysA__stride2`` holds ``sysA``'s answers at
        a coarser step, and must be scored *as* ``sysA``. Without this the system
        would appear in the audit table, and in the pair set, under the name of
        its own re-run directory.
        """
        merged = dict(self.card)
        if card:
            merged.update(card)
        merged["system_id"] = str(system_id)
        merged["answers_read_from"] = self.system_id
        return AnswerTable(
            system_id,
            self.delta_s,
            self.video_ids,
            self.j_min,
            self.pred,
            self.present,
            self.committed,
            self.probs,
            merged,
            self.n_classes,
        )

    def subsample(self, factor: int) -> "AnswerTable":
        """Keep every ``factor``-th column: the coarse-step subsampling of 3.1.5.

        Equivalent to re-running at ``factor * delta_s`` only when the system has
        no state dependence on the prefix end point; E0 tests that claim per
        system (contract section 5.4, track B reports it).
        """
        factor = int(factor)
        if factor < 1:
            raise ValueError("factor must be >= 1")
        keep = np.arange(self.pred.shape[1])
        offset = (-self.j_min) % factor  # keep the column with j == 0
        keep = keep[(keep - offset) % factor == 0]
        new_j_min = int((self.j_min + offset) / factor)
        return AnswerTable(
            self.system_id,
            self.delta_s * factor,
            self.video_ids,
            new_j_min,
            self.pred[:, keep],
            self.present[:, keep],
            self.committed[:, keep],
            None if self.probs is None else self.probs[:, keep, :],
            self.card,
            self.n_classes,
        )


def discover_answer_dirs(spec: str) -> List[str]:
    """Resolve a directory or a glob into the list of system answer directories."""
    if os.path.isdir(spec) and os.path.exists(os.path.join(spec, "answers.csv")):
        return [os.path.abspath(spec)]
    cands: List[str] = []
    if os.path.isdir(spec):
        cands = sorted(glob.glob(os.path.join(spec, "*")))
    else:
        cands = sorted(glob.glob(spec))
    out = [os.path.abspath(c) for c in cands if os.path.isdir(c) and os.path.exists(os.path.join(c, "answers.csv"))]
    return out


# --------------------------------------------------------------------------
# primitive metric functions (operate on a [n, J_H+1] prediction matrix)
# --------------------------------------------------------------------------
def _row_max_skipnan(a: np.ndarray) -> np.ndarray:
    """Row-wise max ignoring NaN, returning NaN for all-NaN rows without warnings."""
    a = np.asarray(a, dtype=float)
    all_nan = np.isnan(a).all(axis=1)
    filled = np.where(np.isnan(a), -np.inf, a)
    out = filled.max(axis=1)
    return np.where(all_nan, np.nan, out)


def correctness(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``1[yhat_{i,j} == y_i]``; ``BOT`` never equals a class code, so it is wrong."""
    return pred == y[:, None]


def stable_correct(correct: np.ndarray) -> np.ndarray:
    """``I^H_{i,j}``: correct from ``j`` through the last window index."""
    return np.logical_and.accumulate(correct[:, ::-1], axis=1)[:, ::-1]


def s_curve(indicator: np.ndarray) -> np.ndarray:
    """``S_H(delta_j)`` averaged over the fixed cohort."""
    if indicator.shape[0] == 0:
        return np.full(indicator.shape[1], np.nan)
    return indicator.mean(axis=0)


def s_curve_macro(indicator: np.ndarray, y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Class-balanced ``S_H``: stable fraction per class, then equal weights."""
    curves = []
    for c in range(n_classes):
        m = y == c
        if m.any():
            curves.append(indicator[m].mean(axis=0))
    if not curves:
        return np.full(indicator.shape[1], np.nan)
    return np.mean(np.vstack(curves), axis=0)


def trapezoid(values: np.ndarray, dx: float, axis: int = -1) -> np.ndarray:
    """Uniform-spacing trapezoid rule, written out to be version independent."""
    values = np.asarray(values, dtype=float)
    if values.shape[axis] < 2:
        return np.zeros(np.delete(np.array(values.shape), axis), dtype=float)
    total = values.sum(axis=axis)
    first = np.take(values, 0, axis=axis)
    last = np.take(values, values.shape[axis] - 1, axis=axis)
    return float(dx) * (total - 0.5 * (first + last))


def rmscd_per_video(indicator: np.ndarray, delta_s: float) -> np.ndarray:
    """``int_0^H (1 - I^H_{i,u}) du`` per video; a never-stable video gives ``H``."""
    return trapezoid(1.0 - indicator.astype(float), delta_s, axis=1)


def rmscd_from_curve(curve: np.ndarray, delta_s: float) -> float:
    """``RMSCD@H`` from the aggregate curve (identical to the mean per-video value)."""
    return float(trapezoid(1.0 - np.asarray(curve, dtype=float), delta_s))


def flip_counts(pred: np.ndarray) -> np.ndarray:
    """``F^H_i``: number of adjacent grid steps at which the Top-1 changes."""
    if pred.shape[1] < 2:
        return np.zeros(pred.shape[0], dtype=np.int64)
    return (pred[:, 1:] != pred[:, :-1]).sum(axis=1).astype(np.int64)


def macro_accuracy(pred_col: np.ndarray, y: np.ndarray, n_classes: int = N_CLASSES) -> float:
    """Class-balanced accuracy at one grid point; classes absent from ``y`` skipped."""
    accs = []
    for c in range(n_classes):
        m = y == c
        if m.any():
            accs.append(float((pred_col[m] == c).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def micro_accuracy(pred_col: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float((pred_col == y).mean())


def first_stable_index(indicator: np.ndarray) -> np.ndarray:
    """First ``j`` with ``I^H_{i,j} = 1``; ``-1`` when the video never stabilises."""
    any_stable = indicator.any(axis=1)
    idx = np.argmax(indicator, axis=1).astype(np.int64)
    return np.where(any_stable, idx, -1)


def commit_stats(
    committed: np.ndarray,
    pred: np.ndarray,
    y: np.ndarray,
    delta_s: float,
) -> Dict[str, object]:
    """Commitment triple ``(rho, tau_c, e_c)`` with horizon censoring.

    ``tau_c`` is filled to ``H`` for videos that never commit inside the window,
    and those videos are **not** deleted; ``e_c`` is defined only on committed
    videos and is therefore always reported together with ``rho``.
    """
    n, n_j = pred.shape
    h_s = float(delta_s) * (n_j - 1)
    if n == 0:
        return {
            "rho": float("nan"),
            "tau_c_mean": float("nan"),
            "tau_c_median": float("nan"),
            "e_c": float("nan"),
            "n_committed": 0,
            "n": 0,
            "h_s": h_s,
            "tau_c_per_video": np.zeros(0),
            "committed_mask": np.zeros(0, dtype=bool),
        }
    has = committed.any(axis=1)
    first = np.where(has, np.argmax(committed, axis=1), n_j - 1).astype(np.int64)
    tau = np.where(has, first.astype(float) * float(delta_s), h_s)
    if has.any():
        commit_pred = pred[np.arange(n)[has], first[has]]
        e_c = float((commit_pred != y[has]).mean())
    else:
        e_c = float("nan")
    return {
        "rho": float(has.mean()),
        "tau_c_mean": float(tau.mean()),
        "tau_c_median": float(np.median(tau)),
        "e_c": e_c,
        "n_committed": int(has.sum()),
        "n": int(n),
        "h_s": h_s,
        "tau_c_per_video": tau,
        "committed_mask": has,
    }


# --------------------------------------------------------------------------
# evaluation of one system under one protocol vector
# --------------------------------------------------------------------------
@dataclass
class EvalResult:
    """Everything one ``(system, pi)`` cell produces, arrays kept for bootstrap."""

    system_id: str
    pi: ProtocolVector
    video_ids: np.ndarray
    source_cluster_id: np.ndarray
    y: np.ndarray
    offsets_s: np.ndarray
    pred: np.ndarray
    present: np.ndarray
    committed: np.ndarray
    msp: Optional[np.ndarray]
    full_clip_pred: np.ndarray
    cohort_info: Dict[str, object]
    lookup_quantization_s: float
    n_missing_lookup: int
    card: Dict[str, object] = field(default_factory=dict)

    # ---- derived arrays ----
    @property
    def h_s(self) -> float:
        return float(self.pi.h_s)

    @property
    def effective_h_s(self) -> float:
        """``floor(H/Delta) * Delta``: the window actually integrated.

        ``H`` need not be a multiple of the step (the frozen horizons come from
        percentiles of the post-anchor length), so the grid stops at the last
        point at or before ``H``. Eligibility still uses the full ``H``, which
        keeps the cohort on the strict side, and both numbers are reported so
        that the small gap is visible rather than assumed away.
        """
        return float(self.delta_s) * (self.pred.shape[1] - 1)

    @property
    def delta_s(self) -> float:
        return float(self.pi.delta_s)

    @property
    def n(self) -> int:
        return int(self.pred.shape[0])

    @property
    def correct(self) -> np.ndarray:
        return correctness(self.pred, self.y)

    @property
    def indicator(self) -> np.ndarray:
        return stable_correct(self.correct)

    @property
    def missing_lookup_rate(self) -> float:
        total = self.pred.size
        return float(self.n_missing_lookup) / total if total else float("nan")

    # ---- scalar / curve metrics on a (possibly resampled) row index ----
    def metrics_on(self, idx: Optional[np.ndarray] = None) -> Dict[str, float]:
        """The frozen metric family plus its companions, on a row subset.

        ``idx`` may contain repeats: that is exactly what the cluster bootstrap
        feeds in.
        """
        if idx is None:
            idx = np.arange(self.n)
        idx = np.asarray(idx, dtype=np.int64)
        pred = self.pred[idx]
        y = self.y[idx]
        ind = stable_correct(correctness(pred, y))
        curve = s_curve(ind)
        curve_macro = s_curve_macro(ind, y)
        flips = flip_counts(pred)
        j_end = pred.shape[1] - 1
        out = {
            "RMSCD@H": rmscd_from_curve(curve, self.delta_s),
            "RMSCD@H_macro": rmscd_from_curve(curve_macro, self.delta_s),
            "end_window_macro_acc": macro_accuracy(pred[:, j_end], y),
            "end_window_micro_acc": micro_accuracy(pred[:, j_end], y),
            "median_flips": float(np.median(flips)) if len(flips) else float("nan"),
            "mean_flips": float(np.mean(flips)) if len(flips) else float("nan"),
            "flip_rate": float((flips > 0).mean()) if len(flips) else float("nan"),
            "full_clip_macro_acc": macro_accuracy(self.full_clip_pred[idx], y),
        }
        return out

    def s_at(self, delta_k_s: float, macro: bool = False) -> float:
        """``S_H(delta_k)`` at a pre-registered grid time; NaN if outside the window."""
        if delta_k_s < -TOL or delta_k_s > self.h_s + TOL:
            return float("nan")
        j = int(round(float(delta_k_s) / self.delta_s))
        if j < 0 or j >= self.pred.shape[1]:
            return float("nan")
        ind = self.indicator
        curve = s_curve_macro(ind, self.y) if macro else s_curve(ind)
        return float(curve[j])

    def s_at_on(self, delta_k_s: float, idx: np.ndarray) -> float:
        if delta_k_s < -TOL or delta_k_s > self.h_s + TOL:
            return float("nan")
        j = int(round(float(delta_k_s) / self.delta_s))
        if j < 0 or j >= self.pred.shape[1]:
            return float("nan")
        pred = self.pred[idx]
        y = self.y[idx]
        ind = stable_correct(correctness(pred, y))
        return float(s_curve(ind)[j])

    def frozen_family(self, s_report_delta_s: Sequence[float], idx: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Contract section 8's frozen metric family ``M``.

        Computed from a single stable-correctness pass. It is the inner loop of
        the cluster bootstrap (one call per replicate per system), so it must not
        route through :meth:`metrics_on` and :meth:`s_at_on`, which would repeat
        that pass three times over and also compute companions the family does
        not contain.
        """
        if idx is None:
            idx = np.arange(self.n)
        idx = np.asarray(idx, dtype=np.int64)
        pred = self.pred[idx]
        y = self.y[idx]
        curve = s_curve(stable_correct(correctness(pred, y)))
        flips = flip_counts(pred)
        j_end = pred.shape[1] - 1
        fam = {
            "RMSCD@H": rmscd_from_curve(curve, self.delta_s),
            "end_window_macro_acc": macro_accuracy(pred[:, j_end], y),
            "median_flips": float(np.median(flips)) if len(flips) else float("nan"),
        }
        for k in s_report_delta_s:
            k = float(k)
            j = int(round(k / self.delta_s))
            fam[f"S_H@{k:g}"] = (
                float(curve[j]) if (-TOL <= k <= self.h_s + TOL and 0 <= j < len(curve)) else float("nan")
            )
        return fam

    def commit(self) -> Dict[str, object]:
        return commit_stats(self.committed, self.pred, self.y, self.delta_s)

    def to_json_dict(self, s_report_delta_s: Sequence[float]) -> Dict[str, object]:
        ind = self.indicator
        curve = s_curve(ind)
        curve_macro = s_curve_macro(ind, self.y)
        c = self.commit()
        base = self.metrics_on(None)
        out: Dict[str, object] = {
            "system_id": self.system_id,
            "pi_hash": self.pi.pi_hash,
            "pi": self.pi.as_dict(),
            "h_s": self.h_s,
            "effective_h_s": self.effective_h_s,
            "delta_s": self.delta_s,
            "grid_offsets_s": [float(x) for x in self.offsets_s],
            "S_H": [float(x) for x in curve],
            "S_H_macro": [float(x) for x in curve_macro],
            "RMSCD": base["RMSCD@H"],
            "RMSCD_macro": base["RMSCD@H_macro"],
            "flips_median": base["median_flips"],
            "flips_mean": base["mean_flips"],
            "flip_rate": base["flip_rate"],
            "end_window_macro_acc": base["end_window_macro_acc"],
            "end_window_micro_acc": base["end_window_micro_acc"],
            "full_clip_macro_acc": base["full_clip_macro_acc"],
            "full_clip_definition": (
                "Top-1 at the longest cached causal prefix that still ends inside "
                "the clip; equals the whole-clip answer for family=clip systems."
            ),
            "commit": {
                "rho": c["rho"],
                "tau_c_mean": c["tau_c_mean"],
                "tau_c_median": c["tau_c_median"],
                "e_c": c["e_c"],
                "n_committed": c["n_committed"],
                "censoring": "tau_c = H for videos that never commit inside the window",
            },
            "N_H": int(self.cohort_info.get("N_H", self.n)),
            "n_clusters": int(self.cohort_info.get("n_clusters", 0)),
            "class_counts": self.cohort_info.get("class_counts", {}),
            "class_fractions": self.cohort_info.get("class_fractions", {}),
            "lookup_quantization_s": self.lookup_quantization_s,
            "missing_lookup_rate": self.missing_lookup_rate,
            "frozen_family": self.frozen_family(s_report_delta_s),
            "card": self.card,
        }
        return out


def evaluate_system(
    manifest: pd.DataFrame,
    answers: AnswerTable,
    pv: ProtocolVector,
    grid_max_s: Optional[float] = None,
) -> EvalResult:
    """Score one system under one protocol vector on the fixed cohort ``E_H``.

    Steps, in the order the protocol requires them: perturb the anchors, fix the
    cohort, build the causal grid, look the cached answers up by absolute time,
    then compute. No metric is allowed to change the cohort.
    """
    eff = perturb_manifest(manifest, pv)
    h_s = float(pv.h_s)
    if grid_max_s is not None and h_s > float(grid_max_s) + TOL:
        raise ValueError(f"H={h_s} exceeds grid_max_s={grid_max_s}")
    sub = eff.loc[eff["post_anchor_length_s_eff"] >= h_s - TOL].sort_values("video_id")
    sub = sub.reset_index(drop=True)

    J_H = n_grid_points(h_s, pv.delta_s)
    offsets = np.arange(J_H + 1, dtype=float) * float(pv.delta_s)
    n = len(sub)

    vid = sub["video_id"].to_numpy()
    anchor_eff = sub["anchor_s_eff"].to_numpy(dtype=float)
    anchor_base = sub["anchor_s"].to_numpy(dtype=float)
    y = sub["class_code"].to_numpy(dtype=np.int64)

    # absolute end time of every (video, grid point), then its index in the cache
    end_s = anchor_eff[:, None] + offsets[None, :]
    base_off = end_s - anchor_base[:, None]
    base_j = np.rint(base_off / answers.delta_s).astype(np.int64)

    pred, present, committed, probs = answers.lookup_2d(vid, base_j)
    msp = None
    if probs is not None:
        msp = _row_max_skipnan(probs.reshape(-1, N_CLASSES)).reshape(n, J_H + 1)

    # commitment is irrevocable: once true it stays true inside the window
    if committed.size:
        committed = np.logical_or.accumulate(committed, axis=1)

    # longest cached causal prefix that still ends inside the clip
    dur = sub["duration_s"].to_numpy(dtype=float)
    max_j_in_clip = np.floor((dur - anchor_base) / answers.delta_s + TOL).astype(np.int64)
    fc_j = np.clip(max_j_in_clip, answers.j_min, answers.j_max)
    full_pred, _, _, _ = answers.lookup(vid, fc_j)

    info = cohort_summary(eff, h_s)
    return EvalResult(
        system_id=answers.system_id,
        pi=pv,
        video_ids=vid,
        source_cluster_id=sub["source_cluster_id"].to_numpy(),
        y=y,
        offsets_s=offsets,
        pred=pred,
        present=present,
        committed=committed,
        msp=msp,
        full_clip_pred=full_pred,
        cohort_info=info,
        lookup_quantization_s=float(answers.delta_s) / 2.0,
        n_missing_lookup=int((~present).sum()),
        card=answers.card,
    )


# --------------------------------------------------------------------------
# alternative accounting conventions (E2 / D2 / D3 contrasts)
# --------------------------------------------------------------------------
def naive_persistence_to_clip_end(
    manifest: pd.DataFrame,
    answers: AnswerTable,
    pv: ProtocolVector,
    grid_max_s: float,
) -> Dict[str, object]:
    """D2: "correct from here to *its own* clip end", the short-clip-friendly rule.

    Kept in the library because idea section 5.3 E2 has to report it next to the
    fixed-horizon rule, not because it is a defensible metric.
    """
    eff = perturb_manifest(manifest, pv)
    J = n_grid_points(grid_max_s, pv.delta_s)
    offsets = np.arange(J + 1, dtype=float) * float(pv.delta_s)
    vid = eff["video_id"].to_numpy()
    anchor_eff = eff["anchor_s_eff"].to_numpy(dtype=float)
    anchor_base = eff["anchor_s"].to_numpy(dtype=float)
    plen = eff["post_anchor_length_s_eff"].to_numpy(dtype=float)
    y = eff["class_code"].to_numpy(dtype=np.int64)

    end_s = anchor_eff[:, None] + offsets[None, :]
    base_j = np.rint((end_s - anchor_base[:, None]) / answers.delta_s).astype(np.int64)
    pred, _, _, _ = answers.lookup_2d(vid, base_j)

    inside = offsets[None, :] <= (plen[:, None] + TOL)
    corr = (pred == y[:, None]) | (~inside)  # points past the clip end do not count
    ind = np.logical_and.accumulate(corr[:, ::-1], axis=1)[:, ::-1]
    first = np.where(ind.any(axis=1), np.argmax(ind, axis=1), -1)
    time_to_stable = np.where(first >= 0, first.astype(float) * pv.delta_s, np.nan)
    return {
        "convention": "per_clip_end",
        "n": int(len(eff)),
        "median_time_to_stable_s": (
            float(np.nanmedian(time_to_stable)) if np.isfinite(time_to_stable).any() else float("nan")
        ),
        "frac_ever_stable": float((first >= 0).mean()),
        "time_to_stable_per_video": time_to_stable,
        "post_anchor_length_s": plen,
    }


def dynamic_denominator_curve(
    manifest: pd.DataFrame,
    answers: AnswerTable,
    pv: ProtocolVector,
    grid_max_s: float,
    h_s: Optional[float] = None,
) -> Dict[str, object]:
    """D3: the shrinking-denominator convention (idea 3.2, contrast D3).

    Two different curves are returned and they must not be confused.

    ``s_dyn`` is **stable correctness with a dynamic risk set**: at ``delta_j``
    the denominator is only the clips that still have video there, and each clip
    is judged correct-from-here to its own end (capped at ``H``). This is the
    curve that is comparable with ``S_H`` -- same quantity, different denominator
    rule -- and it is what table 2 must use. Late grid points contain only long
    clips, which is precisely the survivorship drift the fixed cohort removes.

    ``acc`` is the plain **instantaneous** accuracy among the clips still alive.
    It answers "isn't a plain accuracy curve enough?" but it is a *different
    quantity*, not a different accounting of the same one: it asks whether the
    answer is right now, while ``S_H`` asks whether it is right from now on. Its
    area is roughly ``(1 - accuracy) * H`` and is therefore far smaller than any
    stable-correctness delay; reporting the two in one column would make a noisy
    system look excellent. It is returned under its own keys for that reason.
    """
    h_s = float(pv.h_s if h_s is None else h_s)
    eff = perturb_manifest(manifest, pv)
    J = n_grid_points(grid_max_s, pv.delta_s)
    offsets = np.arange(J + 1, dtype=float) * float(pv.delta_s)
    vid = eff["video_id"].to_numpy()
    anchor_eff = eff["anchor_s_eff"].to_numpy(dtype=float)
    anchor_base = eff["anchor_s"].to_numpy(dtype=float)
    plen = eff["post_anchor_length_s_eff"].to_numpy(dtype=float)
    y = eff["class_code"].to_numpy(dtype=np.int64)

    end_s = anchor_eff[:, None] + offsets[None, :]
    base_j = np.rint((end_s - anchor_base[:, None]) / answers.delta_s).astype(np.int64)
    pred, _, _, _ = answers.lookup_2d(vid, base_j)

    alive = offsets[None, :] <= (plen[:, None] + TOL)
    in_window = offsets[None, :] <= (h_s + TOL)
    judged = alive & in_window  # points that carry a real answer inside the window

    # plain instantaneous accuracy among the survivors
    corr_now = (pred == y[:, None]) & alive
    denom = alive.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(denom > 0, corr_now.sum(axis=0) / np.maximum(denom, 1), np.nan)

    # stable correctness to each clip's own end (capped at H), dynamic risk set
    ok = (pred == y[:, None]) | (~judged)  # points past the clip end do not count
    ind = np.logical_and.accumulate(ok[:, ::-1], axis=1)[:, ::-1]
    risk = judged.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s_dyn = np.where(risk > 0, (ind & judged).sum(axis=0) / np.maximum(risk, 1), np.nan)
    return {
        "convention": "dynamic_denominator",
        "grid_offsets_s": offsets,
        "s_dyn": s_dyn,
        "acc": acc,
        "risk_set": risk.astype(int),
        "denominator": denom.astype(int),
    }


def _length_split_gap(values: np.ndarray, lengths: np.ndarray) -> float:
    """Difference in a per-video quantity between the long and short halves.

    The E2 question in one number: does this accounting convention read
    differently on clips that simply happen to have more post-anchor video left?
    A convention that confounds stability with remaining length shows a large
    gap; the fixed-horizon cohort should show a small one.
    """
    values = np.asarray(values, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    ok = np.isfinite(values) & np.isfinite(lengths)
    if ok.sum() < 4:
        return float("nan")
    values, lengths = values[ok], lengths[ok]
    cut = float(np.median(lengths))
    short, long_ = values[lengths <= cut], values[lengths > cut]
    if len(short) == 0 or len(long_) == 0:
        return float("nan")
    return float(np.mean(long_) - np.mean(short))


def alt_cohort_report(
    manifest: pd.DataFrame,
    answers: AnswerTable,
    pv: ProtocolVector,
    grid_max_s: float,
    s_ref_delta_s: float = 1.0,
) -> Dict[str, object]:
    """The three accounting conventions of E2 / table 2, side by side.

    ``per_clip_end`` (D2) and ``dynamic_denominator`` (D3) are the naive rules the
    protocol argues against; they are computed here in full so that the argument
    is a measurement rather than an assertion. ``fixed_H_cohort`` is the protocol.

    ``conclusion_stable`` is deliberately ``None``: whether the three conventions
    support the same conclusion is a statement about a *set* of systems and is
    decided in the reporting stage, not inside one system's record.
    """
    h_s = float(pv.h_s)
    res = evaluate_system(manifest, answers, pv, grid_max_s)
    ind = res.indicator
    fixed_per_video = rmscd_per_video(ind, res.delta_s)
    eff = perturb_manifest(manifest, pv)
    len_by_id = eff.set_index("video_id")["post_anchor_length_s_eff"]
    fixed_len = len_by_id.reindex(res.video_ids).to_numpy(dtype=float)

    d2 = naive_persistence_to_clip_end(manifest, answers, pv, grid_max_s)
    t = np.asarray(d2["time_to_stable_per_video"], dtype=float)
    t_cens = np.where(np.isfinite(t), np.minimum(t, h_s), h_s)
    d2_len = np.asarray(d2["post_anchor_length_s"], dtype=float)

    d3 = dynamic_denominator_curve(manifest, answers, pv, grid_max_s, h_s)
    offs = np.asarray(d3["grid_offsets_s"], dtype=float)
    inside = offs <= h_s + TOL
    s_dyn = np.asarray(d3["s_dyn"], dtype=float)[inside]
    acc = np.asarray(d3["acc"], dtype=float)[inside]
    j_ref = int(round(float(s_ref_delta_s) / pv.delta_s))

    return {
        "s_ref_delta_s": float(s_ref_delta_s),
        "per_clip_end": {
            "RMSCD": float(np.mean(t_cens)),
            "S_at_ref": float(np.mean(np.isfinite(t) & (t <= float(s_ref_delta_s) + TOL))),
            "n": int(len(t_cens)),
            "frac_ever_stable": float(d2["frac_ever_stable"]),
            "note": (
                "D2: stable to each clip's own end, censored at H for comparison, "
                "denominator = every clip. Short clips are advantaged by "
                "construction: a clip whose video ends before delta_ref counts as "
                "stable there because it never had the chance to err afterwards. "
                "That vacuous credit is the short-clip bias this row exists to show"
            ),
        },
        "dynamic_denominator": {
            "RMSCD": rmscd_from_curve(np.nan_to_num(s_dyn, nan=0.0), pv.delta_s),
            "S_at_ref": float(s_dyn[j_ref]) if 0 <= j_ref < len(s_dyn) else float("nan"),
            "n_at_start": int(np.asarray(d3["risk_set"])[inside][0]),
            "n_at_H": int(np.asarray(d3["risk_set"])[inside][-1]),
            "note": (
                "D3: stable correctness to each clip's own end with a risk set that "
                "shrinks as clips run out; same quantity as the protocol row, "
                "different denominator. The drop from n_at_start to n_at_H is the "
                "survivorship exposure"
            ),
            "plain_accuracy_at_ref": float(acc[j_ref]) if 0 <= j_ref < len(acc) else float("nan"),
            "plain_accuracy_area": rmscd_from_curve(np.nan_to_num(acc, nan=0.0), pv.delta_s),
            "plain_accuracy_note": (
                "instantaneous accuracy among survivors: a DIFFERENT quantity "
                "(right now, not right from now on). Do not place it in the same "
                "column as the other rows' RMSCD / S_at_ref"
            ),
        },
        "fixed_H_cohort": {
            "RMSCD": float(np.mean(fixed_per_video)),
            "S_at_ref": res.s_at(float(s_ref_delta_s)),
            "n": int(res.n),
            "note": "the protocol: one cohort fixed before any grid point is scored",
        },
        "length_stratified_change": {
            "per_clip_end": _length_split_gap(t_cens, d2_len),
            "fixed_H_cohort": _length_split_gap(fixed_per_video, fixed_len),
            "note": (
                "mean(long half) - mean(short half) of the per-video delay, split at "
                "the median post-anchor length; a convention that confounds stability "
                "with remaining length shows a large gap"
            ),
        },
        "conclusion_stable": None,
        "conclusion_stable_note": ("cross-system determination; decided in the reporting stage, not per system"),
    }
