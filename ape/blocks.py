r"""Synthetic gauge blocks: answer sequences whose metrics are known on paper.

Implements idea section 3.6 and contract section 5.6. A block is *not* a model:
it is a rule that produces an answer path on the **real time axis of a real
video** (real ``anchor_s``, real ``duration_s``), so that the protocol's
response can be predicted analytically and compared with what the code
computes. This is the E0 / G1 instrument check: without it, a number that moves
cannot be attributed to the protocol rather than to a model's instability.

Families
--------
``lock(d0)``   answers a fixed wrong class until ``delta = d0``, then the truth
               forever. Stabilisation offset ``d0``; ``F = 1`` inside the window
               (or ``0`` when the switch falls outside it).
``osc(p,d0)``  alternates truth / wrong class with period ``p`` until ``d0``,
               then the truth. Its flip count is near ``floor(min(d0,H)/p)``;
               its stable-correctness curve is that of a lock at
               :func:`osc_tau`, which is at or before ``d0``.
``frac(c)``    answers the truth from a fixed fraction ``c`` of the **clip
               length**, not of the post-anchor window. This is the block that
               separates a seconds-based zero point from a proportion-based one
               (contrast D5).
``rand(eta)``  answers the truth with probability ``eta`` independently at each
               absolute time step; the noise floor.
``oracle``     answers the truth everywhere; the trivial ceiling.

Analytic expectations, in one place
-----------------------------------
Write ``tau_i`` for the offset, measured from the **protocol** anchor, at which
video ``i``'s answer becomes and stays correct. For a block evaluated under
``pi`` with common-mode shift ``eps``:

    lock(d0)   : tau_i = d0 - eps                    (same for every video)
    osc(p,d0)  : tau_i = osc_tau(p, d0) - eps        (see :func:`osc_tau`; this is
                                                      the end of the last *wrong*
                                                      phase, which can be earlier
                                                      than d0)
    frac(c)    : tau_i = c * duration_i - anchor_i - eps
    oracle     : tau_i = -inf

Then, with ``J_H = floor(H/Delta)`` and ``m_i = max(0, ceil(tau_i/Delta))``:

    S_H(delta_j) = mean_i 1[j >= m_i]
    RMSCD@H_i    = 0                 if m_i <= 0
                 = J_H * Delta       if m_i >  J_H     (never stabilises: full H)
                 = (m_i - 0.5)*Delta otherwise

The ``-0.5*Delta`` is not a bug. ``S_H`` is a step function and the protocol
integrates it with the trapezoid rule (contract section 8), so an exactly
grid-aligned stabilisation at ``tau`` is measured as ``tau - Delta/2``. That
offset is a property of the protocol -- it is the delay quantisation floor that
idea section 5.3 E4 asks for on the step axis -- and it is what
:func:`analytic_rmscd_per_video` reproduces exactly. In the ideal continuous
limit the same quantity is ``min(tau, H)``, and
:func:`ideal_rmscd_per_video` returns that for the comparison.

Flip count of ``osc`` (the rounding convention, written out)
------------------------------------------------------------
With ``eps = 0``, ``d0 > 0`` and ``p >= 2*Delta``, let

    m   = ceil(d0/Delta)                     first grid index at or after d0
    M   = min(m, J_H + 1)                    clipped to the window
    K   = #{k >= 1 : ceil(k*p/Delta) <= min(M-1, J_H)}      parity boundaries
    a   = 1 if floor((M-1)*Delta / p) is odd else 0         answer just before d0

then ``F^H = K + 1[M <= J_H and a == 0]``. Writing
``base = floor(min(d0, H) / p)``, this always lands within one of ``base``:
the last parity block before ``d0`` is truncated by the lock, so whether the
lock is itself a flip depends on that block's parity (``base - 1``), and when
``d0 < p`` the lock is a flip although no full period has elapsed
(``base + 1``). :func:`analytic_flips_osc` returns the exact value and the tests
check it against both an independent enumeration of the block rule and the value
the metric code computes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .classes import BOT, N_CLASSES, wrong_class
from .metrics import PROB_COLS
from .protocol import TOL, ProtocolVector, n_grid_points, perturb_manifest

BLOCK_FAMILIES = ("lock", "osc", "frac", "rand", "oracle")

#: Confidence assigned to a block's chosen class. Blocks have no real posterior;
#: this is a declared constant so that confidence-based phenomena (K5, pre-anchor
#: overconfidence) are computable on blocks, and the system card says so.
BLOCK_CONFIDENCE = 0.9


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------
def _fmt_param(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float) and abs(v - round(v)) < TOL:
        return "%d" % int(round(v))
    return ("%g" % v) if isinstance(v, (int, float)) else str(v)


def block_id(
    family: str,
    params: Optional[Dict[str, object]] = None,
    commit_after_s: Optional[float] = None,
) -> str:
    """``block__<family>__<params>`` (contract 5.6).

    ``commit_after_s`` is part of the identity: the same answer path with and
    without an irrevocable commitment is two different systems in the audit.
    """
    params = dict(params or {})
    if commit_after_s is not None:
        params = dict(params)
        params["commit"] = float(commit_after_s)
    if not params:
        return f"block__{family}__default"
    body = "_".join(f"{k}-{_fmt_param(params[k])}" for k in sorted(params))
    return f"block__{family}__{body}"


def _uniform01(*parts) -> np.ndarray:
    """Deterministic uniform draws keyed by arbitrary parts (vectorised over the last)."""
    keys = parts[-1]
    prefix = "|".join(str(p) for p in parts[:-1])
    out = np.empty(len(keys), dtype=float)
    for i, k in enumerate(keys):
        digest = hashlib.blake2b(f"{prefix}|{k}".encode("utf-8"), digest_size=8).digest()
        out[i] = int.from_bytes(digest, "little") / float(2**64)
    return out


# --------------------------------------------------------------------------
# the block answer rule (defined on absolute time)
# --------------------------------------------------------------------------
def block_predict(
    family: str,
    params: Dict[str, object],
    y: int,
    anchor_s: float,
    duration_s: float,
    end_s: np.ndarray,
    video_id: str,
    step_s: float,
) -> np.ndarray:
    """Answers of one block for one video at the given absolute end times.

    The rule never consults the protocol anchor, only the true one: a block is a
    fixed physical behaviour that the protocol then measures, possibly through a
    mis-set zero point. That separation is what makes the shift experiment
    meaningful.
    """
    end_s = np.asarray(end_s, dtype=float)
    y = int(y)
    w = wrong_class(y)
    delta = end_s - float(anchor_s)

    if family == "oracle":
        return np.full(end_s.shape, y, dtype=np.int64)

    if family == "lock":
        d0 = float(params["d0"])
        return np.where(delta >= d0 - TOL, y, w).astype(np.int64)

    if family == "osc":
        p = float(params["p"])
        d0 = float(params["d0"])
        if p <= 0:
            raise ValueError("osc period p must be > 0")
        # +TOL: end_s is reconstructed as anchor + j*step, so delta can come back
        # as 2.9999999999999996 and land in the wrong parity block without it
        phase = np.floor(delta / p + TOL)
        alt = np.where(np.mod(phase, 2.0) == 0.0, w, y)
        return np.where(delta >= d0 - TOL, y, alt).astype(np.int64)

    if family == "frac":
        c = float(params["c"])
        return np.where(end_s >= c * float(duration_s) - TOL, y, w).astype(np.int64)

    if family == "rand":
        eta = float(params["eta"])
        # keyed on absolute time in milliseconds, not on the query step, so that
        # rand is a genuine function of absolute time and sub-sampling a cached
        # fine-step matrix gives the same answers as generating a coarse one
        keys = np.rint(end_s * 1000.0).astype(np.int64)
        u = _uniform01("rand", video_id, eta, keys)
        v = _uniform01("randclass", video_id, eta, keys)
        wrong_pool = [c for c in range(N_CLASSES) if c != y]
        pick = np.asarray(wrong_pool, dtype=np.int64)[
            np.minimum((v * len(wrong_pool)).astype(np.int64), len(wrong_pool) - 1)
        ]
        return np.where(u < eta, y, pick).astype(np.int64)

    raise ValueError(f"unknown block family {family!r}; known: {BLOCK_FAMILIES}")


def osc_tau(p: float, d0: float) -> float:
    """Stabilisation offset of ``osc(p, d0)``: the end of its **last wrong phase**.

    This is deliberately not ``d0``. Stable correctness only asks that no error
    occurs from here to the window end, and ``osc`` alternates, so if the phase
    immediately before ``d0`` already answers correctly then the block became
    stably correct earlier than it "decided". Concretely, with even phases wrong:

        q   = the largest even integer with ``q*p < d0``
        tau = min((q+1)*p, d0)

    For ``osc(1.0, 4.0)`` that gives ``tau = 3.0``, not ``4.0``: phase 3 covers
    ``[3,4)`` and is already correct. The consequence is worth stating plainly --
    ``RMSCD`` measures the last error, not the moment a system settles -- and it
    is exactly why the flip count is reported next to it rather than instead
    of it.
    """
    p, d0 = float(p), float(d0)
    if p <= 0:
        raise ValueError("osc period p must be > 0")
    if d0 <= 0:
        return -np.inf
    k = int(math.ceil(d0 / p - TOL)) - 1  # largest integer q with q*p < d0
    if k < 0:
        return -np.inf
    q = k if k % 2 == 0 else k - 1
    if q < 0:
        return -np.inf
    return float(min((q + 1) * p, d0))


def osc_tau_in_window(p: float, d0: float, h_s: float) -> float:
    """``osc``'s stabilisation offset *as the window can see it*.

    :func:`osc_tau` answers "when does this block stop erring, ever". Inside a
    finite window the answer can be earlier, because stable correctness only
    asks that no error occurs between here and the **window end**. If the window
    closes part-way through a correct phase, the block is stably correct from the
    start of that phase even though it has not settled and will err again
    afterwards.

    Concretely ``osc(2, 15)`` measured at ``H = 10``: phase 5 covers ``[10, 12)``
    and answers correctly, so the last grid point is correct and
    ``tau = 10``, giving ``RMSCD = 9.75`` -- not the ``10.0`` (full window) that
    the global ``osc_tau = 14`` would predict. At ``H = 4`` the same block sits in
    phase 2, which is wrong, so it never stabilises inside the window.

    This is not a quirk of the block; it is the protocol being honest that a
    fixed horizon can only report what happens before it closes.
    """
    p, d0, h_s = float(p), float(d0), float(h_s)
    if d0 <= 0:
        return -np.inf
    if d0 <= h_s + TOL:
        return osc_tau(p, d0)
    q = int(math.floor(h_s / p + TOL))
    if q % 2 == 1:  # odd phase answers correctly
        return float(q * p)
    return np.inf  # wrong at the window end: never stable inside the window


def stabilisation_offsets(
    family: str,
    params: Dict[str, object],
    manifest: pd.DataFrame,
    eps_sys_s: float = 0.0,
    h_s: Optional[float] = None,
) -> np.ndarray:
    """``tau_i``: offset from the protocol anchor at which the answer locks in.

    ``-inf`` means "already stable at the anchor"; ``nan`` means there is no
    deterministic stabilisation (``rand``).
    """
    n = len(manifest)
    if family == "oracle":
        return np.full(n, -np.inf)
    if family == "lock":
        return np.full(n, float(params["d0"]) - float(eps_sys_s))
    if family == "osc":
        p, d0 = float(params["p"]), float(params["d0"])
        # the shift moves the protocol's zero, so the window seen by the block
        # runs to h_s + eps_sys; ask for the stabilisation inside that window
        tau = (
            osc_tau(p, d0)
            if h_s is None
            else osc_tau_in_window(p, d0, float(h_s) + float(eps_sys_s))
        )
        if np.isneginf(tau) or np.isposinf(tau):
            return np.full(n, tau)
        return np.full(n, tau - float(eps_sys_s))
    if family == "frac":
        c = float(params["c"])
        dur = manifest["duration_s"].to_numpy(dtype=float)
        anc = manifest["anchor_s"].to_numpy(dtype=float)
        return c * dur - anc - float(eps_sys_s)
    if family == "rand":
        return np.full(n, np.nan)
    raise ValueError(f"unknown block family {family!r}")


# --------------------------------------------------------------------------
# answer-matrix generation
# --------------------------------------------------------------------------
def generate_block_answers(
    manifest: pd.DataFrame,
    family: str,
    params: Optional[Dict[str, object]] = None,
    delta_fine_s: float = 0.25,
    span_s: float = 10.0,
    pad_s: float = 2.0,
    commit_after_s: Optional[float] = None,
    confidence: float = BLOCK_CONFIDENCE,
) -> pd.DataFrame:
    """Contract 5.4 answer matrix for one block, on the finest step.

    The column range runs from ``j = -ceil(pad_s/Delta_fine)`` to
    ``j = ceil((span_s + pad_s)/Delta_fine)``. The negative part is the raw
    pre-anchor output that idea section 3.1 allows to be recorded (the protocol
    masks it to ``BOT`` at evaluation time); it is what makes a negative
    ``eps_sys`` evaluable from cache instead of requiring a re-run.
    """
    params = dict(params or {})
    j_min = -int(math.ceil(float(pad_s) / float(delta_fine_s) - TOL))
    j_max = int(math.ceil((float(span_s) + float(pad_s)) / float(delta_fine_s) - TOL))
    js = np.arange(j_min, j_max + 1, dtype=np.int64)

    frames: List[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        anchor = float(getattr(row, "anchor_s"))
        end_s = anchor + js.astype(float) * float(delta_fine_s)
        pred = block_predict(
            family,
            params,
            int(getattr(row, "class_code")),
            anchor,
            float(getattr(row, "duration_s")),
            end_s,
            str(getattr(row, "video_id")),
            float(delta_fine_s),
        )
        committed = (
            np.zeros(len(js), dtype=bool)
            if commit_after_s is None
            else (end_s - anchor) >= float(commit_after_s) - TOL
        )
        piece = pd.DataFrame(
            {
                "video_id": str(getattr(row, "video_id")),
                "j": js,
                "delta_s": float(delta_fine_s),
                "pred": pred,
            }
        )
        probs = np.full((len(js), N_CLASSES), (1.0 - confidence) / (N_CLASSES - 1))
        probs[np.arange(len(js)), pred] = confidence
        for c, name in enumerate(PROB_COLS):
            piece[name] = probs[:, c]
        piece["committed"] = committed
        frames.append(piece)
    out = pd.concat(frames, ignore_index=True)
    return out


def answers_for_prefixes(
    prefixes: pd.DataFrame,
    manifest: pd.DataFrame,
    family: str,
    params: Optional[Dict[str, object]] = None,
    step_s: Optional[float] = None,
    confidence: float = BLOCK_CONFIDENCE,
) -> pd.DataFrame:
    """Evaluate a block *exactly* on a given (possibly perturbed) prefix list.

    Unlike :func:`generate_block_answers` this is not quantised to the cached
    finest step, so it is the reference for anchor perturbations smaller than
    ``Delta_fine`` (per-video jitter in particular). Returns one row per prefix
    with the block's answer at that prefix's exact ``end_s``.
    """
    params = dict(params or {})
    prefixes = prefixes.reset_index(drop=True)
    meta = manifest.set_index("video_id")
    out_pred = np.empty(len(prefixes), dtype=np.int64)
    step = float(step_s) if step_s is not None else float(prefixes["delta_s"].iloc[0])
    positions = np.arange(len(prefixes), dtype=np.int64)
    for vid, grp in prefixes.groupby("video_id", sort=False):
        row = meta.loc[str(vid)]
        pred = block_predict(
            family,
            params,
            int(row["class_code"]),
            float(row["anchor_s"]),
            float(row["duration_s"]),
            grp["end_s"].to_numpy(dtype=float),
            str(vid),
            step,
        )
        out_pred[positions[grp.index.to_numpy()]] = pred
    res = prefixes[["video_id", "j", "delta_s"]].copy()
    res["pred"] = out_pred
    probs = np.full((len(res), N_CLASSES), (1.0 - confidence) / (N_CLASSES - 1))
    probs[np.arange(len(res)), out_pred] = confidence
    for c, name in enumerate(PROB_COLS):
        res[name] = probs[:, c]
    res["committed"] = False
    return res


def block_card(
    family: str,
    params: Dict[str, object],
    commit_after_s: Optional[float] = None,
) -> Dict[str, object]:
    """Contract 5.4 system card for a block."""
    return {
        "system_id": block_id(family, params, commit_after_s),
        "family": "block",
        "description": (
            f"synthetic gauge block {family}({params}); answers generated by a rule "
            "on the real video time axis, no model is involved"
        ),
        "backbone": None,
        "trained_on_split": None,
        "dev_tuned_params": None,
        "train_data_unknown": False,
        "cost_note": "negligible; generated analytically",
        "causal": True,
        "parent_system_id": None,
        "block_family": family,
        "block_params": {k: params[k] for k in sorted(params)},
        "commit_after_s": commit_after_s,
        "probabilities": (
            f"synthetic one-hot at confidence {BLOCK_CONFIDENCE}; not a posterior, "
            "do not read as calibration"
        ),
        "analytic_expectations": (
            "see ape.blocks module docstring; ape.blocks.analytic_* reproduce them"
        ),
    }


def write_block(
    out_root: str,
    manifest: pd.DataFrame,
    family: str,
    params: Optional[Dict[str, object]] = None,
    **kw,
) -> str:
    """Write ``outputs/answers/block__<family>__<params>/`` (contract 5.6)."""
    params = dict(params or {})
    commit_after_s = kw.get("commit_after_s")
    sid = block_id(family, params, commit_after_s)
    path = os.path.join(out_root, sid)
    os.makedirs(path, exist_ok=True)
    df = generate_block_answers(manifest, family, params, **kw)
    df.to_csv(os.path.join(path, "answers.csv"), index=False)
    card = block_card(family, params, commit_after_s)
    try:
        import yaml as _yaml

        with open(os.path.join(path, "system_card.yaml"), "w", encoding="utf-8") as fh:
            _yaml.safe_dump(card, fh, sort_keys=True, allow_unicode=True)
    except Exception:  # pragma: no cover
        with open(os.path.join(path, "system_card.yaml"), "w", encoding="utf-8") as fh:
            json.dump(card, fh, indent=2, sort_keys=True)
    return path


# --------------------------------------------------------------------------
# analytic expectations
# --------------------------------------------------------------------------
def analytic_s_curve(tau: np.ndarray, h_s: float, delta_s: float) -> np.ndarray:
    """``S_H`` for a cohort whose per-video stabilisation offsets are ``tau``."""
    J_H = n_grid_points(h_s, delta_s)
    offsets = np.arange(J_H + 1, dtype=float) * float(delta_s)
    tau = np.asarray(tau, dtype=float)
    if len(tau) == 0:
        return np.full(J_H + 1, np.nan)
    return (offsets[None, :] >= tau[:, None] - TOL).mean(axis=0)


def analytic_rmscd_per_video(tau: np.ndarray, h_s: float, delta_s: float) -> np.ndarray:
    """Trapezoid ``RMSCD@H`` per video from the stabilisation offsets.

    Reproduces the ``-Delta/2`` step-integration offset exactly; see the module
    docstring.
    """
    tau = np.asarray(tau, dtype=float)
    J_H = n_grid_points(h_s, delta_s)
    with np.errstate(invalid="ignore"):
        m = np.where(np.isneginf(tau), 0.0, np.ceil(tau / float(delta_s) - TOL))
    m = np.maximum(m, 0.0)
    m = np.where(np.isposinf(tau) | np.isnan(tau), J_H + 1.0, m)
    full = float(J_H) * float(delta_s)
    return np.where(m <= 0.0, 0.0, np.where(m > J_H, full, (m - 0.5) * float(delta_s)))


def analytic_rmscd(tau: np.ndarray, h_s: float, delta_s: float) -> float:
    return float(np.mean(analytic_rmscd_per_video(tau, h_s, delta_s)))


def ideal_rmscd_per_video(tau: np.ndarray, h_s: float) -> np.ndarray:
    """``min(max(tau,0), H)``: the continuous-limit value, for comparison only."""
    tau = np.asarray(tau, dtype=float)
    return np.clip(np.where(np.isneginf(tau), 0.0, tau), 0.0, float(h_s))


def analytic_flips_lock(d0: float, h_s: float, delta_s: float, eps_sys_s: float = 0.0) -> int:
    """``lock`` changes its answer at most once inside the window."""
    tau = float(d0) - float(eps_sys_s)
    J_H = n_grid_points(h_s, delta_s)
    m = max(0, int(math.ceil(tau / float(delta_s) - TOL)))
    return int(1 <= m <= J_H)


def analytic_flips_osc(p: float, d0: float, h_s: float, delta_s: float) -> int:
    """Exact ``F^H`` of ``osc(p, d0)`` at ``eps_sys = 0``; see the module docstring."""
    p, d0 = float(p), float(d0)
    delta_s = float(delta_s)
    J_H = n_grid_points(h_s, delta_s)
    m = max(0, int(math.ceil(d0 / delta_s - TOL)))
    M = min(m, J_H + 1)
    limit = min(M - 1, J_H)
    K = 0
    k = 1
    while True:
        g = int(math.ceil(k * p / delta_s - TOL))
        if g > limit:
            break
        K += 1
        k += 1
        if k > 10_000_000:  # pragma: no cover - defensive
            raise RuntimeError("osc flip count did not terminate")
    lock_flip = 0
    if M <= J_H and M - 1 >= 0:
        a_before = int(math.floor(((M - 1) * delta_s) / p + TOL)) % 2 == 1
        lock_flip = int(not a_before)
    return int(K + lock_flip)


def reference_flips_osc(
    p: float, d0: float, h_s: float, delta_s: float, eps_sys_s: float = 0.0
) -> int:
    """Independent enumeration of the ``osc`` rule; the cross-check for the formula."""
    J_H = n_grid_points(h_s, delta_s)
    u = np.arange(J_H + 1, dtype=float) * float(delta_s)
    delta = u + float(eps_sys_s)
    y, w = 1, 0  # any two distinct symbols
    phase = np.floor(delta / float(p) + TOL)
    alt = np.where(np.mod(phase, 2.0) == 0.0, w, y)
    pred = np.where(delta >= float(d0) - TOL, y, alt)
    return int((pred[1:] != pred[:-1]).sum())


def analytic_s_curve_rand(eta: float, h_s: float, delta_s: float) -> np.ndarray:
    """``E[S_H(delta_j)] = eta^(J_H - j + 1)`` for an i.i.d. random block."""
    J_H = n_grid_points(h_s, delta_s)
    j = np.arange(J_H + 1, dtype=float)
    return float(eta) ** (J_H - j + 1.0)


def block_expectations(
    family: str,
    params: Dict[str, object],
    manifest: pd.DataFrame,
    pv: ProtocolVector,
) -> Dict[str, object]:
    """Analytic expectations for one block under one ``pi``, on ``E_H``.

    Only defined for zero jitter: a per-video jitter realisation moves every
    ``tau_i`` by a different amount, which the tests handle by regenerating the
    block exactly on the perturbed prefix list instead.
    """
    if abs(float(pv.eps_jit_sd_s)) > TOL:
        raise ValueError("analytic expectations assume eps_jit_sd_s == 0")
    eff = perturb_manifest(manifest, pv)
    coh = eff.loc[eff["post_anchor_length_s_eff"] >= float(pv.h_s) - TOL].sort_values(
        "video_id"
    )
    tau = stabilisation_offsets(
        family, params, coh, eps_sys_s=float(pv.eps_sys_s), h_s=float(pv.h_s)
    )
    out: Dict[str, object] = {
        "N_H": int(len(coh)),
        "S_H": analytic_s_curve(tau, pv.h_s, pv.delta_s),
        "RMSCD": analytic_rmscd(tau, pv.h_s, pv.delta_s),
        "ideal_RMSCD": float(np.mean(ideal_rmscd_per_video(tau, pv.h_s)))
        if len(coh)
        else float("nan"),
    }
    if family == "lock":
        out["flips_median"] = float(
            analytic_flips_lock(float(params["d0"]), pv.h_s, pv.delta_s, pv.eps_sys_s)
        )
    elif family == "osc" and abs(float(pv.eps_sys_s)) <= TOL:
        out["flips_median"] = float(
            analytic_flips_osc(float(params["p"]), float(params["d0"]), pv.h_s, pv.delta_s)
        )
    elif family == "oracle":
        out["flips_median"] = 0.0
    return out


def default_block_specs() -> List[Dict[str, object]]:
    """The factor table used by the smoke run when the protocol file lists none."""
    return [
        {"family": "oracle", "params": {}},
        {"family": "lock", "params": {"d0": 1.0}},
        {"family": "lock", "params": {"d0": 3.0}},
        {"family": "lock", "params": {"d0": 3.0}, "commit_after_s": 3.0},
        {"family": "lock", "params": {"d0": 8.0}},
        {"family": "osc", "params": {"p": 1.0, "d0": 4.0}},
        {"family": "osc", "params": {"p": 0.5, "d0": 6.0}},
        {"family": "frac", "params": {"c": 0.6}},
        {"family": "rand", "params": {"eta": 0.5}},
        {"family": "rand", "params": {"eta": 0.9}},
    ]
