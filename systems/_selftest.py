"""Self-checks for the B track's pure-array logic. No video, no GPU, no manifest.

Covers what would silently corrupt the audit if it were wrong:
  1. the stable-correct curve and its area against hand-computed trajectories;
  2. the lock-at-d0 analytic case, RMSCD@H == d0;
  3. causality of all four post-processing arms and both commitment rules --
     truncating the input must not change any earlier output;
  4. commitment semantics: bot before the crossing, frozen label after it,
     never-crossing clips kept as all-bot rather than dropped;
  5. the answer-matrix round trip.

Usage: python -X utf8 systems/_selftest.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import commit as CM  # noqa: E402
from systems import common as C  # noqa: E402
from systems import devsel  # noqa: E402
from systems import postproc as PP  # noqa: E402

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"[{status}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def t_curve():
    # one clip, labels 0; predictions wrong at j=0,1 then right through j=4
    preds = np.array([[1, 1, 0, 0, 0]])
    y = np.array([0])
    curve = devsel.stable_correct_curve(preds, y, j_end=4)
    check("stable curve suffix-AND", np.array_equal(curve, np.array([0, 0, 1, 1, 1])),
          f"got {curve.tolist()}")

    # a late relapse must knock out every earlier index
    preds2 = np.array([[0, 0, 0, 1, 0]])
    curve2 = devsel.stable_correct_curve(preds2, y, j_end=4)
    check("late flip invalidates earlier stability",
          np.array_equal(curve2, np.array([0, 0, 0, 0, 1])), f"got {curve2.tolist()}")


def t_lock_block():
    """beta_lock(d0): wrong before d0, correct from d0 to the window end.

    The exact area of 1 - S_H is d0, but CONTRACT 8 fixes the trapezoidal rule,
    and a step landing on a grid point is averaged across its last interval. The
    value the protocol actually reports is therefore d0 - delta/2, for every d0
    strictly inside the window. That offset is a property of the quantisation, not
    of the system: it shifts every system by the same amount at a given delta, and
    it moves when delta moves. Flagged for the A track in systems/REPORT.md.
    """
    delta, H = 0.25, 6.0
    j_end = int(round(H / delta))
    y = np.zeros(7, dtype=np.int64)
    for d0 in (0.5, 2.0, 4.75):
        j0 = int(round(d0 / delta))
        preds = np.full((7, j_end + 1), 3, dtype=np.int64)
        preds[:, j0:] = 0
        curve = devsel.stable_correct_curve(preds, y, j_end)
        area = devsel.rmscd(curve, delta)
        check(f"lock block RMSCD == d0 - delta/2 (d0={d0})",
              abs(area - (d0 - delta / 2)) < 1e-9, f"got {area}, expected {d0 - delta / 2}")

    # the quantisation offset tracks delta, which is what makes it a protocol
    # property rather than a system property
    for coarse in (0.5, 1.0):
        j_end_c = int(round(H / coarse))
        j0_c = int(round(2.0 / coarse))
        preds = np.full((7, j_end_c + 1), 3, dtype=np.int64)
        preds[:, j0_c:] = 0
        area = devsel.rmscd(devsel.stable_correct_curve(preds, y, j_end_c), coarse)
        check(f"lock block offset follows delta (delta={coarse})",
              abs(area - (2.0 - coarse / 2)) < 1e-9, f"got {area}")

    # a clip that never stabilises must contribute the whole window, not be dropped
    never = np.full((3, j_end + 1), 3, dtype=np.int64)
    area_never = devsel.rmscd(devsel.stable_correct_curve(never, np.zeros(3, np.int64), j_end), delta)
    check("never-stable clips contribute the full window", abs(area_never - H) < 1e-9,
          f"got {area_never}")

    j0 = int(round(2.0 / delta))
    preds = np.full((7, j_end + 1), 3, dtype=np.int64)
    preds[:, j0:] = 0
    check("lock block switches exactly once in the window",
          int(devsel.flips(preds, j_end).max()) == 1)


def t_cohort():
    post = np.array([2.0, 6.0, 6.0, 12.0])
    m = devsel.cohort_mask(post, 6.0)
    check("cohort is post_len >= H", m.tolist() == [False, True, True, True])


def _rand_probs(n=6, j=25, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, j, C.N_CLASSES))
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def t_causality():
    """Truncating the future must not change any earlier output."""
    probs = _rand_probs()
    cut = 13
    for arm, value in (("ema", 0.35), ("majority", 5), ("hysteresis", 0.1), ("patience", 3)):
        full, _ = PP.apply_arm(arm, probs, value)
        trunc, _ = PP.apply_arm(arm, probs[:, :cut], value)
        check(f"postproc/{arm} is causal", np.array_equal(full[:, :cut], trunc))
    for rule, thr in (("msp", 0.5), ("margin", 0.15)):
        full, cf, _ = CM.apply_commit(probs, rule, thr)
        trunc, ct, _ = CM.apply_commit(probs[:, :cut], rule, thr)
        check(f"commit/{rule} is causal",
              np.array_equal(full[:, :cut], trunc) and np.array_equal(cf[:, :cut], ct))


def t_commit_semantics():
    n, j = 4, 10
    probs = np.full((n, j, C.N_CLASSES), 0.2)
    # clip 0 crosses at step 3 on class 2; clip 1 never crosses
    probs[0, 3:] = [0.05, 0.05, 0.8, 0.05, 0.05]
    preds, committed, first = CM.apply_commit(probs, "msp", 0.7)
    check("commit fires at the first crossing", int(first[0]) == 3, f"first={first[0]}")
    check("bot before the commitment", (preds[0, :3] == C.BOT).all())
    check("label frozen after the commitment", (preds[0, 3:] == 2).all())
    check("committed flag matches", (~committed[0, :3]).all() and committed[0, 3:].all())
    check("never-crossing clip stays all bot and is kept",
          (preds[1] == C.BOT).all() and (~committed[1]).all() and preds.shape[0] == n)

    y = np.array([2, 0, 0, 0])
    trip = CM.commit_triplet(first, preds, y, j_end=j - 1, delta_s=0.25)
    check("rho counts only clips that committed in the window", abs(trip["rho"] - 0.25) < 1e-9,
          f"rho={trip['rho']}")
    check("tau_c charges the full window to non-committers",
          abs(trip["tau_c"] - (0.75 + 3 * (j - 1) * 0.25) / 4) < 1e-9, f"tau_c={trip['tau_c']}")
    check("e_c is computed on committed clips only", abs(trip["e_c"] - 0.0) < 1e-9,
          f"e_c={trip['e_c']}")


def t_missing_evidence_is_held():
    """A cell with no probabilities must not poison the rest of the trajectory."""
    probs = _rand_probs(n=3, j=12, seed=5)
    probs[:, :2] = np.nan          # empty prefixes at the head, as at negative j
    for arm, value in (("ema", 0.35), ("majority", 5), ("hysteresis", 0.1), ("patience", 3)):
        preds, q = PP.apply_arm(arm, probs, value)
        check(f"{arm} recovers after missing evidence",
              bool((preds[:, 2:] != C.BOT).all()), f"got {preds[0].tolist()}")
    _p, q = PP.apply_arm("ema", probs, 0.35)
    check("ema state is finite once evidence arrives", bool(np.isfinite(q[:, 2:]).all()))
    _p, f = PP.apply_arm("majority", probs, 5)
    populated = ~np.isnan(f).any(axis=-1)
    check("majority frequencies sum to 1 where populated",
          bool(np.abs(f[populated].sum(axis=-1) - 1.0).max() < 1e-9))
    check("majority leaves unanswerable cells NaN", bool(np.isnan(f[:, :2]).all()))


def t_hysteresis_holds():
    """A margin larger than any achievable gap must freeze the first label."""
    probs = _rand_probs(seed=7)
    preds, _ = PP.apply_arm("hysteresis", probs, 1.5)
    check("hysteresis with an unreachable margin never switches",
          bool((preds == preds[:, :1]).all()))
    preds0, _ = PP.apply_arm("hysteresis", probs, 0.0)
    raw = probs.argmax(axis=-1)
    check("hysteresis with margin 0 still needs a strict gap",
          bool((preds0[:, 0] == raw[:, 0]).all()))


def t_answer_roundtrip():
    vids = ["a", "b", "c"]
    preds = np.array([[0, 1, 2, 3], [4, 4, 4, 4], [-1, -1, 3, 3]])
    probs = np.full((3, 4, 5), 0.2)
    df = C.build_answer_frame(vids, preds, probs, delta_s=0.25, j_offset=-2)
    check("answer frame has the contract columns", list(df.columns) == C.ANSWER_COLUMNS)
    check("answer frame row count", len(df) == 12)
    check("j offset is carried through", sorted(df["j"].unique().tolist()) == [-2, -1, 0, 1])
    v2, js2, p2, q2 = C.answers_to_arrays(df)
    check("answer frame round trip", v2 == vids and np.array_equal(p2, preds)
          and np.allclose(q2, probs))
    check("js round trip", js2.tolist() == [-2, -1, 0, 1])
    check("post-anchor slice starts at j=0", C.post_anchor_slice(js2) == slice(2, 4))
    check("committed defaults to False", bool((~df["committed"]).all()))


def t_two_sided_grid():
    """The delivered grid must straddle the anchor and keep j = 0 on every stride."""
    js = C.grid_js()
    check("grid covers j = -11 .. 88", js[0] == -11 and js[-1] == 88 and len(js) == 100)
    check("grid reaches anchor + 22 s", abs(js[-1] * C.FINEST_DELTA_S - 22.0) < 1e-9)
    check("grid starts at anchor - 2.75 s", abs(js[0] * C.FINEST_DELTA_S + 2.75) < 1e-9)
    for k in (2, 4):
        kept = js[(js % k) == 0]
        check(f"stride {k} keeps j = 0", 0 in kept.tolist())


def t_commit_pre_anchor():
    """With a post-anchor start column, delta < 0 carries the raw Top-1, not a commitment."""
    n, j, start = 3, 12, 4
    probs = np.full((n, j, C.N_CLASSES), 0.2)
    probs[:, :, 1] = 0.2
    probs[0, :start] = [0.05, 0.05, 0.8, 0.05, 0.05]   # confident before the anchor
    probs[0, start + 2:] = [0.8, 0.05, 0.05, 0.05, 0.05]
    preds, committed, first = CM.apply_commit(probs, "msp", 0.7, start_col=start)
    check("no commitment fires before the start column", int(first[0]) == start + 2,
          f"first={first[0]}")
    check("pre-anchor rows carry the raw Top-1", (preds[0, :start] == 2).all(),
          f"got {preds[0, :start].tolist()}")
    check("pre-anchor rows are not marked committed", bool((~committed[0, :start]).all()))
    check("post-anchor pre-crossing rows are bot", (preds[0, start:start + 2] == C.BOT).all())
    check("committed is monotone once set",
          bool((committed[:, :-1] & ~committed[:, 1:]).sum() == 0))


def t_subsample_of_stateless():
    """A stateless-in-j system sub-samples exactly; a stateful one need not."""
    probs = _rand_probs(seed=11)
    raw = probs.argmax(axis=-1)                      # the base classifier: stateless in j
    check("stateless system: stride-2 sub-sample equals a stride-2 rerun",
          np.array_equal(raw[:, ::2], probs[:, ::2].argmax(axis=-1)))
    ema_full, _ = PP.apply_arm("ema", probs, 0.35)
    ema_re, _ = PP.apply_arm("ema", probs[:, ::2], 0.35)
    check("stateful arm: sub-sample is NOT a rerun (documented, not a bug)",
          not np.array_equal(ema_full[:, ::2], ema_re))


def main() -> int:
    for fn in (t_curve, t_lock_block, t_cohort, t_causality, t_commit_semantics,
               t_missing_evidence_is_held, t_hysteresis_holds, t_answer_roundtrip,
               t_two_sided_grid, t_commit_pre_anchor, t_subsample_of_stateless):
        print(f"--- {fn.__name__} ---")
        fn()
    print()
    if FAILS:
        print(f"SELFTEST FAILED: {len(FAILS)} check(s): {FAILS}")
        return 1
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
