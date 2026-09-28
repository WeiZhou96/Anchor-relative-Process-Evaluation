"""Ground-truth checks on four hand-written answer paths (idea S2, contract 6).

Every expected number in this file was worked out on paper before the code ran.
With ``H = 3.0``, ``Delta = 0.5`` the window has seven grid points at offsets
``0.0 .. 3.0``:

``V_stable``            pred 0 0 0 0 0 0 0   correct everywhere
                        I = 1 everywhere, RMSCD = 0, F = 0
``V_wrong_then_right``  pred 1 1 0 0 0 0 0   stable from j = 2 (delta = 1.0 s)
                        1-I = 1 1 0 0 0 0 0, trapezoid = 0.5*(2 - 0.5) = 0.75
``V_right_then_wrong``  pred 0 0 0 0 1 1 1   never stable: the window ends wrong
                        1-I = 1 everywhere, trapezoid = 0.5*(7 - 1) = 3.0 = H
``V_flipping``          pred 1 0 1 0 1 0 0   stable only from j = 5
                        1-I = 1 1 1 1 1 0 0, trapezoid = 0.5*(5 - 0.5) = 2.25
                        F = 5 (six adjacent pairs, the last one repeats)

Cohort curve: S_H = [.25, .25, .5, .5, .5, .75, .75] and
RMSCD@H = mean(0, 0.75, 3.0, 2.25) = 1.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from ape.classes import BOT
from ape.metrics import (
    AnswerTable,
    correctness,
    evaluate_system,
    first_stable_index,
    flip_counts,
    rmscd_per_video,
    stable_correct,
)
from ape.protocol import ProtocolVector

from conftest import (
    HANDMADE_DELTA,
    HANDMADE_H,
    HANDMADE_PATHS,
    HANDMADE_RMSCD,
    HANDMADE_S_CURVE,
    HANDMADE_TRUTH,
    make_answers,
    make_manifest,
)


def _pv(h=HANDMADE_H, delta=HANDMADE_DELTA, **kw):
    return ProtocolVector(delta_s=delta, h_s=h, **kw)


def test_four_trajectories_per_video_truth(handmade):
    mf, at = handmade
    res = evaluate_system(mf, at, _pv())
    assert res.pred.shape == (4, 7)
    order = list(res.video_ids)
    ind = res.indicator
    rmscd = rmscd_per_video(ind, res.delta_s)
    flips = flip_counts(res.pred)
    first = first_stable_index(ind)
    for k, vid in enumerate(order):
        truth = HANDMADE_TRUTH[vid]
        assert rmscd[k] == pytest.approx(truth["rmscd"]), vid
        assert int(flips[k]) == truth["flips"], vid
        assert int(first[k]) == truth["first_stable_j"], vid


def test_four_trajectories_cohort_curve(handmade):
    mf, at = handmade
    res = evaluate_system(mf, at, _pv())
    curve = res.indicator.mean(axis=0)
    np.testing.assert_allclose(curve, HANDMADE_S_CURVE)
    assert res.metrics_on(None)["RMSCD@H"] == pytest.approx(HANDMADE_RMSCD)
    # the aggregate identity: mean of per-video areas equals the area of the mean
    assert float(np.mean(rmscd_per_video(res.indicator, res.delta_s))) == pytest.approx(
        HANDMADE_RMSCD
    )


def test_stable_correctness_requires_persistence_to_window_end(handmade):
    """A single late error erases every earlier stable-correct indicator."""
    mf, at = handmade
    res = evaluate_system(mf, at, _pv())
    k = list(res.video_ids).index("V_right_then_wrong")
    assert res.correct[k, :4].all()  # it was correct for four grid points
    assert not res.indicator[k].any()  # yet it is never "stably correct"


def test_bot_and_missing_answers_are_scored_wrong():
    """``pred = -1`` is an error, not an abstention, and stays in the denominator."""
    mf = make_manifest(
        [
            {"video_id": "A", "anchor_s": 5.0, "post_s": 4.0, "y": 0},
            {"video_id": "B", "anchor_s": 5.0, "post_s": 4.0, "y": 0},
        ]
    )
    at = make_answers({"A": [0] * 7, "B": [BOT] * 7}, delta_s=HANDMADE_DELTA)
    res = evaluate_system(mf, at, _pv())
    assert res.cohort_info["N_H"] == 2  # B is not dropped
    k = list(res.video_ids).index("B")
    assert not res.correct[k].any()
    assert rmscd_per_video(res.indicator, res.delta_s)[k] == pytest.approx(HANDMADE_H)
    assert res.metrics_on(None)["RMSCD@H"] == pytest.approx(HANDMADE_H / 2.0)


def test_missing_cached_cells_are_wrong_and_counted():
    """A coverage hole must show up as ``missing_lookup_rate``, not as a free pass."""
    mf = make_manifest([{"video_id": "A", "anchor_s": 5.0, "post_s": 4.0, "y": 0}])
    at = make_answers({"A": [0, 0, 0]}, delta_s=HANDMADE_DELTA)  # only j = 0..2
    res = evaluate_system(mf, at, _pv())
    assert res.pred.shape == (1, 7)
    assert (res.pred[0, 3:] == BOT).all()
    assert res.n_missing_lookup == 4
    assert res.missing_lookup_rate == pytest.approx(4.0 / 7.0)


def test_flip_count_ignores_repeats_and_counts_every_change():
    mf = make_manifest([{"video_id": "A", "anchor_s": 5.0, "post_s": 4.0, "y": 0}])
    at = make_answers({"A": [0, 0, 1, 1, 2, 2, 0]}, delta_s=HANDMADE_DELTA)
    res = evaluate_system(mf, at, _pv())
    assert int(flip_counts(res.pred)[0]) == 3


def test_constant_wrong_system_has_zero_flips_but_full_delay():
    """Why a stability metric may never be reported without correctness."""
    mf = make_manifest([{"video_id": "A", "anchor_s": 5.0, "post_s": 4.0, "y": 0}])
    at = make_answers({"A": [3] * 7}, delta_s=HANDMADE_DELTA)
    res = evaluate_system(mf, at, _pv())
    m = res.metrics_on(None)
    assert m["median_flips"] == 0.0
    assert m["RMSCD@H"] == pytest.approx(HANDMADE_H)


def test_macro_and_micro_window_end_accuracy_differ_under_imbalance():
    mf = make_manifest(
        [{"video_id": f"A{i}", "anchor_s": 5.0, "post_s": 4.0, "y": 0} for i in range(3)]
        + [{"video_id": "B0", "anchor_s": 5.0, "post_s": 4.0, "y": 1}]
    )
    preds = {f"A{i}": [0] * 7 for i in range(3)}
    preds["B0"] = [0] * 7  # the minority class is always missed
    res = evaluate_system(mf, make_answers(preds, delta_s=HANDMADE_DELTA), _pv())
    m = res.metrics_on(None)
    assert m["end_window_micro_acc"] == pytest.approx(0.75)
    assert m["end_window_macro_acc"] == pytest.approx(0.5)
