import numpy as np
import pytest

from ape.m3_analysis import (
    class_support,
    identity_check,
    kendall,
    own_end_delay,
    pair_disagreement,
    ratio,
    readout,
    selection_regret,
    stable_onset_index,
    stratified_halves,
    top_k_overlap,
    weighted,
)
from ape.method_analysis import point_weights, trajectory_components


def test_onset_and_trapezoid_relation():
    c = np.array([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1]], dtype=bool)
    step = 0.5
    times = np.arange(5) * step
    assert stable_onset_index(c).tolist() == [2, 0, -1, 3]
    delay = trajectory_components(c, times)["delay"]
    assert np.allclose(delay, [2 * step - step / 2, 0.0, times[-1], 3 * step - step / 2])


def test_conditional_delay_identity_and_weights():
    rng = np.random.default_rng(0)
    c = rng.random((40, 9)) < 0.6
    times = np.arange(9) * 0.5
    labels = np.repeat(np.arange(4), 10)
    comp = trajectory_components(c, times)
    end = comp["endpoint"].astype(float)
    for mode in ["micro", "macro"]:
        raw = np.vstack([np.ones(40), 1 + rng.integers(0, 3, 40)]).astype(float)  # every class supported
        rmscd = weighted(comp["delay"], raw, labels, mode)
        acc = weighted(end, raw, labels, mode)
        ct = ratio(end * comp["delay"], end, raw, labels, mode)
        assert np.allclose(rmscd, times[-1] * (1 - acc) + acc * ct)
        assert np.isclose(weighted(comp["delay"], np.ones((1, 40)), labels, mode)[0], point_weights(labels, mode) @ comp["delay"])


def test_macro_undefined_without_class_support():
    labels = np.array([0, 0, 1, 1])
    raw = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=float)
    assert class_support(raw, labels).tolist() == [False, True]
    out = weighted(np.array([1.0, 2.0, 3.0, 4.0]), raw, labels, "macro")
    assert np.isnan(out[0]) and np.isclose(out[1], (1 + 3) / 2)
    assert np.isnan(ratio(np.zeros(4), np.zeros(4), np.ones((1, 4)), labels, "micro")[0])


def test_own_end_delay_matches_fixed_window_when_ends_agree():
    c = np.array([[0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=bool)
    last = np.array([2, 4, 5])
    delay, end, onset = own_end_delay(c, last, 0.25)
    assert end.tolist() == [True, True, True]
    for r in range(3):
        ref = trajectory_components(c[r:r + 1, : last[r] + 1], np.arange(last[r] + 1) * 0.25)["delay"][0]
        assert np.isclose(delay[r], ref)
    assert np.allclose(onset, [0.25, 0.5, 0.0])
    d0, e0, _ = own_end_delay(np.array([[0, 1]], dtype=bool), np.array([0]), 0.25)
    assert d0[0] == 0.0 and not e0[0]


def test_readout_last_point_before_time():
    full = np.zeros((2, 12), dtype=bool)
    full[0, 2 + 3] = True
    full[1, 2 + 8] = True
    t = np.array([[0.5, 0.8, 1.0], [1.0, 1.6, 2.0]])
    assert readout(full, t, 0.25, 2).tolist() == [[False, True, False], [False, False, True]]
    with pytest.raises(ValueError):
        readout(full, np.array([[5.0], [0.0]]), 0.25, 2)


def test_selection_ties_and_overlap():
    ids = ["b", "a", "c"]
    out = selection_regret(ids, np.array([0.5, 0.5, 0.4]), np.array([6.0, 7.0, 5.0]))
    assert (out["selected_by_accuracy"], out["selected_by_rmscd"]) == ("a", "c")
    assert np.isclose(out["rmscd_regret"], 2.0) and np.isclose(out["accuracy_change"], -0.1)
    assert top_k_overlap(ids, np.array([0.5, 0.5, 0.4]), np.array([6.0, 7.0, 5.0]), k=2) == 0.5


def test_disagreement_identity_and_kendall():
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    rmscd, ct, acc = np.array([6.0, 7.0, 5.0]), np.array([3.0, 2.0, 2.0]), np.array([0.4, 0.3, 0.5])
    out = pair_disagreement(rmscd, ct, pairs)
    assert (out["compared"], out["opposite"]) == (2, 1)
    assert identity_check(rmscd, ct, acc, pairs) == 0
    assert identity_check(rmscd, ct, np.array([0.3, 0.4, 0.5]), pairs) == 1
    assert np.isnan(kendall(np.ones(4), np.arange(4.0)))
    assert np.isclose(kendall(np.arange(5.0), np.arange(5.0) * 2), 1.0)


def test_stratified_halves_balance():
    first = np.repeat(np.arange(3), [4, 5, 6])
    half = stratified_halves(first, np.random.default_rng(1))
    assert [int(half[first == c].sum()) for c in range(3)] == [2, 2, 3]
