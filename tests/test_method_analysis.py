"""Scientific counterexamples and invariants for the new analysis."""

from itertools import product

import numpy as np

from ape.method_analysis import (
    cluster_weights,
    delay_bounds,
    integration_weights,
    paired_components,
    point_weights,
    simultaneous_band,
    trajectory_components,
)


def test_equal_accuracy_and_flips_do_not_identify_stability():
    a = np.array([[0, 0, 0, 1, 1], [0, 1, 1, 0, 1]], dtype=bool)
    b = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 0, 1]], dtype=bool)
    np.testing.assert_equal(a.mean(0), b.mean(0))
    np.testing.assert_equal(np.diff(a).sum(1), np.diff(b).sum(1))
    assert trajectory_components(a, np.arange(5))["delay"].mean() == 3
    assert trajectory_components(b, np.arange(5))["delay"].mean() == 2.5


def test_every_partial_pattern_has_sharp_bounds():
    times = np.array([0, 0.2, 1, 3])
    for pattern in product([-1, 0, 1], repeat=4):
        pattern = np.array(pattern)
        observed = pattern >= 0
        lo, hi = delay_bounds((pattern == 1)[None], observed, times)
        candidates = []
        for fill in product([False, True], repeat=int((~observed).sum())):
            full = pattern == 1
            full[~observed] = fill
            candidates.append(trajectory_components(full[None], times)["delay"][0])
        assert lo[0] == min(candidates)
        assert hi[0] == max(candidates)


def test_missed_late_error_and_irregular_coarse_integral():
    correct = np.ones((1, 101), dtype=bool)
    correct[0, 99] = False
    full = trajectory_components(correct, np.arange(101))["delay"][0]
    assert full == 99.5
    assert trajectory_components(correct[:, ::2], np.arange(101)[::2])["delay"][0] == 0
    assert integration_weights(np.array([0, 0.25, 2])).sum() == 2


def test_success_set_term_is_not_accuracy_only():
    a = {
        "delay": np.array([2.0, 10]),
        "endpoint": np.array([1, 0]),
        "error": np.array([2.0, 10]),
        "retracted": np.zeros(2),
    }
    b = {
        "delay": np.array([10.0, 8]),
        "endpoint": np.array([0, 1]),
        "error": np.array([10.0, 8]),
        "retracted": np.zeros(2),
    }
    x = paired_components(a, b, 10).mean(0)
    assert x[0] == x[2] == 3
    assert x[1] == x[5] == 0
    np.testing.assert_allclose(paired_components(a, b, 10), -paired_components(b, a, 10))


def test_cluster_draws_and_class_weights():
    labels = np.array([0, 0, 0, 1, 1, 1])
    clusters = np.array([0, 0, 1, 2, 3, 3])
    weights, _ = cluster_weights(clusters, labels, 100, 1)
    np.testing.assert_allclose(weights["micro"][:, 0], weights["micro"][:, 1])
    np.testing.assert_allclose(weights["macro"][:, labels == 0].sum(1), 0.5)
    np.testing.assert_allclose(weights["macro"].sum(1), 1)
    np.testing.assert_allclose(point_weights(labels, "macro").sum(), 1)


def test_simultaneous_band_translation_permutation_and_pairing():
    rng = np.random.default_rng(3)
    points = np.array([1.0, 2, 0])
    samples = points + rng.normal(size=(500, 3)) * [1, 2, 0]
    a = simultaneous_band(points, samples)
    b = simultaneous_band(points[::-1] + 5, samples[:, ::-1] + 5)
    np.testing.assert_allclose(a["lower"][::-1] + 5, b["lower"])
    assert a["zero_variance"][2]
    assert a["critical"] > 1.9


def test_bonferroni_guard_never_narrows_the_empirical_band():
    rng = np.random.default_rng(8)
    points = np.zeros(9)
    samples = rng.normal(size=(500, 9))
    old = simultaneous_band(points, samples)
    guarded = simultaneous_band(points, samples, degrees_freedom=79)
    assert guarded["bonferroni_t_guard"] > 2.8
    assert np.all(guarded["lower"] <= old["lower"])
    assert np.all(guarded["upper"] >= old["upper"])
