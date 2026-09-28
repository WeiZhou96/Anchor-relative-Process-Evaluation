"""E0 / G1: the gauge blocks' analytic expectations must equal what the code computes.

These are the tests that give the rest of the study its licence. If a number
moves when a protocol knob is turned, it can only be attributed to the protocol
once the metric code is known to reproduce, exactly, the values that can be
derived on paper for answer paths whose rule is fully known.

The expectations live in :mod:`ape.blocks` (see its module docstring for the
derivation) and are computed here without going through :mod:`ape.metrics`.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from ape import blocks as B
from ape.metrics import AnswerTable, evaluate_system, flip_counts
from ape.protocol import ProtocolVector, n_grid_points

SPAN = 10.0
PAD = 2.0
FINE = 0.25


def _table(manifest, family, params, commit_after_s=None, delta_fine=FINE,
           span_s=SPAN, pad_s=PAD):
    df = B.generate_block_answers(
        manifest,
        family,
        params,
        delta_fine_s=delta_fine,
        span_s=span_s,
        pad_s=pad_s,
        commit_after_s=commit_after_s,
    )
    return AnswerTable.from_frame(df, B.block_id(family, params, commit_after_s),
                                 {"family": "block"})


BLOCK_CASES = [
    ("oracle", {}),
    ("lock", {"d0": 1.0}),
    ("lock", {"d0": 3.0}),
    ("lock", {"d0": 8.0}),
    ("lock", {"d0": 12.0}),  # never stabilises inside any horizon
    ("osc", {"p": 1.0, "d0": 4.0}),
    ("osc", {"p": 0.5, "d0": 6.0}),
    ("frac", {"c": 0.6}),
]
GRIDS = [(3.0, 0.5), (6.0, 0.5), (10.0, 1.0), (6.0, 0.25)]


@pytest.mark.parametrize("family,params", BLOCK_CASES)
@pytest.mark.parametrize("h_s,delta_s", GRIDS)
def test_block_curve_and_area_match_the_analytic_expectation(
    manifest, family, params, h_s, delta_s
):
    pv = ProtocolVector(delta_s=delta_s, h_s=h_s)
    at = _table(manifest, family, params)
    res = evaluate_system(manifest, at, pv)
    exp = B.block_expectations(family, params, manifest, pv)

    assert res.cohort_info["N_H"] == exp["N_H"]
    np.testing.assert_allclose(res.indicator.mean(axis=0), exp["S_H"], atol=1e-12)
    assert res.metrics_on(None)["RMSCD@H"] == pytest.approx(exp["RMSCD"], abs=1e-12)


@pytest.mark.parametrize("h_s,delta_s", GRIDS)
def test_oracle_is_the_trivial_ceiling(manifest, h_s, delta_s):
    pv = ProtocolVector(delta_s=delta_s, h_s=h_s)
    res = evaluate_system(manifest, _table(manifest, "oracle", {}), pv)
    assert res.indicator.all()
    m = res.metrics_on(None)
    assert m["RMSCD@H"] == pytest.approx(0.0)
    assert m["median_flips"] == 0.0
    assert m["end_window_macro_acc"] == pytest.approx(1.0)


@pytest.mark.parametrize("d0", [1.0, 2.5, 3.0, 5.0])
def test_lock_rmscd_equals_min_d0_H_minus_half_step(manifest, d0):
    """The documented step-integration offset, stated as an equality not a hope.

    ``S_H`` is a step function and the protocol integrates it with the trapezoid
    rule, so a grid-aligned stabilisation at ``d0`` is measured as
    ``min(d0, H) - Delta/2``. This is a property of the protocol (the delay
    quantisation floor of E4), and the test pins it so it cannot drift silently.
    """
    h_s, delta_s = 6.0, 0.5
    pv = ProtocolVector(delta_s=delta_s, h_s=h_s)
    res = evaluate_system(manifest, _table(manifest, "lock", {"d0": d0}), pv)
    ideal = min(d0, h_s)
    assert res.metrics_on(None)["RMSCD@H"] == pytest.approx(ideal - delta_s / 2.0)
    # and the continuous-limit value is what ideal_rmscd_per_video reports
    tau = B.stabilisation_offsets("lock", {"d0": d0}, manifest)
    assert float(np.mean(B.ideal_rmscd_per_video(tau, h_s))) == pytest.approx(ideal)


@pytest.mark.parametrize("d0", [1.0, 3.0, 8.0, 12.0])
@pytest.mark.parametrize("h_s,delta_s", GRIDS)
def test_lock_flip_count(manifest, d0, h_s, delta_s):
    pv = ProtocolVector(delta_s=delta_s, h_s=h_s)
    res = evaluate_system(manifest, _table(manifest, "lock", {"d0": d0}), pv)
    expected = B.analytic_flips_lock(d0, h_s, delta_s)
    assert set(np.unique(flip_counts(res.pred)).tolist()) == {expected}


OSC_CASES = list(
    itertools.product([0.5, 1.0, 1.5, 2.0], [1.0, 2.0, 3.0, 4.0, 6.0, 12.0])
)


@pytest.mark.parametrize("p,d0", OSC_CASES)
@pytest.mark.parametrize("h_s,delta_s", [(6.0, 0.5), (10.0, 0.5), (3.0, 0.25)])
def test_osc_flip_count_closed_form_reference_and_code_agree(
    manifest, p, d0, h_s, delta_s
):
    """Three-way agreement: paper formula, independent enumeration, metric code."""
    if p < 2 * delta_s:  # the closed form assumes the alternation is resolvable
        pytest.skip("period below twice the step is not resolvable on this grid")
    closed = B.analytic_flips_osc(p, d0, h_s, delta_s)
    enumerated = B.reference_flips_osc(p, d0, h_s, delta_s)
    assert closed == enumerated

    pv = ProtocolVector(delta_s=delta_s, h_s=h_s)
    res = evaluate_system(manifest, _table(manifest, "osc", {"p": p, "d0": d0}), pv)
    assert set(np.unique(flip_counts(res.pred)).tolist()) == {closed}

    # and the rounding convention: F is within one of floor(min(d0,H)/p)
    base = int(np.floor(min(d0, h_s) / p))
    assert abs(closed - base) <= 1, (closed, base)


@pytest.mark.parametrize("p,d0", [(1.0, 4.0), (0.5, 6.0), (2.0, 3.0)])
def test_osc_stabilises_at_its_last_error_not_at_its_decision(manifest, p, d0):
    """RMSCD measures the last error; the flip count measures the churn.

    ``osc(p, d0)`` only "decides" at ``d0``, but if the alternation phase just
    before ``d0`` is already the correct class then no error occurs after
    ``osc_tau(p, d0) <= d0``, and the stable-correctness area says so. This is
    why a stability metric may never be reported on its own: the pair
    (area, flips) is what separates "settled early" from "stopped erring early".
    """
    pv = ProtocolVector(delta_s=0.5, h_s=6.0)
    r_osc = evaluate_system(manifest, _table(manifest, "osc", {"p": p, "d0": d0}), pv)
    r_lock = evaluate_system(manifest, _table(manifest, "lock", {"d0": d0}), pv)
    tau = B.osc_tau(p, d0)
    assert tau <= d0
    np.testing.assert_allclose(
        r_osc.indicator.mean(axis=0),
        B.analytic_s_curve(np.full(r_osc.n, tau), 6.0, 0.5),
        atol=1e-12,
    )
    assert r_osc.metrics_on(None)["RMSCD@H"] <= r_lock.metrics_on(None)["RMSCD@H"] + 1e-12
    assert r_osc.metrics_on(None)["median_flips"] == B.analytic_flips_osc(p, d0, 6.0, 0.5)
    assert r_lock.metrics_on(None)["median_flips"] == B.analytic_flips_lock(d0, 6.0, 0.5)


def test_oscillation_costs_flips_that_a_lock_does_not(manifest):
    """With several periods inside the window the churn is visible as flips."""
    pv = ProtocolVector(delta_s=0.5, h_s=6.0)
    r_osc = evaluate_system(manifest, _table(manifest, "osc", {"p": 1.0, "d0": 4.0}), pv)
    r_lock = evaluate_system(manifest, _table(manifest, "lock", {"d0": 4.0}), pv)
    assert r_osc.metrics_on(None)["median_flips"] > r_lock.metrics_on(None)["median_flips"]


@pytest.mark.parametrize("p,d0,h_s,expected", [
    # the window closes inside a correct phase: stable earlier than the decision
    (2.0, 15.0, 10.0, 10.0),   # phase 5 = [10,12) answers correctly
    (2.0, 15.0, 4.0, np.inf),  # phase 2 = [4,6) answers wrongly: never stable
    (2.0, 15.0, 21.5, 14.0),   # d0 is inside the window: the global answer applies
    (1.0, 4.0, 10.0, 3.0),     # d0 inside the window: the global answer applies
    (0.5, 6.0, 10.0, 5.5),
])
def test_osc_tau_in_window(p, d0, h_s, expected):
    got = B.osc_tau_in_window(p, d0, h_s)
    if np.isinf(expected):
        assert np.isinf(got) and got > 0
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize("h_s", [4.0, 10.0, 21.5])
@pytest.mark.parametrize("family,params", [
    ("oracle", {}),
    ("lock", {"d0": 1.0}), ("lock", {"d0": 3.0}), ("lock", {"d0": 8.0}),
    ("lock", {"d0": 18.0}),
    ("osc", {"p": 1.0, "d0": 4.0}), ("osc", {"p": 0.5, "d0": 6.0}),
    ("osc", {"p": 2.0, "d0": 15.0}),
    ("frac", {"c": 0.6}),
])
def test_frozen_factor_table_matches_analytics_at_the_frozen_horizons(
    manifest, family, params, h_s
):
    """E0 on exactly the combinations S1 froze: the block table x the H list.

    ``osc(2, 15)`` at ``H = 10`` is the case that motivated
    :func:`ape.blocks.osc_tau_in_window`: the window shuts mid-correct-phase, so
    the measured area is 9.75 rather than the full 10.0 a naive global
    stabilisation time would predict.
    """
    pv = ProtocolVector(delta_s=0.5, h_s=h_s)
    # the cache must reach past the horizon, exactly as pi0's grid_max_s does
    at = _table(manifest, family, params, delta_fine=0.25, span_s=21.5, pad_s=2.75)
    res = evaluate_system(manifest, at, pv)
    exp = B.block_expectations(family, params, manifest, pv)
    np.testing.assert_allclose(res.indicator.mean(axis=0), exp["S_H"], atol=1e-12)
    assert res.metrics_on(None)["RMSCD@H"] == pytest.approx(exp["RMSCD"], abs=1e-12)


def test_osc_2_15_at_H10_is_the_documented_window_truncation(manifest):
    pv = ProtocolVector(delta_s=0.5, h_s=10.0)
    at = _table(manifest, "osc", {"p": 2.0, "d0": 15.0}, delta_fine=0.25,
                span_s=21.5, pad_s=2.75)
    res = evaluate_system(manifest, at, pv)
    assert res.metrics_on(None)["RMSCD@H"] == pytest.approx(9.75)
    assert B.osc_tau(2.0, 15.0) == pytest.approx(14.0)          # global
    assert B.osc_tau_in_window(2.0, 15.0, 10.0) == pytest.approx(10.0)  # in-window


@pytest.mark.parametrize("p,d0,expected", [
    (1.0, 4.0, 3.0),   # phase 3 covers [3,4) and is already correct
    (1.0, 3.0, 3.0),   # the last wrong phase [2,3) is cut off exactly at d0
    (0.5, 6.0, 5.5),
    (2.0, 3.0, 2.0),
    (2.0, 1.0, 1.0),   # the first wrong phase is truncated by d0
    (1.0, 0.0, -np.inf),
])
def test_osc_tau_closed_form(p, d0, expected):
    assert B.osc_tau(p, d0) == pytest.approx(expected)


@pytest.mark.parametrize("eta", [0.5, 0.9])
def test_rand_block_matches_its_expected_stable_correctness_curve(manifest, eta):
    """``E[S_H(delta_j)] = eta^(J_H - j + 1)`` for the i.i.d. noise floor."""
    h_s, delta_s = 3.0, 0.5
    pv = ProtocolVector(delta_s=delta_s, h_s=h_s)
    res = evaluate_system(manifest, _table(manifest, "rand", {"eta": eta}), pv)
    expected = B.analytic_s_curve_rand(eta, h_s, delta_s)
    observed = res.indicator.mean(axis=0)
    n = res.n
    sd = np.sqrt(np.clip(expected * (1.0 - expected), 0.0, None) / n)
    np.testing.assert_array_less(np.abs(observed - expected), 4.0 * sd + 1e-9)


def test_frac_block_exposes_the_seconds_versus_proportion_difference(manifest):
    """``frac`` stabilises at a fixed share of the clip, so its delay tracks length."""
    pv = ProtocolVector(delta_s=0.5, h_s=6.0)
    tau = B.stabilisation_offsets("frac", {"c": 0.6}, manifest)
    assert np.std(tau) > 1.0  # the whole point: it is not one number
    res = evaluate_system(manifest, _table(manifest, "frac", {"c": 0.6}), pv)
    exp = B.block_expectations("frac", {"c": 0.6}, manifest, pv)
    assert res.metrics_on(None)["RMSCD@H"] == pytest.approx(exp["RMSCD"], abs=1e-12)


def test_alt_cohorts_show_the_length_confound_the_fixed_cohort_removes(manifest):
    """E2 in one assertion: the naive rule reads differently on longer clips.

    ``per_clip_end`` asks each clip to stay correct to its own end, so a clip
    with little post-anchor video left has an easier task; the long/short gap is
    therefore large. The fixed cohort scores every clip over the same window and
    the gap collapses.
    """
    from ape.metrics import alt_cohort_report

    pv = ProtocolVector(delta_s=0.5, h_s=6.0)
    rep = alt_cohort_report(manifest, _table(manifest, "frac", {"c": 0.6}), pv, 22.0, 1.0)
    assert set(rep) >= {"per_clip_end", "dynamic_denominator", "fixed_H_cohort",
                        "length_stratified_change", "conclusion_stable"}
    assert rep["conclusion_stable"] is None  # decided across systems, not here
    gaps = rep["length_stratified_change"]
    assert abs(gaps["per_clip_end"]) > abs(gaps["fixed_H_cohort"])
    # the dynamic denominator really does shrink over the window
    assert rep["dynamic_denominator"]["n_at_H"] < rep["dynamic_denominator"]["n_at_start"]


def test_dynamic_denominator_row_is_the_same_quantity_as_the_protocol_row(manifest):
    """Regression: table 2's three rows must differ only in the cohort rule.

    ``osc(1,4)`` is correct *at* delta = 1.0 (phase 1) but errs again in [2,3),
    so its instantaneous accuracy there is 1.0 while its stable correctness is
    0.0. Putting instantaneous accuracy in the ``S_at_ref`` column made the
    dynamic-denominator row read 1.000 against the protocol's 0.000 -- an
    artefact of comparing two different quantities, not a survivorship effect.
    """
    from ape.metrics import alt_cohort_report, dynamic_denominator_curve

    pv = ProtocolVector(delta_s=0.5, h_s=10.03)
    at = _table(manifest, "osc", {"p": 1.0, "d0": 4.0})
    rep = alt_cohort_report(manifest, at, pv, 22.0, 1.0)
    dyn, fixed = rep["dynamic_denominator"], rep["fixed_H_cohort"]

    # the comparable column: stable correctness, so both are ~0 at the reference
    assert fixed["S_at_ref"] == pytest.approx(0.0)
    assert dyn["S_at_ref"] < 0.2
    # the plain accuracy curve is kept, under its own name, and is still 1.0
    assert dyn["plain_accuracy_at_ref"] == pytest.approx(1.0)
    # the risk set really does shrink
    assert dyn["n_at_H"] < dyn["n_at_start"]

    d3 = dynamic_denominator_curve(manifest, at, pv, 22.0, 10.03)
    assert np.nanmax(d3["s_dyn"]) <= 1.0 + 1e-12


def test_dynamic_denominator_area_is_a_delay_not_an_error_rate(manifest):
    """``rand(0.9)``: the two curves must not be confused in the RMSCD column.

    The plain accuracy area is about ``(1 - eta) * H``; the stable-correctness
    delay is many times larger because staying correct to the end needs every
    remaining draw to succeed. The old code reported the former in the RMSCD
    column, making a 10%-error system look near perfect.
    """
    from ape.metrics import alt_cohort_report

    pv = ProtocolVector(delta_s=0.5, h_s=10.03)
    at = _table(manifest, "rand", {"eta": 0.9})
    rep = alt_cohort_report(manifest, at, pv, 22.0, 1.0)
    dyn, fixed = rep["dynamic_denominator"], rep["fixed_H_cohort"]

    assert dyn["plain_accuracy_area"] == pytest.approx(0.1 * 10.0, abs=0.35)
    # the comparable row is a delay of the same order as the protocol's
    assert dyn["RMSCD"] > 2.0
    assert dyn["RMSCD"] < fixed["RMSCD"]  # survivorship makes it look better
    assert fixed["RMSCD"] == pytest.approx(6.24, abs=0.6)


def test_alt_cohorts_agree_with_the_main_metric_on_the_fixed_cohort(manifest):
    from ape.metrics import alt_cohort_report

    pv = ProtocolVector(delta_s=0.5, h_s=6.0)
    at = _table(manifest, "lock", {"d0": 3.0})
    rep = alt_cohort_report(manifest, at, pv, 22.0, 1.0)
    res = evaluate_system(manifest, at, pv)
    assert rep["fixed_H_cohort"]["RMSCD"] == pytest.approx(res.metrics_on(None)["RMSCD@H"])
    assert rep["fixed_H_cohort"]["n"] == res.n


def test_block_answers_cover_the_pre_anchor_range(manifest):
    """Cached columns must start before the anchor or a negative shift falls off."""
    at = _table(manifest, "lock", {"d0": 3.0})
    assert at.j_min <= -int(PAD / FINE)
    assert at.j_max >= int(SPAN / FINE)
