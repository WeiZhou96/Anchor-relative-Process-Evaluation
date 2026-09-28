"""Per-metric verdict rulers, and rule-level system identity for reporting."""

from __future__ import annotations

import numpy as np
import pytest

from ape import calib as C
from ape.cli import RULER_DEFINITION, _rulers_by_metric
from ape.systems import arm_rule, arm_value, describe, family_of, parent_of, seed_of


# --------------------------------------------------------------------------
# rulers
# --------------------------------------------------------------------------
def test_ruler_is_computed_per_metric_in_that_metrics_units():
    """One ruler per metric, from the same tied pairs, never shared.

    RMSCD is seconds, the window-end accuracy is a fraction and the flip count is
    an integer. A single ruler in seconds judged all three, which is how
    median_flips got a "normalize" verdict out of MRD = 2 flips vs a 0.37 s
    ruler -- a comparison between unlike quantities.
    """
    fam_point = {
        "a": {"RMSCD@H": 3.0, "end_window_macro_acc": 0.50, "median_flips": 1.0},
        "b": {"RMSCD@H": 3.4, "end_window_macro_acc": 0.52, "median_flips": 3.0},
        "c": {"RMSCD@H": 5.0, "end_window_macro_acc": 0.90, "median_flips": 9.0},
    }
    tied = [("a", "b")]
    r = _rulers_by_metric(fam_point, tied)
    assert r["RMSCD@H"] == pytest.approx(0.4)
    assert r["end_window_macro_acc"] == pytest.approx(0.02)
    assert r["median_flips"] == pytest.approx(2.0)
    # the untied pair does not contribute
    assert r["RMSCD@H"] != pytest.approx(2.0)


def test_ruler_uses_the_median_over_all_tied_pairs():
    fam_point = {k: {"m": v} for k, v in
                 {"a": 0.0, "b": 1.0, "c": 4.0, "d": 10.0}.items()}
    tied = [("a", "b"), ("a", "c"), ("a", "d")]  # gaps 1, 4, 10
    assert _rulers_by_metric(fam_point, tied)["m"] == pytest.approx(4.0)


def test_ruler_is_none_without_tied_pairs_so_the_verdict_stays_undetermined():
    fam_point = {"a": {"m": 1.0}, "b": {"m": 2.0}}
    assert _rulers_by_metric(fam_point, [])["m"] is None
    assert C.verdict(0.95, 0.01, 0.9, None).startswith("undetermined")


def test_ruler_ignores_non_finite_values():
    fam_point = {"a": {"m": float("nan")}, "b": {"m": 2.0}, "c": {"m": 5.0}}
    r = _rulers_by_metric(fam_point, [("a", "b"), ("b", "c")])
    assert r["m"] == pytest.approx(3.0)  # only the b/c pair is usable


def test_verdict_flips_when_the_ruler_is_given_in_the_right_units():
    """The concrete regression: median_flips judged against its own ruler."""
    mrd_flips, ruler_flips, ruler_seconds = 2.0, 3.0, 0.3726
    # against the RMSCD ruler in seconds, 2 flips looked like a large shift
    assert C.verdict(1.0, mrd_flips, 0.9, ruler_seconds) == "normalize"
    # against the flip-count ruler, the same shift is smaller than what
    # separates window-end-tied systems, so the metric pools
    assert C.verdict(1.0, mrd_flips, 0.9, ruler_flips) == "poolable"


def test_rank_floor_still_dominates_the_verdict():
    """A metric that loses orderings is not comparable whatever the ruler says."""
    assert C.verdict(0.85, 0.001, 0.9, 10.0) == "not comparable"


def test_calibrate_metric_reports_the_ruler_it_used():
    rows = []
    for pi, (eps, plaus, vals) in {
        "pi0": (0.0, True, {"s1": 1.0, "s2": 2.0}),
        "e1": (0.25, True, {"s1": 1.1, "s2": 2.1}),
    }.items():
        for sid, v in vals.items():
            rows.append({"system_id": sid, "is_block": False, "pi_hash": pi,
                         "eps_sys_s": eps, "eps_jit_sd_s": 0.0, "delta_s": 0.5,
                         "h_s": 10.0, "in_plaus": plaus, "metric": "median_flips",
                         "value": v})
    import pandas as pd
    scan = pd.DataFrame(rows, columns=C.SCAN_COLUMNS)
    out = C.calibrate_metric(scan, "pi0", "median_flips", [("s1", "s2")], 0.9, ruler=3.0)
    assert out["ruler"] == pytest.approx(3.0)
    assert "own units" in out["ruler_definition"]
    assert out["verdict"] == "poolable"
    assert "own units" in RULER_DEFINITION or "units" in RULER_DEFINITION


# --------------------------------------------------------------------------
# arm_rule
# --------------------------------------------------------------------------
@pytest.mark.parametrize("system_id,expected", [
    ("clip__r18mean__seed20260905", "clip"),
    ("prefix__gru128__seed20260903", "prefix_gru128"),
    ("prefix__gru512__seed20260904", "prefix_gru512"),
    ("postproc__ema0p2__prefix__gru512__seed20260904", "ema"),
    ("postproc__ema0p7__prefix__gru512__seed20260903", "ema"),
    ("postproc__majority9__prefix__gru512__seed20260905", "majority"),
    ("postproc__hysteresis0p05__prefix__gru512__seed20260904", "hysteresis"),
    ("postproc__patience6__prefix__gru512__seed20260905", "patience"),
    ("commit__msp0p9__prefix__gru512__seed20260903", "commit_msp"),
    ("commit__margin0p4__prefix__gru512__seed20260904", "commit_margin"),
    ("trivial__majority", "trivial_majority"),
    ("trivial__constbot", "trivial_constbot"),
    ("trivial__random", "trivial_random"),
    ("trivial__oracle", "trivial_oracle"),
    ("block__osc__d0-4_p-1", "block_osc"),
    ("block__oracle__default", "block_oracle"),
])
def test_arm_rule_strips_the_dev_tuned_value(system_id, expected):
    assert arm_rule(system_id) == expected


def test_arm_rule_groups_the_seed_split_postproc_arms():
    """The point of the field: recover 3 seeds per rule from 1-2 seeds per arm.

    B's post-processing arms carry a per-seed tuned value, so grouping by arm
    string yields one or two seeds per group and no seed spread. Grouping by
    rule restores the three seeds.
    """
    realised = [
        "postproc__ema0p2__prefix__gru512__seed20260904",
        "postproc__ema0p2__prefix__gru512__seed20260905",
        "postproc__ema0p7__prefix__gru512__seed20260903",
    ]
    assert {arm_rule(s) for s in realised} == {"ema"}
    assert {seed_of(s) for s in realised} == {"20260903", "20260904", "20260905"}


def test_arm_value_and_parent_are_recovered():
    s = "commit__margin0p4__prefix__gru512__seed20260904"
    assert arm_value(s) == "0p4"
    assert parent_of(s) == "prefix__gru512__seed20260904"
    assert family_of(s) == "commit"
    # base systems have no arm value or parent
    assert arm_value("prefix__gru512__seed20260903") is None
    assert parent_of("prefix__gru512__seed20260903") is None


def test_describe_returns_every_identity_field():
    d = describe("postproc__patience6__prefix__gru512__seed20260905")
    assert d == {
        "system_id": "postproc__patience6__prefix__gru512__seed20260905",
        "family": "postproc",
        "arm_rule": "patience",
        "arm_value": "6",
        "seed": "20260905",
        "parent": "prefix__gru512__seed20260905",
    }


def test_unknown_shapes_degrade_quietly():
    assert arm_rule("") == "unknown"
    assert seed_of("no_seed_here") is None
    assert arm_rule("weird__thing") == "weird"


def test_a_zero_width_ruler_is_labelled_not_silently_treated_as_a_threshold():
    """median_flips ties at 0 among window-end-tied pairs on the real data.

    An integer metric that takes few values gives ruler_M = 0: systems that tie
    at the window end have the *same* median flip count. Every non-zero MRD then
    exceeds it, so the verdict is always "normalize" -- which is defensible, but
    it rests on the metric being unable to resolve anything rather than on a
    magnitude comparison. The label has to carry that distinction.
    """
    v = C.verdict(1.0, 2.0, 0.9, 0.0)
    assert v.startswith("normalize")
    assert "ruler is zero" in v
    # a real, non-zero ruler still produces the plain verdicts
    assert C.verdict(1.0, 2.0, 0.9, 3.0) == "poolable"
    assert C.verdict(1.0, 4.0, 0.9, 3.0) == "normalize"


def test_calibrate_metric_flags_the_degenerate_ruler():
    import pandas as pd
    rows = []
    for pi, (eps, vals) in {"pi0": (0.0, {"s1": 1.0, "s2": 2.0}),
                            "e1": (0.25, {"s1": 1.5, "s2": 2.5})}.items():
        for sid, v in vals.items():
            rows.append({"system_id": sid, "is_block": False, "pi_hash": pi,
                         "eps_sys_s": eps, "eps_jit_sd_s": 0.0, "delta_s": 0.5,
                         "h_s": 10.0, "in_plaus": True, "metric": "median_flips",
                         "value": v})
    scan = pd.DataFrame(rows, columns=C.SCAN_COLUMNS)
    zero = C.calibrate_metric(scan, "pi0", "median_flips", [("s1", "s2")], 0.9, ruler=0.0)
    assert zero["ruler_degenerate"] is True
    assert "ruler is zero" in zero["verdict"]
    ok = C.calibrate_metric(scan, "pi0", "median_flips", [("s1", "s2")], 0.9, ruler=3.0)
    assert ok["ruler_degenerate"] is False
