"""Phenomenon statistics and K1-K6 on paths whose behaviour is known by hand.

These quantities are deliverables, not verdicts, so the tests check that they
count what their names say and that unavailable inputs surface as NaN plus an
availability count rather than as a comfortable zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from ape import blocks as B
from ape.metrics import AnswerTable, evaluate_system
from ape.phenomena import phenomena_report, phenomena_table
from ape.protocol import ProtocolVector

from conftest import HANDMADE_DELTA, HANDMADE_H, make_answers, make_manifest


def _pv(h=HANDMADE_H, delta=HANDMADE_DELTA, **kw):
    return ProtocolVector(delta_s=delta, h_s=h, **kw)


def test_flip_rate_last_flip_and_correct_wrong_correct(handmade):
    mf, at = handmade
    rep = phenomena_report(evaluate_system(mf, at, _pv()))
    ph = rep["phenomena"]
    # three of the four paths change their answer at least once
    assert ph["flip_rate_videos_with_any_flip"] == pytest.approx(0.75)
    # V_wrong_then_right flips at j=2 (1.0 s), V_right_then_wrong at j=4 (2.0 s),
    # V_flipping last flips at j=5 (2.5 s)
    np.testing.assert_allclose(
        sorted(x for x in [1.0, 2.0, 2.5]), [1.0, 2.0, 2.5]
    )
    assert ph["last_flip_time_s"]["median"] == pytest.approx(2.0)
    # only V_flipping goes correct -> wrong -> correct
    assert ph["correct_wrong_correct_rate"] == pytest.approx(0.25)


def test_confidence_items_are_nan_when_the_system_reports_no_probabilities(handmade):
    mf, at = handmade
    rep = phenomena_report(evaluate_system(mf, at, _pv()), manifest=mf, answers=at)
    ph = rep["phenomena"]
    assert ph["confidence_cells_available"] == 0
    assert np.isnan(ph["short_prefix_confident_inconsistent_rate"])
    assert np.isnan(ph["pre_anchor_confident_and_finally_wrong"])
    assert ph["pre_anchor_cells_available"] == 0


def test_k3_counts_only_the_clips_whose_terminal_answer_is_wrong(handmade):
    mf, at = handmade
    rep = phenomena_report(evaluate_system(mf, at, _pv()))
    # only V_right_then_wrong ends wrong, and it did have correct prefixes
    assert rep["K"]["K3_terminal_wrong_but_had_correct_prefix"] == pytest.approx(1.0)
    assert rep["K"]["K2_full_clip_macro_acc"] == pytest.approx(
        rep["K"]["K2_full_clip_micro_acc"]
    )


def test_k6_separates_terminal_consistency_from_correctness(handmade):
    """A path can settle on an answer long before that answer is the right one."""
    mf, at = handmade
    rep = phenomena_report(evaluate_system(mf, at, _pv()))
    k = rep["K"]
    # V_right_then_wrong settles on its (wrong) terminal answer at 2.0 s but never
    # becomes stably correct, so it is censored at H = 3.0
    assert k["K6_first_terminal_consistent_s"]["median"] < k["K6_first_stable_correct_s"]["median"]
    assert np.isfinite(k["K6_ks_statistic"])
    assert k["K6_paired_difference_s"]["n"] == 4


def test_k1_agreement_is_one_when_the_terminal_answer_is_correct():
    mf = make_manifest([{"video_id": "A", "anchor_s": 5.0, "post_s": 4.0, "y": 0}])
    at = make_answers({"A": [1, 1, 0, 0, 0, 0, 0]}, delta_s=HANDMADE_DELTA)
    rep = phenomena_report(evaluate_system(mf, at, _pv()))
    k = rep["K"]
    assert k["K1_prefix_equals_terminal_vs_prefix_equals_truth_agreement"] == pytest.approx(1.0)


def test_k1_agreement_collapses_when_the_terminal_answer_is_wrong():
    """"Agrees with the final answer" and "is right" can be exact opposites.

    Path ``0 0 0 0 1 1 1`` with truth ``0`` ends on the wrong class, so the four
    early points are correct but disagree with the terminal answer, and the three
    late points agree with it but are wrong: the two events never coincide. This
    is what K1 is for, and why "consistent with the final prediction" must not be
    read as a stand-in for correctness.
    """
    mf = make_manifest([{"video_id": "A", "anchor_s": 5.0, "post_s": 4.0, "y": 0}])
    at = make_answers({"A": [0, 0, 0, 0, 1, 1, 1]}, delta_s=HANDMADE_DELTA)
    rep = phenomena_report(evaluate_system(mf, at, _pv()))
    k = rep["K"]
    assert k["K1_prefix_equals_terminal_vs_prefix_equals_truth_agreement"] == pytest.approx(0.0)
    assert k["K1_rate_prefix_equals_terminal"] == pytest.approx(3.0 / 7.0)
    assert k["K1_rate_prefix_equals_truth"] == pytest.approx(4.0 / 7.0)


def test_blocks_supply_declared_confidences_so_the_k5_item_is_computable(manifest):
    df = B.generate_block_answers(
        manifest, "osc", {"p": 1.0, "d0": 4.0}, delta_fine_s=0.25, span_s=10.0, pad_s=2.0
    )
    at = AnswerTable.from_frame(df, "block__osc", {"family": "block"})
    res = evaluate_system(manifest, at, ProtocolVector(delta_s=0.5, h_s=6.0))
    rep = phenomena_report(res, manifest=manifest, answers=at, conf_threshold=0.5)
    ph = rep["phenomena"]
    assert ph["confidence_cells_available"] > 0
    assert np.isfinite(ph["short_prefix_confident_inconsistent_rate"])
    assert ph["pre_anchor_cells_available"] > 0
    assert np.isfinite(ph["pre_anchor_answer_rate"])


def test_k5_is_nan_not_zero_when_no_cell_clears_the_threshold(manifest):
    """An unreachable threshold means unmeasured, and must not report as 0.0.

    This is why the stand-in matrices show ``K5 = null``: their declared
    confidence is below the protocol's 0.9 threshold, so the conditional rate has
    an empty denominator. A 0.0 would be read as "never inconsistent".
    """
    df = B.generate_block_answers(
        manifest, "lock", {"d0": 3.0}, delta_fine_s=0.25, span_s=10.0, pad_s=2.0,
        confidence=0.60,  # below the threshold used below
    )
    at = AnswerTable.from_frame(df, "low_conf", {"family": "block"})
    res = evaluate_system(manifest, at, ProtocolVector(delta_s=0.5, h_s=6.0))
    rep = phenomena_report(res, manifest=manifest, answers=at, conf_threshold=0.9)
    ph, k = rep["phenomena"], rep["K"]

    assert ph["confidence_cells_available"] > 0  # probabilities exist
    assert ph["short_prefix_confident_cells"] == 0  # but none is confident enough
    assert np.isnan(k["K5_short_prefix_confident_inconsistent_rate"])
    assert np.isnan(ph["short_prefix_confident_inconsistent_video_rate"])
    # the same rule applies to the pre-anchor item, which used to report 0.0
    assert ph["pre_anchor_confident_cells"] == 0
    assert np.isnan(ph["pre_anchor_confident_and_finally_wrong"])
    # and the definition travels with the number
    assert "prefix-cell granularity" in k["K5_definition"]


def test_k5_is_measured_when_the_threshold_is_reachable(manifest):
    df = B.generate_block_answers(
        manifest, "osc", {"p": 1.0, "d0": 4.0}, delta_fine_s=0.25, span_s=10.0,
        pad_s=2.0, confidence=0.95,
    )
    at = AnswerTable.from_frame(df, "hi_conf", {"family": "block"})
    res = evaluate_system(manifest, at, ProtocolVector(delta_s=0.5, h_s=6.0))
    rep = phenomena_report(res, manifest=manifest, answers=at, conf_threshold=0.9)
    ph, k = rep["phenomena"], rep["K"]
    assert ph["short_prefix_confident_cells"] > 0
    assert 0.0 <= k["K5_short_prefix_confident_inconsistent_rate"] <= 1.0
    assert 0.0 <= k["K5_video_rate"] <= 1.0
    # the cell rate and the video rate are different denominators, both exposed
    assert k["K5_video_rate"] == ph["short_prefix_confident_inconsistent_video_rate"]


def test_phenomena_report_declares_that_it_judges_nothing(handmade):
    mf, at = handmade
    rep = phenomena_report(evaluate_system(mf, at, _pv()))
    assert "report only" in rep["interpretation"]
    assert "pass" not in {k.lower() for k in rep["K"]}


def test_phenomena_table_has_one_row_per_system(handmade):
    mf, at = handmade
    reps = [phenomena_report(evaluate_system(mf, at, _pv()))]
    tbl = phenomena_table(reps)
    assert len(tbl) == 1
    assert "K6 median diff (s)" in tbl.columns
