"""Tests for G0's recomputable sample floor (``ape.g0_floor``)."""

from __future__ import annotations

import math
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ape.g0_floor import (  # noqa: E402
    G0FloorNotRegisteredError,
    g0_sample_floor_report,
    macro_metric_support,
    n_clusters_needed,
    s_curve_resolution,
)


# --------------------------------------------------------------------------
# normal quantile and resolution arithmetic
# --------------------------------------------------------------------------
def test_resolution_matches_the_textbook_proportion_interval():
    # At p = 0.5, alpha = 0.05 and n = 100 the half-width is 1.96 * 0.05.
    res = s_curve_resolution(100, p=0.5, alpha=0.05)
    assert res["half_width_single"] == pytest.approx(1.959964 * 0.05, rel=1e-4)
    # A difference of two independent proportions widens by sqrt(2).
    assert res["half_width_unpaired_difference"] == pytest.approx(res["half_width_single"] * math.sqrt(2.0), rel=1e-9)


def test_resolution_improves_as_the_root_of_the_cluster_count():
    wide = s_curve_resolution(100)["half_width_single"]
    narrow = s_curve_resolution(400)["half_width_single"]
    assert narrow == pytest.approx(wide / 2.0, rel=1e-9)


def test_zero_clusters_gives_no_resolution_rather_than_zero():
    res = s_curve_resolution(0)
    assert math.isnan(res["half_width_single"])


def test_required_clusters_and_achieved_resolution_are_consistent():
    need = n_clusters_needed(0.05, p=0.5, alpha=0.05)
    n = need["n_clusters_needed_unpaired_reference"]
    # At exactly the required n the achieved resolution must meet the target.
    achieved = s_curve_resolution(n, p=0.5, alpha=0.05)["half_width_unpaired_difference"]
    assert achieved <= 0.05 + 1e-12
    # One cluster fewer must miss it, or the requirement was not tight.
    worse = s_curve_resolution(n - 1, p=0.5, alpha=0.05)["half_width_unpaired_difference"]
    assert worse > 0.05


def test_finer_target_needs_quadratically_more_clusters():
    coarse = n_clusters_needed(0.10)["n_clusters_needed_unpaired_reference"]
    fine = n_clusters_needed(0.05)["n_clusters_needed_unpaired_reference"]
    assert fine == pytest.approx(4 * coarse, rel=0.02)


def test_non_positive_target_is_rejected():
    with pytest.raises(ValueError):
        n_clusters_needed(0.0)
    with pytest.raises(ValueError):
        n_clusters_needed(-0.1)


def test_paired_requirement_is_only_reported_when_a_correlation_was_measured():
    without = n_clusters_needed(0.05)
    assert without["n_clusters_needed_paired"] is None
    assert without["paired_correlation_used"] is None
    with_rho = n_clusters_needed(0.05, paired_correlation=0.75)
    # Pairing at rho = 0.75 cuts the requirement to a quarter.
    assert with_rho["n_clusters_needed_paired"] == pytest.approx(
        without["n_clusters_needed_unpaired_reference"] * 0.25, rel=0.02
    )
    # The conservative value is unchanged, because it is what the gate uses.
    assert with_rho["n_clusters_needed_unpaired_reference"] == without["n_clusters_needed_unpaired_reference"]


def test_impossible_correlation_is_rejected():
    with pytest.raises(ValueError):
        n_clusters_needed(0.05, paired_correlation=1.0)


# --------------------------------------------------------------------------
# macro support
# --------------------------------------------------------------------------
def test_empty_and_thin_classes_are_reported_separately():
    support = macro_metric_support(
        {"head-on": 0, "rear-end": 0, "t-bone": 471, "sideswipe": 6, "single": 860}, min_per_class=30
    )
    assert support["empty_classes"] == ["head-on", "rear-end"]
    assert support["thin_classes"] == ["sideswipe"]
    assert support["n_classes_with_support"] == 2
    assert support["supported"] is False


def test_full_support_passes():
    support = macro_metric_support({"a": 100, "b": 100}, min_per_class=30, closed_set=["a", "b"])
    assert support["supported"] is True
    assert support["empty_classes"] == [] and support["thin_classes"] == []


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def _rows():
    return [
        {
            "h_f": 41,
            "axis_unit": "frame",
            "N_H": 1381,
            "n_clusters": 1381,
            "class_counts": {"head-on": 0, "rear-end": 0, "t-bone": 471, "sideswipe": 50, "single": 860},
        },
        {
            "h_f": 111,
            "axis_unit": "frame",
            "N_H": 456,
            "n_clusters": 456,
            "class_counts": {"head-on": 0, "rear-end": 0, "t-bone": 157, "sideswipe": 6, "single": 293},
        },
    ]


def test_report_refuses_to_default_its_registered_inputs():
    with pytest.raises(G0FloorNotRegisteredError, match="registered"):
        g0_sample_floor_report(_rows(), target_resolution=None, min_per_class=30, n_boot=1000, n_pairs=100)
    with pytest.raises(G0FloorNotRegisteredError):
        g0_sample_floor_report(_rows(), target_resolution=0.05, min_per_class=None, n_boot=1000, n_pairs=100)


def test_report_applies_both_floors_independently():
    """The two floors must bind separately; clearing one may not mask the other.

    The rows are the real MM-AU development cohorts from the frame-axis smoke run.
    A 0.05 target needs 769 clusters, so the short horizon clears the cluster
    floor and the long one does not -- and both still fail overall, because
    ``head-on`` and ``rear-end`` are empty at every horizon.
    """
    report = g0_sample_floor_report(_rows(), target_resolution=0.05, min_per_class=30, n_boot=1000, n_pairs=100)
    assert report["cluster_requirement"]["n_clusters_needed_unpaired_reference"] == 769
    by_h = {h["horizon"]: h for h in report["per_horizon"]}
    assert by_h[41]["n_clusters"] == 1381 and by_h[41]["cluster_floor_met"] is True
    assert by_h[111]["n_clusters"] == 456 and by_h[111]["cluster_floor_met"] is False
    # The short horizon clears the cluster floor yet still fails, so the macro
    # floor is genuinely independent rather than dominated by the cluster one.
    assert by_h[41]["passes_sample_floor"] is None
    assert by_h[41]["macro_support"]["empty_classes"] == ["head-on", "rear-end"]
    assert by_h[111]["passes_sample_floor"] is None
    assert report["n_horizons_passing"] is None


def test_report_carries_the_correction_resolution_check():
    # 100 pairs and only 1000 replicates: Holm cannot reach alpha with a
    # percentile p, which the report must surface rather than hide.
    report = g0_sample_floor_report(_rows(), target_resolution=0.05, min_per_class=30, n_boot=1000, n_pairs=100)
    check = report["correction_resolution"]
    assert check["percentile_ok"] is False
    assert check["n_boot_needed_for_percentile"] == 2000


def test_report_passes_when_both_floors_are_met():
    rows = [
        {
            "h_f": 41,
            "axis_unit": "frame",
            "N_H": 1000,
            "n_clusters": 1000,
            "class_counts": {"a": 300, "b": 300, "c": 400},
            "closed_set": ["a", "b", "c"],
        },
    ]
    report = g0_sample_floor_report(rows, target_resolution=0.05, min_per_class=30, n_boot=4000, n_pairs=100)
    assert report["per_horizon"][0]["passes_sample_floor"] is None
    assert report["n_horizons_passing"] is None
    assert report["horizons_passing"] == []
    assert report["is_formal_sample_floor"] is False


def test_report_states_what_it_does_not_close():
    report = g0_sample_floor_report(_rows(), target_resolution=0.05, min_per_class=30, n_boot=4000, n_pairs=100)
    joined = " ".join(report["not_covered_by_this_rule"])
    assert "licence" in joined
    assert "A5" in joined
    assert "does not close G0" in report["verdict_scope"]


def test_rule_applies_unchanged_to_a_seconds_axis_cohort_table():
    rows = [
        {
            "h_s": 10.0,
            "N_H": 1113,
            "n_clusters": 900,
            "class_counts": {"head-on": 200, "rear-end": 200, "t-bone": 200, "sideswipe": 200, "single": 313},
        },
    ]
    report = g0_sample_floor_report(rows, target_resolution=0.05, min_per_class=30, n_boot=4000, n_pairs=100)
    row = report["per_horizon"][0]
    assert row["horizon"] == 10.0
    assert row["axis_unit"] == "second"
    assert row["passes_sample_floor"] is None
