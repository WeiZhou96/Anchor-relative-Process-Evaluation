"""Tests for the G6 cross-track direction gate (``ape.g6``).

The cases target the ways a cross-track gate goes wrong rather than only its
arithmetic: a verdict produced from an unregistered threshold, track B being used
to choose the pairs it then confirms, ties quietly dropped so that an unreadable
difference reads as a success, seeds counted as independent families, an interval
computed over pairs that share systems, and an unmet precondition reported as a
failed gate.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ape.g6 import (  # noqa: E402
    DEFAULT_FAMILY_FIELDS,
    G6NotRegisteredError,
    PairOutcome,
    TrackReading,
    agreement_rates,
    bootstrap_agreement_ci,
    cross_track_family_key,
    evaluability,
    g6_report,
    g6_verdict,
    load_g6_config,
    match_families,
    pair_direction,
    select_pairs_on_track_a,
)

DRAFT_PATH = os.path.join(REPO_ROOT, "prereg", "G6_draft_2026-09-20.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_g6_config(DRAFT_PATH)


def _meta(backbone, model, rule, seed, threshold=None, round_="R2"):
    return {
        "backbone": backbone,
        "model_kind": model,
        "arm_rule": rule,
        "commit_threshold": threshold,
        "seed": seed,
        "library_round": round_,
    }


# --------------------------------------------------------------------------
# configuration and registration
# --------------------------------------------------------------------------
def test_draft_config_loads_and_is_not_registered(cfg):
    assert cfg.registered is False
    assert cfg.pass_threshold is None
    assert cfg.null_reference is None
    assert cfg.primary_metric == "RMSCD@H_norm"
    assert cfg.family_fields == DEFAULT_FAMILY_FIELDS
    assert len(cfg.sha256) == 64


def test_draft_leaves_track_b_parameters_unchosen(cfg):
    track_b = dict(cfg.raw["tracks"])["b"]
    # H, step and the closed set depend on a dev subset and a human mapping that
    # do not exist yet; a draft that filled them in would be pre-registering a
    # choice it was not entitled to make.
    assert track_b["h_f"] is None
    assert track_b["delta_f"] is None
    assert track_b["closed_set"] is None
    assert dict(cfg.raw["metrics"])["mmau_s_report_delta_f"] is None


def test_draft_records_the_known_open_preconditions(cfg):
    assert len(cfg.raw["open_preconditions"]) >= 4
    assert cfg.fallback_mode == "abstain"


def test_verdict_refuses_an_unregistered_threshold(cfg):
    outcomes = [PairOutcome("fa", "fb", "RMSCD@H_norm", 1.0, 1.0, "agree", "same sign")]
    gate = evaluability(
        cfg,
        human_mapping_returned=True,
        source_clusters_defined=True,
        decode_verified=True,
        n_populated_classes_track_b=3,
        n_matched_families=10,
        n_selected_pairs=50,
        n_clusters_track_b=500,
    )
    # The draft has no registered floors, so the gate is unmet and the verdict is
    # "not evaluable" rather than a pass.
    assert gate.evaluable is False
    verdict = g6_verdict(cfg, agreement_rates(outcomes), gate)
    assert verdict["verdict"] == "not_evaluable"
    assert verdict["is_failure"] is False

    # Force the gate open but leave the threshold unregistered: now it must raise
    # rather than invent a threshold at report time.
    from ape.g6 import Evaluability

    open_gate = Evaluability(evaluable=True, unmet=[], checked=[])
    with pytest.raises(G6NotRegisteredError, match="registered"):
        g6_verdict(cfg, agreement_rates(outcomes), open_gate)


def test_registered_config_produces_a_verdict(cfg):
    from ape.g6 import Evaluability, G6Config

    raw = dict(cfg.raw)
    raw["registered"] = True
    direction = dict(raw["direction"])
    direction["pass_threshold"] = 0.8
    raw["direction"] = direction
    raw["tracks"] = {
        "b": {
            "h_f": 10,
            "delta_f": 1,
            "grid_max_f": 10,
            "closed_set": ["single", "t-bone"],
            "split_source": "synthetic-dev",
        }
    }
    raw["evaluability"] = {
        "min_matched_families": 2,
        "min_selected_pairs": 1,
        "min_clusters_track_b": 2,
        "min_videos_per_class_track_b": 1,
    }
    registered = G6Config(raw=raw, path=cfg.path, sha256=cfg.sha256, registration_verified=True)
    open_gate = Evaluability(evaluable=True, unmet=[], checked=[])

    agree = [PairOutcome(f"f{i}", "fb", "m", 1.0, 1.0, "agree", "") for i in range(9)]
    verdict = g6_verdict(registered, agreement_rates(agree), open_gate)
    assert verdict["verdict"] == "pass"

    mixed = agree[:5] + [PairOutcome(f"g{i}", "fb", "m", 1.0, -1.0, "reverse", "") for i in range(5)]
    verdict = g6_verdict(registered, agreement_rates(mixed), open_gate)
    assert verdict["verdict"] == "fail"
    assert verdict["is_failure"] is True


# --------------------------------------------------------------------------
# family matching
# --------------------------------------------------------------------------
def test_family_key_excludes_seed_and_library_round():
    a = _meta("r18", "gru128", "ema", seed="1", round_="R1")
    b = _meta("r18", "gru128", "ema", seed="2", round_="R2")
    assert cross_track_family_key(a) == cross_track_family_key(b)
    c = _meta("clip", "gru128", "ema", seed="1")
    assert cross_track_family_key(a) != cross_track_family_key(c)


def test_family_key_rejects_incomplete_metadata():
    with pytest.raises(ValueError, match="missing family fields"):
        cross_track_family_key({"backbone": "r18"})


def test_matching_is_symmetric_and_reports_both_sides():
    meta_a = {
        "a1": _meta("r18", "gru128", "ema", "1"),
        "a2": _meta("r18", "gru128", "ema", "2"),
        "a3": _meta("clip", "mean", "mean_linear", "1"),
    }
    meta_b = {
        "b1": _meta("r18", "gru128", "ema", "1"),
        "b2": _meta("r18", "gru128", "ema", "2"),
        "b3": _meta("vit", "gru512", "hysteresis", "1"),
    }
    report = match_families(meta_a, meta_b, min_seeds_per_family=2)
    assert report["n_matched"] == 1
    assert report["matched"][0].n_seeds_a == 2
    assert report["matched"][0].n_seeds_b == 2
    # Both misses are named, not just track A's.
    assert len(report["only_on_track_a"]) == 1
    assert len(report["only_on_track_b"]) == 1


def test_single_seed_family_is_underpowered_not_matched():
    meta_a = {"a1": _meta("r18", "gru128", "ema", "1")}
    meta_b = {"b1": _meta("r18", "gru128", "ema", "1")}
    report = match_families(meta_a, meta_b, min_seeds_per_family=2)
    assert report["n_matched"] == 0
    assert report["underpowered"][0]["n_seeds_a"] == 1
    # With the floor at one seed the same family does match, so the exclusion is
    # the floor's doing and not a matching bug.
    assert match_families(meta_a, meta_b, min_seeds_per_family=1)["n_matched"] == 1


def test_seeds_of_one_family_are_not_separate_families():
    meta = {f"s{i}": _meta("r18", "gru128", "ema", str(i)) for i in range(3)}
    report = match_families(meta, meta, min_seeds_per_family=2)
    assert report["n_families_a"] == 1
    assert report["n_matched"] == 1


# --------------------------------------------------------------------------
# pair selection
# --------------------------------------------------------------------------
class _FakeTest:
    def __init__(self, a, b, significant):
        self.system_a = a
        self.system_b = b
        self.significant = significant


def test_selection_keeps_only_significant_track_a_pairs():
    tests = [_FakeTest("a", "b", True), _FakeTest("c", "d", False), _FakeTest("e", "f", True)]
    assert select_pairs_on_track_a(tests) == [("a", "b"), ("e", "f")]
    assert len(select_pairs_on_track_a(tests, require_significant=False)) == 3


def test_selection_signature_cannot_see_track_b():
    import inspect

    params = set(inspect.signature(select_pairs_on_track_a).parameters)
    assert not any("b" == p or p.endswith("_b") for p in params)


# --------------------------------------------------------------------------
# per-pair direction
# --------------------------------------------------------------------------
def test_same_sign_agrees_and_opposite_sign_reverses():
    a = TrackReading(diff=0.4, ci_lo=0.2, ci_hi=0.6)
    b_same = TrackReading(diff=0.3, ci_lo=0.1, ci_hi=0.5)
    b_opp = TrackReading(diff=-0.3, ci_lo=-0.5, ci_hi=-0.1)
    assert pair_direction(a, b_same)[0] == "agree"
    assert pair_direction(a, b_opp)[0] == "reverse"


def test_track_b_interval_containing_zero_is_a_tie():
    a = TrackReading(diff=0.4, ci_lo=0.2, ci_hi=0.6)
    b = TrackReading(diff=0.3, ci_lo=-0.1, ci_hi=0.7)
    outcome, reason = pair_direction(a, b)
    assert outcome == "tie"
    assert "interval contains zero" in reason


def test_track_b_difference_below_the_ruler_is_a_tie():
    a = TrackReading(diff=0.4, ci_lo=0.2, ci_hi=0.6)
    # Same sign and a clean interval, but smaller than track B can resolve.
    b = TrackReading(diff=0.01, ci_lo=0.005, ci_hi=0.015, ruler=0.05)
    outcome, reason = pair_direction(a, b)
    assert outcome == "tie"
    assert "minimum resolvable" in reason
    # Without a ruler the same reading is a genuine agreement.
    assert pair_direction(a, TrackReading(0.01, 0.005, 0.015))[0] == "agree"


def test_missing_track_b_reading_is_uncertain_not_a_tie():
    a = TrackReading(diff=0.4, ci_lo=0.2, ci_hi=0.6)
    assert pair_direction(a, None)[0] == "uncertain"
    assert pair_direction(a, TrackReading(np.nan, np.nan, np.nan))[0] == "uncertain"


def test_degenerate_track_a_reading_is_uncertain():
    b = TrackReading(diff=0.3, ci_lo=0.1, ci_hi=0.5)
    assert pair_direction(None, b)[0] == "uncertain"
    assert pair_direction(TrackReading(0.0, -0.1, 0.1), b)[0] == "uncertain"


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def _outcomes(n_agree, n_reverse, n_tie, n_uncertain):
    out = []
    k = 0
    for kind, count in (
        ("agree", n_agree),
        ("reverse", n_reverse),
        ("tie", n_tie),
        ("uncertain", n_uncertain),
    ):
        for _ in range(count):
            out.append(PairOutcome(f"fa{k}", f"fb{k}", "m", 1.0, 1.0, kind, ""))
            k += 1
    return out


def test_ties_count_against_reproduction_in_the_primary_rate():
    outcomes = _outcomes(n_agree=6, n_reverse=2, n_tie=2, n_uncertain=5)
    rates = agreement_rates(outcomes)
    assert rates["n_selected"] == 15
    # Uncertain pairs leave the denominator; ties stay in it.
    assert rates["n_evaluated_excluding_uncertain"] == 10
    assert rates["primary_rate"] == pytest.approx(0.6)
    # The secondary rate drops ties and is therefore always the more flattering
    # of the two; it is reported beside the primary, never instead of it.
    assert rates["n_decided_excluding_ties"] == 8
    assert rates["secondary_rate"] == pytest.approx(0.75)
    assert rates["primary_rate"] <= rates["secondary_rate"]


def test_all_uncertain_gives_no_rate_rather_than_zero():
    rates = agreement_rates(_outcomes(0, 0, 0, 4))
    assert np.isnan(rates["primary_rate"])
    assert np.isnan(rates["secondary_rate"])


def test_counts_cover_every_outcome_name():
    rates = agreement_rates(_outcomes(1, 1, 1, 1))
    assert set(rates["counts"]) == {"agree", "reverse", "tie", "uncertain"}
    assert sum(rates["counts"].values()) == 4


# --------------------------------------------------------------------------
# interval
# --------------------------------------------------------------------------
def test_agreement_interval_resamples_families():
    outcomes = _outcomes(n_agree=8, n_reverse=2, n_tie=0, n_uncertain=0)
    ci = bootstrap_agreement_ci(outcomes, n_boot=400, seed=1)
    assert ci["resample_unit"] == "family"
    assert ci["weight_scheme"] == "dirichlet_multiplier"
    assert 0.0 <= ci["ci_lo"] <= ci["ci_hi"] <= 1.0
    assert ci["n_families"] == 20
    assert ci["ci_lo"] <= agreement_rates(outcomes)["primary_rate"] <= ci["ci_hi"]


def test_no_pair_is_destroyed_by_the_resampling():
    """Every usable pair must survive every replicate.

    The first version of this interval drew multinomial family counts, so a
    family drawn zero times zeroed the weight of every pair containing it. With
    many families and few pairs most pairs then vanished per replicate and the
    interval widened for a reason unrelated to the dependence being modelled.
    Strictly positive Dirichlet weights are what stop that, and this test is what
    would catch a regression back to counts.
    """
    outcomes = _outcomes(n_agree=8, n_reverse=2, n_tie=1, n_uncertain=3)
    ci = bootstrap_agreement_ci(outcomes, n_boot=200, seed=3)
    # 11 of the 14 outcomes are usable; the 3 uncertain ones are excluded by
    # definition, not by resampling.
    assert ci["n_pairs_used"] == 11
    assert ci["n_boot"] == 200  # no replicate collapsed to an empty denominator


def test_interval_narrows_as_evidence_grows_at_a_fixed_rate():
    """Width must be driven by how much evidence there is, at a constant rate."""

    def width(n_pairs):
        n_agree = int(round(0.8 * n_pairs))
        outcomes = _outcomes(n_agree=n_agree, n_reverse=n_pairs - n_agree, n_tie=0, n_uncertain=0)
        ci = bootstrap_agreement_ci(outcomes, n_boot=4000, seed=11)
        assert agreement_rates(outcomes)["primary_rate"] == pytest.approx(0.8)
        return ci["ci_hi"] - ci["ci_lo"]

    w5, w10, w40 = width(5), width(10), width(40)
    assert w5 > w10 > w40


def test_a_family_shared_by_every_pair_contributes_no_uncertainty():
    """Records a known limitation of the ratio statistic, so it is not a surprise.

    The agreement rate is a ratio, so a family appearing in *every* pair has its
    weight cancel from numerator and denominator. An audit whose pairs all hang
    off one hub therefore gets an interval about the spoke families only, and its
    interval is narrower than the same number of pairs over disjoint families.
    This is documented in ``bootstrap_agreement_ci`` and must be stated whenever
    such a design is reported, rather than read as extra precision.
    """
    hub = [PairOutcome("hub", f"f{i}", "m", 1.0, 1.0, "agree" if i < 8 else "reverse", "") for i in range(10)]
    disjoint = _outcomes(n_agree=8, n_reverse=2, n_tie=0, n_uncertain=0)
    assert agreement_rates(hub)["primary_rate"] == agreement_rates(disjoint)["primary_rate"]
    ci_hub = bootstrap_agreement_ci(hub, n_boot=4000, seed=7)
    ci_disjoint = bootstrap_agreement_ci(disjoint, n_boot=4000, seed=7)
    assert ci_hub["n_families"] == 11
    assert ci_disjoint["n_families"] == 20
    assert (ci_hub["ci_hi"] - ci_hub["ci_lo"]) < (ci_disjoint["ci_hi"] - ci_disjoint["ci_lo"])


def test_empty_outcomes_give_no_interval():
    ci = bootstrap_agreement_ci([], n_boot=10, seed=1)
    assert np.isnan(ci["ci_lo"]) and np.isnan(ci["ci_hi"])


def test_all_uncertain_outcomes_give_no_interval():
    ci = bootstrap_agreement_ci(_outcomes(0, 0, 0, 5), n_boot=10, seed=1)
    assert np.isnan(ci["ci_lo"]) and np.isnan(ci["ci_hi"])
    assert ci["n_boot"] == 0


# --------------------------------------------------------------------------
# evaluability
# --------------------------------------------------------------------------
def test_draft_is_not_evaluable_and_names_every_missing_input(cfg):
    gate = evaluability(
        cfg,
        human_mapping_returned=False,
        source_clusters_defined=False,
        decode_verified=False,
        n_populated_classes_track_b=3,
        n_matched_families=0,
        n_selected_pairs=0,
        n_clusters_track_b=None,
    )
    assert gate.evaluable is False
    joined = " | ".join(gate.unmet)
    assert "B8" in joined
    assert "source-video clusters" in joined
    assert "decode-verified" in joined
    assert "draft or has no verified" in joined
    assert "min_matched_families is unregistered" in joined


def test_a_single_populated_class_blocks_the_gate(cfg):
    gate = evaluability(
        cfg,
        human_mapping_returned=True,
        source_clusters_defined=True,
        decode_verified=True,
        n_populated_classes_track_b=1,
        n_matched_families=10,
        n_selected_pairs=10,
        n_clusters_track_b=100,
    )
    assert any("populated classes" in u for u in gate.unmet)


def test_unregistered_floors_are_unmet_rather_than_defaulted(cfg):
    gate = evaluability(
        cfg,
        human_mapping_returned=True,
        source_clusters_defined=True,
        decode_verified=True,
        n_populated_classes_track_b=3,
        n_matched_families=10_000,
        n_selected_pairs=10_000,
        n_clusters_track_b=10_000,
    )
    # Arbitrarily large inputs still do not satisfy a floor nobody registered.
    assert gate.evaluable is False
    floor_messages = [u for u in gate.unmet if "unregistered" in u]
    assert {"min_matched_families", "min_selected_pairs", "min_clusters_track_b", "min_videos_per_class_track_b"} <= {
        u.split(" is unregistered")[0] for u in floor_messages
    }
    # The only other complaint is that the draft itself is not registered; no
    # unmet item may come from a floor that was silently given a default.
    assert set(gate.unmet) - set(floor_messages) == {
        "G6 config is a draft or has no verified external registration receipt"
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def test_report_carries_provenance_and_the_draft_warning(cfg):
    outcomes = _outcomes(3, 1, 1, 1)
    gate = evaluability(
        cfg,
        human_mapping_returned=False,
        source_clusters_defined=False,
        decode_verified=False,
        n_populated_classes_track_b=3,
        n_matched_families=1,
        n_selected_pairs=1,
    )
    report = g6_report(cfg, outcomes, gate)
    assert report["registered"] is False
    assert report["config_sha256"] == cfg.sha256
    assert report["pass_threshold"] is None
    assert "must not be quoted as a G6 result" in report["status_note"]
    assert report["rates"]["primary_rate"] == pytest.approx(0.6)
    assert report["evaluability"]["evaluable"] is False
    assert len(report["outcomes"]) == 6


def test_report_includes_family_matching_summary(cfg):
    meta = {f"s{i}": _meta("r18", "gru128", "ema", str(i)) for i in range(2)}
    family_report = match_families(meta, meta, min_seeds_per_family=2)
    gate = evaluability(cfg, False, False, False, 3, 1, 1)
    report = g6_report(cfg, _outcomes(1, 0, 0, 0), gate, family_report)
    assert report["family_matching"]["n_matched"] == 1
    assert len(report["family_matching"]["matched_keys"]) == 1
