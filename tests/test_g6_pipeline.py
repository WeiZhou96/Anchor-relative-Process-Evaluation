"""Behavioural regression tests for the diagnostic cross-track pipeline."""

import copy
import json

import numpy as np
import pytest

from ape.g6 import G6Config
from ape.g6_pipeline import (
    compare_family_pair,
    rmscd_contributions,
    run_pipeline,
    strict_json,
    system_resamples,
    validate_track,
)


def config():
    return G6Config(
        {
            "registered": False,
            "family_matching": {"min_seeds_per_family": 2},
            "direction": {"n_boot": 100, "seed": 19, "selection": {"alpha": 0.05}},
        }
    )


def track(reverse=False, families=("fast", "slow")):
    payload = {
        "name": "synthetic_fixture",
        "split_role": "synthetic",
        "metric": "RMSCD@H_norm",
        "provenance": {"cohort_sha256": "synthetic-fixture-not-an-experimental-manifest", "protocol_id": "fixture"},
        "systems": {},
        "observations": [],
    }
    for family in families:
        for seed in ("0", "1", "2"):
            sid = family + seed
            payload["systems"][sid] = {
                "backbone": "fixture",
                "model_kind": "fixture",
                "arm_rule": family,
                "commit_threshold": None,
                "seed": seed,
                "library_round": "fixture",
            }
            base = 0.1 if family == "fast" else 0.8
            if reverse:
                base = 1 - base
            for i in range(9):
                payload["observations"].append(
                    {
                        "system_id": sid,
                        "video_id": str(i),
                        "source_cluster_id": str(i // 3),
                        "class_code": str(i % 2),
                        "value": base + i / 1000 + int(seed) / 100,
                    }
                )
    return payload


def test_end_to_end_agree_and_reverse_are_distinguished():
    agree = run_pipeline(config(), track(), track())
    reverse = run_pipeline(config(), track(), track(True))
    assert agree["rates"]["counts"]["agree"] == 1
    assert reverse["rates"]["counts"]["reverse"] == 1
    assert agree["formal_verdict"] == "not_evaluable"
    assert agree["resampling"]["training_seed_uncertainty"].startswith("not resampled")


def test_b_cannot_change_a_selection_and_missing_family_is_reported():
    full = run_pipeline(config(), track(), track())
    missing = run_pipeline(config(), track(), track(families=("fast",)))
    assert missing["selected_pairs_sha256"] == full["selected_pairs_sha256"]
    assert missing["pair_tests_a"] == full["pair_tests_a"]
    assert missing["rates"]["counts"]["uncertain"] == 1
    assert missing["rates"]["evaluation_coverage"] == 0
    assert len(missing["family_matching"]["only_on_track_a"]) == 1
    json.dumps(strict_json(missing), allow_nan=False)


def test_median_paired_seed_difference_is_not_difference_of_medians():
    payload = track()
    values = {"fast0": 0.0, "fast1": 0.1, "fast2": 1.0, "slow0": 0.0, "slow1": 0.9, "slow2": 1.0}
    for row in payload["observations"]:
        row["value"] = values[row["system_id"]]
    data = validate_track(payload, config())
    point, reps = system_resamples(data, 100, 3)
    result = compare_family_pair(data, *sorted(data.families), point, reps, 2, 0.05)
    assert result["diff"] == 0.0  # median([0,-0.8,0]); difference of medians would be -0.8


def test_unequal_clusters_retain_video_weighting():
    payload = track()
    for row in payload["observations"]:
        row["source_cluster_id"] = "small" if row["video_id"] == "0" else "large"
        row["value"] = 1 if row["video_id"] == "0" else 0
    data = validate_track(payload, config())
    point, reps = system_resamples(data, 100, 5)
    assert point["fast0"] == pytest.approx(1 / 9)
    assert set(np.round(reps["fast0"], 8)) == {0, round(1 / 9, 8), 1}
    assert np.array_equal(reps["fast0"], reps["slow2"])


def test_shared_seed_requirement_is_enforced_per_pair():
    payload = track()
    for card in payload["systems"].values():
        if card["arm_rule"] == "slow":
            card["seed"] = str(int(card["seed"]) + 10)
    result = run_pipeline(config(), payload, track())
    assert result["rates"]["n_selected"] == 0
    assert result["pair_tests_a"][0]["reason"] == "insufficient paired seeds"


def test_rmscd_contributions_match_manual_fixed_cohort_curve():
    pred = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1], [0, 0, 0]])
    values = rmscd_contributions(pred, np.zeros(4, int))
    np.testing.assert_allclose(values, [0.75, 0.25, 1, 0])
    assert values.mean() * 4 == 2


@pytest.mark.parametrize("role", ["audit", "test", "train", None])
def test_audit_and_undeclared_roles_are_rejected(role):
    payload = track()
    payload["split_role"] = role
    with pytest.raises(ValueError, match="synthetic or development"):
        run_pipeline(config(), payload, track())


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_row",
        "duplicate_row",
        "cluster_conflict",
        "class_conflict",
        "nan",
        "range",
        "duplicate_seed",
        "unknown_system",
        "missing_cluster",
    ],
)
def test_malformed_inputs_are_rejected(corruption):
    payload = track()
    if corruption == "missing_row":
        payload["observations"].pop()
    elif corruption == "duplicate_row":
        payload["observations"].append(copy.deepcopy(payload["observations"][0]))
    elif corruption == "cluster_conflict":
        payload["observations"][0]["source_cluster_id"] = "different"
    elif corruption == "class_conflict":
        payload["observations"][0]["class_code"] = "different"
    elif corruption == "nan":
        payload["observations"][0]["value"] = float("nan")
    elif corruption == "range":
        payload["observations"][0]["value"] = -0.1
    elif corruption == "duplicate_seed":
        payload["systems"]["fast1"]["seed"] = "0"
    elif corruption == "unknown_system":
        payload["observations"][0]["system_id"] = "other"
    else:
        payload["observations"][0]["source_cluster_id"] = None
    with pytest.raises(ValueError):
        validate_track(payload, config())


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("direction", "n_boot", 100.5),
        ("direction", "n_boot", True),
        ("direction", "seed", -1),
        ("family_matching", "min_seeds_per_family", 2.5),
    ],
)
def test_no_truncated_protocol_parameters(section, field, value):
    cfg = config()
    cfg.raw[section][field] = value
    with pytest.raises(ValueError):
        run_pipeline(cfg, track(), track())


def test_zero_difference_is_tie_and_registered_flag_cannot_enable_audit_verdict():
    b = track()
    for row in b["observations"]:
        row["value"] = 0.5
    cfg = config()
    cfg.raw["registered"] = True
    report = run_pipeline(cfg, track(), b)
    assert report["rates"]["counts"]["tie"] == 1
    assert report["formal_verdict"] == "not_evaluable"


def test_mixed_training_rounds_are_not_silently_pooled():
    payload = track()
    payload["systems"]["fast1"]["library_round"] = "different_recipe"
    with pytest.raises(ValueError, match="mixes library rounds"):
        validate_track(payload, config())


@pytest.mark.parametrize("field,value", [("track", "b"), ("correction", "none"), ("require_significant", False)])
def test_unsupported_selection_is_rejected(field, value):
    cfg = config()
    cfg.raw["direction"]["selection"][field] = value
    with pytest.raises(ValueError, match="diagnostic selector"):
        run_pipeline(cfg, track(), track())
