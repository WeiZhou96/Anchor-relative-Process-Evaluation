"""Split constraints must flag known leakage and never certify all source groups."""

import pytest
from mmau_split_constraints import build_constraints, check_assignments


def test_cross_split_is_flagged_but_exclusion_resolves_only_known_constraint():
    constraints = {"components": [{"members": ["a", "b"], "content_component_id": "c"}]}
    leaked = check_assignments(constraints, {"a": "train", "b": "test"})
    excluded = check_assignments(constraints, {"a": "train", "b": "excluded"})
    assert leaked["n_violations"] == 1
    assert excluded["known_constraints_satisfied"]
    assert not excluded["global_split_certified"]
    assert not check_assignments(constraints, {"a": "train"})["known_constraints_satisfied"]


def test_class_and_anchor_conflict_are_preserved_not_fixed():
    common = {"metadata_total_frames": 10, "accident": 1, "split_official_ara": ["train"]}
    rows = [
        {**common, "hashcode": "a", "native_class": 10, "anchor_frame": 3},
        {**common, "hashcode": "b", "native_class": 11, "anchor_frame": 4},
    ]
    payload = build_constraints({"content_constraint_components": [["b", "a"]], "input_sha256": {}}, rows)
    group = payload["components"][0]
    assert set(group["annotation_conflicts"]) == {"native_class", "anchor_frame"}
    assert group["recommended_action"] == "quarantine_pending_review"
    assert not payload["annotation_changes"]


def test_invalid_components_and_unknown_splits_rejected():
    with pytest.raises(ValueError):
        build_constraints({"content_constraint_components": [["a", "a"]]}, [])
    with pytest.raises(ValueError):
        check_assignments({"components": []}, {"a": "guess"})
