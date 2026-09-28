"""Full-content verification must not promote isolated or reordered frame matches."""

from mmau_overlap_audit import compare_sequences, components, probe_pairs


def test_full_identity_and_embedded_run():
    exact = compare_sequences(["a", "b", "c"], ["a", "b", "c"])
    assert exact["full_sequence_identical"]
    partial = compare_sequences(["x", "a", "b", "c"], ["a", "b", "c", "y"])
    assert partial["relation"] == "aligned_byte_run"
    assert partial["run_start_zero_based_a"] == 1
    assert partial["run_start_zero_based_b"] == 0
    assert not partial["source_video_identity_established"]


def test_shared_black_frame_and_reordered_frames_do_not_define_components():
    blank = compare_sequences(["black"] * 4, ["black"] * 4)
    reordered = compare_sequences(["a", "b", "c"], ["c", "a", "b"])
    assert not blank["leakage_constraint_candidate"]
    assert not reordered["leakage_constraint_candidate"]


def test_probe_self_matches_are_not_pairs_and_components_are_transitive():
    rows = [
        {"hashcode": "a", "probes": [{"file_digest": "x"}] * 3},
        {"hashcode": "b", "probes": [{"file_digest": "x"}]},
    ]
    assert probe_pairs(rows) == [("a", "b")]
    assert components(
        [
            {"a": "a", "b": "b", "leakage_constraint_candidate": True},
            {"a": "b", "b": "c", "leakage_constraint_candidate": True},
            {"a": "c", "b": "d", "leakage_constraint_candidate": False},
        ]
    ) == [["a", "b", "c"]]
