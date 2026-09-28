"""Tests for the candidate ledger and its tiered duplicate evidence.

The point of the ledger is that no sequence is ever silently dropped and no
duplicate claim is ever stronger than its evidence. These cases check both: every
exclusion appears as a named reason on the row it applies to, and the three
duplicate tiers stay separate -- byte-identical (confirmed), same dHash with
different bytes (pending visual confirmation), and near dHash (candidate only).
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MMAU_DIR = os.path.join(REPO_ROOT, "data", "mmau")
if MMAU_DIR not in sys.path:
    sys.path.insert(0, MMAU_DIR)

import mmau_candidates as mc  # noqa: E402


def _preflight_row(hashcode, native_class, status="frame_index_valid", post=100, source="cap", **kw):
    row = {
        "hashcode": hashcode,
        "video_name": f"{native_class}_{hashcode}",
        "source": source,
        "native_class": native_class,
        "relative_image_directory": f"extracted/x/{hashcode}/images",
        "status": status,
        "anchor_frame": 10,
        "post_anchor_frames": post,
        "first_frame": 1,
        "last_frame": 10 + post,
    }
    row.update(kw)
    return row


def _decode_row(hashcode, digests, dhashes, status="all_frames_decoded", geometries=1, failures=0):
    return {
        "hashcode": hashcode,
        "decode_status": status,
        "frame_files": len(digests),
        "decoded_ok": len(digests) - failures,
        "decode_failures": failures,
        "n_distinct_geometries": geometries,
        "probes": [
            {"position": i, "frame": f"{i:06d}", "is_anchor": False, "bytes": 1000, "file_digest": d, "dhash": h}
            for i, (d, h) in enumerate(zip(digests, dhashes))
        ],
    }


def _write(tmp_path, pre_rows, dec_rows, proposal_payload=None):
    pre = tmp_path / "preflight.jsonl"
    pre.write_text("\n".join(json.dumps(r) for r in pre_rows) + "\n", encoding="utf-8")
    dec = tmp_path / "decode.jsonl"
    dec.write_text("\n".join(json.dumps(r) for r in dec_rows) + "\n", encoding="utf-8")
    prop = None
    if proposal_payload is not None:
        prop = tmp_path / "proposal.json"
        prop.write_text(json.dumps(proposal_payload), encoding="utf-8")
    return pre, dec, prop


def _proposal(mapping):
    return {"proposal": [{"code": c, "proposed_class": v, "definition": f"def {c}"} for c, v in mapping.items()]}


# --------------------------------------------------------------------------
# nothing is dropped
# --------------------------------------------------------------------------
def test_every_released_sequence_gets_a_row(tmp_path):
    pre_rows = [
        _preflight_row("a1", 10),
        _preflight_row("a2", 10, status="frame_anomaly"),
        _preflight_row("a3", 10, status="non_accident"),
    ]
    dec_rows = [_decode_row(h, [f"d{h}"], ["0000000000000000"]) for h in ("a1", "a2", "a3")]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    assert ledger["summary"]["n_released_sequences"] == 3
    assert len(ledger["rows"]) == 3
    # An excluded sequence is still present, carrying its reason.
    by_key = {r["hashcode"]: r for r in ledger["rows"]}
    assert by_key["a3"]["is_candidate"] is False
    assert "not_an_accident_sequence" in by_key["a3"]["exclusion_reasons"]


def test_exclusion_reasons_are_named_and_counted(tmp_path):
    pre_rows = [
        _preflight_row("ok", 10),
        _preflight_row("anom", 10, status="frame_anomaly"),
        _preflight_row("zero", 10, post=0),
        _preflight_row("amb", 43),
    ]
    dec_rows = [_decode_row(h, [f"d{h}"], [f"{i:016x}"]) for i, h in enumerate(("ok", "anom", "zero", "amb"))]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone", 43: "ambiguous"}))
    ledger = mc.build_ledger(pre, dec, prop)
    by_key = {r["hashcode"]: r for r in ledger["rows"]}
    assert by_key["ok"]["is_candidate"] is True
    assert "frame_index_anomaly_released_length_disagrees_with_delivery" in by_key["anom"]["exclusion_reasons"]
    assert "anchor_at_last_frame_no_positive_post_anchor_span" in by_key["zero"]["exclusion_reasons"]
    assert "ai_proposal_ambiguous_definition_does_not_entail_geometry" in by_key["amb"]["exclusion_reasons"]
    counts = ledger["summary"]["exclusion_reason_counts"]
    assert counts["anchor_at_last_frame_no_positive_post_anchor_span"] == 1
    assert ledger["summary"]["n_candidates"] == 1


def test_missing_decode_record_excludes_rather_than_assumes_success(tmp_path):
    pre_rows = [_preflight_row("a1", 10), _preflight_row("a2", 10)]
    dec_rows = [_decode_row("a1", ["d1"], ["0000000000000000"])]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    by_key = {r["hashcode"]: r for r in ledger["rows"]}
    assert "decode_not_yet_verified" in by_key["a2"]["exclusion_reasons"]
    assert by_key["a2"]["decode_status"] == "not_run"
    assert ledger["summary"]["decode_complete"] is False


def test_decode_failure_and_geometry_change_are_separate_reasons(tmp_path):
    pre_rows = [_preflight_row("bad", 10), _preflight_row("geo", 10)]
    dec_rows = [
        _decode_row("bad", ["d1", "d2"], ["0" * 16, "1" * 16], status="decode_failure", failures=1),
        _decode_row("geo", ["d3"], ["2" * 16], geometries=2),
    ]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    by_key = {r["hashcode"]: r for r in ledger["rows"]}
    assert "decode_decode_failure" in by_key["bad"]["exclusion_reasons"]
    assert "frame_geometry_changes_within_sequence" in by_key["geo"]["exclusion_reasons"]


def test_native_code_absent_from_the_proposal_is_flagged(tmp_path):
    pre_rows = [_preflight_row("x", 999)]
    dec_rows = [_decode_row("x", ["d1"], ["0" * 16])]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    assert "native_class_absent_from_release_definitions" in ledger["rows"][0]["exclusion_reasons"]


# --------------------------------------------------------------------------
# duplicate tiers stay separate
# --------------------------------------------------------------------------
def test_identical_bytes_is_tier_a_and_excludes(tmp_path):
    shared = "deadbeef" * 4
    pre_rows = [_preflight_row("a", 10), _preflight_row("b", 10)]
    dec_rows = [
        _decode_row("a", [shared, "unique1"], ["0" * 16, "1" * 16]),
        _decode_row("b", [shared, "unique2"], ["2" * 16, "3" * 16]),
    ]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    by_key = {r["hashcode"]: r for r in ledger["rows"]}
    assert by_key["a"]["tier_a_identical_bytes_with"] == ["b"]
    assert by_key["b"]["tier_a_identical_bytes_with"] == ["a"]
    assert "shares_byte_identical_frames_with_another_sequence" in by_key["a"]["exclusion_reasons"]
    assert ledger["summary"]["duplicate_evidence"]["tier_a_identical_bytes_sequences"] == 2


def test_same_dhash_different_bytes_is_tier_b_and_does_not_exclude(tmp_path):
    same_hash = "0f0f0f0f0f0f0f0f"
    pre_rows = [_preflight_row("a", 10), _preflight_row("b", 10)]
    dec_rows = [
        _decode_row("a", ["bytes_a"], [same_hash]),
        _decode_row("b", ["bytes_b"], [same_hash]),
    ]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    by_key = {r["hashcode"]: r for r in ledger["rows"]}
    assert by_key["a"]["tier_b_identical_dhash_with"] == ["b"]
    assert by_key["a"]["tier_a_identical_bytes_with"] == []
    # A re-encoded match is pending confirmation, so it must not silently filter.
    assert "shares_byte_identical_frames_with_another_sequence" not in by_key["a"]["exclusion_reasons"]
    assert by_key["a"]["is_candidate"] is True


def test_tier_b_does_not_double_report_a_tier_a_pair(tmp_path):
    shared_digest = "aa" * 16
    same_hash = "0f0f0f0f0f0f0f0f"
    pre_rows = [_preflight_row("a", 10), _preflight_row("b", 10)]
    dec_rows = [
        _decode_row("a", [shared_digest], [same_hash]),
        _decode_row("b", [shared_digest], [same_hash]),
    ]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    by_key = {r["hashcode"]: r for r in ledger["rows"]}
    assert by_key["a"]["tier_a_identical_bytes_with"] == ["b"]
    assert by_key["a"]["tier_b_identical_dhash_with"] == []


def test_near_dhash_is_counted_at_several_thresholds(tmp_path):
    # Two sequences share one exact dHash, and their other probes differ by a
    # single bit, so the pair is near at every threshold from 0 upwards.
    pre_rows = [_preflight_row("a", 10), _preflight_row("b", 10)]
    dec_rows = [
        _decode_row("a", ["d1", "d2"], ["0f0f0f0f0f0f0f0f", "0000000000000000"]),
        _decode_row("b", ["d3", "d4"], ["0f0f0f0f0f0f0f0f", "0000000000000001"]),
    ]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    counts = ledger["summary"]["duplicate_evidence"]["tier_c_candidate_pairs_by_threshold"]
    assert counts["0"] == 1
    assert counts["8"] == 1
    by_key = {r["hashcode"]: r for r in ledger["rows"]}
    assert by_key["a"]["tier_c_near_dhash_with"] == ["b"]


def test_unrelated_sequences_produce_no_duplicate_evidence(tmp_path):
    pre_rows = [_preflight_row("a", 10), _preflight_row("b", 10)]
    dec_rows = [
        _decode_row("a", ["d1"], ["0000000000000000"]),
        _decode_row("b", ["d2"], ["ffffffffffffffff"]),
    ]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    ev = ledger["summary"]["duplicate_evidence"]
    assert ev["tier_a_identical_bytes_sequences"] == 0
    assert ev["tier_b_identical_dhash_sequences"] == 0
    assert ev["tier_c_candidate_pairs_by_threshold"]["8"] == 0
    assert all(r["is_candidate"] for r in ledger["rows"])


def test_cross_source_byte_duplication_is_counted_separately(tmp_path):
    """The leak that matters: the same footage delivered under both sources."""
    shared = "cc" * 16
    pre_rows = [_preflight_row("a", 10, source="cap"), _preflight_row("b", 10, source="dada")]
    dec_rows = [_decode_row("a", [shared], ["0" * 16]), _decode_row("b", [shared], ["1" * 16])]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    assert ledger["summary"]["duplicate_evidence"]["tier_a_cross_source_pairs"] == 1


def test_same_source_duplication_is_not_counted_as_cross_source(tmp_path):
    shared = "cc" * 16
    pre_rows = [_preflight_row("a", 10, source="cap"), _preflight_row("b", 10, source="cap")]
    dec_rows = [_decode_row("a", [shared], ["0" * 16]), _decode_row("b", [shared], ["1" * 16])]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    ev = ledger["summary"]["duplicate_evidence"]
    assert ev["tier_a_identical_bytes_sequences"] == 2
    assert ev["tier_a_cross_source_pairs"] == 0


# --------------------------------------------------------------------------
# the ledger never asserts what it has not established
# --------------------------------------------------------------------------
def test_rows_leave_split_cluster_and_fps_unset(tmp_path):
    pre_rows = [_preflight_row("a", 10)]
    dec_rows = [_decode_row("a", ["d1"], ["0" * 16])]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    row = mc.build_ledger(pre, dec, prop)["rows"][0]
    assert row["split"] == "unassigned"
    assert row["source_cluster_id"] is None
    assert row["fps"] is None
    assert row["time_unit"] == "frame"
    assert row["ai_proposed_class_is_human_judgement"] is False


def test_summary_lists_what_is_not_established(tmp_path):
    pre_rows = [_preflight_row("a", 10)]
    dec_rows = [_decode_row("a", ["d1"], ["0" * 16])]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    summary = mc.build_ledger(pre, dec, prop)["summary"]
    joined = " ".join(summary["not_established"])
    assert "source-video clusters" in joined
    assert "B8" in joined
    assert "no FPS" in joined
    assert "detector" in summary["duplicate_evidence"]["interpretation"]


def test_ledger_is_versioned_and_hashed(tmp_path):
    pre_rows = [_preflight_row("a", 10)]
    dec_rows = [_decode_row("a", ["d1"], ["0" * 16])]
    pre, dec, prop = _write(tmp_path, pre_rows, dec_rows, _proposal({10: "t-bone"}))
    ledger = mc.build_ledger(pre, dec, prop)
    paths = mc.write_outputs(ledger, tmp_path / "out")
    assert len(paths["sha256"]) == 64
    summary = json.loads((tmp_path / "out" / "candidates_summary.json").read_text(encoding="utf-8"))
    assert summary["ledger_version"] == mc.LEDGER_VERSION
    assert summary["candidates_jsonl_sha256"] == paths["sha256"]
    # The ledger is deterministic, so the same inputs hash the same way.
    again = mc.write_outputs(mc.build_ledger(pre, dec, prop), tmp_path / "out2")
    assert again["sha256"] == paths["sha256"]


def test_ledger_runs_without_a_proposal_or_decode_pass(tmp_path):
    """A partial run must still produce a ledger, with the gaps named."""
    pre_rows = [_preflight_row("a", 10)]
    pre, dec, _ = _write(tmp_path, pre_rows, [])
    ledger = mc.build_ledger(pre, None, None)
    row = ledger["rows"][0]
    assert "decode_not_yet_verified" in row["exclusion_reasons"]
    assert "native_class_absent_from_release_definitions" in row["exclusion_reasons"]
    assert row["ai_proposed_class"] is None
