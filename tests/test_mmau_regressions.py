"""Independent regressions for defects reproduced during code review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ape.frame_axis import FrameAnswerTable, FrameProtocolVector, evaluate_system_frames, frame_horizon_candidates
from ape.g0_floor import macro_metric_support, n_clusters_needed
from ape.g6 import Evaluability, G6Config, G6NotRegisteredError, g6_verdict, load_g6_config
from data.mmau.mmau_candidates import build_ledger
from data.mmau.mmau_decode import done_keys, run


def manifest(n: int = 4, last: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                video_id=str(i),
                source_cluster_id=str(i),
                first_frame=1,
                anchor_frame=1,
                last_frame=last,
                post_anchor_frames=last - 1,
                class_code=0,
                split="test",
            )
            for i in range(n)
        ]
    )


def answers(predictions: list[list[int]], delta: int = 2) -> FrameAnswerTable:
    return FrameAnswerTable.from_frame(
        pd.DataFrame(
            [
                dict(video_id=str(i), j=j, delta_f=delta, pred=p)
                for i, row in enumerate(predictions)
                for j, p in enumerate(row)
            ]
        ),
        "regression-fixture",
    )


def test_hand_calculation_retains_refusals_and_late_regressions() -> None:
    result = evaluate_system_frames(
        manifest(), answers([[0, 1, 0], [-1, 0, 0], [0, 0, -1], [0, 0, 0]]), FrameProtocolVector(h_f=4, delta_f=2)
    )
    assert [result.s_at_f(k) for k in [0, 2, 4]] == [0.25, 0.5, 0.75]
    assert result.metrics_on()["RMSCD@H_frames"] == 2


@pytest.mark.parametrize(
    "kwargs", [{"delta_f": 1.9}, {"h_f": 4.7}, {"h_f": 5, "delta_f": 2}, {"eps_jit_sd_f": np.nan}, {"delta_f": True}]
)
def test_protocol_inputs_are_not_silently_coerced(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        FrameProtocolVector(**kwargs)


@pytest.mark.parametrize(
    "field,value", [("class_code", 1.8), ("class_code", 99), ("source_cluster_id", None), ("anchor_frame", 1.000001)]
)
def test_evaluator_validates_manifest_at_its_boundary(field: str, value: object) -> None:
    data = manifest(1)
    data[field] = [value]
    with pytest.raises(ValueError):
        evaluate_system_frames(data, answers([[0, 0, 0]]), FrameProtocolVector(h_f=4, delta_f=2))


def test_cached_tail_is_not_reported_as_a_full_clip_result() -> None:
    result = evaluate_system_frames(manifest(1, last=11), answers([[0, 0, 0]]), FrameProtocolVector(h_f=4, delta_f=2))
    assert "full_clip_macro_acc" not in result.metrics_on()
    assert result.metrics_on()["cached_tail_macro_acc"] == 1


def test_test_split_cannot_select_the_horizon() -> None:
    with pytest.raises(ValueError, match="development"):
        frame_horizon_candidates(manifest())


@pytest.mark.parametrize("field,value", [("delta_f", 1.1), ("j", 0.5), ("pred", 1.2), ("pred", 99)])
def test_answer_cache_rejects_fractional_and_invalid_codes(field: str, value: object) -> None:
    row = dict(video_id="a", j=0, delta_f=1, pred=0)
    row[field] = value
    with pytest.raises(ValueError):
        FrameAnswerTable.from_frame(pd.DataFrame([row]), "fixture")


def test_decode_recovers_torn_tail_without_losing_the_next_result(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (16, 16), (20, 30, 40)).save(images / "0001.jpg")
    pre = tmp_path / "preflight.jsonl"
    pre.write_text(
        json.dumps(dict(hashcode="a", video_name="fixture", relative_image_directory="images", anchor_frame=1)) + "\n"
    )
    out = tmp_path / "out"
    out.mkdir()
    ledger = out / "decode.jsonl"
    ledger.write_text('{"hashcode":"torn')
    run(tmp_path, pre, out, workers=1, limit=None)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["decoded_ok"] == 1
    assert len(list(out.glob("*.before-recovery-*"))) == 1


def test_task_errors_are_retryable_but_observed_bad_frames_are_terminal(tmp_path: Path) -> None:
    path = tmp_path / "decode.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(dict(hashcode=k, decode_status=s))
            for k, s in [("retry", "worker_error"), ("checked", "decode_failure")]
        )
        + "\n"
    )
    assert done_keys(path) == {"checked"}
    assert [json.loads(line)["hashcode"] for line in path.read_text().splitlines()] == ["checked"]


def _candidate_fixture(tmp_path: Path, hashes: tuple[str, str]) -> tuple[Path, Path]:
    pre = tmp_path / "pre.jsonl"
    dec = tmp_path / "decode.jsonl"
    pre.write_text(
        "".join(
            json.dumps(
                dict(hashcode=k, native_class=13, status="frame_index_valid", post_anchor_frames=20, source="cap")
            )
            + "\n"
            for k in ["a", "b"]
        )
    )
    dec.write_text(
        "".join(
            json.dumps(
                dict(
                    hashcode=k,
                    decode_status="all_frames_decoded",
                    n_distinct_geometries=1,
                    probes=[dict(dhash=h, file_digest=k)],
                )
            )
            + "\n"
            for k, h in zip(["a", "b"], hashes)
        )
    )
    return pre, dec


def test_near_hash_retrieval_requires_no_exact_shared_probe(tmp_path: Path) -> None:
    pre, dec = _candidate_fixture(tmp_path, ("0000000000000000", "0000000000000001"))
    counts = build_ledger(pre, dec, None)["summary"]["duplicate_evidence"]["tier_c_candidate_pairs_by_threshold"]
    assert counts == {"0": 0, "2": 1, "4": 1, "8": 1}


def test_nine_bit_difference_is_outside_registered_search_radius(tmp_path: Path) -> None:
    pre, dec = _candidate_fixture(tmp_path, ("0000000000000000", "00000000000001ff"))
    assert (
        build_ledger(pre, dec, None)["summary"]["duplicate_evidence"]["tier_c_candidate_pairs_by_threshold"]["8"] == 0
    )


def test_equal_record_counts_do_not_hide_wrong_identifiers(tmp_path: Path) -> None:
    pre, dec = _candidate_fixture(tmp_path, ("0000000000000000", "0000000000000001"))
    dec.write_text(dec.read_text().replace('"hashcode": "b"', '"hashcode": "unrelated"'))
    with pytest.raises(ValueError, match="absent"):
        build_ledger(pre, dec, None)


def test_g6_cannot_bypass_registration_with_a_manually_opened_gate() -> None:
    with pytest.raises(G6NotRegisteredError):
        g6_verdict(
            G6Config({"registered": False, "direction": {"pass_threshold": 0.6}}),
            {"primary_rate": 1.0},
            Evaluability(True),
        )


def test_config_hash_uses_exact_bytes_and_receipt_pins_them(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    content = b"registered: true\r\ng6_version: test\r\n"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    assert load_g6_config(str(path)).sha256 == digest
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            dict(
                config_sha256=digest,
                registered_at="2026-09-20T09:00:00+08:00",
                development_split="dev",
                registered_before_first_audit=True,
            )
        )
    )
    assert load_g6_config(str(path), str(receipt)).registration_verified
    path.write_bytes(content + b"# changed\r\n")
    with pytest.raises(ValueError, match="receipt"):
        load_g6_config(str(path), str(receipt))


def test_negative_correlation_invalidates_the_old_conservative_claim() -> None:
    estimate = n_clusters_needed(0.1, paired_correlation=-0.9)
    assert estimate["n_clusters_needed_paired"] > estimate["n_clusters_needed_unpaired_reference"]
    assert estimate["assumptions_verified"] is False


def test_absent_classes_do_not_pass_macro_support() -> None:
    support = macro_metric_support({}, 10)
    assert not support["supported"] and len(support["empty_classes"]) == 5


@pytest.mark.parametrize("rate", [np.nan, np.inf, -0.1, 1.1])
def test_g6_invalid_rate_is_not_a_pass_or_failure(rate: float) -> None:
    config = G6Config(
        raw={
            "registered": True,
            "direction": {"pass_threshold": 0.6},
            "tracks": {
                "b": {
                    "h_f": 4,
                    "delta_f": 1,
                    "grid_max_f": 4,
                    "closed_set": ["a", "b"],
                    "split_source": "synthetic-dev",
                }
            },
            "evaluability": {
                "min_matched_families": 2,
                "min_selected_pairs": 1,
                "min_clusters_track_b": 2,
                "min_videos_per_class_track_b": 1,
            },
        },
        registration_verified=True,
    )
    result = g6_verdict(config, {"primary_rate": rate}, Evaluability(True))
    assert result["verdict"] == "not_evaluable"
    assert result["is_failure"] is False


def test_near_search_recalls_eight_bits_spread_over_eight_bands(tmp_path: Path) -> None:
    value = sum(1 << (7 * i) for i in range(8))
    pre, dec = _candidate_fixture(tmp_path, ("0000000000000000", f"{value:016x}"))
    counts = build_ledger(pre, dec, None)["summary"]["duplicate_evidence"]["tier_c_candidate_pairs_by_threshold"]
    assert counts == {"0": 0, "2": 0, "4": 0, "8": 1}


@pytest.mark.parametrize("value", [np.inf, np.nan, -0.1, 2])
def test_nonfinite_or_out_of_range_planning_targets_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        n_clusters_needed(value)
