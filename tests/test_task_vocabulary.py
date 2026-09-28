"""Prevent native-task caches from silently using five-class semantics."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ape.frame_axis import (
    FrameAnswerTable,
    FrameProtocolVector,
    evaluate_system_frames,
    frame_cohort_summary,
    make_frame_prefixes,
    validate_frame_manifest,
)
from ape.vocabulary import ACCIDENT_VOCABULARY, TaskVocabulary


def vocab() -> TaskVocabulary:
    return TaskVocabulary(
        "native-pilot", "dev-v1", (8, 10, 11, 12, 14, 24, 43, 48, 50), tuple(f"class-{i}" for i in range(9))
    )


def fixtures() -> tuple[TaskVocabulary, pd.DataFrame, pd.DataFrame]:
    v = vocab()
    # The high-index class must contribute to macro scores, even if its support differs.
    m = pd.DataFrame(
        [
            dict(
                video_id=str(i),
                source_cluster_id=str(i),
                first_frame=1,
                anchor_frame=2,
                last_frame=6,
                post_anchor_frames=4,
                class_code=c,
                vocabulary_hash=v.vocabulary_hash,
                split="dev",
            )
            for i, c in enumerate([0, 0, 8])
        ]
    )
    a = pd.DataFrame(
        [
            dict(
                video_id=str(i),
                j=j,
                delta_f=1,
                pred=(0 if i == 2 else c),
                vocabulary_hash=v.vocabulary_hash,
                **{f"p{k}": float(k == (0 if i == 2 else c)) for k in range(9)},
            )
            for i, c in enumerate([0, 0, 8])
            for j in range(5)
        ]
    )
    return v, m, a


def test_ninth_class_and_dynamic_probability_roundtrip() -> None:
    v, m, a = fixtures()
    answers = FrameAnswerTable.from_frame(a, "sys", vocabulary=v)
    pv = FrameProtocolVector(h_f=4, vocabulary_hash=v.vocabulary_hash)
    r = evaluate_system_frames(m, answers, pv, vocabulary=v)
    assert r.metrics_on()["end_window_macro_acc"] == 0.5
    assert r.metrics_on()["RMSCD@H_frames_macro"] == 2
    assert r.metrics_on()["RMSCD@H_frames"] == pytest.approx(4 / 3)
    assert r.s_at_f(2, macro=True) == 0.5
    assert r.frozen_family([1])["end_window_macro_acc"] == 0.5
    assert len(r.cohort_info["class_counts"]) == 9
    assert r.cohort_info["class_counts"]["class-8"] == 1
    assert r.cohort_info["class_counts"]["class-7"] == 0
    for table in [answers, answers.with_identity("renamed"), answers.subsample(2)]:
        frame = table.to_frame()
        assert {f"p{i}" for i in range(9)}.issubset(frame)
        loaded = FrameAnswerTable.from_frame(frame, "reload", vocabulary=v)
        assert loaded.lookup(["0", "2"], np.array([0, 0]))[3].shape == (2, 9)
        assert loaded.lookup_2d(["0"], np.array([[0, 1]]))[3].shape == (1, 2, 9)
    assert not make_frame_prefixes(m, pv, 4, [4], vocabulary=v).empty


@pytest.mark.parametrize("mode", ["missing", "mismatch", "mixed"])
def test_manifest_and_answers_require_consistent_task_identity(mode: str) -> None:
    v, m, a = fixtures()
    for table in (m, a):
        if mode == "missing":
            table.drop(columns="vocabulary_hash", inplace=True)
        elif mode == "mismatch":
            table["vocabulary_hash"] = "f" * 64
        else:
            table.loc[0, "vocabulary_hash"] = "f" * 64
    with pytest.raises(ValueError, match="vocabulary"):
        validate_frame_manifest(m, v)
    with pytest.raises(ValueError, match="vocabulary"):
        FrameAnswerTable.from_frame(a, "sys", vocabulary=v)


def test_permuted_vocabularies_cannot_reuse_protocol_or_cache() -> None:
    v, m, a = fixtures()
    reverse = TaskVocabulary(v.task_id, v.version, v.native_codes[::-1], v.class_names[::-1])
    assert reverse.vocabulary_hash != v.vocabulary_hash
    table = FrameAnswerTable.from_frame(a, "sys", vocabulary=v)
    with pytest.raises(ValueError, match="protocol vocabulary"):
        evaluate_system_frames(m, table, FrameProtocolVector(h_f=4), vocabulary=v)
    with pytest.raises(ValueError, match="vocabularies differ"):
        evaluate_system_frames(
            m, table, FrameProtocolVector(h_f=4, vocabulary_hash=reverse.vocabulary_hash), vocabulary=reverse
        )
    with pytest.raises(ValueError, match="vocabulary"):
        table.with_identity("oops", {"vocabulary_hash": reverse.vocabulary_hash})
    pv = FrameProtocolVector(h_f=4, vocabulary_hash=v.vocabulary_hash)
    assert pv.pi_hash != pv.replace(vocabulary_hash=reverse.vocabulary_hash).pi_hash


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_probability_dimension_is_not_silently_truncated(change: str) -> None:
    v, _, a = fixtures()
    if change == "missing":
        a.drop(columns="p8", inplace=True)
    else:
        a["p9"] = 0
    with pytest.raises(ValueError, match="probability columns"):
        FrameAnswerTable.from_frame(a, "sys", vocabulary=v)


def test_native_codes_are_not_internal_indices() -> None:
    v, m, _ = fixtures()
    assert [v.decode(v.encode(c)) for c in v.native_codes] == list(v.native_codes)
    m.loc[0, "class_code"] = 50
    with pytest.raises(ValueError, match="closed set"):
        frame_cohort_summary(m, 4, v)
    with pytest.raises(ValueError):
        v.encode(True)
    with pytest.raises(ValueError):
        v.decode(-1)


def test_native_labels_cannot_disagree_with_encoded_indices() -> None:
    v, manifest, _ = fixtures()
    manifest["native_class"] = [8, 8, 50]
    validate_frame_manifest(manifest, v)
    manifest.loc[2, "native_class"] = 48
    with pytest.raises(ValueError, match="native_class and class_code disagree"):
        validate_frame_manifest(manifest, v)


def test_legacy_hash_and_five_probability_columns_remain_unchanged() -> None:
    pv = FrameProtocolVector(h_f=4)
    old_payload = {
        "eps_sys_f": 0,
        "eps_jit_sd_f": 0.0,
        "delta_f": 1,
        "h_f": 4,
        "jit_seed": 20260920,
        "pre_anchor_outputs_bot": True,
        "protocol_version": "0.1-frame-unfrozen",
        "axis_unit": "frame",
    }
    expected = hashlib.blake2b(
        json.dumps(old_payload, sort_keys=True, separators=(",", ":")).encode(), digest_size=6
    ).hexdigest()
    assert pv.pi_hash == expected
    assert "vocabulary_hash" not in pv.as_dict()
    a = pd.DataFrame([dict(video_id="v", j=0, delta_f=1, pred=4)])
    table = FrameAnswerTable.from_frame(a, "legacy")
    assert table.vocabulary == ACCIDENT_VOCABULARY
    assert "p4" in table.to_frame() and "p5" not in table.to_frame()


@pytest.mark.parametrize(
    "codes,names", [((1, 1), ("a", "b")), ((1, 2), ("a", "a")), ((1,), ("a",)), ((True, 2), ("a", "b"))]
)
def test_invalid_vocabulary(codes: tuple, names: tuple) -> None:
    with pytest.raises(ValueError):
        TaskVocabulary("task", "v1", codes, names)
