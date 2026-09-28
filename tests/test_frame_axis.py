"""Tests for the native frame-axis adaptation (``ape.frame_axis``).

The cases are written so that a regression in the protocol *discipline* fails,
not only a regression in arithmetic: exact eligibility, a cohort that no metric
can move, strict causality, never-correct and never-committing clips kept in the
denominator, off-grid lookups reported rather than rounded, and normalisation by
the common horizon rather than by each clip's own remaining length.

The last test in the file pins the seconds-axis reference hash, so that a change
made for the frame axis cannot silently alter the frozen seconds track.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ape.frame_axis import (  # noqa: E402
    FrameAnswerTable,
    FrameProtocolVector,
    check_no_future_reads,
    evaluate_system_frames,
    frame_cohort,
    frame_cohort_summary,
    frame_eligible_mask,
    frame_horizon_candidates,
    jitter_draws_frames,
    make_frame_prefixes,
    n_grid_points_f,
    perturb_frame_manifest,
    validate_frame_manifest,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _manifest(rows) -> pd.DataFrame:
    """Build a frame manifest from ``(video_id, cluster, first, anchor, last, class)``."""
    return validate_frame_manifest(
        pd.DataFrame(
            [
                {
                    "video_id": vid,
                    "source_cluster_id": cluster,
                    "first_frame": first,
                    "anchor_frame": anchor,
                    "last_frame": last,
                    "post_anchor_frames": last - anchor,
                    "class_code": cls,
                    "split": "test",
                }
                for vid, cluster, first, anchor, last, cls in rows
            ]
        )
    )


@pytest.fixture()
def manifest_f() -> pd.DataFrame:
    # post-anchor lengths: 40, 39, 41, 100  -> H=40 keeps three of four
    return _manifest(
        [
            ("v_exact", "c1", 1, 60, 100, 0),
            ("v_short", "c1", 1, 61, 100, 1),
            ("v_long", "c2", 1, 59, 100, 2),
            ("v_huge", "c3", 1, 50, 150, 3),
        ]
    )


def _answers(manifest: pd.DataFrame, delta_f: int, j_lo: int, j_hi: int, pred_fn) -> FrameAnswerTable:
    """Cache one answer per ``(video, j)`` from ``pred_fn(video_id, truth, j)``."""
    truth = dict(zip(manifest["video_id"], manifest["class_code"]))
    rows = []
    for vid in manifest["video_id"]:
        for j in range(j_lo, j_hi + 1):
            rows.append(
                {
                    "video_id": vid,
                    "j": j,
                    "delta_f": delta_f,
                    "pred": int(pred_fn(vid, int(truth[vid]), j)),
                    "committed": False,
                }
            )
    return FrameAnswerTable.from_frame(pd.DataFrame(rows), "sys_test")


# --------------------------------------------------------------------------
# protocol vector
# --------------------------------------------------------------------------
def test_protocol_vector_rejects_unrepresentable_knobs():
    with pytest.raises(ValueError, match="whole number"):
        FrameProtocolVector(eps_sys_f=1.5)
    with pytest.raises(ValueError):
        FrameProtocolVector(eps_jit_sd_f=-1.0)
    with pytest.raises(ValueError):
        FrameProtocolVector(delta_f=0)
    with pytest.raises(ValueError):
        FrameProtocolVector(h_f=0)
    with pytest.raises(ValueError, match="axis_unit"):
        FrameProtocolVector(axis_unit="second")


def test_pi_hash_is_stable_and_separates_axes():
    a = FrameProtocolVector(delta_f=2, h_f=40)
    b = FrameProtocolVector(delta_f=2, h_f=40)
    c = FrameProtocolVector(delta_f=2, h_f=42)
    assert a.pi_hash == b.pi_hash
    assert a.pi_hash != c.pi_hash
    # The frame vector's payload declares its axis, so a seconds vector carrying
    # the same numbers cannot produce the same hash and overwrite its artefacts.
    from ape.protocol import ProtocolVector

    seconds = ProtocolVector(eps_sys_s=0.0, eps_jit_sd_s=0.0, delta_s=2.0, h_s=40.0, jit_seed=a.jit_seed)
    assert a.pi_hash != seconds.pi_hash
    assert a.as_dict()["axis_unit"] == "frame"


# --------------------------------------------------------------------------
# manifest validation
# --------------------------------------------------------------------------
def test_manifest_refuses_seconds_columns(manifest_f):
    bad = manifest_f.copy()
    bad["anchor_s"] = 2.0
    with pytest.raises(ValueError, match="seconds column"):
        validate_frame_manifest(bad)


def test_manifest_refuses_inconsistent_length(manifest_f):
    bad = manifest_f.copy()
    bad.loc[0, "post_anchor_frames"] = 999
    with pytest.raises(ValueError, match="post_anchor_frames"):
        validate_frame_manifest(bad)


def test_manifest_refuses_anchor_outside_clip(manifest_f):
    bad = manifest_f.copy()
    bad.loc[0, "anchor_frame"] = 500
    bad.loc[0, "post_anchor_frames"] = bad.loc[0, "last_frame"] - 500
    with pytest.raises(ValueError, match="outside|>= 0"):
        validate_frame_manifest(bad)


def test_manifest_refuses_fractional_frames(manifest_f):
    bad = manifest_f.copy()
    bad["anchor_frame"] = bad["anchor_frame"].astype(float) + 0.5
    with pytest.raises(ValueError, match="whole number"):
        validate_frame_manifest(bad)


def test_manifest_refuses_duplicate_ids(manifest_f):
    bad = pd.concat([manifest_f, manifest_f.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated video_id"):
        validate_frame_manifest(bad)


# --------------------------------------------------------------------------
# eligibility: exact, no tolerance
# --------------------------------------------------------------------------
def test_eligibility_is_exact_integer_comparison(manifest_f):
    mask = frame_eligible_mask(manifest_f, 40)
    got = dict(zip(manifest_f["video_id"], mask))
    assert got["v_exact"] is np.True_ or bool(got["v_exact"])  # L+ == H is in
    assert not bool(got["v_short"])  # L+ == H-1 is out, with no tolerance slack
    assert bool(got["v_long"])
    assert bool(got["v_huge"])
    assert frame_cohort_summary(manifest_f, 40)["N_H"] == 3


def test_eligibility_has_no_off_by_one_slack(manifest_f):
    # A tolerance of even one frame would pull v_short in; assert it does not.
    assert frame_cohort(manifest_f, 41)["video_id"].tolist() == ["v_huge", "v_long"]
    assert frame_cohort(manifest_f, 101)["video_id"].tolist() == []


def test_cohort_is_independent_of_the_system(manifest_f):
    pv = FrameProtocolVector(delta_f=1, h_f=40)
    good = _answers(manifest_f, 1, 0, 200, lambda v, t, j: t)
    bad = _answers(manifest_f, 1, 0, 200, lambda v, t, j: -1)
    a = evaluate_system_frames(manifest_f, good, pv)
    b = evaluate_system_frames(manifest_f, bad, pv)
    assert a.video_ids.tolist() == b.video_ids.tolist()
    assert a.cohort_info["N_H"] == b.cohort_info["N_H"] == 3


# --------------------------------------------------------------------------
# grid and strict causality
# --------------------------------------------------------------------------
def test_grid_stops_at_or_before_the_horizon():
    assert n_grid_points_f(40, 1) == 40
    assert n_grid_points_f(40, 3) == 13  # 13*3 = 39 <= 40
    with pytest.raises(ValueError):
        n_grid_points_f(40, 0)


def test_nondivisible_horizon_is_rejected(manifest_f):
    with pytest.raises(ValueError, match="divisible"):
        FrameProtocolVector(delta_f=3, h_f=40)


def test_no_future_reads_inside_the_eligible_window(manifest_f):
    pv = FrameProtocolVector(delta_f=1, h_f=40)
    prefixes = make_frame_prefixes(manifest_f, pv, grid_max_f=40, h_list_f=[40])
    assert check_no_future_reads(prefixes, manifest_f, 40).empty


def test_future_read_check_actually_catches_a_violation(manifest_f):
    pv = FrameProtocolVector(delta_f=1, h_f=40)
    prefixes = make_frame_prefixes(manifest_f, pv, grid_max_f=40, h_list_f=[40])
    # Shorten a clip without shrinking its grid: the cohort and the grid have
    # come apart, and the checker must say so rather than silently pass.
    tampered = manifest_f.copy()
    tampered.loc[tampered["video_id"] == "v_exact", "last_frame"] = 80
    bad = check_no_future_reads(prefixes, tampered, 40)
    assert not bad.empty
    assert set(bad["video_id"]) == {"v_exact"}


def test_prefix_end_frames_are_anchor_plus_offset(manifest_f):
    pv = FrameProtocolVector(delta_f=2, h_f=40)
    prefixes = make_frame_prefixes(manifest_f, pv, grid_max_f=40, h_list_f=[40])
    row = prefixes.loc[(prefixes["video_id"] == "v_exact") & (prefixes["j"] == 5)].iloc[0]
    assert int(row["end_frame"]) == 60 + 5 * 2
    assert int(row["offset_f"]) == 10
    assert bool(row["base_on_grid"])


# --------------------------------------------------------------------------
# analytic metric values
# --------------------------------------------------------------------------
def test_s_curve_and_rmscd_match_the_closed_form(manifest_f):
    pv = FrameProtocolVector(delta_f=1, h_f=40)
    j0 = 7
    answers = _answers(manifest_f, 1, 0, 200, lambda v, t, j: t if j >= j0 else (t + 1) % 5)
    res = evaluate_system_frames(manifest_f, answers, pv)
    # Stable correctness is 0 before j0 and 1 from j0 to the window end, so the
    # trapezoid of 1 - S over a unit step is exactly j0 - 0.5.
    assert res.s_at_f(j0 - 1) == pytest.approx(0.0)
    assert res.s_at_f(j0) == pytest.approx(1.0)
    m = res.metrics_on()
    assert m["RMSCD@H_frames"] == pytest.approx(j0 - 0.5)
    assert m["RMSCD@H_norm"] == pytest.approx((j0 - 0.5) / 40.0)
    assert m["end_window_macro_acc"] == pytest.approx(1.0)


def test_never_correct_clips_stay_in_the_denominator(manifest_f):
    pv = FrameProtocolVector(delta_f=1, h_f=40)
    answers = _answers(manifest_f, 1, 0, 200, lambda v, t, j: (t + 1) % 5)
    res = evaluate_system_frames(manifest_f, answers, pv)
    assert res.n == 3  # nothing dropped
    m = res.metrics_on()
    # An all-wrong system integrates 1 over the whole window: exactly H_eff.
    assert m["RMSCD@H_frames"] == pytest.approx(float(res.effective_h_f))
    assert m["RMSCD@H_norm"] == pytest.approx(1.0)
    assert (res.first_stable_offset_f() == -1).all()


def test_one_never_correct_clip_is_not_deleted_from_the_mean(manifest_f):
    pv = FrameProtocolVector(delta_f=1, h_f=40)
    answers = _answers(manifest_f, 1, 0, 200, lambda v, t, j: (t + 1) % 5 if v == "v_exact" else t)
    res = evaluate_system_frames(manifest_f, answers, pv)
    # Two of three clips are correct throughout; S_H is 2/3 everywhere, not 1.
    assert res.s_at_f(0) == pytest.approx(2.0 / 3.0)
    assert res.metrics_on()["RMSCD@H_frames"] == pytest.approx(40.0 / 3.0)


def test_bot_at_non_negative_offset_is_wrong_not_an_abstention(manifest_f):
    pv = FrameProtocolVector(delta_f=1, h_f=40)
    answers = _answers(manifest_f, 1, 0, 200, lambda v, t, j: -1 if j < 5 else t)
    res = evaluate_system_frames(manifest_f, answers, pv)
    assert res.s_at_f(4) == pytest.approx(0.0)
    assert res.s_at_f(5) == pytest.approx(1.0)
    assert res.n == 3


# --------------------------------------------------------------------------
# lookup behaviour
# --------------------------------------------------------------------------
def test_offgrid_lookup_is_reported_and_scored_wrong(manifest_f):
    # Cached step 2, common-mode shift 1 frame: every requested end frame is odd
    # relative to the cached grid, so nothing can be answered.
    pv = FrameProtocolVector(delta_f=2, h_f=40, eps_sys_f=1)
    answers = _answers(manifest_f, 2, 0, 100, lambda v, t, j: t)
    res = evaluate_system_frames(manifest_f, answers, pv)
    assert res.n_offgrid_lookup == res.pred.size
    assert res.missing_lookup_rate == pytest.approx(1.0)
    assert (res.pred == -1).all()
    assert res.metrics_on()["RMSCD@H_norm"] == pytest.approx(1.0)


def test_on_grid_shift_is_answered_from_the_cache(manifest_f):
    pv = FrameProtocolVector(delta_f=2, h_f=40, eps_sys_f=2)
    answers = _answers(manifest_f, 2, 0, 100, lambda v, t, j: t)
    res = evaluate_system_frames(manifest_f, answers, pv)
    assert res.n_offgrid_lookup == 0
    assert res.missing_lookup_rate == pytest.approx(0.0)


def test_missing_cached_rows_count_as_missing_not_as_success(manifest_f):
    pv = FrameProtocolVector(delta_f=1, h_f=40)
    answers = _answers(manifest_f, 1, 0, 10, lambda v, t, j: t)  # cache too short
    res = evaluate_system_frames(manifest_f, answers, pv)
    assert res.n_missing_lookup > 0
    assert res.s_at_f(40) == pytest.approx(0.0)


def test_coarse_step_must_divide_the_cached_step(manifest_f):
    pv = FrameProtocolVector(delta_f=3, h_f=39)
    with pytest.raises(ValueError, match="whole multiple"):
        make_frame_prefixes(manifest_f, pv, grid_max_f=40, h_list_f=[40], delta_fine_f=2)


def test_subsample_scales_the_frame_step(manifest_f):
    answers = _answers(manifest_f, 1, 0, 100, lambda v, t, j: t)
    coarse = answers.subsample(2)
    assert coarse.delta_f == 2


# --------------------------------------------------------------------------
# perturbation
# --------------------------------------------------------------------------
def test_jitter_is_keyed_by_video_and_seed_not_by_row_order(manifest_f):
    ids = manifest_f["video_id"].tolist()
    _, full = jitter_draws_frames(ids, 3.0, 42)
    _, reordered = jitter_draws_frames(list(reversed(ids)), 3.0, 42)
    assert full.tolist() == list(reversed(reordered.tolist()))
    _, subset = jitter_draws_frames(ids[1:3], 3.0, 42)
    assert subset.tolist() == full[1:3].tolist()
    _, other_seed = jitter_draws_frames(ids, 3.0, 43)
    assert other_seed.tolist() != full.tolist()


def test_subframe_jitter_is_quantised_and_reported(manifest_f):
    raw, whole = jitter_draws_frames(manifest_f["video_id"].tolist(), 0.05, 7)
    assert (np.abs(raw) < 1.0).all()
    assert (whole == 0).all()  # a sub-frame jitter cannot move a frame index
    pv = FrameProtocolVector(h_f=40, eps_jit_sd_f=0.05)
    res = evaluate_system_frames(manifest_f, _answers(manifest_f, 1, 0, 200, lambda v, t, j: t), pv)
    assert res.jitter_quantization_frames == pytest.approx(0.5)


def test_perturbation_moves_the_cohort_as_well_as_the_grid(manifest_f):
    pv = FrameProtocolVector(h_f=40, eps_sys_f=2)
    eff = perturb_frame_manifest(manifest_f, pv)
    # Shifting the anchor later shortens every post-anchor length by 2 frames,
    # so v_exact (40) and v_long (41) drop below H=40.
    assert dict(zip(eff["video_id"], eff["post_anchor_frames_eff"]))["v_exact"] == 38
    assert frame_cohort_summary(eff, 40)["N_H"] == 1


def test_anchor_clipping_is_recorded(manifest_f):
    pv = FrameProtocolVector(h_f=10, eps_sys_f=-100)
    eff = perturb_frame_manifest(manifest_f, pv)
    assert eff["anchor_clipped"].all()
    assert (eff["anchor_frame_eff"] == eff["first_frame"]).all()


# --------------------------------------------------------------------------
# commitment
# --------------------------------------------------------------------------
def test_never_committing_clips_are_censored_at_h_not_dropped(manifest_f):
    pv = FrameProtocolVector(delta_f=1, h_f=40)
    answers = _answers(manifest_f, 1, 0, 200, lambda v, t, j: t)
    res = evaluate_system_frames(manifest_f, answers, pv)
    commit = res.commit()
    assert commit["rho"] == pytest.approx(0.0)
    assert commit["n"] == 3
    assert commit["n_committed"] == 0
    assert commit["tau_c_median_frames"] == pytest.approx(float(res.effective_h_f))
    assert "h_s" not in commit and commit["h_f"] == res.effective_h_f


def test_commitment_is_irrevocable_inside_the_window(manifest_f):
    truth = dict(zip(manifest_f["video_id"], manifest_f["class_code"]))
    rows = []
    for vid in manifest_f["video_id"]:
        for j in range(0, 201):
            rows.append(
                {
                    "video_id": vid,
                    "j": j,
                    "delta_f": 1,
                    "pred": int(truth[vid]),
                    "committed": j == 12,  # a single, non-persistent flag
                }
            )
    answers = FrameAnswerTable.from_frame(pd.DataFrame(rows), "sys_commit")
    res = evaluate_system_frames(manifest_f, answers, FrameProtocolVector(delta_f=1, h_f=40))
    assert not res.committed[:, 11].any()
    assert res.committed[:, 12].all()
    assert res.committed[:, 40].all()  # stays committed once committed
    assert res.commit()["tau_c_median_frames"] == pytest.approx(12.0)


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
def test_normalisation_uses_the_common_horizon_not_the_clip_tail():
    # Two clips with very different remaining lengths but the same H. If the
    # normaliser were per-clip, the reading would depend on v_huge's extra tail.
    m = _manifest(
        [
            ("v_tight", "c1", 1, 60, 100, 0),  # L+ = 40, exactly H
            ("v_huge", "c2", 1, 50, 1050, 1),  # L+ = 1000, far beyond H
        ]
    )
    pv = FrameProtocolVector(delta_f=1, h_f=40)
    answers = _answers(m, 1, 0, 1100, lambda v, t, j: t if j >= 10 else (t + 1) % 5)
    res = evaluate_system_frames(m, answers, pv)
    m_out = res.metrics_on()
    assert res.effective_h_f == 40
    assert m_out["RMSCD@H_frames"] == pytest.approx(9.5)
    assert m_out["RMSCD@H_norm"] == pytest.approx(9.5 / 40.0)
    # Same normaliser for both clips: it is a property of the track, not the clip.
    assert m_out["RMSCD@H_norm"] * res.effective_h_f == pytest.approx(m_out["RMSCD@H_frames"])


# --------------------------------------------------------------------------
# read-out offsets and horizon candidates must be registered, not defaulted
# --------------------------------------------------------------------------
def test_frozen_family_refuses_to_invent_readout_offsets(manifest_f):
    pv = FrameProtocolVector(delta_f=1, h_f=40)
    res = evaluate_system_frames(manifest_f, _answers(manifest_f, 1, 0, 200, lambda v, t, j: t), pv)
    with pytest.raises(ValueError, match="registered"):
        res.frozen_family([])
    fam = res.frozen_family([5, 20])
    assert set(fam) >= {"RMSCD@H_frames", "RMSCD@H_norm", "S_H@5f", "S_H@20f"}
    assert fam["S_H@5f"] == pytest.approx(1.0)


def test_offgrid_readout_is_nan_not_a_rounded_neighbour(manifest_f):
    pv = FrameProtocolVector(delta_f=4, h_f=40)
    res = evaluate_system_frames(manifest_f, _answers(manifest_f, 1, 0, 200, lambda v, t, j: t), pv)
    assert np.isnan(res.s_at_f(5))  # 5 is not a multiple of the 4-frame step
    assert res.s_at_f(8) == pytest.approx(1.0)
    assert np.isnan(res.s_at_f(-1))
    assert np.isnan(res.s_at_f(10_000))


def test_horizon_candidates_are_candidates_only(manifest_f):
    out = frame_horizon_candidates(manifest_f.assign(split="dev"), quantiles=(0.5,))
    assert out["status"] == "candidates_only_not_frozen"
    assert out["n_dev"] == 4
    assert out["quantiles"]["q0.5"]["h_f"] == 40
    with pytest.raises(ValueError, match="empty"):
        frame_horizon_candidates(manifest_f.iloc[:0].assign(split="dev"))


# --------------------------------------------------------------------------
# the seconds axis must not move
# --------------------------------------------------------------------------
def test_seconds_axis_reference_hash_is_unchanged():
    """Drift guard for the frozen seconds track.

    ``pi0`` of ``protocol/pi0.yaml`` hashed to ``8ac32aae418b`` before the frame
    axis existed. The frame axis is additive by construction, so this value must
    still hold; if it does not, a frame-axis change has reached into the frozen
    seconds protocol and every cached seconds artefact is invalid.
    """
    from ape.protocol import load_protocol

    cfg = load_protocol(os.path.join(REPO_ROOT, "protocol", "pi0.yaml"))
    assert cfg.pi0().pi_hash == "8ac32aae418b"
    assert cfg.pi0().as_dict()["h_s"] == 10.0
    assert cfg.h_list_s == [4.0, 10.0, 21.5]
