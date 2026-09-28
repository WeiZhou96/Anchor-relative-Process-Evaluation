"""Protocol-layer behaviour: cohorts, censoring, sub-sampling and perturbations.

These tests cover the claims the protocol makes about itself:

* the eligibility cohort is fixed once per ``(pi, H)`` and is shared by every
  system and every grid point;
* a video that never stabilises stays in the denominator and contributes the
  full ``H``;
* evaluating a coarse step by sub-sampling the cached finest-step matrix gives
  the same answer as generating that coarse step directly (the caching
  condition of idea 3.1 item 5);
* a common-mode anchor shift moves every system's delay by the same amount and
  leaves the ordering alone (prediction P-a);
* per-video jitter below the cache resolution is invisible to a cached matrix
  and needs exact regeneration, which is stated rather than hidden;
* the commitment triple is censored at ``H`` and never drops a video.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from ape import blocks as B
from ape.cohort import cohort, cohort_summary
from ape.metrics import AnswerTable, commit_stats, evaluate_system
from ape.protocol import (
    ProtocolVector,
    eligible_col,
    h_label,
    jitter_draws,
    make_prefixes,
    perturb_manifest,
)

from conftest import make_answers, make_manifest

FINE = 0.25
SPAN = 12.0
PAD = 3.0


def _table(manifest, family, params, delta_fine=FINE, commit_after_s=None):
    df = B.generate_block_answers(
        manifest, family, params, delta_fine_s=delta_fine, span_s=SPAN, pad_s=PAD,
        commit_after_s=commit_after_s,
    )
    return AnswerTable.from_frame(
        df, B.block_id(family, params, commit_after_s), {"family": "block"}
    )


def _long_manifest(n=24, post_lo=20.0, post_hi=30.0):
    """Clips long enough that no plausible shift can change the cohort."""
    posts = np.linspace(post_lo, post_hi, n)
    return make_manifest(
        [
            {
                "video_id": f"L{i:02d}",
                "anchor_s": 5.0,
                "post_s": float(posts[i]),
                "y": i % 5,
                "cluster": f"C{i // 2:02d}",
            }
            for i in range(n)
        ]
    )


# --------------------------------------------------------------------------
# identity and prefix lists
# --------------------------------------------------------------------------
@pytest.mark.parametrize("video_id,expected", [
    ("3Xd6PxEvbNk_11_00", "3Xd6PxEvbNk"),
    ("3Xd6PxEvbNk_00", "3Xd6PxEvbNk"),
    ("Z4kg2Ev3vhk_00", "Z4kg2Ev3vhk"),
    ("Z4kg2Ev3vhk", "Z4kg2Ev3vhk"),
    ("real_videos/Z4kg2Ev3vhk_00.mp4", "real_videos/Z4kg2Ev3vhk"),
    ("abc_1234_00", "abc_1234"),  # four digits are not a clip index
])
def test_cluster_rule_strips_trailing_numeric_segments_repeatedly(video_id, expected):
    """Two clips from one source video must land in one cluster at any nesting."""
    from ape.cohort import derive_source_cluster_id

    assert derive_source_cluster_id(video_id) == expected


def test_fixture_manifest_agrees_with_the_cluster_rule(manifest):
    from ape.cohort import check_cluster_rule

    assert len(check_cluster_rule(manifest)) == 0
    # the fixture really does contain a nested suffix, so the rule is exercised
    assert any("_00_00" in v for v in manifest["video_id"])


def test_effective_horizon_is_reported_when_H_is_not_a_multiple_of_the_step(manifest):
    pv = ProtocolVector(delta_s=0.5, h_s=10.03)
    res = evaluate_system(manifest, _table(manifest, "lock", {"d0": 3.0}), pv)
    assert res.h_s == pytest.approx(10.03)
    assert res.effective_h_s == pytest.approx(10.0)
    # eligibility still uses the full H, so the cohort stays on the strict side
    assert res.n == cohort_summary(manifest, 10.03)["N_H"]
    assert res.metrics_on(None)["RMSCD@H"] <= res.effective_h_s + 1e-9


def test_grid_lookup_matches_the_flat_lookup_cell_for_cell(manifest):
    """The fast grid-shaped lookup must be a pure refactor of the flat one."""
    at = _table(manifest, "osc", {"p": 1.0, "d0": 4.0})
    vid = manifest["video_id"].to_numpy()[:40]
    anchor = manifest.set_index("video_id").loc[vid, "anchor_s"].to_numpy()
    offsets = np.arange(0, 25, dtype=float) * 0.5
    base_j = np.rint(offsets[None, :] / FINE).astype(np.int64) + np.zeros(
        (len(vid), 1), dtype=np.int64
    )
    # include out-of-range columns on purpose, both below j_min and above j_max
    base_j[:5, :3] -= 10_000
    base_j[5:10, -3:] += 10_000

    p2, pr2, c2, pb2 = at.lookup_2d(vid, base_j)
    p1, pr1, c1, pb1 = at.lookup(np.repeat(vid, base_j.shape[1]), base_j.reshape(-1))
    np.testing.assert_array_equal(p2, p1.reshape(p2.shape))
    np.testing.assert_array_equal(pr2, pr1.reshape(pr2.shape))
    np.testing.assert_array_equal(c2, c1.reshape(c2.shape))
    np.testing.assert_allclose(
        pb2.reshape(-1, pb2.shape[-1]), pb1, equal_nan=True
    )
    # the out-of-range cells are BOT and flagged missing, not silently valid
    assert (p2[:5, :3] == -1).all() and not pr2[:5, :3].any()
    assert (p2[5:10, -3:] == -1).all() and not pr2[5:10, -3:].any()


def test_grid_lookup_handles_unknown_video_ids(manifest):
    at = _table(manifest, "lock", {"d0": 3.0})
    vid = np.array(["not_a_real_clip", manifest["video_id"].iloc[0]])
    base_j = np.zeros((2, 4), dtype=np.int64)
    pred, present, _, _ = at.lookup_2d(vid, base_j)
    assert (pred[0] == -1).all() and not present[0].any()
    assert present[1].all()


def test_frozen_family_matches_the_general_metric_path(manifest):
    """The bootstrap fast path must return exactly what the general path does."""
    at = _table(manifest, "osc", {"p": 1.0, "d0": 4.0})
    res = evaluate_system(manifest, at, ProtocolVector(delta_s=0.5, h_s=10.03))
    rng = np.random.default_rng(0)
    for idx in [None, rng.integers(0, res.n, res.n), np.arange(res.n // 2)]:
        fam = res.frozen_family([1.0, 3.0], idx)
        ref_idx = np.arange(res.n) if idx is None else idx
        base = res.metrics_on(ref_idx)
        assert fam["RMSCD@H"] == pytest.approx(base["RMSCD@H"], nan_ok=True)
        assert fam["end_window_macro_acc"] == pytest.approx(
            base["end_window_macro_acc"], nan_ok=True
        )
        assert fam["median_flips"] == pytest.approx(base["median_flips"], nan_ok=True)
        for k in (1.0, 3.0):
            assert fam[f"S_H@{k:g}"] == pytest.approx(
                res.s_at_on(k, ref_idx), nan_ok=True
            )


def test_pi_hash_is_stable_and_separates_settings():
    a = ProtocolVector(delta_s=0.5, h_s=6.0)
    b = ProtocolVector(delta_s=0.5, h_s=6.0)
    c = ProtocolVector(delta_s=0.25, h_s=6.0)
    assert a.pi_hash == b.pi_hash
    assert a.pi_hash != c.pi_hash
    assert a.pi_hash != a.replace(eps_sys_s=0.25).pi_hash
    assert a.pi_hash != a.replace(eps_jit_sd_s=0.1).pi_hash
    assert len(a.pi_hash) == 12


def test_h_label_matches_the_contract_column_naming():
    assert h_label(3.0) == "H3"
    assert h_label(10.0) == "H10"
    assert h_label(2.5) == "H25"
    assert eligible_col(6.0) == "eligible_H6"


def test_prefix_list_is_causal_and_consistent(manifest):
    pv = ProtocolVector(delta_s=0.5, h_s=6.0)
    pref = make_prefixes(manifest, pv, 10.0, [3.0, 6.0, 10.0], FINE)
    assert pref["is_post_anchor"].all()
    assert pref["j"].min() == 0 and pref["j"].max() == 20
    eff = perturb_manifest(manifest, pv).set_index("video_id")
    anchor = eff.loc[pref["video_id"], "anchor_s_eff"].to_numpy()
    np.testing.assert_allclose(
        pref["end_s"].to_numpy(), anchor + pref["j"].to_numpy() * 0.5
    )
    fps = eff.loc[pref["video_id"], "fps"].to_numpy()
    np.testing.assert_array_equal(
        pref["end_frame"].to_numpy(), np.floor(pref["end_s"].to_numpy() * fps + 1e-9)
    )
    for h in (3.0, 6.0, 10.0):
        n_elig = int(pref.groupby("video_id")[eligible_col(h)].first().sum())
        assert n_elig == cohort_summary(manifest, h)["N_H"]


# --------------------------------------------------------------------------
# fixed cohort and censoring
# --------------------------------------------------------------------------
def test_cohort_is_the_same_for_every_system_and_every_grid_point(manifest):
    pv = ProtocolVector(delta_s=0.5, h_s=6.0)
    ids = None
    for family, params in [("oracle", {}), ("lock", {"d0": 3.0}), ("rand", {"eta": 0.5})]:
        res = evaluate_system(manifest, _table(manifest, family, params), pv)
        if ids is None:
            ids = list(res.video_ids)
        assert list(res.video_ids) == ids
        # the denominator never changes along the curve
        assert res.pred.shape[0] == len(ids)
    assert len(ids) == cohort_summary(manifest, 6.0)["N_H"]


def test_cohort_shrinks_with_the_horizon_and_the_dropped_clips_are_the_short_ones(manifest):
    sizes = [cohort_summary(manifest, h)["N_H"] for h in (3.0, 6.0, 10.0)]
    assert sizes == sorted(sizes, reverse=True)
    small = set(cohort(manifest, 10.0)["video_id"])
    big = set(cohort(manifest, 3.0)["video_id"])
    assert small < big
    dropped = manifest.loc[manifest["video_id"].isin(big - small)]
    assert float(dropped["post_anchor_length_s"].max()) < 10.0


def test_never_stabilising_videos_contribute_the_full_horizon(manifest):
    """No deletion of failures: that is the D6 survivorship trap."""
    h_s, delta_s = 6.0, 0.5
    pv = ProtocolVector(delta_s=delta_s, h_s=h_s)
    res = evaluate_system(manifest, _table(manifest, "lock", {"d0": 99.0}), pv)
    assert not res.indicator.any()
    assert res.metrics_on(None)["RMSCD@H"] == pytest.approx(h_s)
    assert res.cohort_info["N_H"] == cohort_summary(manifest, h_s)["N_H"]


def test_a_mixed_cohort_keeps_the_failures_in_the_denominator():
    mf = _long_manifest(n=10)
    early = set(mf.iloc[:5]["video_id"].tolist())
    paths = {}
    for vid, y in zip(mf["video_id"], mf["class_code"]):
        y, w = int(y), (int(y) + 1) % 5
        # early clips stabilise at j = 2; the rest are wrong for the whole window
        paths[vid] = [w, w, y, y, y, y, y] if vid in early else [w] * 7
    at = make_answers(paths, delta_s=0.5)
    res = evaluate_system(mf, at, ProtocolVector(delta_s=0.5, h_s=3.0))
    assert res.n == 10
    # five videos at 0.75, five at the full H = 3.0
    assert res.metrics_on(None)["RMSCD@H"] == pytest.approx((5 * 0.75 + 5 * 3.0) / 10)


# --------------------------------------------------------------------------
# sub-sampling equivalence (idea 3.1 item 5, RQ0)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family,params", [
    ("oracle", {}),
    ("lock", {"d0": 3.0}),
    ("osc", {"p": 1.0, "d0": 4.0}),
    ("frac", {"c": 0.6}),
    ("rand", {"eta": 0.7}),
])
@pytest.mark.parametrize("factor", [2, 4])
def test_coarse_subsampling_equals_direct_generation(manifest, family, params, factor):
    coarse = FINE * factor
    direct = _table(manifest, family, params, delta_fine=coarse)
    sub = _table(manifest, family, params, delta_fine=FINE).subsample(factor)
    assert sub.delta_s == pytest.approx(coarse)

    lo = max(sub.j_min, direct.j_min)
    hi = min(sub.j_max, direct.j_max)
    assert hi > lo
    cols_s = slice(lo - sub.j_min, hi - sub.j_min + 1)
    cols_d = slice(lo - direct.j_min, hi - direct.j_min + 1)
    assert list(sub.video_ids) == list(direct.video_ids)
    np.testing.assert_array_equal(sub.pred[:, cols_s], direct.pred[:, cols_d])

    pv = ProtocolVector(delta_s=coarse, h_s=6.0)
    a = evaluate_system(manifest, sub, pv).metrics_on(None)
    b = evaluate_system(manifest, direct, pv).metrics_on(None)
    for k in a:
        assert a[k] == pytest.approx(b[k], nan_ok=True), k


def test_evaluating_a_coarse_step_from_the_fine_cache_matches_the_coarse_cache(manifest):
    """The path the CLI actually takes: one cached matrix, several steps."""
    fine = _table(manifest, "osc", {"p": 1.0, "d0": 4.0}, delta_fine=FINE)
    for factor in (2, 4):
        coarse = _table(manifest, "osc", {"p": 1.0, "d0": 4.0}, delta_fine=FINE * factor)
        pv = ProtocolVector(delta_s=FINE * factor, h_s=6.0)
        a = evaluate_system(manifest, fine, pv).metrics_on(None)
        b = evaluate_system(manifest, coarse, pv).metrics_on(None)
        for k in a:
            assert a[k] == pytest.approx(b[k], nan_ok=True), (factor, k)


# --------------------------------------------------------------------------
# P-a: common-mode shift
# --------------------------------------------------------------------------
# four blocks with four *distinct* stabilisation offsets (2.0, 3.0, 5.0, 4.5):
# osc(1.0, 4.0) is deliberately avoided here because osc_tau(1.0, 4.0) == 3.0
# would tie it with lock(3.0) and a tie is not an ordering to preserve
SHIFT_BLOCKS = [("lock", {"d0": 2.0}), ("lock", {"d0": 3.0}),
                ("lock", {"d0": 5.0}), ("osc", {"p": 0.5, "d0": 5.0})]
SHIFTS = [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_common_mode_shift_moves_every_system_by_minus_epsilon():
    """P-a, measured on gauge blocks whose true behaviour cannot drift."""
    mf = _long_manifest()
    h_s, delta_s = 8.0, 0.5
    tables = {B.block_id(f, p): _table(mf, f, p) for f, p in SHIFT_BLOCKS}
    pv0 = ProtocolVector(delta_s=delta_s, h_s=h_s)
    base = {
        sid: evaluate_system(mf, at, pv0).metrics_on(None)["RMSCD@H"]
        for sid, at in tables.items()
    }
    n_ref = evaluate_system(mf, next(iter(tables.values())), pv0).n
    for eps in SHIFTS:
        pv = ProtocolVector(delta_s=delta_s, h_s=h_s, eps_sys_s=eps)
        for sid, at in tables.items():
            res = evaluate_system(mf, at, pv)
            assert res.n == n_ref  # the cohort is untouched by the shift here
            v = res.metrics_on(None)["RMSCD@H"]
            assert v == pytest.approx(base[sid] - eps, abs=1e-9), (sid, eps)


def test_common_mode_shift_leaves_the_ordering_unchanged():
    """The other half of P-a: ``b`` moves, ``R`` does not."""
    mf = _long_manifest()
    h_s, delta_s = 8.0, 0.5
    tables = {B.block_id(f, p): _table(mf, f, p) for f, p in SHIFT_BLOCKS}
    values = {}
    for eps in SHIFTS:
        pv = ProtocolVector(delta_s=delta_s, h_s=h_s, eps_sys_s=eps)
        values[eps] = {
            sid: evaluate_system(mf, at, pv).metrics_on(None)["RMSCD@H"]
            for sid, at in tables.items()
        }
    sids = sorted(tables)
    for a, b in itertools.combinations(sids, 2):
        s0 = np.sign(values[0.0][a] - values[0.0][b])
        assert s0 != 0
        for eps in SHIFTS:
            assert np.sign(values[eps][a] - values[eps][b]) == s0, (a, b, eps)


def test_common_mode_shift_can_still_change_the_cohort_on_short_clips(manifest):
    """The documented second-order route: a shift also moves who is eligible."""
    h_s = 10.0
    sizes = {
        eps: cohort_summary(
            perturb_manifest(manifest, ProtocolVector(eps_sys_s=eps, h_s=h_s)), h_s
        )["N_H"]
        for eps in (-1.0, 0.0, 1.0)
    }
    assert sizes[-1.0] >= sizes[0.0] >= sizes[1.0]
    assert sizes[-1.0] != sizes[1.0]


# --------------------------------------------------------------------------
# P-b: independent jitter
# --------------------------------------------------------------------------
def test_jitter_is_deterministic_and_independent_of_row_order(manifest):
    ids = manifest["video_id"].tolist()
    a = jitter_draws(ids, 0.25, 20260903)
    b = jitter_draws(list(reversed(ids)), 0.25, 20260903)[::-1]
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, jitter_draws(ids, 0.25, 7))
    np.testing.assert_allclose(jitter_draws(ids, 0.0, 1), np.zeros(len(ids)))


def test_sub_resolution_jitter_needs_exact_regeneration(manifest):
    """Cached matrices quantise the anchor; the library says so and offers a fix.

    A jitter standard deviation well below the cache step is invisible to a
    cached matrix, because the lookup rounds the shifted end time back onto the
    cached grid. Regenerating the block exactly on the perturbed prefix list
    resolves it. Both numbers are available; neither is silently substituted for
    the other.
    """
    h_s, delta_s, d0 = 6.0, 0.5, 3.0
    at = _table(manifest, "lock", {"d0": d0})
    pv0 = ProtocolVector(delta_s=delta_s, h_s=h_s)
    pv = ProtocolVector(delta_s=delta_s, h_s=h_s, eps_jit_sd_s=0.05)

    base = evaluate_system(manifest, at, pv0).metrics_on(None)["RMSCD@H"]
    cached = evaluate_system(manifest, at, pv)
    assert cached.lookup_quantization_s == pytest.approx(FINE / 2.0)
    # jitter of 0.05 s is far below the 0.25 s cache step: the cached reading
    # barely moves
    assert cached.metrics_on(None)["RMSCD@H"] == pytest.approx(base, abs=0.05)

    pref = make_prefixes(manifest, pv, 10.0, [h_s], FINE)
    exact_df = B.answers_for_prefixes(pref, manifest, "lock", {"d0": d0})
    exact = AnswerTable.from_frame(exact_df, "lock_exact", {"family": "block"})
    # the exact table is already indexed on the perturbed grid, so it is read at
    # the reference protocol: this isolates the grid effect of the jitter from
    # its effect on cohort membership
    res_exact = evaluate_system(manifest, exact, pv0)

    ids = list(cohort(manifest, h_s)["video_id"])
    tau = d0 - jitter_draws(ids, 0.05, pv.jit_seed)
    expected = float(np.mean(B.analytic_rmscd_per_video(tau, h_s, delta_s)))
    assert res_exact.metrics_on(None)["RMSCD@H"] == pytest.approx(expected, abs=1e-12)


def test_sub_step_jitter_randomises_rather_than_shifts_the_reading(manifest):
    """Why the two anchor perturbations must be scanned separately (P-a vs P-b).

    A common-mode shift slides the reading. Independent jitter of the same size
    does not: ``ceil`` puts each clip on one side or the other of a grid step, so
    the effect is a scatter around the reference value, not a translation by the
    mean jitter. Averaging the two kinds of anchor error into one noise level
    would hide this.
    """
    h_s, delta_s, d0 = 6.0, 0.5, 3.0
    ids = list(cohort(manifest, h_s)["video_id"])
    jit = jitter_draws(ids, 0.05, 20260903)
    base = float(np.mean(B.analytic_rmscd_per_video(np.full(len(ids), d0), h_s, delta_s)))
    jittered = float(np.mean(B.analytic_rmscd_per_video(d0 - jit, h_s, delta_s)))
    linear_guess = base - float(np.mean(jit))
    assert abs(jittered - base) > 10.0 * abs(linear_guess - base)


def test_jitter_changes_the_cohort_membership(manifest):
    """P-b's gating route: jitter moves clips across the eligibility boundary.

    Membership is what matters, not the count: swaps in both directions can leave
    ``N_H`` unchanged while the cohort is a different set of crashes.
    """
    h_s = 3.0
    a = set(cohort(perturb_manifest(manifest, ProtocolVector(h_s=h_s)), h_s)["video_id"])
    b = set(
        cohort(
            perturb_manifest(manifest, ProtocolVector(h_s=h_s, eps_jit_sd_s=0.5)), h_s
        )["video_id"]
    )
    assert a != b
    assert a.symmetric_difference(b)


# --------------------------------------------------------------------------
# commitment triple
# --------------------------------------------------------------------------
def test_commitment_triple_is_censored_at_H_and_drops_nobody():
    mf = _long_manifest(n=3)
    vids = mf["video_id"].tolist()
    y = dict(zip(mf["video_id"], mf["class_code"]))
    # commits correctly at j = 2; commits wrongly at j = 4; never commits
    preds = {
        vids[0]: [(y[vids[0]] + 1) % 5, (y[vids[0]] + 1) % 5] + [y[vids[0]]] * 5,
        vids[1]: [(y[vids[1]] + 1) % 5] * 7,
        vids[2]: [(y[vids[2]] + 1) % 5] * 7,
    }
    committed = {
        vids[0]: [False, False, True, True, True, True, True],
        vids[1]: [False, False, False, False, True, True, True],
        vids[2]: [False] * 7,
    }
    at = make_answers(preds, delta_s=0.5, committed=committed)
    res = evaluate_system(mf, at, ProtocolVector(delta_s=0.5, h_s=3.0))
    c = res.commit()
    assert c["n"] == 3  # nothing is deleted
    assert c["rho"] == pytest.approx(2.0 / 3.0)
    assert c["n_committed"] == 2
    # tau: 1.0 s, 2.0 s, and 3.0 s for the never-committing clip
    np.testing.assert_allclose(sorted(c["tau_c_per_video"]), [1.0, 2.0, 3.0])
    assert c["e_c"] == pytest.approx(0.5)  # one of the two commitments was wrong


def test_commitment_outside_the_window_counts_as_no_commitment():
    mf = _long_manifest(n=4)
    at = _table(mf, "lock", {"d0": 5.0}, commit_after_s=5.0)
    inside = evaluate_system(mf, at, ProtocolVector(delta_s=0.5, h_s=8.0)).commit()
    outside = evaluate_system(mf, at, ProtocolVector(delta_s=0.5, h_s=3.0)).commit()
    assert inside["rho"] == pytest.approx(1.0)
    assert inside["tau_c_mean"] == pytest.approx(5.0)
    assert inside["e_c"] == pytest.approx(0.0)
    assert outside["rho"] == pytest.approx(0.0)
    assert outside["tau_c_mean"] == pytest.approx(3.0)  # censored at H
    assert np.isnan(outside["e_c"])


def test_commitment_is_irrevocable_inside_the_window():
    mf = _long_manifest(n=2)
    vids = mf["video_id"].tolist()
    y = dict(zip(mf["video_id"], mf["class_code"]))
    preds = {v: [int(y[v])] * 7 for v in vids}
    committed = {vids[0]: [False, True, False, False, True, False, False],
                 vids[1]: [False] * 7}
    res = evaluate_system(
        mf, make_answers(preds, delta_s=0.5, committed=committed),
        ProtocolVector(delta_s=0.5, h_s=3.0),
    )
    assert res.committed[0].tolist() == [False] + [True] * 6
