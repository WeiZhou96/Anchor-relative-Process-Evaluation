"""Cluster bootstrap, Holm correction, and the characterisation quantities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ape import calib as C
from ape import stats as S
from ape.cohort import cluster_codes


def test_holm_matches_the_worked_example():
    p = [0.01, 0.04, 0.03, 0.005]
    out = S.holm(p, alpha=0.05)
    # sorted: .005 (x4), .01 (x3), .03 (x2), .04 (x1), then made monotone
    np.testing.assert_allclose(out["adjusted"], [0.03, 0.06, 0.06, 0.02])
    assert out["reject"].tolist() == [True, False, False, True]


def test_holm_is_monotone_and_never_exceeds_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50)
    adj = S.holm(p)["adjusted"]
    assert (adj <= 1.0 + 1e-12).all()
    order = np.argsort(p)
    assert (np.diff(adj[order]) >= -1e-12).all()


def test_bootstrap_resamples_whole_clusters():
    """A cluster is drawn as a block: all of its rows come along or none do."""
    codes = np.array([0, 0, 0, 1, 1, 2])
    reps = S.cluster_bootstrap_indices(codes, 50, seed=1)
    for idx in reps:
        counts = np.bincount(codes[idx], minlength=3)
        assert counts[0] % 3 == 0  # cluster 0 has three rows
        assert counts[1] % 2 == 0  # cluster 1 has two
        assert len(idx) == counts.sum()
        # exactly three clusters are drawn each time (with replacement)
        assert counts[0] // 3 + counts[1] // 2 + counts[2] == 3
    lengths = {len(i) for i in reps}
    assert len(lengths) > 1  # unequal cluster sizes make replicate sizes vary


def test_cluster_bootstrap_is_seed_reproducible():
    codes = np.repeat(np.arange(10), 3)
    a = S.cluster_bootstrap_indices(codes, 20, seed=7)
    b = S.cluster_bootstrap_indices(codes, 20, seed=7)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_paired_bootstrap_separates_a_real_gap_and_not_a_zero_gap():
    rng = np.random.default_rng(3)
    n_clusters, per = 40, 2
    codes = np.repeat(np.arange(n_clusters), per)
    base = rng.normal(0, 1, n_clusters * per)
    a = base
    b = base + 1.5  # a large, consistent, paired difference
    same = S.paired_bootstrap_diff(
        codes, lambda idx: float(a[idx].mean() - a[idx].mean()), 500, seed=1
    )
    diff = S.paired_bootstrap_diff(
        codes, lambda idx: float(a[idx].mean() - b[idx].mean()), 500, seed=1
    )
    assert same["p_value"] > 0.5
    assert diff["p_value"] < 0.05
    assert diff["ci_hi"] < 0


def test_significant_pairs_applies_the_correction():
    """A large consistent gap separates; independent noise of the same size does not."""
    codes = np.repeat(np.arange(30), 2)
    rng = np.random.default_rng(5)
    base = rng.normal(0, 1, 60)
    noise = rng.normal(0, 1, 60)  # unrelated to base: the paired difference is centred
    fns = {
        "sysA": lambda idx: float(base[idx].mean()),
        "sysB": lambda idx: float(base[idx].mean() + 2.0),
        "sysC": lambda idx: float(noise[idx].mean()),
    }
    pairs, tests = S.significant_pairs(fns, codes, "m", 400, seed=2, alpha=0.05)
    named = {(t.system_a, t.system_b): t for t in tests}
    assert named[("sysA", "sysB")].significant
    assert not named[("sysA", "sysC")].significant
    assert all(t.p_adjusted >= t.p_value - 1e-12 for t in tests)
    assert set(pairs) <= set(named)


def test_a_tiny_but_perfectly_consistent_gap_is_still_significant():
    """The paired test is about consistency, not about effect size.

    Two systems separated by a constant 0.0001 on every cluster are separated by
    the test. That is correct behaviour and it is precisely why ``MRD_M`` exists:
    significance says the ordering is real, ``MRD_M`` says whether the gap is
    large enough to survive the protocol's own wobble.
    """
    codes = np.repeat(np.arange(30), 2)
    base = np.random.default_rng(5).normal(0, 1, 60)
    fns = {
        "sysA": lambda idx: float(base[idx].mean()),
        "sysC": lambda idx: float(base[idx].mean() + 1e-4),
    }
    _, tests = S.significant_pairs(fns, codes, "m", 400, seed=2, alpha=0.05)
    assert tests[0].significant
    assert abs(tests[0].diff) == pytest.approx(1e-4, rel=1e-6)


def test_percentile_p_cannot_survive_holm_on_a_large_library():
    """Regression: an empty P must be diagnosed, not mistaken for a lost ordering.

    A percentile bootstrap cannot report a p below 1/n_boot. Holm multiplies the
    smallest p by the number of pairs. With 190 pairs and 200 replicates the
    best reachable adjusted p is 190/200 = 0.95, so *no* pair can ever be
    significant, P comes back empty and every R_M is NaN -- which reads exactly
    like "the protocol destroyed every ordering". The guard makes that visible.
    """
    g = S.correction_resolution_ok(n_boot=200, n_pairs=190, alpha=0.05)
    assert g["percentile_p_floor"] == pytest.approx(0.005)
    assert g["smallest_adjusted_p_reachable"] == pytest.approx(0.95)
    assert g["percentile_ok"] is False
    assert g["n_boot_needed_for_percentile"] == 3800
    # the frozen budget of 1000 is still not enough for a 20 system library
    assert S.correction_resolution_ok(1000, 190, 0.05)["percentile_ok"] is False
    # a small library is fine
    assert S.correction_resolution_ok(1000, 6, 0.05)["percentile_ok"] is True


def test_normal_p_has_no_resolution_floor_and_recovers_significance():
    """The normal-approximation p is what makes a corrected test usable here."""
    codes = np.repeat(np.arange(40), 2)
    rng = np.random.default_rng(11)
    base = rng.normal(0, 1, 80)
    # 20 systems -> 190 pairs, exactly the situation above
    point = {f"s{i:02d}": float(i) for i in range(20)}
    reps = S.cluster_bootstrap_indices(codes, 200, seed=3)
    samples = {
        f"s{i:02d}": np.array([float(base[idx].mean() + i) for idx in reps])
        for i in range(20)
    }
    pairs_pct, tests_pct = S.significant_pairs_from_samples(
        point, samples, "m", 0.05, "holm", p_method="percentile"
    )
    pairs_nrm, tests_nrm = S.significant_pairs_from_samples(
        point, samples, "m", 0.05, "holm", p_method="normal"
    )
    assert len(tests_pct) == 190
    assert len(pairs_pct) == 0  # the floor makes every corrected p fail
    assert len(pairs_nrm) > 100  # the systems are separated by whole units
    # both p variants are recorded on every test either way
    assert all(np.isfinite(t.p_percentile) and np.isfinite(t.p_normal) for t in tests_nrm)
    assert tests_nrm[0].p_method == "normal"


def test_tied_at_window_end_uses_interval_overlap():
    ci = {"a": (0.50, 0.60), "b": (0.55, 0.65), "c": (0.80, 0.90)}
    assert S.tied_at_window_end(ci) == [("a", "b")]


# --------------------------------------------------------------------------
# characterisation
# --------------------------------------------------------------------------
def _scan_table():
    """Three systems, four protocol points, one metric; ranks flip only at pi3."""
    rows = []
    values = {
        "pi0": {"s1": 1.0, "s2": 2.0, "s3": 3.0},
        "pi1": {"s1": 1.5, "s2": 2.5, "s3": 3.5},  # pure shift, ranks kept
        "pi2": {"s1": 1.1, "s2": 1.9, "s3": 3.2},  # spread, ranks kept
        "pi3": {"s1": 3.0, "s2": 2.0, "s3": 1.0},  # ranks fully reversed
    }
    meta = {
        "pi0": (0.0, 0.0, 0.5, 6.0, True),
        "pi1": (0.25, 0.0, 0.5, 6.0, True),
        "pi2": (-0.25, 0.0, 0.5, 6.0, True),
        "pi3": (1.0, 0.0, 0.5, 6.0, False),
    }
    for pi, vals in values.items():
        e, j, d, h, plaus = meta[pi]
        for sid, v in vals.items():
            rows.append(
                {
                    "system_id": sid,
                    "is_block": False,
                    "pi_hash": pi,
                    "eps_sys_s": e,
                    "eps_jit_sd_s": j,
                    "delta_s": d,
                    "h_s": h,
                    "in_plaus": plaus,
                    "metric": "RMSCD@H",
                    "value": v,
                }
            )
    return pd.DataFrame(rows, columns=C.SCAN_COLUMNS)


def test_offset_and_spread_separate_a_shift_from_a_scatter():
    scan = _scan_table()
    bs = C.offsets_and_spread(scan, "pi0", "RMSCD@H").set_index("pi_hash")
    assert bs.loc["pi0", "b_M"] == pytest.approx(0.0)
    assert bs.loc["pi1", "b_M"] == pytest.approx(0.5)
    assert bs.loc["pi1", "s_M"] == pytest.approx(0.0)  # a pure shift has no scatter
    assert bs.loc["pi2", "s_M"] > 0.0


def test_rank_preservation_and_the_comparability_region():
    scan = _scan_table()
    pairs = [("s1", "s2"), ("s1", "s3"), ("s2", "s3")]
    rt = C.rank_preservation(scan, "pi0", "RMSCD@H", pairs).set_index("pi_hash")
    assert rt.loc["pi0", "R_M"] == pytest.approx(1.0)
    assert rt.loc["pi1", "R_M"] == pytest.approx(1.0)
    assert rt.loc["pi3", "R_M"] == pytest.approx(0.0)
    region = C.comparability_region(rt.reset_index(), r0=0.9).set_index("pi_hash")
    assert bool(region.loc["pi1", "in_region"])
    assert not bool(region.loc["pi3", "in_region"])


def test_rank_preservation_counts_a_new_tie_as_a_loss():
    scan = _scan_table()
    scan.loc[
        (scan["pi_hash"] == "pi1") & (scan["system_id"] == "s2"), "value"
    ] = 1.5  # s1 and s2 now equal at pi1
    rt = C.rank_preservation(scan, "pi0", "RMSCD@H", [("s1", "s2")]).set_index("pi_hash")
    assert rt.loc["pi1", "R_M"] == pytest.approx(0.0)


def test_mrd_is_the_95th_percentile_of_the_plausible_shifts():
    scan = _scan_table()
    m = C.mrd(scan, "pi0", "RMSCD@H", plaus_only=True)
    # plausible shifts are |0.5| x3 and |0.1|,|0.1|,|0.2|
    assert m["n"] == 6
    assert m["MRD"] == pytest.approx(np.percentile([0.5, 0.5, 0.5, 0.1, 0.1, 0.2], 95))
    m_all = C.mrd(scan, "pi0", "RMSCD@H", plaus_only=False)
    assert m_all["n"] == 9 and m_all["MRD"] >= m["MRD"]


def _scan_table_two_horizons():
    """pi0 at H=10, one shift point at H=10, one horizon point at H=22.

    RMSCD is bounded by H, so the H=22 column sits on a different scale.
    """
    rows = []
    for pi, (eps, h, vals) in {
        "pi0": (0.0, 10.0, {"s1": 3.0, "s2": 4.0}),
        "eps": (0.25, 10.0, {"s1": 3.1, "s2": 4.1}),
        "hlong": (0.0, 22.0, {"s1": 12.0, "s2": 14.0}),
    }.items():
        for sid, v in vals.items():
            rows.append({"system_id": sid, "is_block": False, "pi_hash": pi,
                         "eps_sys_s": eps, "eps_jit_sd_s": 0.0, "delta_s": 0.5,
                         "h_s": h, "in_plaus": True, "metric": "RMSCD@H",
                         "value": v})
    return pd.DataFrame(rows, columns=C.SCAN_COLUMNS)


def test_mrd_does_not_mix_horizons_for_an_h_scaled_metric():
    """Regression: RMSCD at H=22 is not a moved reading of RMSCD at H=10.

    Folding the horizon axis into MRD inflated it to several seconds and made
    every metric look incomparable. The horizon spread is still reported, under
    its own key, because it is a change of scale rather than a measurement error.
    """
    scan = _scan_table_two_horizons()
    assert C.is_h_scaled("RMSCD@H")
    m = C.mrd(scan, "pi0", "RMSCD@H")
    assert m["same_h_only"] is True
    # only the eps point contributes: shifts of 0.1 and 0.1
    assert m["n"] == 2
    assert m["MRD"] == pytest.approx(0.1)
    # the horizon axis is measured and reported separately, not silently dropped
    assert m["h_axis_shift"] == pytest.approx(np.percentile([9.0, 10.0], 95))
    # opting in explicitly restores the old, scale-mixing behaviour
    mixed = C.mrd(scan, "pi0", "RMSCD@H", same_h_only=False)
    assert mixed["n"] == 4 and mixed["MRD"] > 5.0


def test_mrd_still_uses_every_axis_for_a_scale_free_metric():
    scan = _scan_table_two_horizons()
    scan = scan.assign(metric="end_window_macro_acc")
    assert not C.is_h_scaled("end_window_macro_acc")
    m = C.mrd(scan, "pi0", "end_window_macro_acc")
    assert m["same_h_only"] is False
    assert m["n"] == 4  # the horizon axis is legitimately comparable here


def test_eps_max_stops_at_the_first_failing_shell():
    rt = pd.DataFrame(
        {
            "pi_hash": ["a", "b", "c", "d", "e"],
            "eps_sys_s": [0.0, 0.25, -0.25, 0.5, -0.5],
            "eps_jit_sd_s": 0.0,
            "delta_s": 0.5,
            "h_s": 6.0,
            "in_plaus": True,
            "metric": "m",
            "n_pairs": 3,
            "R_M": [1.0, 0.95, 0.95, 0.80, 1.0],
        }
    )
    out = C.eps_max(rt, r0=0.9, pi0_hash="a")
    assert out["eps_max"] == pytest.approx(0.25)
    assert out["saturated"] is False
    assert out["scanned_max_abs_eps"] == pytest.approx(0.5)


def test_eps_max_finds_the_shift_axis_in_a_full_axis_scan():
    """Regression: the shift axis must not be filtered away by the delta/H axes.

    An axis scan has one ``eps == 0`` row per step and per horizon. Picking the
    reference (delta, H) from whichever of those rows comes first leaves a single
    point on the shift axis and silently reports ``eps_max = 0`` with nothing
    scanned. The reference must come from ``pi0_hash``.
    """
    rows = []

    def add(pi_hash, eps, d, h, r):
        rows.append({"pi_hash": pi_hash, "eps_sys_s": eps, "eps_jit_sd_s": 0.0,
                     "delta_s": d, "h_s": h, "in_plaus": True, "metric": "m",
                     "n_pairs": 3, "R_M": r})

    add("pi0", 0.0, 0.5, 10.03, 1.0)
    for eps, r in [(-1.0, 0.60), (-0.5, 0.95), (-0.25, 1.0),
                   (0.25, 1.0), (0.5, 0.95), (1.0, 0.70)]:
        add(f"eps{eps}", eps, 0.5, 10.03, r)
    # the other axes also sit at eps == 0 and must not be mistaken for it;
    # these are deliberately ordered before pi0 to reproduce the original failure
    add("dfine", 0.0, 0.25, 10.03, 1.0)
    add("dcoarse", 0.0, 1.0, 10.03, 1.0)
    add("hshort", 0.0, 0.5, 4.11, 1.0)
    add("hlong", 0.0, 0.5, 21.82, 1.0)
    rt = pd.DataFrame(rows).sort_values("pi_hash").reset_index(drop=True)

    out = C.eps_max(rt, r0=0.9, pi0_hash="pi0")
    assert out["scanned_max_abs_eps"] == pytest.approx(1.0)  # the axis was found
    assert out["n_points"] == 4  # |eps| in {0, 0.25, 0.5, 1.0}
    assert out["eps_max"] == pytest.approx(0.5)  # +-1.0 drops R below r0
    assert out["saturated"] is False
    assert out["reference_delta_s"] == pytest.approx(0.5)
    assert out["reference_h_s"] == pytest.approx(10.03)


def test_eps_max_reports_saturation_when_the_whole_grid_holds():
    rt = pd.DataFrame(
        {
            "pi_hash": ["a", "b", "c"],
            "eps_sys_s": [0.0, 0.25, -0.25],
            "eps_jit_sd_s": 0.0,
            "delta_s": 0.5,
            "h_s": 6.0,
            "in_plaus": True,
            "metric": "m",
            "n_pairs": 3,
            "R_M": [1.0, 1.0, 1.0],
        }
    )
    out = C.eps_max(rt, r0=0.9, pi0_hash="a")
    assert out["eps_max"] == pytest.approx(0.25)
    assert out["saturated"] is True


def test_delta_star_finds_the_saturation_step():
    out = C.delta_star({1.0: 2.0, 0.5: 4.0, 0.25: 4.1, 0.125: 4.1}, rel_tol=0.05)
    assert out["delta_star"] == pytest.approx(0.5)
    assert out["saturated"] is True


def test_delta_star_reports_no_saturation_when_still_growing():
    out = C.delta_star({1.0: 2.0, 0.5: 4.0, 0.25: 8.0}, rel_tol=0.05)
    assert out["delta_star"] == pytest.approx(0.25)
    assert out["saturated"] is False  # only the finest step qualifies: not saturated


def test_verdict_rules():
    assert C.verdict(0.5, 0.01, 0.9, 0.5) == "not comparable"
    assert C.verdict(0.95, 0.01, 0.9, 0.5) == "poolable"
    assert C.verdict(0.95, 0.9, 0.9, 0.5) == "normalize"
    assert C.verdict(0.95, 0.01, 0.9, None).startswith("undetermined")


def test_calibrate_metric_end_to_end():
    scan = _scan_table()
    out = C.calibrate_metric(
        scan, "pi0", "RMSCD@H", [("s1", "s2"), ("s2", "s3")], r0=0.9, ruler=0.8
    )
    assert out["min_R_M_plaus"] == pytest.approx(1.0)
    assert out["min_R_M_full"] == pytest.approx(0.0)
    assert out["verdict"] == "poolable"
    assert out["MRD_plaus"] <= out["MRD_full_grid"]
