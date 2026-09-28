"""End-to-end CLI smoke: make-prefixes -> blocks -> eval -> scan -> phenomena.

Runs the whole chain on the fixture manifest with a small bootstrap budget and
checks the shape and the internal consistency of every artefact, not just that
the commands exit zero.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from ape.cli import main as cli_main
from ape.protocol import load_protocol

from conftest import MANIFEST_MIN, PI0_PATH, REPO_ROOT

sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "fixtures"))
import make_standins  # noqa: E402


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    with open(PI0_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["bootstrap_n"] = 60  # keep the smoke fast; the real run uses the frozen value
    # This copy is edited, so it is by definition not the frozen protocol. Saying
    # frozen: false is the honest label and keeps the freeze guard meaningful --
    # the guard exists precisely to stop an edited file from claiming otherwise.
    cfg["frozen"] = False
    cfg["protocol_version"] = "smoke-test-not-frozen"
    pi_path = root / "pi_smoke.yaml"
    with open(pi_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    answers = root / "outputs" / "answers"
    answers.mkdir(parents=True)
    assert cli_main(["make-prefixes", "--pi", str(pi_path), "--manifest", MANIFEST_MIN,
                     "--out", str(root / "outputs" / "prefix_lists")]) == 0
    assert cli_main(["blocks", "--pi", str(pi_path), "--manifest", MANIFEST_MIN,
                     "--out", str(answers)]) == 0
    sys.argv = ["make_standins", "--manifest", MANIFEST_MIN, "--out", str(answers)]
    make_standins.main()
    assert cli_main(["eval", "--pi", str(pi_path), "--manifest", MANIFEST_MIN,
                     "--answers", str(answers), "--out", str(root / "outputs" / "metrics")]) == 0
    assert cli_main(["scan", "--pi", str(pi_path), "--manifest", MANIFEST_MIN,
                     "--answers", str(answers), "--out", str(root / "outputs" / "calib")]) == 0
    assert cli_main(["phenomena", "--pi", str(pi_path), "--manifest", MANIFEST_MIN,
                     "--answers", str(answers),
                     "--out", str(root / "outputs" / "phenomena")]) == 0
    assert cli_main(["report-json", "--metrics", str(root / "outputs" / "metrics"),
                     "--out", str(root / "outputs" / "report_index.json")]) == 0
    return root, load_protocol(str(pi_path))


def test_prefix_lists_written_for_every_horizon(smoke):
    root, cfg = smoke
    d = root / "outputs" / "prefix_lists"
    csvs = sorted(p for p in os.listdir(d) if p.endswith(".csv"))
    assert len(csvs) == len(cfg.h_list_s)
    df = pd.read_csv(d / csvs[0])
    assert {"video_id", "j", "delta_s", "end_s", "end_frame", "is_post_anchor",
            "base_j"} <= set(df.columns)
    assert any(c.startswith("eligible_H") for c in df.columns)


def test_blocks_written_and_readable(smoke):
    root, cfg = smoke
    d = root / "outputs" / "answers"
    made = [p for p in os.listdir(d) if p.startswith("block__")]
    assert len(made) == len(cfg.block_factors)
    one = pd.read_csv(d / made[0] / "answers.csv")
    assert {"video_id", "j", "delta_s", "pred", "p0", "committed"} <= set(one.columns)
    assert one["j"].min() < 0  # pre-anchor columns exist


def test_metrics_json_is_complete_and_self_consistent(smoke):
    root, cfg = smoke
    for h in cfg.h_list_s:
        pv = cfg.pi0(h)
        d = root / "outputs" / "metrics" / pv.pi_hash
        files = [f for f in os.listdir(d) if f.endswith(".json") and not f.startswith("_")]
        assert files
        for f in files:
            with open(d / f, "r", encoding="utf-8") as fh:
                m = json.load(fh)
            assert m["h_s"] == pytest.approx(h)
            assert len(m["S_H"]) == len(m["grid_offsets_s"])
            assert m["N_H"] > 0
            assert 0.0 <= m["RMSCD"] <= h + 1e-9
            # the reported area must match the reported curve
            s = np.asarray(m["S_H"], dtype=float)
            dx = m["delta_s"]
            expect = dx * ((1 - s).sum() - 0.5 * ((1 - s[0]) + (1 - s[-1])))
            assert m["RMSCD"] == pytest.approx(expect)
            # keys the reporting stage consumes
            assert m["track"] and m["family"]
            assert m["effective_h_s"] <= m["h_s"] + 1e-9
            assert set(m["alt_cohorts"]) >= {
                "per_clip_end", "dynamic_denominator", "fixed_H_cohort",
                "length_stratified_change", "conclusion_stable",
            }
            assert set(m["phenomena"]) >= {
                "flip_rate", "last_flip_median_s", "correct_wrong_correct",
                "pre_anchor_overconfidence", "K1", "K2", "K3", "K5", "K6",
            }
            assert m["commit"]["tau_c"] == m["commit"]["tau_c_mean"]
            assert m["alt_cohorts"]["fixed_H_cohort"]["RMSCD"] == pytest.approx(m["RMSCD"])
            assert set(m["frozen_family"]) == {
                "RMSCD@H", "end_window_macro_acc", "median_flips"
            } | {f"S_H@{k:g}" for k in cfg.s_report_delta_s}
            for name, b in m["bootstrap"].items():
                if np.isfinite(b["ci_lo"]) and np.isfinite(b["ci_hi"]):
                    assert b["ci_lo"] <= b["ci_hi"]


def test_oracle_and_random_blocks_land_at_the_two_extremes(smoke):
    root, cfg = smoke
    pv = cfg.pi0()
    d = root / "outputs" / "metrics" / pv.pi_hash
    vals = {}
    for f in os.listdir(d):
        if f.endswith(".json") and not f.startswith("_"):
            with open(d / f, "r", encoding="utf-8") as fh:
                m = json.load(fh)
            vals[m["system_id"]] = m["RMSCD"]
    oracle = vals["block__oracle__default"]
    noisy = vals["block__rand__eta-0.5"]
    assert oracle == pytest.approx(0.0)
    assert noisy > oracle
    # A coin-flip block almost never stays correct to the window end, so its
    # delay sits just under the full window. Bounding it by the window rather
    # than by a fixed number keeps the test honest on a small cohort, where the
    # exact value is sampling noise.
    eff_h = cfg.delta_s * int(cfg.h_ref_s / cfg.delta_s)
    assert 0.85 * eff_h <= noisy <= eff_h + 1e-9


def test_pairs_file_separates_real_systems_from_blocks(smoke):
    root, cfg = smoke
    pv = cfg.pi0()
    with open(root / "outputs" / "metrics" / pv.pi_hash / "_pairs.json",
              "r", encoding="utf-8") as fh:
        p = json.load(fh)
    assert all(s.startswith("block__") for s in p["systems_block"])
    assert all(not s.startswith("block__") for s in p["systems_real"])
    assert len(p["systems_real"]) == 4  # the stand-ins
    assert "RMSCD@H" in p
    assert isinstance(p["RMSCD@H"]["significant_pairs"], list)


def test_calibration_json_has_every_frozen_metric(smoke):
    root, cfg = smoke
    pv = cfg.pi0()
    d = root / "outputs" / "calib" / pv.pi_hash
    scan = pd.read_csv(d / "scan_table.csv")
    assert set(scan["metric"].unique()) == {
        "RMSCD@H", "end_window_macro_acc", "median_flips"
    } | {f"S_H@{k:g}" for k in cfg.s_report_delta_s}
    with open(d / "calibration.json", "r", encoding="utf-8") as fh:
        cal = json.load(fh)
    assert cal["pi0_hash"] == pv.pi_hash
    for metric in ["RMSCD@H", "end_window_macro_acc", "median_flips"]:
        assert metric in cal["metrics"]
        assert f"{metric} [with blocks]" in cal["metrics"]
        entry = cal["metrics"][metric]
        assert "verdict" in entry and "MRD_plaus" in entry and "eps_max" in entry
        # short aliases for the table-3 columns
        assert {"reference_range", "max_b", "max_s", "min_R", "MRD"} <= set(entry)
        assert entry["max_b"] == entry["max_abs_b_M_plaus"]
        assert entry["min_R"] == entry["min_R_M_plaus"]
    assert cal["delta_star"]


def test_scan_reproduces_the_reference_row_of_the_eval_stage(smoke):
    root, cfg = smoke
    pv = cfg.pi0()
    scan = pd.read_csv(root / "outputs" / "calib" / pv.pi_hash / "scan_table.csv")
    ref = scan.loc[(scan["pi_hash"] == pv.pi_hash) & (scan["metric"] == "RMSCD@H")]
    d = root / "outputs" / "metrics" / pv.pi_hash
    for _, row in ref.iterrows():
        with open(d / f"{row['system_id']}.json", "r", encoding="utf-8") as fh:
            m = json.load(fh)
        assert m["RMSCD"] == pytest.approx(row["value"])


def test_phenomena_outputs_a_table_row_per_system(smoke):
    root, cfg = smoke
    pv = cfg.pi0()
    d = root / "outputs" / "phenomena" / pv.pi_hash
    tbl = pd.read_csv(d / "table5.csv")
    assert len(tbl) == len(cfg.block_factors) + 4
    assert tbl["Flip rate"].between(0.0, 1.0).all()


def test_report_index_collects_every_metric_file(smoke):
    root, cfg = smoke
    with open(root / "outputs" / "report_index.json", "r", encoding="utf-8") as fh:
        idx = json.load(fh)
    assert idx["n_rows"] == (len(cfg.block_factors) + 4) * len(cfg.h_list_s)
    assert os.path.exists(root / "outputs" / "report_index.csv")
