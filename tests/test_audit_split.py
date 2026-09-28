"""The audit set is the frozen test split, and every output says which split it is.

This is a protocol-integrity rule, not a convenience. The horizons, the step
candidates, the perturbation bounds and every system hyper-parameter were chosen
on train/dev; a number computed there is a number about the data the protocol was
tuned on, and it would look identical to an audit number once written to json.
Hence: restrict before any cohort is formed, and stamp ``audit_split`` on
everything.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from ape.cli import main as cli_main
from ape.cohort import cohort_summary, select_split, split_summary
from ape.protocol import load_manifest, load_protocol

from conftest import MANIFEST_MIN, PI0_PATH


@pytest.fixture(scope="module")
def full_manifest():
    return load_manifest(MANIFEST_MIN)


def test_select_split_restricts_before_any_cohort_is_formed(full_manifest):
    test = select_split(full_manifest, "test")
    assert len(test) < len(full_manifest)
    assert set(test["split"].astype(str)) == {"test"}
    # the cohort formed on the restricted manifest is smaller than the global one
    for h in (4.0, 10.0, 21.5):
        assert cohort_summary(test, h)["N_H"] < cohort_summary(full_manifest, h)["N_H"]


def test_select_split_defaults_to_test(full_manifest):
    assert select_split(full_manifest).equals(select_split(full_manifest, "test"))


def test_select_split_rejects_an_unknown_or_empty_split(full_manifest):
    with pytest.raises(ValueError, match="empty"):
        select_split(full_manifest, "nosuchsplit")


def test_select_split_refuses_a_manifest_without_a_split_column(full_manifest):
    """Silently scoring every clip would be the worst possible default."""
    m = full_manifest.drop(columns=["split"])
    with pytest.raises(ValueError, match="no 'split' column"):
        select_split(m, "test")
    # 'all' is the explicit opt-out and stays available for diagnostics
    assert len(select_split(m, "all")) == len(m)


def test_fixture_splits_never_straddle_a_cluster(full_manifest):
    """Mirrors the leak repair in manifest v2: a source video stays in one split."""
    straddle = full_manifest.groupby("source_cluster_id")["split"].nunique()
    assert int((straddle > 1).sum()) == 0


def test_eligibility_survives_floating_point_subtraction():
    """The real boundary case: a clip with exactly H seconds must stay eligible.

    Manifest v2 clip ``KUBbn-T3XYI_00`` has anchor 12.08 and duration 22.08, so
    it holds exactly 10.0 s after the anchor. ``22.08 - 12.08`` is
    ``9.999999999999998`` in binary floating point, and a strict ``>= H`` drops
    it -- making cohort membership a property of the arithmetic rather than of
    the video. This is why eligibility carries a 1e-9 slack.
    """
    from conftest import make_manifest

    anchor, duration = 12.08, 22.08
    post = duration - anchor
    assert post < 10.0  # the artefact this test exists for
    m = make_manifest([{"video_id": "boundary", "anchor_s": anchor, "post_s": post}])
    assert cohort_summary(m, 10.0)["N_H"] == 1
    # and a clip that is genuinely short is still excluded: the slack is 1e-9,
    # far below one frame at the fastest source rate (0.02 s at 50 fps)
    short = make_manifest([{"video_id": "short", "anchor_s": 1.0, "post_s": 9.99}])
    assert cohort_summary(short, 10.0)["N_H"] == 0


def test_split_summary_reports_every_split(full_manifest):
    s = split_summary(full_manifest)
    assert set(s) == {"train", "dev", "test"}
    assert sum(s.values()) == len(full_manifest)


# --------------------------------------------------------------------------
# end to end through the CLI
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def unfrozen_pi(tmp_path_factory):
    root = tmp_path_factory.mktemp("splitpi")
    with open(PI0_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["frozen"] = False
    cfg["protocol_version"] = "split-test-not-frozen"
    cfg["bootstrap_n"] = 30
    p = root / "pi.yaml"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return str(p), root


def _run(pi, root, split, tag):
    answers = root / f"answers_{tag}"
    answers.mkdir(exist_ok=True)
    assert cli_main(["blocks", "--pi", pi, "--manifest", MANIFEST_MIN,
                     "--out", str(answers)]) == 0
    out = root / f"metrics_{tag}"
    args = ["eval", "--pi", pi, "--manifest", MANIFEST_MIN, "--answers",
            str(answers), "--out", str(out), "--h", "10.0"]
    if split is not None:
        args += ["--split", split]
    assert cli_main(args) == 0
    cfg = load_protocol(pi)
    d = out / cfg.pi0(10.0).pi_hash
    f = next(x for x in os.listdir(d) if x.endswith(".json") and not x.startswith("_"))
    with open(d / f, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_eval_defaults_to_the_test_split_and_stamps_it(unfrozen_pi, full_manifest):
    pi, root = unfrozen_pi
    m = _run(pi, root, None, "default")
    assert m["audit_split"] == "test"
    expected = cohort_summary(select_split(full_manifest, "test"), 10.0)["N_H"]
    assert m["N_H"] == expected
    # and that is strictly fewer clips than the whole manifest would give
    assert expected < cohort_summary(full_manifest, 10.0)["N_H"]


def test_dev_split_is_available_but_labelled(unfrozen_pi, full_manifest):
    pi, root = unfrozen_pi
    m = _run(pi, root, "dev", "dev")
    assert m["audit_split"] == "dev"
    assert m["N_H"] == cohort_summary(select_split(full_manifest, "dev"), 10.0)["N_H"]


def test_blocks_are_generated_for_every_clip_not_just_the_audit_split(unfrozen_pi,
                                                                     full_manifest):
    """The gauge must exist wherever it might later be read."""
    pi, root = unfrozen_pi
    answers = root / "answers_default"
    df = pd.read_csv(answers / "block__oracle__default" / "answers.csv",
                     usecols=["video_id"])
    assert df["video_id"].nunique() == len(full_manifest)
