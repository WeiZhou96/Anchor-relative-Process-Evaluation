"""The S1 freeze guard: a frozen protocol file may not drift from its digest.

After S1 the protocol is fixed. The danger is not a deliberate change -- that is
allowed, loudly, via ``ape.cli freeze --force`` -- but a quiet one: an edited
file that still claims ``frozen: true`` would silently change every ``pi_hash``
and invalidate every cached artefact while looking untouched. These tests pin
the refusal, the field-level diff, and the fact that the real ``protocol/pi0.yaml``
is currently frozen and self-consistent.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest
import yaml

from ape.cli import main as cli_main
from ape.protocol import (
    ProtocolFrozenError,
    content_sha256,
    diff_against_snapshot,
    frozen_hash_path,
    frozen_snapshot_path,
    load_protocol,
    load_protocol_checked,
    read_frozen_hash,
    verify_frozen,
    write_freeze,
)

from conftest import MANIFEST_MIN, PI0_PATH


def _copy_protocol(tmp_path, **overrides):
    with open(PI0_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.update(overrides)
    p = tmp_path / "pi.yaml"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return str(p)


# --------------------------------------------------------------------------
# the shipped protocol
# --------------------------------------------------------------------------
def test_shipped_protocol_is_frozen_and_matches_its_digest():
    cfg = load_protocol(PI0_PATH)
    assert cfg.frozen is True
    assert cfg.protocol_version == "1.0-S1-2026-09-03"
    assert os.path.exists(frozen_hash_path(PI0_PATH))
    assert os.path.exists(frozen_snapshot_path(PI0_PATH))
    info = verify_frozen(cfg)
    assert info["checked"] is True
    assert info["sha256"] == content_sha256(PI0_PATH)
    assert read_frozen_hash(PI0_PATH) == content_sha256(PI0_PATH)


def test_frozen_values_are_the_S1_decision():
    cfg = load_protocol(PI0_PATH)
    assert cfg.delta_s == 0.5
    assert cfg.h_list_s == [4.0, 10.0, 21.5]
    assert cfg.h_ref_s == 10.0
    assert cfg.grid_max_s == 21.5
    assert cfg.r0 == 0.9
    assert cfg.bootstrap_n == 1000
    assert cfg.alpha_pairs == 0.05
    assert cfg.s_report_delta_s == [1.0, 3.0]
    assert cfg.perturb["delta_s"] == [0.25, 0.5, 1.0]
    assert cfg.raw["frozen_by"] == "authors"
    assert cfg.raw["frozen_at"]
    # provenance of the inputs the freeze was taken against, from track C's
    # prereg artefact and independently recomputed on the server
    assert cfg.raw["manifest_md5"] == "b7a976594deeaed8cfff0ea1630dbd96"
    assert cfg.raw["dev_hash"] == (
        "31ed09e84d2c12ffa37786ed1810b9d4932bb8063ee36d1497cff48252025565"
    )
    assert cfg.raw["prereg_file"] == "prereg/S1_freeze_2026-09-03.yaml"
    # the audit set is the test split and the config says so
    assert cfg.raw["audit_split"] == "test"
    # the perturbation H axis must be the frozen horizons, not the raw quantiles
    assert cfg.perturb["h_s"] == [4.0, 10.0, 21.5]
    assert cfg.perturb["plaus"]["h_s"] == [4.0, 21.5]


def test_every_frozen_horizon_is_a_whole_number_of_steps():
    """H must land on the grid, so the integrated window equals H exactly."""
    cfg = load_protocol(PI0_PATH)
    for h in cfg.h_list_s:
        assert abs(h / cfg.delta_s - round(h / cfg.delta_s)) < 1e-9
        assert h <= cfg.grid_max_s + 1e-9


def test_the_three_frozen_pi_hashes_are_stable():
    """If this test fails, every cached artefact under outputs/ is stale."""
    cfg = load_protocol(PI0_PATH)
    got = {h: cfg.pi0(h).pi_hash for h in cfg.h_list_s}
    assert got == {
        4.0: "77200b351bf1",
        10.0: "8ac32aae418b",
        21.5: "3fb251dc210c",
    }


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------
def test_a_modified_frozen_file_is_refused_with_a_field_level_diff(tmp_path):
    p = _copy_protocol(tmp_path, frozen=True)
    write_freeze(p)
    verify_frozen(load_protocol(p))  # clean

    cfg = yaml.safe_load(open(p, encoding="utf-8"))
    cfg["r0"] = 0.8
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    with pytest.raises(ProtocolFrozenError) as exc:
        load_protocol_checked(p)
    msg = str(exc.value)
    assert "FROZEN PROTOCOL MODIFIED" in msg
    assert "r0" in msg and "0.9" in msg and "0.8" in msg
    assert any("r0" in line for line in diff_against_snapshot(p))


def test_a_frozen_file_without_a_recorded_digest_is_refused(tmp_path):
    p = _copy_protocol(tmp_path, frozen=True)
    with pytest.raises(ProtocolFrozenError) as exc:
        load_protocol_checked(p)
    assert "is missing" in str(exc.value)


def test_an_unfrozen_file_is_never_checked(tmp_path):
    p = _copy_protocol(tmp_path, frozen=False)
    info = verify_frozen(load_protocol(p))
    assert info == {"frozen": False, "checked": False}
    load_protocol_checked(p)  # must not raise even with no digest recorded


def test_line_ending_changes_alone_do_not_trip_the_guard(tmp_path):
    p = _copy_protocol(tmp_path, frozen=True)
    write_freeze(p)
    raw = open(p, "rb").read()
    open(p, "wb").write(raw.replace(b"\n", b"\r\n"))
    verify_frozen(load_protocol(p))  # CRLF is not a protocol change


def test_cli_refuses_to_run_on_a_tampered_frozen_protocol(tmp_path, capsys):
    p = _copy_protocol(tmp_path, frozen=True)
    write_freeze(p)
    cfg = yaml.safe_load(open(p, encoding="utf-8"))
    cfg["delta_s"] = 0.25
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    rc = cli_main(["make-prefixes", "--pi", p, "--manifest", MANIFEST_MIN,
                   "--out", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "FROZEN PROTOCOL MODIFIED" in err
    assert "delta_s" in err
    assert not os.path.exists(tmp_path / "out" / "index.json")


def test_freeze_subcommand_refuses_to_overwrite_without_force(tmp_path, capsys):
    p = _copy_protocol(tmp_path, frozen=True)
    assert cli_main(["freeze", "--pi", p]) == 0
    first = read_frozen_hash(p)

    cfg = yaml.safe_load(open(p, encoding="utf-8"))
    cfg["seed"] = 1
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    assert cli_main(["freeze", "--pi", p]) == 1
    assert read_frozen_hash(p) == first  # unchanged
    out = capsys.readouterr().out
    assert "seed" in out and "--force" in out

    assert cli_main(["freeze", "--pi", p, "--force"]) == 0
    assert read_frozen_hash(p) != first


def test_bootstrap_override_on_a_frozen_protocol_is_announced(tmp_path, capsys):
    from ape.cli import _resolve_bootstrap_n

    cfg = load_protocol(PI0_PATH)
    assert _resolve_bootstrap_n(cfg, None) == 1000
    assert _resolve_bootstrap_n(cfg, 200) == 200
    out = capsys.readouterr().out
    assert "FROZEN" in out and "1000" in out and "200" in out
