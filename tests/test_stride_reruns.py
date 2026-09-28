"""Per-step re-runs for systems that are not sub-sampling equivalent.

Contract 5.4 allows coarse prefix steps to be obtained by sub-sampling the
finest-step answer matrix, but only when the system has no state dependence on
the prefix end point (idea 3.1 item 5). A system whose card sets
``subsampling_equivalent: false`` must be re-run per step; track B supplies
those as ``<system_id>__stride<factor>`` directories.

The failure this guards against is quiet: sub-sampling a system that declared
the equivalence invalid produces a plausible-looking number on the step axis of
E4 that nothing supports.
"""

from __future__ import annotations

import numpy as np
import pytest

from ape.cli import (
    _needs_rerun_per_step,
    _split_stride_tables,
    _table_for_step,
)

from conftest import make_answers


def _table(sid, pred, card=None):
    return make_answers({"A": pred}, delta_s=0.25, system_id=sid, family="prefix")


def _with_card(table, **card):
    table.card.update(card)
    return table


def test_split_separates_stride_dirs_from_their_base_system():
    base = _table("sysA", [0] * 8, None)
    s2 = _table("sysA__stride2", [0] * 8, None)
    s4 = _table("sysA__stride4", [0] * 8, None)
    other = _table("sysB", [0] * 8, None)
    tables, strides = _split_stride_tables([base, s2, s4, other], "__stride")
    assert sorted(t.system_id for t in tables) == ["sysA", "sysB"]
    assert sorted(strides) == ["sysA"]
    assert sorted(strides["sysA"]) == [2, 4]


def test_split_leaves_an_orphan_stride_dir_as_its_own_system():
    """A stride dir with no base system is not silently swallowed."""
    orphan = _table("ghost__stride2", [0] * 8, None)
    tables, strides = _split_stride_tables([orphan], "__stride")
    assert [t.system_id for t in tables] == ["ghost__stride2"]
    assert strides == {}


def test_equivalent_system_is_subsampled_and_needs_no_rerun():
    t = _with_card(_table("sysA", [0] * 8, None), subsampling_equivalent=True)
    assert not _needs_rerun_per_step(t)
    notes = []
    used, factor = _table_for_step(t, {}, 1.0, 0.25, notes)
    assert used is t and factor == 4 and notes == []


def test_non_equivalent_system_reads_its_per_step_rerun():
    base = _with_card(_table("sysA", [0] * 8, None), subsampling_equivalent=False)
    s2 = _table("sysA__stride2", [1] * 8, None)
    s4 = _table("sysA__stride4", [2] * 8, None)
    strides = {"sysA": {2: s2, 4: s4}}
    notes = []
    assert _table_for_step(base, strides, 0.25, 0.25, notes)[0] is base  # finest step

    got2, _ = _table_for_step(base, strides, 0.5, 0.25, notes)
    got4, _ = _table_for_step(base, strides, 1.0, 0.25, notes)
    # the ANSWERS come from the re-run ...
    np.testing.assert_array_equal(got2.pred, s2.pred)
    np.testing.assert_array_equal(got4.pred, s4.pred)
    # ... but the system is scored under its own identity, never the directory's
    assert got2.system_id == "sysA" and got4.system_id == "sysA"
    assert got2.card["answers_read_from"] == "sysA__stride2"
    assert got4.card["answers_read_from"] == "sysA__stride4"
    assert notes == []


def test_a_stride_standin_never_becomes_a_separate_entrant():
    """A re-run directory must not show up as its own row in the audit table.

    It is the same system measured at a coarser step, so it must not enter the
    system set, the significant-pair set P, or any ranking under the name of its
    directory -- that would double-count one system and inflate the pair set.
    """
    base = _with_card(_table("sysA", [0] * 8, None), subsampling_equivalent=False)
    s2 = _table("sysA__stride2", [1] * 8, None)
    tables, strides = _split_stride_tables([base, s2], "__stride")
    assert [t.system_id for t in tables] == ["sysA"]  # not an entrant
    scored = {_table_for_step(t, strides, 0.5, 0.25, [])[0].system_id for t in tables}
    assert scored == {"sysA"}


def test_missing_rerun_falls_back_but_records_a_loud_note():
    base = _with_card(_table("sysA", [0] * 8, None), subsampling_equivalent=False)
    notes = []
    used, factor = _table_for_step(base, {}, 1.0, 0.25, notes)
    assert used is base and factor == 4
    assert len(notes) == 1
    assert "subsampling_equivalent=false" in notes[0]
    assert "NOT supported by a re-run" in notes[0]


def test_a_card_without_the_field_is_treated_as_equivalent():
    """Silence means the default; only an explicit false triggers re-runs."""
    t = _table("sysA", [0] * 8, None)
    assert not _needs_rerun_per_step(t)
    assert _needs_rerun_per_step(_with_card(t, subsampling_equivalent="false"))
    assert not _needs_rerun_per_step(_with_card(t, subsampling_equivalent="true"))
