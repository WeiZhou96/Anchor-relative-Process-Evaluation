"""Guard components must retain transitive evidence through excluded rows."""

from data.mmau.native_development import build_guards


def row(key: str, part: str = "train", source: str = "cap") -> dict:
    return {"hashcode": key, "released_partition": part, "source": source}


def decoded(key: str, hashes: list[str]) -> dict:
    return {"hashcode": key, "probes": [{"dhash": value} for value in hashes]}


def test_cross_partition_bridge_is_quarantined_before_filters() -> None:
    inventory = [row("a"), row("excluded_bridge"), row("b", "test"), row("c", "val")]
    probes = [decoded("a", ["x"]), decoded("excluded_bridge", ["x", "y"]), decoded("b", ["y"]), decoded("c", ["z"])]
    groups, membership, _ = build_guards(inventory, probes, {"components": []})
    by_id = {g["guard_component_id"]: g for g in groups}
    assert membership["a"] == membership["b"]
    assert by_id[membership["a"]]["quarantine"]
    assert not by_id[membership["c"]]["quarantine"]
    assert not by_id[membership["c"]]["source_identity_certified"]
    reversed_groups, reversed_membership, _ = build_guards(inventory[::-1], probes[::-1], {"components": []})
    assert reversed_groups == groups and reversed_membership == membership


def test_source_conflict_and_known_byte_components_are_respected() -> None:
    inventory = [row("a"), row("b", source="dada"), row("c"), row("d", "val")]
    probes = [decoded("a", ["same", "same"]), decoded("b", ["same"]), decoded("c", []), decoded("d", [])]
    groups, membership, edges = build_guards(inventory, probes, {"components": [{"members": ["c", "d"]}]})
    assert len(groups) == 2 and all(g["quarantine"] for g in groups)
    assert edges[0]["shared_distinct_hashes"] == 1
    assert membership["c"] == membership["d"]
