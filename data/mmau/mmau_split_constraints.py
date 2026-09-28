"""Known-content split constraints; never certify global source independence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def build_constraints(overlap: dict[str, Any], preflight: list[dict[str, Any]]) -> dict[str, Any]:
    """Record only established content components and observed annotation conflicts."""
    rows = {row["hashcode"]: row for row in preflight}
    if len(rows) != len(preflight):
        raise ValueError("duplicate preflight IDs")
    components = []
    seen: set[str] = set()
    for group in overlap["content_constraint_components"]:
        members = sorted(group)
        if len(members) < 2 or len(set(members)) != len(members) or set(members) & seen:
            raise ValueError("invalid or overlapping content components")
        if not set(members) <= set(rows):
            raise ValueError("unknown component IDs")
        seen.update(members)
        content_id = "content_" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        conflicts = {}
        for field in ("native_class", "anchor_frame", "metadata_total_frames", "accident"):
            values = {member: rows[member][field] for member in members}
            if len(set(values.values())) > 1:
                conflicts[field] = values
        splits = sorted({split for member in members for split in rows[member]["split_official_ara"]})
        components.append(
            {
                "content_component_id": content_id,
                "members": members,
                "official_splits": splits,
                "annotation_conflicts": conflicts,
                "requires_joint_split_or_exclusion": True,
                "recommended_action": (
                    "quarantine_pending_review"
                    if conflicts or len(splits) > 1
                    else "deduplicate_or_keep_in_same_partition"
                ),
            }
        )
    return {
        "version": "known-content-constraints-v1",
        "population": len(rows),
        "components": components,
        "source_clusters_complete": False,
        "is_frozen_split": False,
        "annotation_changes": [],
        "input_sha256": overlap["input_sha256"],
        "limits": "Passing these constraints does not establish absence of other duplicates or source leakage.",
    }


def check_assignments(constraints: dict[str, Any], assignments: dict[str, str]) -> dict[str, Any]:
    """Reject separation of known shared content; excluded clips do not enter any split."""
    allowed = {"train", "dev", "val", "test", "audit", "unassigned", "excluded"}
    if any(value not in allowed for value in assignments.values()):
        raise ValueError("unrecognised split name")
    violations, unresolved = [], []
    for group in constraints["components"]:
        resolved = {member: assignments.get(member, "unassigned") for member in group["members"]}
        names = set(resolved.values()) - {"unassigned", "excluded"}
        record = {"content_component_id": group["content_component_id"], "assignments": resolved}
        if len(names) > 1:
            violations.append(record)
        if "unassigned" in resolved.values():
            unresolved.append(record)
    return {
        "known_content_split_violations": violations,
        "n_violations": len(violations),
        "unresolved_components": unresolved,
        "known_constraints_satisfied": not violations and not unresolved,
        "global_split_certified": False,
        "source_clusters_complete": False,
    }


def main() -> None:
    """Build evidence and check the released ArA assignments, without altering either."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("overlap", "preflight", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    overlap = json.loads(args.overlap.read_text(encoding="utf-8"))
    blob = args.preflight.read_bytes()
    if hashlib.sha256(blob).hexdigest() != overlap["input_sha256"]["preflight"]:
        raise ValueError("preflight differs from content-audit provenance")
    rows = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
    constraints = build_constraints(overlap, rows)
    assignments = {}
    for row in rows:
        splits = row["split_official_ara"]
        if len(splits) > 1:
            raise ValueError("ambiguous released assignment")
        assignments[row["hashcode"]] = splits[0] if splits else "unassigned"
    constraints["official_ara_check"] = check_assignments(constraints, assignments)
    constraints["overlap_sha256"] = hashlib.sha256(args.overlap.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(constraints, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
