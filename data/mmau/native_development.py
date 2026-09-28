"""Build provisional development partitions respecting observed overlap guards.

Exact dHash probe agreement is a conservative isolation heuristic, not source
identity. All connected components spanning release partitions or CAP/DADA are
quarantined. Unknown singleton origins remain unknown. No audit manifest is
produced and no new video/anchor label is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_guards(
    inventory: list[dict[str, Any]], decoded: list[dict[str, Any]], known: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Build order-invariant components before applying any label/cohort filter."""
    by_id = {r["hashcode"]: r for r in inventory}
    if len(by_id) != len(inventory) or len(decoded) != len(by_id) or {r["hashcode"] for r in decoded} != set(by_id):
        raise ValueError("duplicate or mismatched inventory/decode identifiers")
    parent = {key: key for key in by_id}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    buckets: dict[str, set[str]] = defaultdict(set)
    for row in decoded:
        for probe in row["probes"]:
            buckets[probe["dhash"]].add(row["hashcode"])
    pairs: Counter[tuple[str, str]] = Counter()
    for members in buckets.values():
        pairs.update(itertools.combinations(sorted(members), 2))
    evidence = []
    for (a, b), count in sorted(pairs.items()):
        union(a, b)
        evidence.append({"a": a, "b": b, "kind": "exact_dhash_probe_candidate", "shared_distinct_hashes": count})
    for group in known["components"]:
        members = sorted(group["members"])
        if not set(members).issubset(by_id):
            raise ValueError("known component refers to an unknown identifier")
        for key in members[1:]:
            union(members[0], key)
            evidence.append({"a": members[0], "b": key, "kind": "known_full_byte_content"})
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in sorted(by_id):
        grouped[find(key)].append(key)
    groups, membership = [], {}
    for members in sorted(grouped.values()):
        digest = hashlib.sha256("\n".join(members).encode()).hexdigest()[:20]
        group_id = "guard_" + digest
        partitions = sorted({by_id[key]["released_partition"] for key in members})
        sources = sorted({by_id[key]["source"] for key in members})
        quarantine = len(partitions) > 1 or len(sources) > 1
        groups.append(
            {
                "guard_component_id": group_id,
                "members": members,
                "released_partitions": partitions,
                "sources": sources,
                "quarantine": quarantine,
                "source_identity_certified": False,
                "basis": "observed_overlap_guard" if len(members) > 1 else "unverified_singleton",
            }
        )
        membership.update({key: group_id for key in members})
    return groups, membership, evidence


def run(ledger: Path, decode: Path, constraints: Path, pilot: Path, output: Path) -> dict[str, Any]:
    """Keep the previously proposed vocabulary fixed and inventory its attrition."""
    inventory, decoded = read_rows(ledger), read_rows(decode)
    known = json.loads(constraints.read_text(encoding="utf-8"))
    proposal = json.loads(pilot.read_text(encoding="utf-8"))
    codes = proposal["native_codes"]
    names = {r["native_code"]: r["definition"] for r in proposal["vocabulary"]}
    vocabulary = {
        "task_id": "mmau-native-nine-development",
        "version": "candidate-v1",
        "native_codes": codes,
        "class_names": [names[c] for c in codes],
    }
    payload = json.dumps(vocabulary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    vocabulary["vocabulary_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    groups, membership, evidence = build_guards(inventory, decoded, known)
    by_group = {g["guard_component_id"]: g for g in groups}
    candidate_all, native, decisions = [], [], []
    for row in inventory:
        key = row["hashcode"]
        group = by_group[membership[key]]
        reasons = list(row["planning_exclusions"])
        if group["quarantine"]:
            reasons.append("observed_component_crosses_released_partition_or_source")
        if row["source"] != "cap" or row["released_partition"] not in {"train", "val"}:
            reasons.append("outside_CAP_development_partitions")
        selected = not reasons
        decisions.append(
            {
                "video_id": key,
                "guard_component_id": membership[key],
                "exclusions": reasons,
                "retained_development": selected,
                "native_pilot": selected and row["native_class"] in codes,
            }
        )
        if not selected:
            continue
        candidate_all.append(row)
        if row["native_class"] not in codes:
            continue
        native.append(
            {
                "video_id": key,
                "source_cluster_id": membership[key],
                "source_cluster_basis": group["basis"],
                "source_identity_certified": False,
                "first_frame": row["first_frame"],
                "anchor_frame": row["anchor_frame"],
                "last_frame": row["last_frame"],
                "post_anchor_frames": row["post_anchor_frames"],
                "class_code": codes.index(row["native_class"]),
                "native_class": row["native_class"],
                "vocabulary_hash": vocabulary["vocabulary_hash"],
                "split": "dev" if row["released_partition"] == "val" else "train",
                "released_partition": row["released_partition"],
                "relative_image_directory": row["relative_image_directory"],
                "usage": "development_only_not_source_independent_audit",
            }
        )
    windows = {}
    for horizon in (1, 39, 67, 105):
        windows[str(horizon)] = {}
        for part in ("train", "dev"):
            selected = [r for r in native if r["split"] == part and r["post_anchor_frames"] >= horizon]
            count = Counter(r["native_class"] for r in selected)
            windows[str(horizon)][part] = {
                "n": len(selected),
                "by_native_class": {str(c): count[c] for c in codes},
                "observed_guard_components": len({r["source_cluster_id"] for r in selected}),
            }
    supported = [
        c
        for c in codes
        if windows["67"]["train"]["by_native_class"][str(c)] >= 50
        and windows["67"]["dev"]["by_native_class"][str(c)] >= 10
    ]
    # The two partitions must not share any component in the observed guard graph.
    train_guards = {r["source_cluster_id"] for r in native if r["split"] == "train"}
    dev_guards = {r["source_cluster_id"] for r in native if r["split"] == "dev"}
    if train_guards & dev_guards:
        raise AssertionError("guard component leakage")
    summary = {
        "status": "provisional_development_only",
        "source_clusters_complete": False,
        "global_split_certified": False,
        "formal_verdict": "not_evaluable",
        "guard_rule": (
            "Connect every pair sharing >=1 exact dHash probe and all known full-byte components; "
            "quarantine components crossing release partitions or CAP/DADA, before filtering."
        ),
        "guard_is_source_identity": False,
        "guard_false_positives_possible": True,
        "raw_inventory": len(inventory),
        "exact_dhash_candidate_pairs": sum(e["kind"] == "exact_dhash_probe_candidate" for e in evidence),
        "nontrivial_guard_components": sum(len(g["members"]) > 1 for g in groups),
        "largest_guard_component": max(len(g["members"]) for g in groups),
        "quarantined_components": sum(g["quarantine"] for g in groups),
        "quarantined_raw_members": sum(len(g["members"]) for g in groups if g["quarantine"]),
        "development_all_classes": dict(Counter(r["released_partition"] for r in candidate_all)),
        "development_native_nine": dict(Counter(r["split"] for r in native)),
        "observed_guard_overlap_train_dev": 0,
        "native_codes_fixed_from_previous_proposal": codes,
        "windows": windows,
        "classes_still_meeting_50_10_at_67": supported,
        "all_nine_support_rule_still_satisfied": supported == codes,
        "audit_predictions_used": False,
        "audit_metadata_used_only_for_guard_quarantine": True,
        "cohort_policy": "Use span>=67 common cohort for paired H39/H67 comparisons, class set fixed.",
        "input_sha256": {
            k: hashlib.sha256(p.read_bytes()).hexdigest()
            for k, p in {
                "planning_ledger": ledger,
                "decode": decode,
                "constraints": constraints,
                "prior_pilot": pilot,
            }.items()
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in [
        ("summary.json", summary),
        ("vocabulary.json", vocabulary),
        ("guard_components.json", groups),
        ("guard_edges.json", evidence),
    ]:
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename, records in [("development_manifest.jsonl", native), ("decisions.jsonl", decisions)]:
        with (output / filename).open("w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ledger", "decode", "constraints", "pilot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = run(args.ledger, args.decode, args.constraints, args.pilot, args.output)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "nontrivial_guard_components",
                    "largest_guard_component",
                    "quarantined_components",
                    "quarantined_raw_members",
                    "development_native_nine",
                    "classes_still_meeting_50_10_at_67",
                    "windows",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
