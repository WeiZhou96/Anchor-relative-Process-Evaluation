"""Audit full byte sequences of probe-overlap candidates without relabelling sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a unique-ID diagnostic ledger."""
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len({r["hashcode"] for r in rows}) != len(rows):
        raise ValueError("duplicate ledger identifiers")
    return rows


def probe_pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Candidates share original file-byte digests; confirm using SHA-256 later."""
    buckets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for probe in row.get("probes", []):
            buckets[probe["file_digest"]].add(row["hashcode"])
    return sorted({pair for ids in buckets.values() for pair in combinations(sorted(ids), 2)})


def compare_sequences(a: list[str], b: list[str]) -> dict[str, Any]:
    """Find contiguous byte-equal runs, retaining low-information coincidences."""
    if not a or not b:
        raise ValueError("empty sequence")
    previous: dict[int, int] = {}
    positions: dict[str, list[int]] = defaultdict(list)
    for j, digest in enumerate(b):
        positions[digest].append(j)
    longest = 0
    best = (0, 0)
    for i, digest in enumerate(a):
        current = {}
        for j in positions.get(digest, []):
            length = previous.get(j - 1, 0) + 1
            current[j] = length
            if length > longest:
                longest, best = length, (i - length + 1, j - length + 1)
        previous = current
    distinct = len(set(a[best[0] : best[0] + longest]))
    exact = a == b
    temporal = longest >= 3 and distinct >= 3
    relation = "full_sequence_identical" if exact else "aligned_byte_run" if temporal else "isolated_shared_bytes"
    return {
        "relation": relation,
        "full_sequence_identical": exact,
        "n_shared_unique_frame_digests": len(set(a) & set(b)),
        "longest_equal_run": longest,
        "run_start_zero_based_a": best[0],
        "run_start_zero_based_b": best[1],
        "distinct_contents_in_longest_run": distinct,
        "matched_fraction_a": longest / len(a),
        "matched_fraction_b": longest / len(b),
        "leakage_constraint_candidate": temporal or (exact and distinct >= 3),
        "source_video_identity_established": False,
    }


def components(pairs: list[dict[str, Any]]) -> list[list[str]]:
    """Connected components of verified byte-run constraints, not source labels."""
    neighbours: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        if pair["leakage_constraint_candidate"]:
            a, b = pair["a"], pair["b"]
            neighbours[a].add(b)
            neighbours[b].add(a)
    result = []
    seen: set[str] = set()
    for first in sorted(neighbours):
        if first in seen:
            continue
        queue = [first]
        seen.add(first)
        group = []
        while queue:
            item = queue.pop()
            group.append(item)
            for other in sorted(neighbours[item] - seen):
                seen.add(other)
                queue.append(other)
        result.append(sorted(group))
    return result


def audit(root: Path, decode: Path, preflight: Path, output: Path) -> dict[str, Any]:
    """Read all image bytes for probe-connected clips and write immutable evidence."""
    if output.exists():
        raise FileExistsError(output)
    rows = read_rows(decode)
    pre = {r["hashcode"]: r for r in read_rows(preflight)}
    dec = {r["hashcode"]: r for r in rows}
    if set(pre) != set(dec):
        raise ValueError("preflight/decode populations differ")
    pairs = probe_pairs(rows)
    selected = sorted({item for pair in pairs for item in pair})
    sequences = {}
    bytes_read = 0
    for key in selected:
        directory = (root / dec[key]["relative_image_directory"]).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError("image path leaves the dataset")
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(files) != dec[key]["frame_files"]:
            raise ValueError("image inventory changed since decoding")
        frames = []
        for path in files:
            content = path.read_bytes()
            bytes_read += len(content)
            frames.append({"name": path.name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
        sequences[key] = {"video_name": pre[key]["video_name"], "frames": frames}
    findings = []
    for a, b in pairs:
        record = compare_sequences(
            [f["sha256"] for f in sequences[a]["frames"]],
            [f["sha256"] for f in sequences[b]["frames"]],
        )
        record.update(
            {
                "a": a,
                "b": b,
                "video_a": pre[a]["video_name"],
                "video_b": pre[b]["video_name"],
                "source_a": pre[a]["source"],
                "source_b": pre[b]["source"],
                "official_splits_a": pre[a]["split_official_ara"],
                "official_splits_b": pre[b]["split_official_ara"],
                "native_class_a": pre[a]["native_class"],
                "native_class_b": pre[b]["native_class"],
            }
        )
        record["crosses_official_split"] = bool(
            record["official_splits_a"]
            and record["official_splits_b"]
            and set(record["official_splits_a"]).isdisjoint(record["official_splits_b"])
        )
        findings.append(record)
    payload = {
        "status": "content_overlap_evidence_only",
        "input_sha256": {
            "decode": hashlib.sha256(decode.read_bytes()).hexdigest(),
            "preflight": hashlib.sha256(preflight.read_bytes()).hexdigest(),
        },
        "population": len(rows),
        "sequences_audited": len(selected),
        "candidate_pairs": len(pairs),
        "frames_hashed": sum(len(s["frames"]) for s in sequences.values()),
        "bytes_read": bytes_read,
        "pairs": findings,
        "content_constraint_components": components(findings),
        "sequences": sequences,
        "source_clusters_complete": False,
        "limits": [
            "Only byte-identical probe candidates are audited; recompressed or unsampled overlap may be missed.",
            "Byte runs establish duplicated content, not unique source uploads or independence of other clips.",
            "Components constrain a future split; no split, class, raw file or source-cluster label is modified.",
            "Three distinct consecutive frames is a diagnostic filter, not a calibrated statistical threshold.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in payload.items() if k not in {"sequences", "pairs"}}


def main() -> None:
    """Run the read-only content audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "decode", "preflight", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.decode, args.preflight, args.output), indent=2))


if __name__ == "__main__":
    main()
