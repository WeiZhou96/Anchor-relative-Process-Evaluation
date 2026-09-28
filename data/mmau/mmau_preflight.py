"""Inspect MM-AU image sequences without modifying the dataset or inferring FPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def signature(row: dict[str, Any], metadata: bool = False) -> tuple[int, ...]:
    """Match release records using identifiers and all available temporal fields."""
    fields = (
        ("type", "id", "total_frames", "t_co", "t_ai", "t_ae", "accident occurred")
        if metadata
        else ("type", "video", "total_frames", "anchor", "abnormal_start", "abnormal_end", "accident")
    )
    return tuple(
        int(row["video_name"].rsplit("_", 1)[1]) if metadata and field == "id" else int(row[field]) for field in fields
    )


def sequence_info(path: Path) -> dict[str, Any]:
    """Check numeric image indices; decoding is a separate, unperformed check."""
    ids = []
    unexpected = []
    zero_bytes = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if not entry.is_file():
                unexpected.append(entry.name)
                continue
            p = Path(entry.name)
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not p.stem.isdigit():
                unexpected.append(entry.name)
                continue
            ids.append(int(p.stem))
            zero_bytes += entry.stat().st_size == 0
    unique = set(ids)
    lo, hi = (min(unique), max(unique)) if unique else (None, None)
    return dict(
        frame_count=len(ids),
        first_frame=lo,
        last_frame=hi,
        missing_indices=(hi - lo + 1 - len(unique)) if unique else None,
        duplicate_indices=len(ids) - len(unique),
        unexpected_entries=unexpected,
        zero_byte_frames=zero_bytes,
        frame_ids=unique,
    )


def audit(root: Path, annotations: Path, out: Path) -> dict[str, Any]:
    """Write a diagnostic ledger; never expose it as a ready APE manifest."""
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "official_metadata/video_metadata.json"
    meta = json.loads(metadata_path.read_text())
    index: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in json.loads(annotations.read_text()):
        index[signature(row)].append(row)
    paths: dict[tuple[str, int, int], list[Path]] = defaultdict(list)
    for source, pattern in [("cap", "CAP-DATA/*/*/*/images"), ("dada", "Origin/DADA2000/DADA2000/*/*/images")]:
        for path in (root / "extracted").glob(pattern):
            paths[(source, int(path.parent.parent.name), int(path.parent.name))].append(path)
    split_names: dict[tuple[int, int], set[str]] = defaultdict(set)
    for split in ["train", "val", "test"]:
        for entry in json.loads((root / f"official_metadata/ArA label/{split}.json").read_text()):
            code, video = map(int, entry["video_name"].split("-"))
            split_names[(code, video)].add(split)
    counts: Counter[str] = Counter()
    used_paths: Counter[str] = Counter()
    post: dict[str, list[int]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    total_images = 0
    with (out / "preflight.jsonl").open("w", encoding="utf-8") as stream:
        for i, (hashcode, m) in enumerate(sorted(meta.items()), 1):
            row: dict[str, Any] = dict(
                hashcode=hashcode,
                video_name=m["video_name"],
                metadata_id=m["id"],
                native_class=int(m["type"]),
                anchor_frame=int(m["t_co"]),
                metadata_total_frames=int(m["total_frames"]),
                accident=int(m["accident occurred"]),
                fps=None,
                time_unit="frame",
                split="unassigned",
                source_cluster_id=None,
                map_status="pending",
                decode_status="not_tested",
            )
            matches = index.get(signature(m, True), [])
            row["annotation_match_count"] = len(matches)
            if len(matches) != 1:
                row["status"] = "annotation_unresolved"
            else:
                a = matches[0]
                row.update(source=a["source"], annotation_sheet_row=a["sheet_row"], original_video_number=a["video"])
                official = sorted(split_names.get((a["type"], a["video"]), set())) if a["source"] == "cap" else []
                row["split_official_ara"] = official
                split_counts[",".join(official) if official else "unassigned"] += 1
                found = paths.get((a["source"], a["type"], a["video"]), [])
                row["directory_match_count"] = len(found)
                if len(found) != 1:
                    row["status"] = "directory_unresolved"
                else:
                    path = found[0]
                    row["relative_image_directory"] = str(path.relative_to(root))
                    used_paths[str(path)] += 1
                    info = sequence_info(path)
                    ids = info.pop("frame_ids")
                    row.update(info)
                    total_images += info["frame_count"]
                    row["anchor_present"] = row["anchor_frame"] in ids
                    row["post_anchor_frames"] = (
                        info["last_frame"] - row["anchor_frame"] if row["accident"] and row["anchor_present"] else None
                    )
                    row["total_frames_match"] = info["frame_count"] == row["metadata_total_frames"]
                    row["status"] = (
                        "non_accident"
                        if not row["accident"]
                        else (
                            "frame_index_valid"
                            if row["anchor_present"]
                            and row["total_frames_match"]
                            and info["first_frame"] == 1
                            and info["missing_indices"] == 0
                            and info["duplicate_indices"] == 0
                            and info["zero_byte_frames"] == 0
                            and not info["unexpected_entries"]
                            else "frame_anomaly"
                        )
                    )
                    if row["status"] == "frame_index_valid":
                        post[a["source"]].append(row["post_anchor_frames"])
            counts[row["status"]] += 1
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 1000 == 0:
                logging.info("Processed %d/%d", i, len(meta))

    def stats(values: list[int]) -> dict[str, Any]:
        values = sorted(values)
        return dict(
            n=len(values),
            min=values[0],
            median=(values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2,
            max=values[-1],
            positive=sum(v > 0 for v in values),
            zero=sum(v == 0 for v in values),
        )

    cap_keys = {(code, video) for source, code, video in paths if source == "cap"}
    summary = dict(
        metadata_rows=len(meta),
        image_directories=sum(map(len, paths.values())),
        status_counts=dict(counts),
        inspected_image_files=total_images,
        official_ara_split_counts=dict(split_counts),
        official_ara_keys_equal_cap_directories=set(split_names) == cap_keys,
        official_ara_overlap_keys=[list(k) for k, v in split_names.items() if len(v) > 1],
        referenced_directories=len(used_paths),
        duplicate_directory_assignments={k: v for k, v in used_paths.items() if v > 1},
        unreferenced_directories=[
            str(p.relative_to(root)) for group in paths.values() for p in group if str(p) not in used_paths
        ],
        post_anchor_frames={s: stats(v) for s, v in post.items()},
        metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        annotation_index_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),
        limitations=[
            "No image decoding",
            "No FPS or seconds inferred",
            "No source-video deduplication",
            "No human mappings",
            "No experimental split or H selected",
            "No model outcomes inspected",
        ],
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.annotations, args.out)
    logging.info("Status: %s; referenced directories: %d", result["status_counts"], result["referenced_directories"])
