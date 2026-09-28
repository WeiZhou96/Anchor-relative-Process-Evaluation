"""Root-cause evidence for the 25 MM-AU frame anomalies.

The frame-index preflight recorded *that* 25 accident sequences disagree with
their released metadata. This script asks *why*, using only released files, and
separates what the evidence settles from what it merely suggests.

Four questions, each answerable from the release:

1. **Do the two official sources agree with each other?** ``video_metadata.json``
   and the CAP/DADA annotation sheet both publish ``total_frames``. If they agree
   and both disagree with the images, the fault is not a metadata transcription
   slip between the two files.
2. **Is the anchor still usable?** An anomaly whose ``t_co`` still lies inside the
   actual frame range is a length bookkeeping problem; one whose anchor falls
   outside it is a different and worse failure.
3. **Is the disagreement one-directional?** Systematically more images than
   announced would suggest padding; a mix suggests independent per-sequence slips.
4. **Are there frame-count swaps?** A pair of records each carrying the other's
   count is a bookkeeping hypothesis. It is reported strictly as a *candidate*:
   matching integers are not evidence that two sequences were exchanged, and
   nothing here relabels or moves any data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_preflight(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    return rows


def annotation_lookup(path: Path) -> Dict[tuple, Dict[str, Any]]:
    """Index the release annotation rows by ``(source, type, video)``."""
    out: Dict[tuple, Dict[str, Any]] = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        out[(row["source"], int(row["type"]), int(row["video"]))] = row
    return out


def analyse(preflight: Path, annotations: Path) -> Dict[str, Any]:
    rows = load_preflight(preflight)
    ann = annotation_lookup(annotations)
    anomalies = [r for r in rows if r.get("status") == "frame_anomaly"]

    # Actual image count of every sequence that has one, for the swap search.
    actual_by_key: Dict[tuple, int] = {}
    meta_count_index: Dict[int, List[str]] = defaultdict(list)
    actual_count_index: Dict[int, List[str]] = defaultdict(list)
    for r in rows:
        if r.get("frame_count") is None or r.get("source") is None:
            continue
        key = (r["source"], int(r["native_class"]), int(r["original_video_number"]))
        actual_by_key[key] = int(r["frame_count"])
        meta_count_index[int(r["metadata_total_frames"])].append(r["video_name"])
        actual_count_index[int(r["frame_count"])].append(r["video_name"])

    findings: List[Dict[str, Any]] = []
    sources_agree = 0
    sources_disagree = 0
    anchor_inside = 0
    anchor_outside = 0
    direction: Counter = Counter()
    kinds: Counter = Counter()

    for r in anomalies:
        key = (r.get("source"), int(r["native_class"]), int(r.get("original_video_number", -1)))
        a = ann.get(key)
        meta_total = int(r["metadata_total_frames"])
        actual = r.get("frame_count")
        first = r.get("first_frame")
        last = r.get("last_frame")
        anchor = int(r["anchor_frame"])

        ann_total = int(a["total_frames"]) if a else None
        agree = ann_total is not None and ann_total == meta_total
        sources_agree += int(bool(agree))
        sources_disagree += int(not agree)

        # Which defect is this?
        if first is not None and first != 1:
            kind = "numbering_does_not_start_at_1"
        elif actual is not None and actual != meta_total:
            kind = "count_mismatch"
        else:
            kind = "other"
        kinds[kind] += 1

        if actual is not None:
            category = (
                "same_count_numbering_anomaly"
                if actual == meta_total
                else ("more_images_than_announced" if actual > meta_total else "fewer_images_than_announced")
            )
            direction[category] += 1

        inside = last is not None and first is not None and first <= anchor <= last
        anchor_inside += int(bool(inside))
        anchor_outside += int(not inside)

        # Swap candidates: another record announcing exactly this record's actual
        # count. The reciprocal form -- that record's actual count is this
        # record's announced count -- is the stronger version and is flagged.
        #
        # Both sides must actually disagree with their own images. Without that
        # condition the test fires trivially on any record whose count is correct:
        # 3_11665 announces 150 and delivers 150, and 503 other sequences also
        # hold 150 frames, so it would report 503 "reciprocal" partners that carry
        # no information about a swap at all.
        swap_candidates: List[Dict[str, Any]] = []
        self_mismatched = actual is not None and actual != meta_total
        if self_mismatched:
            for name in meta_count_index.get(int(actual), []):
                if name == r["video_name"]:
                    continue
                other = next((x for x in rows if x["video_name"] == name), None)
                if other is None or other.get("frame_count") is None:
                    continue
                other_announced = int(other["metadata_total_frames"])
                other_actual = int(other["frame_count"])
                if other_actual == other_announced:
                    continue  # that record is self-consistent; nothing was swapped into it
                swap_candidates.append(
                    {
                        "other_video_name": name,
                        "other_announced": other_announced,
                        "other_actual": other_actual,
                        "reciprocal": other_actual == meta_total,
                        "other_status": other.get("status"),
                        "other_source": other.get("source"),
                        "other_native_class": other.get("native_class"),
                        "other_original_video_number": other.get("original_video_number"),
                    }
                )
        reciprocal = [c for c in swap_candidates if c["reciprocal"]]

        findings.append(
            {
                "video_name": r["video_name"],
                "hashcode": r["hashcode"],
                "source": r.get("source"),
                "native_class": r["native_class"],
                "relative_image_directory": r.get("relative_image_directory"),
                "defect_kind": kind,
                "announced_total_frames_metadata_json": meta_total,
                "announced_total_frames_annotation_sheet": ann_total,
                "official_sources_agree": agree,
                "actual_image_count": actual,
                "first_frame": first,
                "last_frame": last,
                "delta_actual_minus_announced": (None if actual is None else actual - meta_total),
                "anchor_frame": anchor,
                "anchor_inside_actual_range": inside,
                "post_anchor_frames_on_actual_range": (None if last is None else last - anchor),
                "missing_indices": r.get("missing_indices"),
                "duplicate_indices": r.get("duplicate_indices"),
                "zero_byte_frames": r.get("zero_byte_frames"),
                "n_swap_candidates": len(swap_candidates),
                "n_reciprocal_swap_candidates": len(reciprocal),
                "reciprocal_swap_candidates": reciprocal[:5],
            }
        )

    return {
        "artifact": "mmau_frame_anomaly_root_cause",
        "generated_on": "2026-09-20",
        "population": "the 25 accident sequences the frame-index preflight marked frame_anomaly",
        "n_anomalies": len(anomalies),
        "summary": {
            "defect_kinds": dict(kinds),
            "official_sources_agree_with_each_other": sources_agree,
            "official_sources_disagree_with_each_other": sources_disagree,
            "anchor_inside_actual_range": anchor_inside,
            "anchor_outside_actual_range": anchor_outside,
            "direction_of_count_mismatch": dict(direction),
            "n_with_reciprocal_swap_candidate": sum(1 for f in findings if f["n_reciprocal_swap_candidates"] > 0),
            "n_with_any_swap_candidate": sum(1 for f in findings if f["n_swap_candidates"] > 0),
            "integrity_within_the_delivered_images": {
                "sequences_with_missing_indices": sum(1 for f in findings if f["missing_indices"]),
                "sequences_with_duplicate_indices": sum(1 for f in findings if f["duplicate_indices"]),
                "sequences_with_zero_byte_frames": sum(1 for f in findings if f["zero_byte_frames"]),
            },
        },
        "interpretation_limits": [
            "A matching integer count is not evidence that two sequences were exchanged. "
            "Swap candidates are listed so the hypothesis can be checked against the release, "
            "and no label, file or directory is changed on their basis.",
            "This pass reads released metadata and image file names only. Whether the extra or "
            "missing frames are themselves decodable is answered by the separate decode pass.",
            "No FPS is inferred anywhere, so none of these counts is converted to seconds.",
        ],
        "anomalies": findings,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# MM-AU 25 条帧异常：成因证据",
        "",
        f"生成：{report['generated_on']}。口径：{report['population']}。只读发布件，未改动任何原始数据。",
        "",
        "## 汇总",
        "",
        f"- 异常总数 {report['n_anomalies']}，缺陷分型：{s['defect_kinds']}",
        f"- 两个官方来源（`video_metadata.json` 与 CAP/DADA 标注表）**相互一致** "
        f"{s['official_sources_agree_with_each_other']} 条，相互矛盾 "
        f"{s['official_sources_disagree_with_each_other']} 条",
        f"- 碰撞锚仍落在实际帧号范围内 {s['anchor_inside_actual_range']} 条，落在范围外 "
        f"{s['anchor_outside_actual_range']} 条",
        f"- 数量偏差方向：{s['direction_of_count_mismatch']}",
        "- 交付图片自身的完整性：缺号 "
        f"{s['integrity_within_the_delivered_images']['sequences_with_missing_indices']} 条、"
        f"重号 {s['integrity_within_the_delivered_images']['sequences_with_duplicate_indices']} 条、"
        f"零字节帧 {s['integrity_within_the_delivered_images']['sequences_with_zero_byte_frames']} 条",
        f"- 存在互换候选 {s['n_with_any_swap_candidate']} 条，其中构成**互为**对方公布值的互换候选 "
        f"{s['n_with_reciprocal_swap_candidate']} 条",
        "",
        "## 逐条",
        "",
        "| 视频 | 来源 | 类 | 分型 | 公布(metadata/标注表) | 实际 | 差 | 锚 | 锚在范围内 | 互换候选/互为 |",
        "|---|---|---:|---|---|---:|---:|---:|---|---|",
    ]
    for f in report["anomalies"]:
        lines.append(
            f"| {f['video_name']} | {f['source']} | {f['native_class']} | {f['defect_kind']} | "
            f"{f['announced_total_frames_metadata_json']}/{f['announced_total_frames_annotation_sheet']} | "
            f"{f['actual_image_count']} | {f['delta_actual_minus_announced']:+d} | {f['anchor_frame']} | "
            f"{'是' if f['anchor_inside_actual_range'] else '否'} | "
            f"{f['n_swap_candidates']}/{f['n_reciprocal_swap_candidates']} |"
        )
    lines += ["", "## 解读边界", ""]
    for item in report["interpretation_limits"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)

    report = analyse(args.preflight, args.annotations)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
