"""Decode evidence, tiered duplicate evidence, and the versioned candidate ledger.

Consumes the frame-index preflight (``preflight.jsonl``), the decode pass
(``decode.jsonl``) and the automatic mapping proposal, and emits one row per released
sequence recording every reason it is or is not an APE candidate. The ledger is a
*filter-reason record*, not an experimental manifest: it assigns no split, freezes
no horizon, and converts nothing to seconds.

Duplicate evidence is reported in three tiers that are never collapsed, because
they support different claims:

``tier_a_identical_bytes``
    Two sequences share at least one probe frame whose file digest is identical.
    The same bytes were delivered twice. This is *confirmed* duplication of
    delivered content.

``tier_b_identical_dhash_different_bytes``
    Same 64-bit dHash, different bytes: a visual-similarity candidate or hash collision, but
    a dHash is an 8x8 gradient summary, so it is recorded as *pending visual
    confirmation* rather than as proof.

``tier_c_near_dhash``
    dHash within a small Hamming distance. A *candidate* only. Visual similarity
    is not identity: consecutive frames of one clip, or two clips of the same
    junction, can be near-identical without coming from one source video. This
    matches the ACCIDENT-track finding that a perceptual hash is a detector and
    does not define a cluster.

None of the tiers is treated as a source-video cluster. Establishing that unit
for MM-AU is a separate, still-open task, and this file reports what the evidence
does and does not license.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

LEDGER_VERSION = "candidates-v2-supervised-2026-09-20"

#: Hamming distance at or below which two dHashes are a tier-C candidate.
#: Deliberately tight. The ACCIDENT track used a perceptual-hash threshold of at
#: most 11 over a 64-bit hash as a *detector*; 8 is tighter still, and the count
#: at several thresholds is reported so the choice is visible rather than load-bearing.
NEAR_DHASH_THRESHOLDS = (0, 2, 4, 8)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def build_ledger(
    preflight: Path,
    decode: Optional[Path],
    proposal: Optional[Path],
) -> Dict[str, Any]:
    pre_rows = list(_read_jsonl(preflight))
    pre = {r["hashcode"]: r for r in pre_rows}
    if len(pre) != len(pre_rows):
        raise ValueError("duplicate preflight identifiers")
    dec: Dict[str, Dict[str, Any]] = {}
    if decode is not None and decode.exists():
        decode_rows = list(_read_jsonl(decode))
        dec = {r["hashcode"]: r for r in decode_rows}
        if len(dec) != len(decode_rows):
            raise ValueError("duplicate decode identifiers; recover the retry ledger before analysis")
        extra_ids = set(dec) - set(pre)
        if extra_ids:
            raise ValueError(f"decode identifiers absent from preflight: {sorted(extra_ids)[:5]}")

    verdict_of: Dict[int, str] = {}
    if proposal is not None and proposal.exists():
        payload = json.loads(proposal.read_text(encoding="utf-8"))
        verdict_of = {int(r["code"]): str(r["proposed_class"]) for r in payload["proposal"]}

    # ---- duplicate indices over the decode probes ----
    by_digest: Dict[str, List[str]] = defaultdict(list)
    by_dhash: Dict[str, List[str]] = defaultdict(list)
    dhashes_of: Dict[str, List[int]] = {}
    digests_of: Dict[str, set] = {}
    for key, row in dec.items():
        probes = row.get("probes") or []
        dhashes_of[key] = [int(p["dhash"], 16) for p in probes]
        digests_of[key] = {p["file_digest"] for p in probes}
        for p in probes:
            by_digest[p["file_digest"]].append(key)
            by_dhash[p["dhash"]].append(key)

    tier_a: Dict[str, set] = defaultdict(set)
    for digest, keys in by_digest.items():
        uniq = sorted(set(keys))
        if len(uniq) > 1:
            for a, b in combinations(uniq, 2):
                tier_a[a].add(b)
                tier_a[b].add(a)

    tier_b: Dict[str, set] = defaultdict(set)
    for dhash, keys in by_dhash.items():
        uniq = sorted(set(keys))
        if len(uniq) > 1:
            for a, b in combinations(uniq, 2):
                # Only tier B if the bytes are not already known to be identical.
                if b in tier_a.get(a, set()):
                    continue
                tier_b[a].add(b)
                tier_b[b].add(a)

    # Multi-index search over 9 disjoint bands. At Hamming radius <= 8,
    # at least one band must match exactly (pigeonhole principle). Candidate
    # verification uses all 64 bits; unlike the initial full-hash buckets this
    # retrieves near hashes even when no probe hash is identical.
    widths = (7,) * 8 + (8,)
    buckets = [defaultdict(list) for _ in widths]
    hashes = sorted(int(h, 16) for h in by_dhash)
    partners = {int(h, 16): sorted(set(keys)) for h, keys in by_dhash.items()}
    distances = {}
    for value in hashes:
        candidates = {value}
        shift = 0
        bands = []
        for band, width in enumerate(widths):
            key = (value >> shift) & ((1 << width) - 1)
            bands.append(key)
            candidates.update(buckets[band].get(key, ()))
            shift += width
        for other in candidates:
            distance = (value ^ other).bit_count()
            if distance > max(NEAR_DHASH_THRESHOLDS):
                continue
            for a in partners[value]:
                for b in partners[other]:
                    if a == b:
                        continue
                    pair = tuple(sorted((a, b)))
                    distances[pair] = min(distances.get(pair, 65), distance)
        for band, key in enumerate(bands):
            buckets[band][key].append(value)
    tier_c_counts = {t: sum(d <= t for d in distances.values()) for t in NEAR_DHASH_THRESHOLDS}
    tier_c = defaultdict(set)
    for a, b in distances:
        tier_c[a].add(b)
        tier_c[b].add(a)

    # ---- one row per released sequence ----
    rows: List[Dict[str, Any]] = []
    reasons: Counter = Counter()
    for key, p in sorted(pre.items()):
        d = dec.get(key)
        code = p.get("native_class")
        verdict = verdict_of.get(int(code)) if code is not None else None
        exclude: List[str] = []

        status = p.get("status")
        if status == "non_accident":
            exclude.append("not_an_accident_sequence")
        elif status == "frame_anomaly":
            exclude.append("frame_index_anomaly_released_length_disagrees_with_delivery")
        elif status != "frame_index_valid":
            exclude.append(f"preflight_status_{status}")

        post = p.get("post_anchor_frames")
        if status == "frame_index_valid" and (post is None or int(post) <= 0):
            exclude.append("anchor_at_last_frame_no_positive_post_anchor_span")

        if d is None:
            exclude.append("decode_not_yet_verified")
        else:
            if d.get("decode_status") != "all_frames_decoded":
                exclude.append(f"decode_{d.get('decode_status')}")
            if int(d.get("n_distinct_geometries") or 0) > 1:
                exclude.append("frame_geometry_changes_within_sequence")

        if verdict is None:
            exclude.append("native_class_absent_from_release_definitions")
        elif verdict == "ambiguous":
            exclude.append("ai_proposal_ambiguous_definition_does_not_entail_geometry")
        elif verdict == "out_of_scope":
            exclude.append("ai_proposal_out_of_scope_for_the_closed_set")

        if tier_a.get(key):
            exclude.append("shares_byte_identical_frames_with_another_sequence")

        for r in exclude:
            reasons[r] += 1
        rows.append(
            {
                "hashcode": key,
                "video_name": p.get("video_name"),
                "source": p.get("source"),
                "native_class": code,
                "relative_image_directory": p.get("relative_image_directory"),
                "preflight_status": status,
                "anchor_frame": p.get("anchor_frame"),
                "post_anchor_frames": post,
                "decode_status": (d or {}).get("decode_status", "not_run"),
                "frame_files": (d or {}).get("frame_files"),
                "decode_failures": (d or {}).get("decode_failures"),
                "n_distinct_geometries": (d or {}).get("n_distinct_geometries"),
                "ai_proposed_class": verdict,
                "ai_proposed_class_is_human_judgement": False,
                "tier_a_identical_bytes_with": sorted(tier_a.get(key, set()))[:8],
                "tier_b_identical_dhash_with": sorted(tier_b.get(key, set()))[:8],
                "tier_c_near_dhash_with": sorted(tier_c.get(key, set()))[:8],
                "exclusion_reasons": exclude,
                "is_candidate": not exclude,
                # Split, horizon and source cluster are deliberately unset.
                "split": "unassigned",
                "source_cluster_id": None,
                "time_unit": "frame",
                "fps": None,
            }
        )

    # ---- cross-source duplication, which is the leak that matters ----
    cross_source_tier_a = 0
    for key, partners in tier_a.items():
        src = pre.get(key, {}).get("source")
        for other in partners:
            if pre.get(other, {}).get("source") != src:
                cross_source_tier_a += 1
    cross_source_tier_a //= 2

    n_candidates = sum(1 for r in rows if r["is_candidate"])
    summary = {
        "ledger_version": LEDGER_VERSION,
        "n_released_sequences": len(rows),
        "n_decode_records_available": len(dec),
        "decode_complete": set(dec) == set(pre)
        and all(r.get("decode_status") in {"all_frames_decoded", "decode_failure", "empty"} for r in dec.values()),
        "all_sequences_decode_successful": set(dec) == set(pre)
        and all(r.get("decode_status") == "all_frames_decoded" for r in dec.values()),
        "input_sha256": {
            "preflight": hashlib.sha256(preflight.read_bytes()).hexdigest(),
            "decode": (
                hashlib.sha256(decode.read_bytes()).hexdigest() if decode is not None and decode.exists() else None
            ),
            "proposal": (
                hashlib.sha256(proposal.read_bytes()).hexdigest()
                if proposal is not None and proposal.exists()
                else None
            ),
            "generator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "n_candidates": n_candidates,
        "exclusion_reason_counts": dict(reasons.most_common()),
        "decode_status_counts": dict(Counter(r["decode_status"] for r in rows)),
        "duplicate_evidence": {
            "tier_a_identical_bytes_sequences": len(tier_a),
            "tier_a_cross_source_pairs": cross_source_tier_a,
            "tier_b_identical_dhash_sequences": len(tier_b),
            "tier_c_scope": "all released decode probes only, not exhaustive all-frame/source-event deduplication",
            "tier_c_candidate_pairs_by_threshold": {str(k): v for k, v in sorted(tier_c_counts.items())},
            "interpretation": (
                "Tier A confirms shared probe-frame bytes, not whole-video identity. Tier B is a visual "
                "candidate, potentially a hash collision, pending confirmation. Tier C is a candidate only: a "
                "perceptual hash is a detector, it does not define a source-video cluster, and "
                "no sequence is merged, relabelled or moved on its basis."
            ),
        },
        "not_established": [
            "source-video clusters for MM-AU (hashcode is a release identifier, not a verified cluster)",
            "experimental split (ArA covers CAP only; DADA has no ArA split)",
            "frame horizon H and step (must be chosen on a development subset and registered)",
            "final class labels (B8 two-person human mapping not returned; only an automatic proposal exists)",
            "any seconds-valued quantity (the release publishes no FPS)",
        ],
    }
    return {"summary": summary, "rows": rows}


def write_outputs(ledger: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "candidates.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in ledger["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    summary = dict(ledger["summary"])
    summary["candidates_jsonl_sha256"] = digest
    summary_path = out_dir / "candidates_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": str(rows_path), "summary": str(summary_path), "sha256": digest}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--decode", type=Path, default=None)
    parser.add_argument("--proposal", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ledger = build_ledger(args.preflight, args.decode, args.proposal)
    paths = write_outputs(ledger, args.out)
    print(json.dumps(ledger["summary"], ensure_ascii=False, indent=2))
    logging.info("ledger sha256 %s", paths["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
