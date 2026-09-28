#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the APE manifest (CONTRACT.md section 5.1) from the ACCIDENT_2026 metadata CSVs.

Real set  : full manifest with md5, OpenCV decode probe, fps, post-anchor length,
            source-event clusters and a cluster-level dev split carved out of the
            official in-distribution train split.
Synthetic : registration only (no md5, no decode probe).

All comments and logs are in English on purpose; the Chinese report lives in REPORT.md.
"""

import argparse
import csv
import hashlib
import os
import random
import sys
import time

DATA_ROOT_DEFAULT = os.environ.get("APE_ACCIDENT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "ACCIDENT_2026"))
SEED_DEFAULT = 20260903
DEV_FRACTION_DEFAULT = 0.20

# CONTRACT.md section 4. Populated/verified at runtime against the CSV unique values.
CLASS_CODES = {
    "head-on": 0,
    "rear-end": 1,
    "t-bone": 2,
    "sideswipe": 3,
    "single": 4,
}

MANIFEST_COLUMNS = [
    "track_id",
    "dataset_id",
    "video_id",
    "path",
    "source_cluster_id",
    "native_anchor_field",
    "anchor_s",
    "anchor_frame",
    "fps",
    "duration_s",
    "n_frames",
    "post_anchor_length_s",
    "class_code",
    "class_name",
    "map_status",
    "split",
    "split_geo",
    "quality",
    "day_time",
    "scene_layout",
    "region",
    "license_note",
    "decode_ok",
    "decode_hash",
    # v2 addition, appended last so positional readers of the contract section 5.1
    # column order are unaffected: the official split before the leak repair.
    "split_official",
]

LICENSE_NOTE = "academic-noncommercial; see paper vs kaggle"


def log(msg):
    sys.stdout.write("[build_manifest] %s\n" % msg)
    sys.stdout.flush()


def cluster_of(video_id):
    """Source event cluster = source video id.

    ACCIDENT real clips carry two naming shapes, both verified on the 2027 file names
    (data/_probe_names.py):
        <source_video_id>_<NN>              e.g. Z4kg2Ev3vhk_00
        <source_video_id>_<k>_<NN>          e.g. 3Xd6PxEvbNk_11_00, AuQz_-J2kzc_25_00
    so trailing index segments are stripped repeatedly, not just once: stripping only
    once leaves 42 source videos split across several clusters, which would leak the
    same source across splits.

    A segment is treated as an index only when it is 1 to 3 digits long. YouTube ids
    themselves may contain '_' (ftk_fSc4haM, kUp_0rJe_OI) but their segments are not
    short digit runs, so they survive. The residual risk is an id whose own last
    segment is a short digit run; over-merging is the safe direction for leakage
    control, at the cost of a little statistical power.
    """
    parts = video_id.split("_")
    while len(parts) > 1 and parts[-1].isdigit() and 1 <= len(parts[-1]) <= 3:
        parts = parts[:-1]
    return "_".join(parts)


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def decode_probe(path):
    """Return (decode_ok, cv_fps, cv_n_frames, note) using OpenCV."""
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - environment guard
        return False, "", "", "cv2-import-failed: %s" % exc
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False, "", "", "open-failed"
        cv_fps = cap.get(cv2.CAP_PROP_FPS)
        cv_n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, cv_fps, cv_n, "first-frame-read-failed"
        return True, cv_fps, cv_n, ""
    except Exception as exc:
        return False, "", "", "exception: %s" % exc
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def read_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=-1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def assign_dev(rows, seed, dev_fraction):
    """Carve a dev subset out of the official train split at cluster granularity.

    Returns a set of cluster ids assigned to dev. Only clips whose own official split
    is 'train' are moved to dev; a cluster that leaks across the official split keeps
    its test clips in test (the leak itself is reported by clusters.py).
    """
    train_rows = [r for r in rows if r["_split_raw"] == "train"]
    if not train_rows:
        return set(), 0, 0
    counts = {}
    for r in train_rows:
        counts[r["source_cluster_id"]] = counts.get(r["source_cluster_id"], 0) + 1
    clusters = sorted(counts.keys())
    rng = random.Random(seed)
    rng.shuffle(clusters)
    target = dev_fraction * len(train_rows)
    dev_clusters = set()
    acc = 0
    for cid in clusters:
        if acc >= target:
            break
        dev_clusters.add(cid)
        acc += counts[cid]
    return dev_clusters, acc, len(train_rows)


def repair_leaks(rows):
    """Manifest v2: no source video may straddle a split.

    The official in-distribution split leaves a few source videos with clips on both
    sides. A cluster is the unit of analysis for the bootstrap, so a straddling cluster
    both leaks training material into the audit set and breaks the resampling unit.

    Repair policy frozen at S1 (2026-09-03): the whole cluster follows the training
    side, so the audit set stays a strict subset of the official test split and no clip
    ever moves *into* test. A cluster that already contains dev clips goes to dev as a
    whole; otherwise it goes to train. The dev subset itself is not re-drawn.

    Returns the list of moved clips as (video_id, cluster, from_split, to_split).
    """
    by_cluster = {}
    for r in rows:
        by_cluster.setdefault(r["source_cluster_id"], []).append(r)
    moved = []
    for cid, members in sorted(by_cluster.items()):
        splits = set(r["split"] for r in members)
        if len(splits) <= 1:
            continue
        target = "dev" if "dev" in splits else "train"
        for r in members:
            if r["split"] != target:
                moved.append((r["video_id"], cid, r["split"], target))
                r["split"] = target
    if moved:
        log("leak repair: moved %d clips across %d clusters"
            % (len(moved), len(set(m[1] for m in moved))))
        for vid, cid, a, b in moved:
            log("  %s (cluster %s): %s -> %s" % (vid, cid, a, b))
    else:
        log("leak repair: no cluster straddles a split; nothing moved")
    return moved


def build_real(args):
    csv_path = os.path.join(args.data_root, "metadata-real.csv")
    rows_in = read_rows(csv_path)
    log("read %d rows from %s" % (len(rows_in), csv_path))

    types = sorted(set(r["type"] for r in rows_in))
    log("unique type values: %s" % types)
    unknown = [t for t in types if t not in CLASS_CODES]
    if unknown:
        log("WARNING: type values missing from CLASS_CODES: %s" % unknown)

    rows = []
    for r in rows_in:
        path = r["path"]
        video_id = os.path.splitext(os.path.basename(path))[0]
        duration_s = to_float(r["duration"])
        n_frames = to_int(r["no_frames"])
        anchor_s = to_float(r["accident_time"])
        fps = (n_frames / duration_s) if duration_s and duration_s > 0 else float("nan")
        cname = r["type"]
        rows.append(
            {
                "track_id": "roadside",
                "dataset_id": "ACCIDENT_real",
                "video_id": video_id,
                "path": path,
                "source_cluster_id": cluster_of(video_id),
                "native_anchor_field": "accident_time",
                "anchor_s": anchor_s,
                "anchor_frame": to_int(r["accident_frame"]),
                "fps": fps,
                "duration_s": duration_s,
                "n_frames": n_frames,
                "post_anchor_length_s": duration_s - anchor_s,
                "class_code": CLASS_CODES.get(cname, -1),
                "class_name": cname,
                "map_status": "unique" if cname in CLASS_CODES else "out_of_scope",
                "split": "",  # filled below
                "split_geo": r.get("split_geo_aware", ""),
                "quality": r.get("quality", ""),
                "day_time": r.get("day_time", ""),
                "scene_layout": r.get("scene_layout", ""),
                "region": r.get("region", ""),
                "license_note": LICENSE_NOTE,
                "decode_ok": "",
                "decode_hash": "",
                "_split_raw": r.get("split_in_distribution", ""),
            }
        )

    if args.limit:
        rows = rows[: args.limit]
        log("SMOKE MODE: truncated to %d rows" % len(rows))

    n_clusters = len(set(r["source_cluster_id"] for r in rows))
    log("source clusters: %d for %d clips (%.2f clips per cluster)"
        % (n_clusters, len(rows), len(rows) / max(1, n_clusters)))

    dev_clusters, dev_n, train_n = assign_dev(rows, args.seed, args.dev_fraction)
    log(
        "dev split: %d clusters, %d of %d train clips (%.1f%%), seed=%d"
        % (
            len(dev_clusters),
            dev_n,
            train_n,
            (100.0 * dev_n / train_n) if train_n else 0.0,
            args.seed,
        )
    )
    for r in rows:
        raw = r["_split_raw"]
        r["split_official"] = raw
        if raw == "train" and r["source_cluster_id"] in dev_clusters:
            r["split"] = "dev"
        else:
            r["split"] = raw

    if args.repair_leaks:
        repair_leaks(rows)

    probe_rows = []
    t0 = time.time()
    for idx, r in enumerate(rows):
        abs_path = os.path.join(args.data_root, r["path"])
        if args.md5:
            try:
                r["decode_hash"] = md5_of(abs_path)
            except Exception as exc:
                r["decode_hash"] = ""
                log("md5 failed for %s: %s" % (abs_path, exc))
        if args.decode_check:
            ok, cv_fps, cv_n, note = decode_probe(abs_path)
            r["decode_ok"] = "True" if ok else "False"
            probe_rows.append(
                {
                    "video_id": r["video_id"],
                    "decode_ok": r["decode_ok"],
                    "cv_fps": cv_fps,
                    "cv_n_frames": cv_n,
                    "csv_fps": r["fps"],
                    "csv_n_frames": r["n_frames"],
                    "note": note,
                }
            )
        if (idx + 1) % 100 == 0:
            log("processed %d/%d (%.1fs)" % (idx + 1, len(rows), time.time() - t0))

    out_path = args.out or os.path.join(args.repo_root, "data", "manifest", "manifest_real.csv")
    write_manifest(rows, out_path)
    log("wrote %s (%d rows, %.1fs)" % (out_path, len(rows), time.time() - t0))
    report_splits(rows, out_path)

    if probe_rows:
        probe_path = os.path.join(os.path.dirname(out_path), "decode_probe.csv")
        with open(probe_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["video_id", "decode_ok", "cv_fps", "cv_n_frames", "csv_fps", "csv_n_frames", "note"],
            )
            w.writeheader()
            w.writerows(probe_rows)
        n_bad = sum(1 for p in probe_rows if p["decode_ok"] != "True")
        log("wrote %s (decode failures: %d)" % (probe_path, n_bad))


def build_synthetic(args):
    csv_path = os.path.join(args.data_root, "metadata-synthetic.csv")
    rows_in = read_rows(csv_path)
    log("read %d rows from %s" % (len(rows_in), csv_path))
    types = sorted(set(r["type"] for r in rows_in))
    log("unique type values (synthetic): %s" % types)

    rows = []
    for r in rows_in:
        path = r["rgb_path"]
        video_id = os.path.splitext(os.path.basename(path))[0]
        duration_s = to_float(r["duration"])
        n_frames = to_int(r["no_frames"])
        anchor_s = to_float(r["accident_time"])
        fps = (n_frames / duration_s) if duration_s and duration_s > 0 else float("nan")
        cname = r["type"]
        rows.append(
            {
                "track_id": "synthetic",
                "dataset_id": "ACCIDENT_synthetic",
                "video_id": video_id,
                "path": path,
                "source_cluster_id": cluster_of(video_id),
                "native_anchor_field": "accident_time",
                "anchor_s": anchor_s,
                "anchor_frame": to_int(r["accident_frame"]),
                "fps": fps,
                "duration_s": duration_s,
                "n_frames": n_frames,
                "post_anchor_length_s": duration_s - anchor_s,
                "class_code": CLASS_CODES.get(cname, -1),
                "class_name": cname,
                "map_status": "unique" if cname in CLASS_CODES else "out_of_scope",
                "split": "unassigned",  # no official split column in the synthetic CSV
                "split_geo": "",
                "quality": "",
                "day_time": "",
                "scene_layout": r.get("map", ""),
                "region": r.get("map", ""),
                "license_note": LICENSE_NOTE,
                "decode_ok": "",  # registration only, no decode probe by contract
                "decode_hash": "",
                "split_official": "unassigned",
            }
        )
    out_path = args.out or os.path.join(args.repo_root, "data", "manifest", "manifest_synthetic.csv")
    write_manifest(rows, out_path)
    log("wrote %s (%d rows, registration only)" % (out_path, len(rows)))


def report_splits(rows, out_path):
    """Print the split/cluster tallies and the two digests the S1 freeze records."""
    for label, key in (("official", "split_official"), ("repaired", "split")):
        parts = []
        for sp in ("train", "dev", "test"):
            sub = [r for r in rows if r.get(key) == sp]
            parts.append("%s %d clips / %d clusters"
                         % (sp, len(sub), len(set(r["source_cluster_id"] for r in sub))))
        log("%s split: %s" % (label, "; ".join(parts)))

    n_multi = 0
    by_cluster = {}
    for r in rows:
        by_cluster.setdefault(r["source_cluster_id"], set()).add(r["class_name"])
    n_multi = sum(1 for v in by_cluster.values() if len(v) > 1)
    log("clusters carrying more than one collision type: %d of %d" % (n_multi, len(by_cluster)))

    dev_ids = sorted(r["video_id"] for r in rows if r.get("split") == "dev")
    dev_sha = hashlib.sha256("\n".join(dev_ids).encode("utf-8")).hexdigest()
    log("dev subset: %d clips, video_id list sha256 = %s" % (len(dev_ids), dev_sha))
    log("dev hash recipe: sha256 of the sorted video_id list joined by a single '\\n', "
         "UTF-8, no trailing newline")
    try:
        log("manifest md5 = %s" % md5_of(out_path))
    except Exception as exc:
        log("manifest md5 failed: %s" % exc)


def write_manifest(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Build APE manifest from ACCIDENT_2026 metadata.")
    ap.add_argument("--kind", choices=["real", "synthetic"], default="real")
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke mode: only the first N rows")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--dev-fraction", type=float, default=DEV_FRACTION_DEFAULT)
    ap.add_argument("--md5", dest="md5", action="store_true", default=True)
    ap.add_argument("--no-md5", dest="md5", action="store_false")
    ap.add_argument("--decode-check", dest="decode_check", action="store_true", default=True)
    ap.add_argument("--no-decode-check", dest="decode_check", action="store_false")
    ap.add_argument("--repair-leaks", dest="repair_leaks", action="store_true", default=True,
                    help="manifest v2 default: no source cluster straddles a split")
    ap.add_argument("--no-repair-leaks", dest="repair_leaks", action="store_false",
                    help="reproduce the v1 manifest, which kept the official split verbatim")
    args = ap.parse_args()

    if args.kind == "real":
        build_real(args)
    else:
        args.md5 = False
        args.decode_check = False
        build_synthetic(args)


if __name__ == "__main__":
    main()
