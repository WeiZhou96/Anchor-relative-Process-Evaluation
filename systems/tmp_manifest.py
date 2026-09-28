"""Temporary manifest for the B track (CONTRACT 7.4).

Reads ``metadata-real.csv`` directly and emits the CONTRACT 5.1 columns to
``outputs/tmp_manifest_b.csv``.  It is a stand-in until the C track publishes
``data/manifest/manifest_real.csv``; every consumer in ``systems/`` reads the
manifest through ``common.load_manifest``, so switching over is a path change.

Usage:
    python -X utf8 systems/tmp_manifest.py [--no-decode-check] [--no-md5] [--out PATH]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402


def carve_dev(man: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Carve a dev subset out of the IID train split at source-cluster granularity."""
    train = man[man["split"] == "train"]
    clusters = np.array(sorted(train["source_cluster_id"].unique().tolist()))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clusters))
    n_dev = int(round(fraction * len(clusters)))
    dev_clusters = set(clusters[order[:n_dev]].tolist())
    is_dev = man["split"].eq("train") & man["source_cluster_id"].isin(dev_clusters)
    man = man.copy()
    man.loc[is_dev, "split"] = "dev"
    return man


def check_decode(abs_path: str) -> bool:
    import cv2

    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def build(csv_path: str, data_root: str, do_decode_check: bool, do_md5: bool) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    unknown = sorted(set(raw["type"].astype(str)) - set(C.CLASS_TO_CODE))
    if unknown:
        raise ValueError(f"unexpected type strings in CSV, not in CONTRACT 4 table: {unknown}")

    video_id = raw["path"].astype(str).str.rsplit("/", n=1).str[-1].str.replace(r"\.mp4$", "", regex=True)
    fps = raw["no_frames"].astype(float) / raw["duration"].astype(float)

    man = pd.DataFrame({
        "track_id": C.TRACK_ID,
        "dataset_id": C.DATASET_ID,
        "video_id": video_id,
        "path": raw["path"].astype(str),
        "source_cluster_id": [C.source_cluster_id(v) for v in video_id],
        "native_anchor_field": "accident_time",
        "anchor_s": raw["accident_time"].astype(float),
        "anchor_frame": raw["accident_frame"].astype("int64"),
        "fps": fps,
        "duration_s": raw["duration"].astype(float),
        "n_frames": raw["no_frames"].astype("int64"),
        "post_anchor_length_s": raw["duration"].astype(float) - raw["accident_time"].astype(float),
        "class_code": raw["type"].astype(str).map(C.CLASS_TO_CODE).astype("int64"),
        "class_name": raw["type"].astype(str),
        "map_status": "unique",
        "split": raw["split_in_distribution"].astype(str),
        "split_geo": raw["split_geo_aware"].astype(str),
        "quality": raw["quality"],
        "day_time": raw["day_time"],
        "scene_layout": raw["scene_layout"],
        "region": raw["region"],
        "license_note": C.LICENSE_NOTE,
    })

    man = carve_dev(man, C.DEV_FRACTION, C.SEED)

    decode_ok = []
    decode_hash = []
    for rel in man["path"].tolist():
        abs_path = os.path.join(data_root, rel)
        if not os.path.exists(abs_path):
            decode_ok.append(False)
            decode_hash.append("")
            continue
        decode_ok.append(check_decode(abs_path) if do_decode_check else True)
        decode_hash.append(C.md5_of_file(abs_path) if do_md5 else "")
    man["decode_ok"] = decode_ok
    man["decode_hash"] = decode_hash
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=C.METADATA_REAL)
    ap.add_argument("--data-root", default=C.DATA_ROOT)
    ap.add_argument("--out", default=C.TMP_MANIFEST_B)
    ap.add_argument("--no-decode-check", action="store_true")
    ap.add_argument("--no-md5", action="store_true")
    args = ap.parse_args()

    man = build(args.csv, args.data_root, not args.no_decode_check, not args.no_md5)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    man.to_csv(args.out, index=False)

    print(f"wrote {args.out}  rows={len(man)}  cols={len(man.columns)}")
    print("split counts:")
    print(man["split"].value_counts().to_string())
    print("clusters per split:")
    print(man.groupby("split", observed=True)["source_cluster_id"].nunique().to_string())
    print("class_code counts (all):")
    print(man["class_code"].value_counts().sort_index().to_string())
    print("class_code x split:")
    print(pd.crosstab(man["split"], man["class_code"]).to_string())
    print(f"decode_ok False: {int((~man['decode_ok']).sum())}")
    post = man["post_anchor_length_s"]
    print("post_anchor_length_s q25/q50/q75: "
          f"{post.quantile(0.25):.3f} {post.quantile(0.50):.3f} {post.quantile(0.75):.3f}")
    for h in C.H_PLACEHOLDERS:
        print(f"eligible@H={h}: all={int((post >= h).sum())} "
              f"test={int(((post >= h) & man['split'].eq('test')).sum())} "
              f"dev={int(((post >= h) & man['split'].eq('dev')).sum())}")
    # leakage check: a source cluster must not straddle splits
    straddle = man.groupby("source_cluster_id", observed=True)["split"].nunique()
    print(f"clusters straddling splits: {int((straddle > 1).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
