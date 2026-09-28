"""Build the minimal synthetic manifest used by the tests and the smoke run.

Not real data. It imitates the shape of the ACCIDENT manifest (contract 5.1):
60 clips, 5 classes, post-anchor lengths spanning 2 to 20 s, mixed frame rates,
and repeated source video clusters so that the cluster-level bootstrap and the
"whole cluster moves together" rule are actually exercised.

    python -X utf8 tests/fixtures/make_fixture.py
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

SEED = 20260903
N_VIDEOS = 60
CLASS_NAMES = ["head-on", "rear-end", "t-bone", "sideswipe", "single"]
FPS_CHOICES = [4.0, 10.0, 15.0, 24.0, 25.0, 30.0, 50.0]


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    # 42 source videos; 12 of them contribute more than one clip
    n_sources = 42
    sources = [f"SRC{idx:03d}" for idx in range(n_sources)]
    cluster_of = list(sources)
    while len(cluster_of) < N_VIDEOS:
        cluster_of.append(sources[int(rng.integers(0, 12))])
    cluster_of = cluster_of[:N_VIDEOS]

    # Splits are assigned to whole clusters, never to individual clips: the real
    # v2 manifest is leak-repaired so that a source video never straddles a
    # split, and a fixture that violated that would let leakage bugs pass. The
    # proportions mirror the real manifest (test is the large audit split).
    split_rng = np.random.default_rng(SEED + 1)
    uniq_clusters = sorted(set(cluster_of))
    order = split_rng.permutation(len(uniq_clusters))
    n_test = int(round(0.70 * len(uniq_clusters)))
    n_dev = int(round(0.10 * len(uniq_clusters)))
    split_of = {}
    for rank, idx in enumerate(order):
        c = uniq_clusters[int(idx)]
        split_of[c] = "test" if rank < n_test else (
            "dev" if rank < n_test + n_dev else "train"
        )

    seq: dict = {}
    rows = []
    # post-anchor lengths sweep 2..30 s so that every candidate horizon (the
    # 4.11 / 10.03 / 21.82 s bands from the development subset included) has a
    # non-trivial cohort and the longest one censors a large part of the set
    post_lengths = np.round(np.linspace(2.0, 30.0, N_VIDEOS), 2)
    order = rng.permutation(N_VIDEOS)
    post_lengths = post_lengths[order]

    for i in range(N_VIDEOS):
        cluster = cluster_of[i]
        k = seq.get(cluster, 0)
        seq[cluster] = k + 1
        # every fifth clip gets a nested numeric suffix, so that the repeated
        # trailing-digit stripping of the cluster rule is actually exercised
        video_id = f"{cluster}_{k:02d}" if i % 5 else f"{cluster}_{k:02d}_00"
        fps = float(FPS_CHOICES[int(rng.integers(0, len(FPS_CHOICES)))])
        pre = float(np.round(rng.uniform(3.0, 12.0), 2))
        post = float(post_lengths[i])
        anchor_s = pre
        duration_s = float(np.round(pre + post, 2))
        post = float(np.round(duration_s - anchor_s, 6))
        code = int(i % len(CLASS_NAMES))
        rows.append(
            {
                "track_id": "roadside",
                "dataset_id": "SYNTH_fixture",
                "video_id": video_id,
                "path": f"fixture_videos/{video_id}.mp4",
                "source_cluster_id": cluster,
                "native_anchor_field": "accident_time",
                "anchor_s": anchor_s,
                "anchor_frame": int(np.floor(anchor_s * fps)),
                "fps": fps,
                "duration_s": duration_s,
                "n_frames": int(np.floor(duration_s * fps)),
                "post_anchor_length_s": post,
                "class_code": code,
                "class_name": CLASS_NAMES[code],
                "map_status": "unique",
                "split": split_of[cluster],
                "split_geo": "train" if i % 3 else "test",
                "quality": "good",
                "day_time": "day" if i % 2 else "night",
                "scene_layout": "intersection" if i % 3 else "road",
                "region": "synthetic",
                "license_note": "synthetic fixture; not real data",
                "decode_ok": True,
                "decode_hash": hashlib.md5(video_id.encode("utf-8")).hexdigest(),
            }
        )
    return pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    df = build()
    out = os.path.join(here, "manifest_min.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} clips, "
          f"{df['source_cluster_id'].nunique()} clusters, "
          f"post-anchor {df['post_anchor_length_s'].min():g}..."
          f"{df['post_anchor_length_s'].max():g} s")
    print("  splits:", df["split"].value_counts().to_dict())
    straddle = df.groupby("source_cluster_id")["split"].nunique()
    print("  clusters straddling a split:", int((straddle > 1).sum()))
    test = df.loc[df["split"] == "test"]
    for h in (3.0, 4.0, 6.0, 10.0, 21.5):
        n_all = int((df["post_anchor_length_s"] >= h).sum())
        n_test = int((test["post_anchor_length_s"] >= h).sum())
        print(f"  eligible@H{h:g}: all={n_all} test={n_test}")


if __name__ == "__main__":
    main()
