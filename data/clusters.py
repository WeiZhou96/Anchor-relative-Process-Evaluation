#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Source-event cluster diagnostics for the APE manifest.

Three jobs, all report-only (the manifest itself is never rewritten):
  1. cluster size distribution;
  2. cross-split leakage check (one cluster whose clips land in different splits),
     plus an advisory repaired split;
  3. perceptual-hash (pHash) distance distribution over sampled clip pairs at the
     anchor frame, giving a threshold candidate and a spot-check table.

Freeze discipline (idea section 3.8): everything looked at before the S1 freeze must
stay inside the training portion. The pHash sampling therefore never touches clips
whose split is 'test'.

Output: data/manifest/clusters_report.md
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT_DEFAULT = os.environ.get("APE_ACCIDENT", os.path.join(REPO_ROOT, "external", "ACCIDENT_2026"))


def log(msg):
    sys.stdout.write("[clusters] %s\n" % msg)
    sys.stdout.flush()


def read_manifest(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def size_distribution(rows):
    sizes = Counter()
    for r in rows:
        sizes[r["source_cluster_id"]] += 1
    hist = Counter(sizes.values())
    return sizes, hist


def leakage(rows):
    """Return {cluster: {split: count}} for clusters spanning more than one split."""
    by_cluster = defaultdict(Counter)
    for r in rows:
        by_cluster[r["source_cluster_id"]][r["split"]] += 1
    return {c: dict(s) for c, s in by_cluster.items() if len(s) > 1}


def repair_split(splits_of_cluster):
    """The repair policy frozen at S1 (2026-09-03): the whole cluster follows the
    training side, so the audit set stays a strict subset of the official test split and
    no clip is ever moved *into* test. A cluster containing dev clips goes to dev.

    Same rule as data/build_manifest.py: repair_leaks. Manifest v2 already applies it, so
    this report should now find no straddling cluster; the function stays here to catch a
    regression if the manifest is ever rebuilt with --no-repair-leaks.
    """
    for s in ("dev", "train"):
        if s in splits_of_cluster:
            return s
    return sorted(splits_of_cluster)[0]


def read_anchor_frame(data_root, rel_path, anchor_frame):
    import cv2

    cap = cv2.VideoCapture(os.path.join(data_root, rel_path))
    if not cap.isOpened():
        return None
    try:
        idx = max(0, int(anchor_frame))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def phash_pairs(rows, data_root, n_pairs, seed):
    """Sample within-cluster and cross-cluster pairs, return the spot-check table."""
    import imagehash
    from PIL import Image

    pool = [r for r in rows if r["split"] in ("train", "dev") and r.get("decode_ok", "True") != "False"]
    by_cluster = defaultdict(list)
    for r in pool:
        by_cluster[r["source_cluster_id"]].append(r)
    multi = [c for c, v in by_cluster.items() if len(v) >= 2]

    rng = random.Random(seed)
    want_within = n_pairs // 2

    # Enumerate every distinct within-cluster pair available in train/dev; there are few,
    # so sampling with replacement would just repeat the same handful of clips.
    all_within = []
    for c in sorted(multi):
        members = sorted(by_cluster[c], key=lambda r: r["video_id"])
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                all_within.append((members[i], members[j], "within"))
    n_within_available = len(all_within)
    rng.shuffle(all_within)
    within = all_within[:want_within]

    seen = set()
    cross = []
    clusters = sorted(by_cluster.keys())
    want_cross = n_pairs - len(within)
    for _ in range(want_cross * 20):
        if len(cross) >= want_cross:
            break
        c1, c2 = rng.sample(clusters, 2)
        a, b = rng.choice(by_cluster[c1]), rng.choice(by_cluster[c2])
        key = tuple(sorted((a["video_id"], b["video_id"])))
        if key in seen:
            continue
        seen.add(key)
        cross.append((a, b, "cross"))

    pairs = within + cross
    log("sampled %d pairs (%d within-cluster of %d distinct available, %d cross-cluster) "
        "from %d train/dev clips in %d clusters (%d multi-clip)"
        % (len(pairs), len(within), n_within_available, len(cross), len(pool),
           len(by_cluster), len(multi)))

    cache = {}

    def get_hash(r):
        vid = r["video_id"]
        if vid in cache:
            return cache[vid]
        frame = read_anchor_frame(data_root, r["path"], float(r["anchor_frame"] or 0))
        h = None if frame is None else imagehash.phash(Image.fromarray(frame))
        cache[vid] = h
        return h

    table = []
    for a, b, kind in pairs:
        ha, hb = get_hash(a), get_hash(b)
        if ha is None or hb is None:
            table.append({"kind": kind, "video_a": a["video_id"], "video_b": b["video_id"], "hamming": ""})
            continue
        table.append(
            {
                "kind": kind,
                "video_a": a["video_id"],
                "video_b": b["video_id"],
                "cluster_a": a["source_cluster_id"],
                "cluster_b": b["source_cluster_id"],
                "hamming": int(ha - hb),
            }
        )
    meta = {
        "pool": len(pool),
        "clusters": len(by_cluster),
        "multi_clusters": len(multi),
        "within_available": n_within_available,
    }
    return table, meta


def summarize(values):
    vals = sorted(values)
    if not vals:
        return {}
    def q(p):
        k = (len(vals) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)
    return {"n": len(vals), "min": vals[0], "q25": q(0.25), "median": q(0.5), "q75": q(0.75), "max": vals[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "data", "manifest", "manifest_real.csv"))
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "manifest", "clusters_report.md"))
    ap.add_argument("--repair-out", default=os.path.join(REPO_ROOT, "data", "manifest", "split_repair_suggestion.csv"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--no-phash", action="store_true")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    log("read %d manifest rows" % len(rows))

    sizes, hist = size_distribution(rows)
    leaks = leakage(rows)
    log("clusters: %d; leaking clusters: %d" % (len(sizes), len(leaks)))

    table, meta = [], {}
    if not args.no_phash:
        try:
            table, meta = phash_pairs(rows, args.data_root, args.pairs, args.seed)
        except Exception as exc:
            log("pHash step failed: %s" % exc)
            table, meta = [], {}

    within_d = [t["hamming"] for t in table if t["kind"] == "within" and t["hamming"] != ""]
    cross_d = [t["hamming"] for t in table if t["kind"] == "cross" and t["hamming"] != ""]

    # Threshold candidate: below the smallest cross-cluster distance seen, i.e. a value
    # that would not merge any sampled unrelated pair. Reported as a candidate only.
    if cross_d:
        thr_candidate = max(0, min(cross_d) - 1)
    else:
        thr_candidate = None

    if leaks:
        os.makedirs(os.path.dirname(args.repair_out), exist_ok=True)
        with open(args.repair_out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["source_cluster_id", "splits_observed", "repaired_split", "n_clips"])
            w.writeheader()
            for c, s in sorted(leaks.items()):
                w.writerow(
                    {
                        "source_cluster_id": c,
                        "splits_observed": ";".join("%s:%d" % kv for kv in sorted(s.items())),
                        "repaired_split": repair_split(s),
                        "n_clips": sum(s.values()),
                    }
                )
        log("wrote %s" % args.repair_out)

    lines = []
    lines.append("# 源事件簇报告（clusters_report.md）\n")
    lines.append("由 `data/clusters.py` 生成，只读报告，不改写 manifest 的簇定义。\n")
    lines.append("manifest：`%s`，共 %d 段。\n" % (args.manifest, len(rows)))

    lines.append("\n## 1. 簇规模分布\n")
    lines.append("簇总数 %d，平均每簇 %.2f 段。\n" % (len(sizes), len(rows) / max(1, len(sizes))))
    lines.append("\n| 每簇片段数 | 簇数 | 覆盖片段数 |\n|---|---|---|\n")
    for k in sorted(hist):
        lines.append("| %d | %d | %d |\n" % (k, hist[k], k * hist[k]))
    top = sorted(sizes.items(), key=lambda kv: -kv[1])[:10]
    lines.append("\n最大的 10 个簇：%s\n" % ", ".join("%s(%d)" % kv for kv in top))

    lines.append("\n## 2. 跨 split 泄漏检查\n")
    if not leaks:
        lines.append("\n未发现同簇片段被切到不同 split 的情况（泄漏数 0）。"
                     "manifest v2 已在生成时应用整簇归并，本项应恒为 0；若不为 0，"
                     "说明 manifest 是用 `--no-repair-leaks` 生成的 v1。\n")
    else:
        lines.append("\n发现 %d 个簇跨 split，共 %d 段。修正建议见 `split_repair_suggestion.csv`；"
                     "修正规则与 S1 冻结一致：整簇跟随训练侧（含 dev 则整簇进 dev，否则整簇进 train），"
                     "审计集因此始终是官方 test 的子集，任何片段都不会被移入 test。\n"
                     % (len(leaks), sum(sum(s.values()) for s in leaks.values())))
        lines.append("\n| 簇 | 观察到的 split | 修正后 |\n|---|---|---|\n")
        for c, s in sorted(leaks.items())[:40]:
            lines.append("| %s | %s | %s |\n" % (c, ", ".join("%s:%d" % kv for kv in sorted(s.items())), repair_split(s)))
        if len(leaks) > 40:
            lines.append("\n（只列前 40 个，全量见 csv）\n")

    lines.append("\n## 3. 感知哈希（pHash，锚点帧）抽检\n")
    if not table:
        lines.append("\n本次未执行 pHash 抽检（`--no-phash` 或依赖缺失）。\n")
    else:
        lines.append("\n抽样限定在 split 为 train 或 dev 的片段内，审计集（test）不参与，遵守 idea §3.8 的冻结纪律。"
                     "抽样池 %d 段、%d 个簇，其中多片段簇 %d 个，可枚举的同簇对共 %d 个（本次全部取用或按上限截取）。\n"
                     % (meta.get("pool", 0), meta.get("clusters", 0), meta.get("multi_clusters", 0),
                        meta.get("within_available", 0)))
        lines.append("\n| 组 | 对数 | min | q25 | median | q75 | max |\n|---|---|---|---|---|---|---|\n")
        for name, vals in (("同簇", within_d), ("跨簇", cross_d)):
            s = summarize(vals)
            if s:
                lines.append("| %s | %d | %g | %g | %g | %g | %g |\n"
                             % (name, s["n"], s["min"], s["q25"], s["median"], s["q75"], s["max"]))
            else:
                lines.append("| %s | 0 | - | - | - | - | - |\n" % name)
        separable = bool(within_d) and bool(cross_d) and max(within_d) < min(cross_d)
        if thr_candidate is not None and separable:
            lines.append("\n**pHash 阈值候选**：汉明距离 ≤ %d 判为同源。两组分布不重叠，该值取跨簇对最小距离减一，"
                         "落在两组之间；正式阈值仍须在开发子集上按预注册的误合率上限冻结。\n" % thr_candidate)
        elif thr_candidate is not None:
            lines.append("\n**pHash 阈值候选与它的实际含义**：本次抽样中同簇与跨簇的距离分布完全重叠"
                         "（同簇 %g–%g，跨簇 %g–%g），不存在能把同簇对判进、把跨簇对挡在外的阈值。"
                         "原因是同一源视频切出的多个片段本来就是同一路摄像头下不同时刻的不同事故，锚点帧画面并不相似；"
                         "pHash 在这里不承担簇的定义。\n"
                         "\n因此阈值候选只能按另一个用途给：把 pHash 当**近重复检测器**，用一个严格阈值"
                         "（建议 ≤ %d，即远小于本次观察到的任何一对距离 %g）去找“源视频 ID 不同但画面近乎重复”的转载片段。"
                         "本次 %d 对抽样中没有任何一对落到该阈值以下，即未观察到 ID 看不见的近重复。"
                         "该阈值仍须在开发子集上按预注册的误合率上限冻结，冻结前不得据此剔除任何片段。\n"
                         "\n本库的簇定义因此仍以源视频 ID 为准：pHash 只用于补查 ID 看不见的漏合，"
                         "不用于拆分已由 ID 合并的簇。\n"
                         % (min(within_d), max(within_d), min(cross_d), max(cross_d),
                            max(0, min(min(within_d), min(cross_d)) // 2),
                            min(min(within_d), min(cross_d)), len(within_d) + len(cross_d)))
        if not within_d:
            lines.append("\n注意：本次抽样在 train/dev 内没有取到同簇对（该划分内的簇多为单片段），"
                         "因此没有正样本一侧的距离分布，阈值候选只有上界一侧的证据；"
                         "补正样本需要动用 test 侧的多片段簇，那会违反冻结纪律，本轮不做。\n")
        elif meta.get("within_available", 0) < 20:
            lines.append("\n注意：train/dev 内可枚举的同簇对只有 %d 个，正样本一侧的分布来自极少数源视频，"
                         "分位数只作参考；正式阈值冻结前应在开发子集上扩大正样本来源，"
                         "或直接承认 pHash 只作源视频 ID 之外的补充判据。\n" % meta.get("within_available", 0))
        lines.append("\n### 抽检表（前 40 对，按距离升序）\n")
        lines.append("\n| 组 | 片段 A | 片段 B | 汉明距离 |\n|---|---|---|---|\n")
        ordered = sorted([t for t in table if t["hamming"] != ""], key=lambda t: t["hamming"])[:40]
        for t in ordered:
            lines.append("| %s | %s | %s | %s |\n" % (t["kind"], t["video_a"], t["video_b"], t["hamming"]))
        n_fail = sum(1 for t in table if t["hamming"] == "")
        if n_fail:
            lines.append("\n有 %d 对因锚点帧读取失败未计入。\n" % n_fail)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    log("wrote %s" % args.out)


if __name__ == "__main__":
    main()
