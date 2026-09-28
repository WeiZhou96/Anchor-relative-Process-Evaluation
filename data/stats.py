#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Descriptive statistics of the APE manifest, plus the H-tier candidates.

Produces:
  data/manifest/stats.md      class counts, post-anchor length quantiles, fps and
                              duration distributions, per-split attrition table,
                              and the three H candidates read off the dev subset.
  outputs/figs/data_*.png     matplotlib figures, English labels, white background.

The H candidates are reported only. Freezing them into protocol/pi0.yaml is track A's
job (idea section 3.2 rule 1: the 25/50/75 percentiles of L+ on the dev subset).
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_ORDER = ["train", "dev", "test", "unassigned", ""]
CLASS_ORDER = [(0, "head-on"), (1, "rear-end"), (2, "t-bone"), (3, "sideswipe"), (4, "single")]

# Cool, low-saturation palette; white background; print-safe when scaled down.
PALETTE = ["#3C6E9F", "#6E9BC5", "#9FBBD6", "#2F4858", "#8FA9BF"]


def log(msg):
    sys.stdout.write("[stats] %s\n" % msg)
    sys.stdout.flush()


def read_manifest(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def quantile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def describe(values):
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return dict(n=0, mean=float("nan"), min=float("nan"), q25=float("nan"),
                    median=float("nan"), q75=float("nan"), max=float("nan"))
    return dict(
        n=len(vals),
        mean=sum(vals) / len(vals),
        min=vals[0],
        q25=quantile(vals, 0.25),
        median=quantile(vals, 0.5),
        q75=quantile(vals, 0.75),
        max=vals[-1],
    )


def md_stat_row(name, s):
    return "| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n" % (
        name, s["n"], s["mean"], s["min"], s["q25"], s["median"], s["q75"], s["max"])


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDE3E9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 160,
    })
    return plt


def fig_post_anchor(plt, rows, h_candidates, out_dir):
    all_vals = [fnum(r, "post_anchor_length_s") for r in rows]
    dev_vals = [fnum(r, "post_anchor_length_s") for r in rows if r["split"] == "dev"]
    all_vals = [v for v in all_vals if not math.isnan(v)]
    dev_vals = [v for v in dev_vals if not math.isnan(v)]
    # Densities, not counts: the dev subset is 5% of the clips and would be invisible
    # on a shared count axis.
    hi = max(all_vals) if all_vals else 1.0
    bins = [hi * k / 40.0 for k in range(41)]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.hist(all_vals, bins=bins, density=True, color=PALETTE[2], edgecolor="white",
            linewidth=0.4, label="all clips (n=%d)" % len(all_vals))
    if dev_vals:
        ax.hist(dev_vals, bins=bins, density=True, histtype="step", color=PALETTE[3],
                linewidth=1.3, label="dev subset (n=%d)" % len(dev_vals))
    ymax = ax.get_ylim()[1]
    for k, (h, style) in enumerate(zip(h_candidates, ["--", "-.", ":"])):
        ax.axvline(h, color=PALETTE[0], linestyle=style, linewidth=1.0)
        ax.annotate("H=%.1fs" % h, xy=(h, ymax * (0.95 - 0.09 * k)), xytext=(3, 0),
                    textcoords="offset points", fontsize=7, color=PALETTE[0])
    ax.set_xlabel("post-anchor length $L^{+}$ (s)")
    ax.set_ylabel("density")
    ax.set_title("Post-anchor remaining length")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "data_post_anchor_length.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_simple_hist(plt, values, xlabel, title, fname, out_dir, bins=40):
    vals = [v for v in values if not math.isnan(v)]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.hist(vals, bins=bins, color=PALETTE[1], edgecolor="white", linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("clips")
    ax.set_title(title)
    fig.tight_layout()
    p = os.path.join(out_dir, fname)
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_class_by_split(plt, rows, splits, out_dir):
    counts = defaultdict(Counter)
    for r in rows:
        counts[r["split"]][r["class_name"]] += 1
    names = [n for _, n in CLASS_ORDER]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    width = 0.8 / max(1, len(splits))
    xs = range(len(names))
    for k, sp in enumerate(splits):
        vals = [counts[sp][n] for n in names]
        ax.bar([x + k * width for x in xs], vals, width=width, color=PALETTE[k % len(PALETTE)],
               edgecolor="white", linewidth=0.5, label=sp)
    ax.set_xticks([x + width * (len(splits) - 1) / 2.0 for x in xs])
    ax.set_xticklabels(names)
    ax.set_ylabel("clips")
    ax.set_title("Collision-type counts by split")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "data_class_by_split.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_cohort_curve(plt, rows, splits, h_candidates, out_dir):
    grid = [x * 0.5 for x in range(1, 61)]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for k, sp in enumerate(splits):
        vals = [fnum(r, "post_anchor_length_s") for r in rows if r["split"] == sp]
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            continue
        ys = [100.0 * sum(1 for v in vals if v >= h) / len(vals) for h in grid]
        ax.plot(grid, ys, color=PALETTE[k % len(PALETTE)], linewidth=1.4, label="%s (n=%d)" % (sp, len(vals)))
    for h, style in zip(h_candidates, ["--", "-.", ":"]):
        ax.axvline(h, color=PALETTE[3], linestyle=style, linewidth=0.9)
    ax.set_xlabel("horizon $H$ (s)")
    ax.set_ylabel("eligible cohort size (% of split)")
    ax.set_title("Eligibility cohort vs horizon")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "data_cohort_vs_h.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "data", "manifest", "manifest_real.csv"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "manifest", "stats.md"))
    ap.add_argument("--figs", default=os.path.join(REPO_ROOT, "outputs", "figs"))
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    log("read %d rows from %s" % (len(rows), args.manifest))
    os.makedirs(args.figs, exist_ok=True)

    present = [s for s in SPLIT_ORDER if any(r["split"] == s for r in rows)]
    splits = [s for s in present if s]

    dev_lplus = sorted(v for v in (fnum(r, "post_anchor_length_s") for r in rows if r["split"] == "dev")
                       if not math.isnan(v))
    if dev_lplus:
        h_candidates = [round(quantile(dev_lplus, p), 2) for p in (0.25, 0.5, 0.75)]
        h_source = "dev 子集 L+ 的 25/50/75 分位（n=%d）" % len(dev_lplus)
    else:
        all_lplus = sorted(v for v in (fnum(r, "post_anchor_length_s") for r in rows) if not math.isnan(v))
        h_candidates = [round(quantile(all_lplus, p), 2) for p in (0.25, 0.5, 0.75)]
        h_source = "全集 L+ 的 25/50/75 分位（dev 为空，退化口径）"
    log("H candidates: %s" % h_candidates)

    plt = setup_mpl()
    figs = []
    figs.append(fig_post_anchor(plt, rows, h_candidates, args.figs))
    figs.append(fig_simple_hist(plt, [fnum(r, "fps") for r in rows], "fps (= n_frames / duration)",
                                "Frame-rate distribution", "data_fps.png", args.figs))
    figs.append(fig_simple_hist(plt, [fnum(r, "duration_s") for r in rows], "clip duration (s)",
                                "Clip duration distribution", "data_duration.png", args.figs))
    figs.append(fig_class_by_split(plt, rows, splits, args.figs))
    figs.append(fig_cohort_curve(plt, rows, splits, h_candidates, args.figs))
    for p in figs:
        log("wrote %s" % p)

    L = []
    L.append("# manifest 统计（stats.md）\n")
    L.append("\n由 `data/stats.py` 生成，输入 `%s`，共 %d 段。图见 `outputs/figs/data_*.png`。\n" % (args.manifest, len(rows)))

    L.append("\n## 1. 按 split 的片段数与簇数\n\n| split | 片段数 | 源事件簇数 |\n|---|---|---|\n")
    for sp in splits:
        sub = [r for r in rows if r["split"] == sp]
        L.append("| %s | %d | %d |\n" % (sp, len(sub), len(set(r["source_cluster_id"] for r in sub))))
    L.append("| **合计** | %d | %d |\n" % (len(rows), len(set(r["source_cluster_id"] for r in rows))))

    L.append("\n## 2. 类分布（按 split）\n\n| class_code | class_name | %s | 合计 |\n|---|---|%s---|\n"
             % (" | ".join(splits), "---|" * len(splits)))
    for code, name in CLASS_ORDER:
        cells = []
        total = 0
        for sp in splits:
            c = sum(1 for r in rows if r["split"] == sp and r["class_name"] == name)
            cells.append(str(c))
            total += c
        L.append("| %d | %s | %s | %d |\n" % (code, name, " | ".join(cells), total))
    other = [r for r in rows if r["class_name"] not in [n for _, n in CLASS_ORDER]]
    if other:
        L.append("| -1 | (未映射) | %s | %d |\n"
                 % (" | ".join(str(sum(1 for r in other if r["split"] == sp)) for sp in splits), len(other)))

    L.append("\n## 3. 锚后剩余长度 L+、时长与帧率\n\n")
    L.append("| 量 | n | mean | min | q25 | median | q75 | max |\n|---|---|---|---|---|---|---|---|\n")
    L.append(md_stat_row("L+ 全集 (s)", describe(fnum(r, "post_anchor_length_s") for r in rows)))
    for sp in splits:
        L.append(md_stat_row("L+ %s (s)" % sp,
                             describe(fnum(r, "post_anchor_length_s") for r in rows if r["split"] == sp)))
    L.append(md_stat_row("duration 全集 (s)", describe(fnum(r, "duration_s") for r in rows)))
    L.append(md_stat_row("fps 全集", describe(fnum(r, "fps") for r in rows)))
    L.append(md_stat_row("anchor_s 全集 (s)", describe(fnum(r, "anchor_s") for r in rows)))

    L.append("\n## 4. H 三档候选（只报告，不写入 protocol）\n")
    L.append("\n依据：%s。\n" % h_source)
    L.append("\n| 档 | 分位 | H 候选 (s) |\n|---|---|---|\n")
    for tier, p, h in zip(["短", "中", "长"], ["25%", "50%", "75%"], h_candidates):
        L.append("| %s | %s | %.2f |\n" % (tier, p, h))
    L.append("\n各档在各 split 上的合格集合规模（`L+ >= H` 的片段数与占比）：\n")
    L.append("\n| split | 片段数 | %s |\n" % " | ".join("N_H(H=%.2f)" % h for h in h_candidates))
    L.append("|---|---|" + "---|" * len(h_candidates) + "\n")
    for sp in splits:
        sub = [fnum(r, "post_anchor_length_s") for r in rows if r["split"] == sp]
        sub = [v for v in sub if not math.isnan(v)]
        cells = []
        for h in h_candidates:
            n = sum(1 for v in sub if v >= h)
            cells.append("%d (%.1f%%)" % (n, 100.0 * n / len(sub) if sub else 0.0))
        L.append("| %s | %d | %s |\n" % (sp, len(sub), " | ".join(cells)))
    L.append("\n注：H 档由 A 在 S1 冻结进 `protocol/pi0.yaml`；本文件只给候选值，不代表已冻结。\n")

    L.append("\n### 4.1 锚后长度退化的片段\n")
    L.append("\n锚点落在片尾附近时，锚后网格上一个评测点都排不出，这类片段在任何 H 档下都不入合格集合，"
             "但会出现在原始段数里，报 attrition 时不能忽略。\n")
    L.append("\n| 判据 | 片段数 | 占比 |\n|---|---|---|\n")
    lall = [fnum(r, "post_anchor_length_s") for r in rows]
    lall = [v for v in lall if not math.isnan(v)]
    for label, thr in [("L+ = 0", 0.0), ("L+ < 0.25 s", 0.25), ("L+ < 0.5 s", 0.5), ("L+ < 1 s", 1.0)]:
        n = sum(1 for v in lall if (v <= 0.0 if thr == 0.0 else v < thr))
        L.append("| %s | %d | %.2f%% |\n" % (label, n, 100.0 * n / len(lall) if lall else 0.0))
    n_neg = sum(1 for v in lall if v < 0.0)
    if n_neg:
        L.append("\n另有 %d 段 `L+ < 0`（锚点时刻超过片长），属字段异常，应在 S1 前按预注册规则处理。\n" % n_neg)

    L.append("\n### 4.2 dev 与 audit（test）的 L+ 分布是否一致\n")
    L.append("\nH 档在 dev 上取分位、合格集合却在 audit 集上形成。两边 L+ 分布若不一致，"
             "H 的三档在 audit 集上就不再对应 25/50/75 的位置，合格集合会系统性地偏大或偏小。"
             "这不是错误，但必须在正文写出来，并进 G0 的判据。\n")
    dev_v = sorted(v for v in (fnum(r, "post_anchor_length_s") for r in rows if r["split"] == "dev")
                   if not math.isnan(v))
    test_v = sorted(v for v in (fnum(r, "post_anchor_length_s") for r in rows if r["split"] == "test")
                    if not math.isnan(v))
    if dev_v and test_v:
        L.append("\n| 量 | dev | test | 差 |\n|---|---|---|---|\n")
        for label, p in [("L+ 25% 分位", 0.25), ("L+ 50% 分位", 0.5), ("L+ 75% 分位", 0.75)]:
            a, b = quantile(dev_v, p), quantile(test_v, p)
            L.append("| %s | %.2f s | %.2f s | %+.2f s |\n" % (label, a, b, b - a))
        for h in h_candidates:
            a = 100.0 * sum(1 for v in dev_v if v >= h) / len(dev_v)
            b = 100.0 * sum(1 for v in test_v if v >= h) / len(test_v)
            L.append("| 合格比例 @H=%.2f | %.1f%% | %.1f%% | %+.1f pt |\n" % (h, a, b, b - a))
        gaps = [100.0 * sum(1 for v in test_v if v >= h) / len(test_v)
                - 100.0 * sum(1 for v in dev_v if v >= h) / len(dev_v) for h in h_candidates]
        if max(abs(g) for g in gaps) >= 10.0:
            n_tr = sum(1 for r in rows if r["split"] in ("train", "dev"))
            n_te = sum(1 for r in rows if r["split"] == "test")
            L.append("\n**注意**：某一档上两边的合格比例相差 %.1f 个百分点。"
                     "官方 IID 划分本身就把长锚后片段更多地放在了 test 一侧"
                     "（训练划分 %d 段，其 L+ 中位 %.2f s；审计集 %d 段，其 L+ 中位 %.2f s），"
                     "所以这不是本库切 dev 时引入的。处理办法二选一，须在 S1 前定死并写进预注册："
                     "① 仍按 dev 分位取 H，正文报出两边分布差与各档的实际 N_H；"
                     "② 改按 train+dev 整个训练划分的分位取 H。两种都不允许在看到审计结果之后再换。\n"
                     % (max(abs(g) for g in gaps), n_tr,
                        quantile(sorted(v for v in (fnum(r, "post_anchor_length_s") for r in rows
                                                    if r["split"] in ("train", "dev"))
                                        if not math.isnan(v)), 0.5),
                        n_te, quantile(test_v, 0.5)))

    L.append("\n## 5. attrition 表（按 split）\n\n")
    L.append("| split | Raw | Readable | Anchor-valid | Closed-set unique | Deduplicated (簇数) | %s |\n"
             % " | ".join("Eligible@H=%.2f" % h for h in h_candidates))
    L.append("|---|---|---|---|---|---|" + "---|" * len(h_candidates) + "\n")
    for sp in splits + ["ALL"]:
        sub = rows if sp == "ALL" else [r for r in rows if r["split"] == sp]
        raw = len(sub)
        readable = sum(1 for r in sub if r.get("decode_ok", "") != "False")
        anchor_ok = sum(1 for r in sub
                        if r.get("decode_ok", "") != "False"
                        and not math.isnan(fnum(r, "anchor_s"))
                        and 0.0 <= fnum(r, "anchor_s") <= fnum(r, "duration_s"))
        uniq = sum(1 for r in sub
                   if r.get("decode_ok", "") != "False"
                   and 0.0 <= fnum(r, "anchor_s") <= fnum(r, "duration_s")
                   and r["map_status"] == "unique")
        dedup = len(set(r["source_cluster_id"] for r in sub
                        if r.get("decode_ok", "") != "False" and r["map_status"] == "unique"))
        cells = []
        for h in h_candidates:
            cells.append(str(sum(1 for r in sub
                                 if r.get("decode_ok", "") != "False"
                                 and r["map_status"] == "unique"
                                 and fnum(r, "post_anchor_length_s") >= h)))
        L.append("| %s | %d | %d | %d | %d | %d | %s |\n" % (sp, raw, readable, anchor_ok, uniq, dedup, " | ".join(cells)))
    n_bad = sum(1 for r in rows if r.get("decode_ok", "") == "False")
    L.append("\nOpenCV 解码失败 %d 段。Readable 列按 `decode_ok != False` 统计；若 manifest 生成时关闭了解码检查，该列等于 Raw。\n" % n_bad)

    L.append("\n## 6. 图\n\n")
    for p in figs:
        L.append("- `%s`\n" % p)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("".join(L))
    log("wrote %s" % args.out)


if __name__ == "__main__":
    main()
