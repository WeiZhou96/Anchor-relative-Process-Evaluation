#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render idea section 5.5 tables 1 to 5 from track A's real outputs into outputs/tables/.

Reads the layout documented in report/ape_io.py. Nothing is invented: a value A did not
produce renders as 'n/a' and the reason is stated in the table notes.

Discipline enforced here:
  - gauge blocks (family=block) are marked and excluded from ranks and from any
    cross-system determination (contract section 8);
  - stand-in systems (family=standin) are marked PLACEHOLDER in every row and every
    note, and no conclusion may rest on them;
  - the main table 3 uses the metric entries without '[with blocks]'; the block-inclusive
    series is written alongside as table3b and never merged.

Tables 1, 2 and 5 are produced once per reference protocol (one H each, since A's
pi_hash encodes H); the mid-H copy is also written under the plain name.
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ape_io as io  # noqa: E402
import r2_tables

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    sys.stdout.write("[make_tables] %s\n" % msg)
    sys.stdout.flush()


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        return v
    try:
        f = float(v)
        if math.isnan(f):
            return "n/a"
        return ("%%.%df" % nd) % f
    except (TypeError, ValueError):
        return str(v)


def htag(h):
    return ("H%s" % fmt(h, 2)).replace(".", "p")


def write_table(out_dir, name, header, rows, title, notes):
    if any(len(row) != len(header) for row in rows):
        raise ValueError("Header/data width mismatch in " + name)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, name + ".csv")
    md_path = os.path.join(out_dir, name + ".md")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# %s\n\n" % title)
        fh.write("| " + " | ".join(header) + " |\n")
        fh.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", "<br>") for c in r) + " |\n")
        if notes:
            fh.write("\n")
            for n in notes:
                fh.write("- %s\n" % n)
    log("wrote %s (%d rows)" % (md_path, len(rows)))
    return md_path


def ranks_within(values, ascending, groups):
    out = [None] * len(values)
    for g in sorted(set(groups)):
        idx = [i for i, v in enumerate(values)
               if groups[i] == g and v is not None and not (isinstance(v, float) and math.isnan(v))]
        for r, i in enumerate(sorted(idx, key=lambda i: values[i], reverse=not ascending), start=1):
            out[i] = r
    return out


# --------------------------------------------------------------------------- table 1

def table1(recs, pairs, h_s, pi_hash, out_dir, suffix):
    header = ["Track", "System", "Class", "arm_rule", "Arm value", "Seed", "backbone", "model_kind", "library_round", "train_data_unknown",
              "Full macro-Acc", "End-window macro-Acc",
              "RMSCD@H", "RMSCD 95% CI", "Median flips", "Commit (rho, tau_c, e_c)",
              "N_H", "Rank by End-window", "Rank by Progress"]
    rows, end_accs, rmscds, tracks, rankable = [], [], [], [], []
    for rec in sorted(recs, key=lambda r: (io.is_block(r), r.get("system_id", ""))):
        cls = io.system_class(rec)
        commit = rec.get("commit") or {}
        rho = commit.get("rho")
        commit_s = "n/a"
        if rho is not None and (rho or commit.get("n_committed")):
            commit_s = "(%s, %s, %s)" % (fmt(rho, 2), fmt(commit.get("tau_c"), 2), fmt(commit.get("e_c"), 3))
        elif rho is not None:
            commit_s = "(0.00, %s, n/a) never commits" % fmt(commit.get("tau_c"), 2)
        boot = (rec.get("bootstrap") or {}).get("RMSCD@H") or {}
        ci = "n/a"
        if boot.get("ci_lo") is not None:
            ci = "[%s, %s]" % (fmt(boot.get("ci_lo")), fmt(boot.get("ci_hi")))
        tracks.append(rec.get("track", "n/a"))
        # blocks are instruments, not candidates: they never enter a ranking
        rankable.append(not io.is_block(rec))
        end_accs.append(rec.get("end_window_macro_acc") if not io.is_block(rec) else None)
        rmscds.append(rec.get("RMSCD") if not io.is_block(rec) else None)
        rows.append([rec.get("track", "n/a"), rec.get("system_id", "?"), cls,
                     io.arm_rule_of(rec) or "-", io.arm_value_of(rec) or "-",
                     io.seed_of(rec) or "-", rec.get("backbone","-"), rec.get("model_kind","-"),
                     rec.get("library_round","-"), rec.get("train_data_unknown",(rec.get("card") or {}).get("train_data_unknown")),
                     fmt(rec.get("full_clip_macro_acc")), fmt(rec.get("end_window_macro_acc")),
                     fmt(rec.get("RMSCD")), ci, fmt(rec.get("flips_median"), 1), commit_s,
                     rec.get("N_H", "n/a")])
    r_end = ranks_within(end_accs, ascending=False, groups=tracks)
    r_prog = ranks_within(rmscds, ascending=True, groups=tracks)
    for i, row in enumerate(rows):
        row.append("-" if not rankable[i] else ("n/a" if r_end[i] is None else str(r_end[i])))
        row.append("-" if not rankable[i] else ("n/a" if r_prog[i] is None else str(r_prog[i])))

    n_ph = sum(1 for r in recs if io.is_placeholder(r))
    n_bl = sum(1 for r in recs if io.is_block(r))
    notes = [
        "协议：`pi_hash=%s`，H = %s s，delta = %s s。曲线各时点同一合格集合。" %
        (pi_hash, fmt(h_s, 2), fmt((recs[0].get("delta_s") if recs else None), 2)),
        "两列名次在轨内排；量块（gauge block）是仪器不是应考者，一律不进名次，标 `-`。",
        "标 `trivial anchor` 的四个平凡系统（多数类 / 随机 / oracle / 常数 ⊥）是系统库的极端锚点，"
        "进名次是为了检验退化行为：oracle 排第一是尺子的性质，不是审计发现，不得作为结论引用。",
        "承诺三元组三者成对；从不承诺的系统 rho=0、tau_c 按 H 计满、e_c 无定义。",
        "Arm rule 是臂的身份（规则名）；Arm value 是该种子在 dev 上选出的参数，"
        "同一 arm_rule 的三个种子可以取到不同参数值。按规则分组的三种子均值与标准差见 "
        "`table1b_audit_by_arm_rule*.md`。",
        "audit_split = %s，N_H 为审计集内的合格集合规模。" %
        ((recs[0].get("audit_split") if recs else None) or "n/a"),
        "RMSCD 95%% CI 为簇级配对 bootstrap（n_boot=%s）。" %
        fmt((pairs or {}).get("n_boot"), 0),
    ]
    if n_ph:
        notes.append("**本表 %d 个系统是 PLACEHOLDER（family=standin）**：由规则生成的作答矩阵，"
                     "只为在 B 的真实系统落地前把管线跑通。它们的数字不得进入任何结论、"
                     "不得用于回答 E3，也不得写进论文。" % n_ph)
    if n_bl:
        notes.append("另有 %d 个合成量块，用于仪器自检与解析对照，主结果不含量块。" % n_bl)
    notes.append("B 的真实系统落地后，本脚本无需改动即可重跑：它按 `family` 与 card 判类，不按系统名写死。")
    return write_table(out_dir, "table1_audit" + suffix, header, rows,
                       "表 1 审计主表（H = %s s）" % fmt(h_s, 2), notes)


# --------------------------------------------------------------------------- table 2

def cohort_cell(d):
    if not isinstance(d, dict):
        return fmt(d)
    return "RMSCD=%s, S@ref=%s, n=%s" % (fmt(d.get("RMSCD")), fmt(d.get("S_at_ref")),
                                         d.get("n", d.get("n_at_H", "n/a")))


def table2(recs, h_s, pi_hash, out_dir, suffix, use_rule=True, gate_rows=None):
    header = ["System", "Class", "Per-clip end", "Dynamic denominator", "Fixed H / cohort",
              "Length-strat. change (per-clip end)", "Length-strat. change (fixed H)",
              "Conclusion stable?", "G3 own tercile gap", "G3 fixed tercile gap", "G3 MRD_plaus", "G3 original consequence", "G3 revised own gap", "G3 revised fixed gap", "G3 revised consequence"]
    # A leaves conclusion_stable null on purpose: it is a cross-system determination.
    # Rule applied here (switchable with --no-conclusion-rule): a system's conclusion is
    # stable when its rank by RMSCD among non-block systems is identical under all three
    # cohort definitions.
    pool = [r for r in recs if not io.is_block(r)]
    stable = {}
    if use_rule and len(pool) >= 2:
        order = {}
        for key in ("per_clip_end", "dynamic_denominator", "fixed_H_cohort"):
            vals, ids = [], []
            for r in pool:
                d = (r.get("alt_cohorts") or {}).get(key) or {}
                vals.append(d.get("RMSCD"))
                ids.append(r.get("system_id"))
            rk = ranks_within(vals, ascending=True, groups=[""] * len(vals))
            order[key] = dict(zip(ids, rk))
        for r in pool:
            sid = r.get("system_id")
            rs = [order[k].get(sid) for k in order]
            stable[sid] = (None if any(x is None for x in rs) else len(set(rs)) == 1)

    rows = []
    for rec in sorted(recs, key=lambda r: (io.is_block(r), r.get("system_id", ""))):
        alt = rec.get("alt_cohorts") or {}
        lsc = alt.get("length_stratified_change")
        lsc_pce = lsc.get("per_clip_end") if isinstance(lsc, dict) else lsc
        lsc_fix = lsc.get("fixed_H_cohort") if isinstance(lsc, dict) else None
        sid = rec.get("system_id")
        if io.is_block(rec):
            st = "-"
        elif not use_rule:
            st = "n/a"
        else:
            st = fmt(stable.get(sid))
        rows.append([sid, io.system_class(rec),
                     cohort_cell(alt.get("per_clip_end")), cohort_cell(alt.get("dynamic_denominator")),
                     cohort_cell(alt.get("fixed_H_cohort")), fmt(lsc_pce), fmt(lsc_fix), st])
        gr = (gate_rows or {}).get(sid, {})
        rows[-1].extend([fmt((gr.get('own') or {}).get('gap_long_minus_short'), 6),
                         fmt((gr.get('fixed') or {}).get('gap_long_minus_short'), 6),
                         fmt(gr.get('MRD_plaus'), 6), fmt(gr.get('consequence')),
                         fmt((gr.get('revised',{}).get('own') or {}).get('gap_long_minus_short'),6),
                         fmt((gr.get('revised',{}).get('fixed') or {}).get('gap_long_minus_short'),6),
                         fmt(gr.get('revised',{}).get('consequence'))])

    s_ref = None
    for r in recs:
        s_ref = (r.get("alt_cohorts") or {}).get("s_ref_delta_s")
        if s_ref is not None:
            break
    notes = [
        "协议：`pi_hash=%s`，H = %s s；S@ref 取 delta = %s s。" % (pi_hash, fmt(h_s, 2), fmt(s_ref, 2)),
        "三种口径：D2 各自片尾（在 H 处删失以便比较，全部片段都算，短片被设计性地占便宜）、"
        "D3 动态分母（分母随时间缩小，start 与 H 的样本量差就是幸存者暴露）、本文固定 H 与固定合格集合。",
        "旧 Length-strat. change = 按锚后长度中位切两半后，长半与短半的每视频延迟均值之差；"
        "把稳定性与剩余长度混在一起的口径，这个差会明显更大。",
    ]
    notes.append("G3新列使用固定dev三分位切点、长层减短层；旧两半列仅留作追溯。"
                 "原规则要求|own gap|>MRD且|fixed gap|<MRD；修订规则使用E_H内dev三分位、own差95%CI排除0且大于MRD、fixed差CI含0或绝对差小于尺子。两规则并列；空层n/a。逐层与区间见tableS_G3_strata和tableS_G3_revised。")
    if use_rule:
        notes.append("**Conclusion stable? 由本脚本判定，不是 A 的输出**（A 恒置 null，因为这是跨系统判断）。"
                     "规则：该系统按 RMSCD 在非量块系统中的名次，在三种口径下完全一致才算 stable。"
                     "用 `--no-conclusion-rule` 可关闭，关闭后该列一律 n/a。")
    else:
        notes.append("Conclusion stable? 列已按 `--no-conclusion-rule` 关闭。")
    if any(io.is_placeholder(r) for r in recs):
        notes.append("**含 PLACEHOLDER 系统**，本表当前只证明口径对照算得出来，不构成对 E2 的回答。")
    return write_table(out_dir, "table2_cohort_definitions" + suffix, header, rows,
                       "表 2 观察窗与合格集合口径（H = %s s）" % fmt(h_s, 2), notes)



REQUIRED_SEEDS = ("20260903", "20260904", "20260905")


def mean_sd(vals):
    """mean +- sample sd over the seeds; returns (text, mean). Blank when any seed is
    missing, so a partially trained arm never looks like a finished measurement."""
    xs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return "n/a", None
    if len(xs) != len(vals):
        return "n/a (undefined in %d/%d seeds)" % (len(vals)-len(xs),len(vals)), None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return "%.3f (n=1)" % m, m
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return "%.3f +- %.3f" % (m, var ** 0.5), m


def group_by_rule(recs):
    """Group the seeded systems so that the spread reported is seed spread, nothing else.

    Two kinds of arm live under arm_rule:
      - dev-selected parameter (ema / majority / hysteresis / patience): each seed picks
        its own parameter on dev, so one seed contributes exactly one system and the
        rule name alone is the identity;
      - threshold sweep (commit_msp / commit_margin): every threshold is a separate
        system present in every seed, so grouping by rule alone would average across
        thresholds and report threshold spread as if it were seed spread.
    The kind is detected, not hard-coded: a rule contributing more than one arm_value
    within a single seed is a sweep and is keyed by (rule, value).

    Returns (complete, partial) keyed by a display label.
    """
    if recs and all(r.get('group_key') for r in recs):
        groups={}
        for rec in recs:
            if io.is_block(rec) or io.is_trivial(rec): continue
            groups.setdefault(rec['group_key'],[]).append(rec)
        complete,partial={},{}
        for label,items in groups.items():
            seeds=[io.seed_of(x) for x in items]
            if len(seeds)!=len(set(seeds)): raise ValueError('Repeated seed inside '+label)
            (complete if set(seeds)==set(REQUIRED_SEEDS) else partial)[label]=items
        return complete,partial
    by_rule = {}
    for rec in recs:
        rule = io.arm_rule_of(rec)
        if not rule or io.seed_of(rec) is None:
            continue
        by_rule.setdefault(rule, []).append(rec)

    groups = {}
    for rule, items in by_rule.items():
        per_seed = {}
        for r in items:
            per_seed.setdefault(io.seed_of(r), set()).add(io.arm_value_of(r))
        is_sweep = any(len(v) > 1 for v in per_seed.values())
        for r in items:
            if is_sweep:
                label = "%s @ %s" % (rule, io.arm_value_of(r) or "-")
            else:
                label = rule
            groups.setdefault(label, []).append(r)

    complete, partial = {}, {}
    for label, items in groups.items():
        seeds = set(io.seed_of(r) for r in items)
        (complete if set(REQUIRED_SEEDS).issubset(seeds) else partial)[label] = items
    return complete, partial


def table1_by_rule(recs, h_s, pi_hash, out_dir, suffix):
    complete, partial = group_by_rule(recs)
    header = ["Arm rule (group)", "backbone", "model_kind", "arm_rule", "commit_threshold", "library_round", "Family", "Seeds", "Arm values (per seed)",
              "Full macro-Acc", "End-window macro-Acc", "RMSCD@H", "Median flips",
              "Commit rho", "Commit tau_c", "Commit e_c"]
    rows = []
    order = sorted(complete.items(), key=lambda kv: (io.family_of(kv[1][0]), kv[0]))
    for rule, items in order:
        items = sorted(items, key=lambda r: io.seed_of(r) or "")
        vals = lambda f: [f(r) for r in items]
        avs = ", ".join("%s:%s" % (io.seed_of(r)[-1], io.arm_value_of(r) or "-") for r in items)
        commit = [r.get("commit") or {} for r in items]
        rows.append([rule, *[items[0].get(k,"-") for k in ["backbone","model_kind","arm_rule","commit_threshold","library_round"]], io.family_of(items[0]), len(items), avs,
                     mean_sd(vals(lambda r: r.get("full_clip_macro_acc")))[0],
                     mean_sd(vals(lambda r: r.get("end_window_macro_acc")))[0],
                     mean_sd(vals(lambda r: r.get("RMSCD")))[0],
                     mean_sd(vals(lambda r: r.get("flips_median")))[0],
                     mean_sd([c.get("rho") for c in commit])[0],
                     mean_sd([c.get("tau_c") for c in commit])[0],
                     mean_sd([c.get("e_c") for c in commit])[0]])
    notes = [
        "协议：`pi_hash=%s`，H = %s s。均值 +- 样本标准差，统计单位是三个随机种子"
        "（%s），只对三种子齐全的 arm_rule 出行。" % (pi_hash, fmt(h_s, 2), " / ".join(REQUIRED_SEEDS)),
        "完整库分组键为轮次、backbone、model_kind、arm_rule、commit_threshold；轮次隔离第一轮与R2基础分类器。后处理参数在dev逐种子选，"
        "同一规则的三个种子可能落在不同参数上，Arm values 列逐种子列出（只写种子尾号）。",
        "承诺臂（commit_msp / commit_margin）是**阈值扫描**而非 dev 选参：每个阈值在三个种子里都在。"
        "若按规则名合并，报出来的就是阈值离散而不是种子离散，因此这类规则自动改按 "
        "`规则 @ 阈值` 分组。判据是「同一种子内是否出现多个 arm_value」，由脚本检测，不写死名单。",
        "平凡系统与量块没有种子维度，不进本表；它们在表 1 逐行报告。",
        "承诺三列必须成对读：从不承诺的规则 rho=0、tau_c 计满到 H、e_c 无定义（n/a）；某种子未定义时不报告跨种子均值，并注明未定义种子数。",
    ]
    if partial:
        notes.append("以下 arm_rule 的三种子尚未齐全，已按纪律排除、不做部分平均：%s。"
                     % "、".join("%s(%d 个种子)" % (k, len(set(io.seed_of(r) for r in v)))
                                 for k, v in sorted(partial.items())))
    return write_table(out_dir, "table1b_audit_by_arm_rule" + suffix, header, rows,
                       "表 1b 按 arm_rule 分组的三种子均值与标准差（H = %s s）" % fmt(h_s, 2), notes)


def table5_by_rule(recs, h_s, pi_hash, out_dir, suffix):
    complete, partial = group_by_rule(recs)
    header = ["Arm rule (group)", "Family", "Seeds", "Flip rate", "Last-flip median (s)",
              "Correct-wrong-correct", "Pre-anchor overconfidence",
              "K1", "K2", "K3", "K5", "K6"]
    rows = []
    order = sorted(complete.items(), key=lambda kv: (io.family_of(kv[1][0]), kv[0]))
    for rule, items in order:
        phs = [r.get("phenomena") or {} for r in items]
        g = lambda k: mean_sd([p.get(k) for p in phs])[0]
        rows.append([rule, io.family_of(items[0]), len(items),
                     g("flip_rate"), g("last_flip_median_s"), g("correct_wrong_correct"),
                     g("pre_anchor_overconfidence"),
                     g("K1"), g("K2"), g("K3"), g("K5"), g("K6")])
    notes = [
        "协议：`pi_hash=%s`，H = %s s。均值 +- 样本标准差，跨三个随机种子；"
        "只对三种子齐全的 arm_rule 出行。本表只报告，不判定。" % (pi_hash, fmt(h_s, 2)),
        "分组键同表 1b：dev 选参的规则按规则名合并，阈值扫描按 `规则 @ 阈值` 合并。"
        "平凡系统与量块无种子维度，不进本表。",
    ]
    if partial:
        notes.append("三种子未齐全、已排除：%s。" % "、".join(sorted(partial.keys())))
    return write_table(out_dir, "table5b_phenomena_by_arm_rule" + suffix, header, rows,
                       "表 5b 按 arm_rule 分组的现象统计（H = %s s）" % fmt(h_s, 2), notes)


# --------------------------------------------------------------------------- table 3

def table3_from(metrics, rulers, calibration, out_dir, name, title, extra_notes):
    header = ["Metric M", "Ruler (same units)", "MRD_plaus / ruler", "Reference range",
              "Max b_M", "Max s_M", "Min R_M", "MRD_M (plaus)", "MRD_M (full grid)",
              "eps_max", "n pairs", "Verdict"]
    rows, degenerate = [], []
    names = sorted(set(list(metrics.keys()) + list(rulers.keys())))
    for mname in names:
        m = metrics.get(mname) or {}
        ruler = m.get("ruler", rulers.get(mname))
        rng = m.get("reference_range")
        rng_s = "n/a" if not rng else "[%s, %s]" % (fmt(rng[0]), fmt(rng[1]))
        em = m.get("eps_max")
        if isinstance(em, dict):
            em_s = fmt(em.get("eps_max"), 2)
            if em.get("scanned_max_abs_eps") in (0, 0.0):
                em_s += "（未扫到非零偏移）"
        else:
            em_s = fmt(em, 2)
        mrd = m.get("MRD_plaus", m.get("MRD"))
        # The actionable readout: protocol wobble measured against the smallest
        # difference this study would report at all. >= 1 means the wobble swallows it.
        if mrd is not None and ruler:
            ratio = "%.2f" % (mrd / ruler)
        elif mrd is not None and ruler == 0:
            ratio = "尺子为零，比值无定义"
        else:
            ratio = "n/a"
        verdict = m.get("verdict", "n/a")
        if m.get("ruler_degenerate") or (ruler == 0 and mname in rulers):
            degenerate.append(mname)
            verdict = "%s —— 尺子为零，不可分辨同分系统" % verdict
        rows.append([mname, fmt(ruler, 4), ratio, rng_s,
                     fmt(m.get("max_b")), fmt(m.get("max_s")), fmt(m.get("min_R")),
                     fmt(mrd), fmt(m.get("MRD_full_grid")), em_s,
                     m.get("n_pairs", "n/a"), verdict])
    notes = list(extra_notes)
    notes.append("Ruler = 窗末同分系统对上 |M(a)-M(b)| 的中位数，与该指标同量纲，"
                 "即本研究认为值得报告的最小差；定义取自 A 的 `ruler_definition`。")
    notes.append("MRD_plaus / ruler 由本脚本现算：≥ 1 表示协议扰动带来的挪动已经吞掉了"
                 "该指标值得报告的最小差，此时跨文比较该指标不成立。")
    if degenerate:
        notes.append("**%s 的尺子为 0**：窗末同分对在该指标上完全同值，尺子无法分辨它们，"
                     "任何基于该尺子的可比判断都不成立，其 verdict 只能读作"
                     "「该指标在本设置下不承担区分同分系统的功能」。" % "、".join(degenerate))
    if calibration:
        n_pi = calibration.get("n_pi")
        mode = calibration.get("scan_mode")
        mode_text = ("plaus product, %s points" % n_pi) if mode == "full" else                     ("axis, %s points" % n_pi if mode == "axis" else "%s (%s points)" % (mode, n_pi))
        notes.append("r0 = %s；π0 = `%s`，scan = `%s`，audit_split = `%s`，"
                     "显著性 p 用 %s 近似、%s 校正，系统数 %s。" %
                     (fmt(calibration.get("r0"), 2), calibration.get("pi0_hash"),
                      mode_text, calibration.get("audit_split"),
                      calibration.get("p_method"), (calibration.get("correction") or "holm"),
                      calibration.get("n_systems")))
        cr = calibration.get("correction_resolution") or {}
        if cr and not cr.get("percentile_ok", True):
            notes.append("分辨率守卫：n_boot=%s 时 percentile p 的下限是 %s，"
                         "Holm 校正后最小可达 p 为 %s，已超过 alpha=%s；"
                         "因此 p 值改用正态近似。要用 percentile 需 n_boot ≈ %s。" %
                         (cr.get("n_boot"), cr.get("percentile_p_floor"),
                          fmt(cr.get("smallest_adjusted_p_reachable"), 3), cr.get("alpha"),
                          cr.get("n_boot_needed_for_percentile")))
        notes.append("Verdict 直接取 A 的 `verdict` 字段，本脚本不重新判定，只在尺子退化时加注。")
    else:
        notes.append("**该 H 没有 calibration.json**：扰动扫描只在参考 H 上跑，"
                     "因此 b / s / R / MRD / eps_max 各列为 n/a，本表在该 H 上只承载尺子。"
                     "不得把参考 H 的标定量搬到这一档。")
    return write_table(out_dir, name, header, rows, title, notes)


def table3(calibration, rulers, h_s, pi_hash, out_dir, suffix, extra_notes, variant="plaus"):
    """variant='plaus' -> the main table, read from the plausible product grid;
       variant='axis'  -> the one-knob-at-a-time scan, kept only as a propagation
       reference and never as the comparability verdict."""
    main = io.main_metrics(calibration) if calibration else {}
    blocks = io.block_metrics(calibration) if calibration else {}
    if variant == "plaus":
        name = "table3_characterization" + suffix
        title = "表 3 主标定表（真实系统，不含量块；H = %s s）" % fmt(h_s, 2)
        if calibration:
            notes = ["**本表是主结论**：读的是 Π_plaus 的**乘积网格（%s 点，四个旋钮同时取值）**，"
                     "不是逐轴扫描。冻结的可比域判据以此为准。" % calibration.get("n_pi"),
                     "主表只用不含量块的条目；量块并排版见同名 `table3b_*`，两者不得合并。",
                     "逐轴（axis）版本另存 `table3c_characterization_axis*`，"
                     "只作传播曲线对照，不得用于可比域判定。"]
        else:
            notes = ["本档**没有跑乘积网格标定**，所以本表只承载该 H 的尺子，"
                     "标定量各列为 n/a。可比域判据只在参考档 H=10.0 上成立。"]
        notes = notes + extra_notes
        block_name = "table3b_characterization_with_blocks" + suffix
        block_title = "表 3b 标定表（含量块的并排版本，不进主结论；H = %s s）" % fmt(h_s, 2)
        block_notes = ["本表含合成量块，按契约 §8 单独标注并排报，主结果不用它。",
                       "同样取自 Π_plaus 的乘积网格（54 点）。"]
    else:
        name = "table3c_characterization_axis" + suffix
        title = "表 3c 逐轴扫描对照表（axis，14 点；H = %s s）" % fmt(h_s, 2)
        notes = ["**本表不是主结论**：它是每次只动一个旋钮的星形扫描（14 点），"
                 "只用来看单个旋钮的传播形状。可比域与最小可报差以主表 "
                 "`table3_characterization*`（乘积网格 54 点）为准。",
                 "逐轴扫描永远看不到旋钮之间的交互，因此它给出的 min R_M 系统性偏乐观："
                 "把它当判据会高估可比性。"] + extra_notes
        block_name = None
        block_title = None
        block_notes = None

    p1 = table3_from(main, rulers, calibration, out_dir, name, title, notes)
    if blocks and block_name:
        table3_from(blocks, rulers, calibration, out_dir, block_name, block_title, block_notes)
    return p1


ELIGIBILITY_TOL_S = 1e-9   # S1 v2 amendment A3; matches track A's implementation


def table4(manifest_path, protocols, metrics_root, out_dir):
    hs = [h for _, h in protocols if h is not None]
    header = ["Track", "Scope", "Raw", "Readable", "Anchor-valid", "Closed-set unique",
              "Deduplicated"] + ["Eligible@H=%s" % fmt(h, 2) for h in hs] + ["Ordering summary"]
    rows, notes = [], []
    if not os.path.isfile(manifest_path):
        notes.append("**manifest 不存在**：%s。" % manifest_path)
        return write_table(out_dir, "table4_attrition", header, rows,
                           "表 4 两轨方向一致性与 attrition", notes)
    mrows = io.read_csv_rows(manifest_path)

    def f(r, k):
        return io.to_float(r.get(k))

    # The audit split is what the metrics are scored on; the whole manifest is shown
    # alongside so the attrition from "all clips" to "audit cohort" is visible.
    audit_split = None
    for ph, _h in protocols:
        recs = io.load_metrics(metrics_root, ph)
        if recs:
            audit_split = recs[0].get("audit_split")
            break

    def block(sub, scope_label, tr):
        readable = [r for r in sub if r.get("decode_ok", "") != "False"]
        anchor_ok = [r for r in readable
                     if f(r, "anchor_s") is not None and f(r, "duration_s") is not None
                     and 0.0 <= f(r, "anchor_s") <= f(r, "duration_s")]
        uniq = [r for r in anchor_ok if r.get("map_status") == "unique"]
        dedup = len(set(r["source_cluster_id"] for r in uniq))
        cells = [str(sum(1 for r in uniq
                         if (f(r, "post_anchor_length_s") or -1) >= h - ELIGIBILITY_TOL_S))
                 for h in hs]
        return [tr, scope_label, len(sub), len(readable), len(anchor_ok), len(uniq), dedup] +                cells + ["见表 1 的两列 Rank"]

    for tr in sorted(set(r["track_id"] for r in mrows)):
        sub = [r for r in mrows if r["track_id"] == tr]
        rows.append(block(sub, "all clips", tr))
        if audit_split:
            rows.append(block([r for r in sub if r.get("split") == audit_split],
                              "audit split = %s" % audit_split, tr))

    checks = []
    for ph, h in protocols:
        recs = io.load_metrics(metrics_root, ph)
        a_nh = sorted(set(r.get("N_H") for r in recs if r.get("N_H") is not None))
        c_nh = sum(1 for r in mrows
                   if (audit_split is None or r.get("split") == audit_split)
                   and r.get("decode_ok", "") != "False" and r.get("map_status") == "unique"
                   and (io.to_float(r.get("post_anchor_length_s")) or -1) >= h - ELIGIBILITY_TOL_S)
        if len(a_nh) == 1 and a_nh[0] == c_nh:
            checks.append("H=%s 一致（%d）" % (fmt(h, 2), c_nh))
        else:
            checks.append("H=%s **不一致**：C 算 %d，A 报 %s" % (fmt(h, 2), c_nh, a_nh))
    notes.append("Raw → Eligible@H 全部由 `data/manifest/manifest_real.csv` 现算，独立于 A 的产出。")
    notes.append("合格判据用 `L+ >= H - 1e-9` 的容差（S1 v2 修订 A3）。"
                 "不加容差时 `KUBbn-T3XYI_00` 会被漏掉：它的 duration 22.08 减 anchor 12.08 "
                 "在浮点下是 9.9999999999999982，恰好落在 H=10.0 之下，H=4.0 与 H=21.5 两档不受影响。")
    notes.append("与 A 的 N_H 交叉核对（同一 audit split、同一容差）：%s。" % "；".join(checks))
    notes.append("Ordering summary 需要两轨的系统排序方向，MM-AU 轨本轮未下载"
                 "（约 526 GB、需邮件申请），该列与车载轨行、58→5 映射 attrition 一并留空。")
    return write_table(out_dir, "table4_attrition", header, rows,
                       "表 4 两轨方向一致性与 attrition", notes)


# --------------------------------------------------------------------------- table 5

def table5(recs, h_s, pi_hash, out_dir, suffix):
    header = ["System", "Class", "Arm rule", "Seed", "backbone", "model_kind", "library_round", "train_data_unknown", "Flip rate", "Last-flip median (s)",
              "Correct-wrong-correct", "Pre-anchor overconfidence",
              "K1", "K2", "K3", "K5", "K6"]
    rows, missing_K5 = [], 0
    for rec in sorted(recs, key=lambda r: (io.is_block(r), r.get("system_id", ""))):
        ph = rec.get("phenomena") or {}
        if ph.get("K5") is None:
            missing_K5 += 1
        rows.append([rec.get("system_id", "?"), io.system_class(rec),
                     io.arm_rule_of(rec) or "-", io.seed_of(rec) or "-", rec.get("backbone","-"), rec.get("model_kind","-"),
                     rec.get("library_round","-"), rec.get("train_data_unknown",(rec.get("card") or {}).get("train_data_unknown")),
                     fmt(ph.get("flip_rate")), fmt(ph.get("last_flip_median_s"), 2),
                     fmt(ph.get("correct_wrong_correct")), fmt(ph.get("pre_anchor_overconfidence")),
                     fmt(ph.get("K1")), fmt(ph.get("K2")), fmt(ph.get("K3")),
                     fmt(ph.get("K5")), fmt(ph.get("K6"), 3)])
    notes = [
        "协议：`pi_hash=%s`，H = %s s。本表是交付物，只报告不判定，不设通过与否。" % (pi_hash, fmt(h_s, 2)),
        "K1 前缀预测等于终态预测与等于真值两事件的样本位重合率；K2 整段 macro 准确率；"
        "K3 终态错误视频中曾有正确前缀的比例；K5 短前缀高置信不一致率；"
        "K6 首次终态一致时刻与首次持续正确时刻的分布差。K4（改口）已由 Flip rate 与 Last-flip median 承担。",
    ]
    if missing_K5:
        notes.append("K5 在 %d 个系统上为 n/a。这些系统的 `phenomena.full.phenomena` 里 "
                     "`short_prefix_confident_inconsistent_rate` 为 null，但同处另有一个"
                     "视频粒度的 `short_prefix_confident_inconsistent_video_rate` 有值。"
                     "两者粒度不同（前缀单元对视频单元），本脚本**不做替换**，留 n/a："
                     "该选哪一个是 A 的定义问题，不是渲染问题。" % missing_K5)
    if any(io.is_placeholder(r) for r in recs):
        notes.append("**含 PLACEHOLDER 系统**：现象数字来自规则生成的作答，不描述任何真实模型的行为。")
    return write_table(out_dir, "table5_phenomena" + suffix, header, rows,
                       "表 5 现象统计与终态一致检查（H = %s s）" % fmt(h_s, 2), notes)


# --------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-root", default=os.path.join(REPO_ROOT, "outputs"))
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "data", "manifest", "manifest_real.csv"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--pi-hash", default="", help="restrict to one protocol")
    ap.add_argument("--no-conclusion-rule", dest="conclusion_rule", action="store_false", default=True,
                    help="leave table 2's Conclusion stable? column as n/a instead of applying C's rule")
    args = ap.parse_args()
    from r2c_tables import available_root
    kc_root = available_root(args.outputs_root)
    if kc_root is not None:
        import r2c_tables
        return r2c_tables.render(kc_root,args,sys.modules[__name__])
    args.outputs_root = io.active_outputs_root(args.outputs_root)

    metrics_root = os.path.join(args.outputs_root, "metrics")
    # The main comparability verdict comes from the plausible PRODUCT grid; the axis
    # scan is kept only as a propagation reference (S1 v2 amendment A5).
    calib_plaus_root = os.path.join(args.outputs_root, "calib_plaus")
    calib_axis_root = os.path.join(args.outputs_root, "calib")
    out_dir = args.out_dir or os.path.join(args.outputs_root, "tables")

    protocols = io.reference_protocols(args.outputs_root, metrics_root)
    if args.pi_hash:
        protocols = [(p, h) for p, h in protocols if p == args.pi_hash]
    if not protocols:
        log("no protocols found under %s" % metrics_root)
        return
    log("protocols: %s" % ", ".join("%s(H=%s)" % (p, h) for p, h in protocols))

    mid = protocols[len(protocols) // 2][0]
    ref = args.pi_hash or mid
    n_ph_total = 0
    for ph, h in protocols:
        recs = io.load_metrics(metrics_root, ph)
        pairs = io.load_pairs(metrics_root, ph)
        rulers = io.rulers_of(pairs)
        n_ph = sum(1 for r in recs if io.is_placeholder(r))
        n_ph_total = max(n_ph_total, n_ph)
        log("%s (H=%s): %d systems (%d blocks, %d placeholders), rulers=%s"
            % (ph, h, len(recs), sum(1 for r in recs if io.is_block(r)), n_ph,
               {k: round(v, 4) for k, v in sorted(rulers.items())}))
        _, cal_plaus = io.load_calibration(calib_plaus_root, ph)
        _, cal_axis = io.load_calibration(calib_axis_root, ph)
        for suffix in ([""] if ph == ref else []) + ["_%s" % htag(h)]:
            table1(recs, pairs, h, ph, out_dir, suffix)
            table1_by_rule(recs, h, ph, out_dir, suffix)
            table2(recs, h, ph, out_dir, suffix, use_rule=args.conclusion_rule, gate_rows=r2_tables.g3_rows(args.outputs_root,h))
            table5(recs, h, ph, out_dir, suffix)
            table5_by_rule(recs, h, ph, out_dir, suffix)
            extra = []
            if n_ph:
                extra.append("**含 %d 个 PLACEHOLDER 系统**，标定数字不构成可比域结论。" % n_ph)
            if ph == ref:
                extra.append("本 H 是参考档，主表以它为准。")
            else:
                extra.append("参考档是 H=%s s；本档只作对照。" % fmt(dict(protocols).get(ref), 2))
            table3(cal_plaus, rulers, h, ph, out_dir, suffix, extra, variant="plaus")
            if cal_axis:
                table3(cal_axis, rulers, h, ph, out_dir, suffix, extra, variant="axis")

    table4(args.manifest, protocols, metrics_root, out_dir)
    r2_tables.render(args.outputs_root, out_dir, write_table, fmt)
    log("done -> %s" % out_dir)


if __name__ == "__main__":
    main()
