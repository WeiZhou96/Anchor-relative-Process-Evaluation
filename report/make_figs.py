#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render idea section 5.5 figures 1 to 4 from track A's real outputs into outputs/figs/.

fig1  two end-window-tied systems, two S_H curves on one cohort (teaser)
fig2  one system under three cohort definitions, plus the length-stratified gap
fig3  protocol response: R_M along each knob axis, with r0 and the comparable band
      (A's scan_mode is 'axis', a star around pi0; if a filled 2-D grid ever appears the
      contour form is drawn instead, without editing this script)
fig4  small multiples by track

Style: white background, cool low-saturation palette, English in-figure text.
Systems whose family is 'standin' are placeholders: every figure that shows one carries
a visible PLACEHOLDER stamp, because their numbers may not support any conclusion.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ape_io as io  # noqa: E402
import r2_figs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PALETTE = ["#2F5D8A", "#7FA8C9", "#4E7C59", "#8FA9BF", "#2F4858", "#A8BDD0"]
SHADE = "#CBD9E6"
MUTED = "#8A97A3"


def log(msg):
    sys.stdout.write("[make_figs] %s\n" % msg)
    sys.stdout.flush()


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E1E7ED", "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.6, "figure.dpi": 200,
    })
    return plt


def stamp_placeholder(fig, on):
    if on:
        fig.text(0.5, 0.5, "PLACEHOLDER", fontsize=30, color="#C8CED4", alpha=0.45,
                 ha="center", va="center", rotation=22, zorder=10)


def by_id(recs):
    return {r.get("system_id"): r for r in recs}


def curve(rec):
    return rec.get("grid_offsets_s"), rec.get("S_H")


# ------------------------------------------------------------------------------ fig 1

def fig1(plt, recs, pairs, h_s, pi_hash, out_dir, want=None, selection_note=None):
    idx = by_id(recs)
    tied = [tuple(p) for p in (pairs.get("window_end_tied_pairs") or [])]

    def eligible_for_teaser(rec):
        """Gauge blocks and trivial anchors cannot carry the teaser: idea 5.3 E3 forbids
        resting the progress-difference claim on weak or deliberately degenerate
        systems, and a curve against trivial__random would be exactly that."""
        return not io.is_block(rec) and not io.is_trivial(rec)

    def usable(pair, strict=True):
        ra, rb = idx.get(pair[0]), idx.get(pair[1])
        if not (ra and rb and curve(ra)[1] and curve(rb)[1]):
            return None
        if strict and not (eligible_for_teaser(ra) and eligible_for_teaser(rb)):
            return None
        return (ra, rb)

    chosen, rule_note = None, ""
    if want and len(want) == 2 and usable(tuple(want), strict=False):
        chosen = usable(tuple(want), strict=False)
        rule_note = selection_note or "pair given on the command line"
    if chosen is None:
        # A window-end-tied pair of two *different* systems, taking the largest RMSCD
        # gap. Two seeds of one system are also "tied", but a teaser built from seed
        # noise would not show what the protocol is for. The selection rule is printed
        # under the figure so the choice is not a silent one.
        scored = []
        for pr in tied:
            got = usable(pr)
            if not got:
                continue
            ra, rb = got
            same_system = (io.arm_rule_of(ra) == io.arm_rule_of(rb)
                           and io.arm_value_of(ra) == io.arm_value_of(rb))
            gap = abs((ra.get("RMSCD") or 0.0) - (rb.get("RMSCD") or 0.0))
            scored.append((same_system, -gap, ra.get("system_id"), rb.get("system_id"), got))
        scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        if scored:
            chosen = scored[0][4]
            rule_note = ("largest RMSCD gap among window-end-tied pairs of two different systems"
                         if not scored[0][0] else
                         "largest RMSCD gap among window-end-tied pairs (only same-system pairs available)")
            rule_note += "; gauge blocks and trivial anchors excluded"
    if chosen is None:
        pool = [r for r in recs if eligible_for_teaser(r) and curve(r)[1]]
        best, bd = None, None
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                ai = pool[i].get("end_window_macro_acc")
                aj = pool[j].get("end_window_macro_acc")
                if ai is None or aj is None:
                    continue
                d = abs(ai - aj)
                if bd is None or d < bd:
                    best, bd = (pool[i], pool[j]), d
        chosen = best
        rule_note = "closest end-window accuracy (no tied-pair list available)"
    if chosen is None:
        log("fig1 skipped: no two systems with S_H curves")
        return None
    log("fig1 pair: %s vs %s (%s)" % (chosen[0].get("system_id"), chosen[1].get("system_id"), rule_note))

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for k, rec in enumerate(chosen):
        g, s = curve(rec)
        acc = rec.get("end_window_macro_acc")
        rm = rec.get("RMSCD")
        # Keep the entry short: system ids can be long, and a legend inside the axes
        # would run off the panel. The two numbers stay, one line each, below the axes.
        ax.plot(g, s, color=PALETTE[k], label="%s\nend-window acc %s, RMSCD %s" % (
            "%s / %s / %s / %s; seed %s" % (rec.get("library_round",""),rec.get("backbone",""),rec.get("model_kind",""),io.arm_rule_of(rec),io.seed_of(rec)),
            "%.3f" % acc if acc is not None else "n/a",
            "%.2f s" % rm if rm is not None else "n/a"))
    ax.set_xlabel(r"time after anchor $\delta$ (s)")
    ax.set_ylabel(r"stable-correct fraction $S_H(\delta)$")
    ax.set_title("Tied at the end of the window, different course\n$H$ = %.2f s, $N_H$ = %s" %
                 (h_s, chosen[0].get("N_H")))
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=7.5, handlelength=1.6, columnspacing=1.4)
    fig.text(0.5, -0.075, "pair selection: %s" % rule_note, ha="center", fontsize=6.5, color=MUTED)
    stamp_placeholder(fig, any(io.is_placeholder(r) for r in chosen))
    p = os.path.join(out_dir, "fig1_teaser_two_systems.png")
    # bbox_inches='tight' is what keeps a legend placed outside the axes inside the file
    fig.savefig(p, bbox_inches="tight", dpi=300)
    fig.savefig(os.path.splitext(p)[0]+".pdf", bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------ fig 2

def fig2(plt, recs, h_s, out_dir, system=None):
    pool = [r for r in recs if not io.is_block(r) and (r.get("alt_cohorts") or {}).get("fixed_H_cohort")]
    if not pool:
        log("fig2 skipped: no non-block system carries alt_cohorts")
        return None
    if system:
        pool = [r for r in pool if r.get("system_id") == system] or pool
    rec = pool[0]
    alt = rec.get("alt_cohorts") or {}
    keys = [("per_clip_end", "per-clip end (D2)"),
            ("dynamic_denominator", "dynamic denominator (D3)"),
            ("fixed_H_cohort", "fixed $H$ / cohort (ours)")]
    s_ref = alt.get("s_ref_delta_s")

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))

    # left: the protocol curve with the three definitions' S at the reference offset
    g, s = curve(rec)
    ax = axes[0]
    if g and s:
        ax.plot(g, s, color=PALETTE[0], label="$S_H(\\delta)$, fixed cohort")
    for k, (key, label) in enumerate(keys):
        d = alt.get(key) or {}
        if d.get("S_at_ref") is not None and s_ref is not None:
            ax.plot([s_ref], [d["S_at_ref"]], marker="o", ms=5, color=PALETTE[k],
                    linestyle="none", label="%s: %.3f" % (label, d["S_at_ref"]))
    if s_ref is not None:
        ax.axvline(s_ref, color=MUTED, linestyle=":", linewidth=0.9)
    ax.set_xlabel(r"$\delta$ (s)")
    ax.set_ylabel("stable-correct fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Stable correctness and cohort definitions")
    ax.legend(loc="upper left", fontsize=7)

    # middle: RMSCD under the three definitions, for every non-block system
    ax = axes[1]
    sids = [r.get("system_id") for r in pool]
    width = 0.8 / len(keys)
    for k, (key, label) in enumerate(keys):
        vals = [((r.get("alt_cohorts") or {}).get(key) or {}).get("RMSCD") for r in pool]
        vals = [v if v is not None else float("nan") for v in vals]
        ax.scatter([i + k * width for i in range(len(pool))], vals, s=9,
                   color=PALETTE[k], label=label, alpha=.7)
    ax.set_xticks([i + width for i in range(len(pool))])
    ax.set_xticklabels([str(i+1) if i % max(5, int(math.ceil(len(sids)/8))) == 0 else "" for i in range(len(sids))], fontsize=7)
    ax.set_xlabel("System index (table order)")
    ax.set_ylabel("RMSCD@H (s)")
    ax.grid(False, axis="x")
    ax.set_ylim(bottom=0)
    ax.set_title("Same systems, three cohort definitions")
    ax.legend(fontsize=7)

    # right: the length-stratified gap each definition leaves behind
    ax = axes[2]
    pce, fix = [], []
    for r in pool:
        lsc = (r.get("alt_cohorts") or {}).get("length_stratified_change")
        lsc = lsc if isinstance(lsc, dict) else {}
        pce.append(lsc.get("per_clip_end", float("nan")))
        fix.append(lsc.get("fixed_H_cohort", float("nan")))
    ax.scatter([i - 0.2 for i in range(len(pool))], pce, s=9, color=PALETTE[0], label="per-clip end")
    ax.scatter([i + 0.2 for i in range(len(pool))], fix, s=9, color=PALETTE[2], label="fixed $H$ / cohort")
    ax.axhline(0, color=MUTED, linewidth=.7)
    ax.set_xticks(range(len(pool)))
    ax.set_xticklabels([str(i+1) if i % max(5, int(math.ceil(len(sids)/8))) == 0 else "" for i in range(len(sids))], fontsize=7)
    ax.set_xlabel("System index (table order)")
    ax.grid(False, axis="x")
    ax.set_ylabel("long-half minus short-half delay (s)")
    ax.set_title("Residual dependence on remaining length")
    ax.legend(fontsize=7)

    stamp_placeholder(fig, any(io.is_placeholder(r) for r in pool))
    fig.suptitle("Cohort definitions at $H$ = %.2f s" % h_s, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out_dir, "fig2_cohort_definitions.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(p)[0]+".pdf", bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------ fig 3

def fig3(plt, calibration, out_dir, placeholder, metric_order=None):
    """Comparability region from the plausible PRODUCT grid when it is filled.

    Three slices (eps_sys, delta), (eps_sys, H), (delta, H), each holding the remaining
    two knobs at pi0. The plausible grid is 3 x 2 x 3 x 3, so a slice is 3 x 3: too
    coarse for honest contour interpolation, and drawn as an annotated heat map with the
    numbers in the cells. Cells below r0 are outlined, pi0 is marked. If the grid is not
    filled (an axis scan), the per-knob curves are drawn instead and the title says so.
    """
    R = io.load_R_tables(calibration, with_blocks=False)
    if not R:
        log("fig3 skipped: no R_*.csv in the calibration directory")
        return None
    pi0 = calibration.get("pi0") or {}
    r0 = calibration.get("r0", 0.9)
    n_pi = calibration.get("n_pi")
    mode = calibration.get("scan_mode")
    metrics = metric_order or sorted(R.keys())
    slices = [("eps_sys_s", "delta_s"), ("eps_sys_s", "h_s"), ("delta_s", "h_s")]

    griddable = [m for m in metrics
                 if all(io.grid_slice(R[m], a, b, pi0) is not None for a, b in slices)]

    if griddable:
        import matplotlib
        nrow = len(griddable)
        # constrained_layout keeps the per-row metric titles clear of the axis labels of
        # the row above; plain tight_layout cannot, because the colour bar steals space.
        fig, axes = plt.subplots(nrow, 3, figsize=(10.2, 2.9 * nrow), squeeze=False,
                                 constrained_layout=True)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "ape_cool", ["#B4553F", "#D98E77", "#E8D9C5", "#CBD9E6", "#2F5D8A"])
        # Diverging about r0: the question the panel answers is "is this cell inside the
        # comparability region", so r0 is the colour midpoint, warm below, cool above.
        lo = min(v for m in griddable for a, b in slices
                 for row in io.grid_slice(R[m], a, b, pi0)[2] for v in row if v is not None)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=min(lo, r0 - 0.01), vcenter=r0, vmax=1.0)
        im = None
        for i, m in enumerate(griddable):
            for j, (xk, yk) in enumerate(slices):
                ax = axes[i][j]
                xs, ys, Z = io.grid_slice(R[m], xk, yk, pi0)
                im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                               extent=(-0.5, len(xs) - 0.5, -0.5, len(ys) - 0.5))
                for a, yv in enumerate(ys):
                    for b, xv in enumerate(xs):
                        v = Z[a][b]
                        if v is None:
                            continue
                        below = v < r0
                        ax.text(b, a, "%.3f" % v, ha="center", va="center", fontsize=6.5,
                                color="white" if v > .95 else "#7A1F0F" if below else "#16324A",
                                fontweight="bold" if below else "normal")
                        if below:
                            ax.add_patch(matplotlib.patches.Rectangle(
                                (b - 0.5, a - 0.5), 1, 1, fill=False,
                                edgecolor="#B4553F", linewidth=1.6))
                        # explicit None checks: pi0 values are legitimately 0.0 for
                        # both anchor knobs, and `0.0 or fallback` would discard them
                        px, py = io.to_float(pi0.get(xk)), io.to_float(pi0.get(yk))
                        if (px is not None and py is not None
                                and abs(xv - px) < 1e-9 and abs(yv - py) < 1e-9):
                            ax.plot([b - 0.36], [a - 0.36], marker="P", ms=6, mew=0.6,
                                    color="#16324A", markeredgecolor="white")
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels(["%g" % v for v in xs], fontsize=7)
                ax.set_yticks(range(len(ys)))
                ax.set_yticklabels(["%g" % v for v in ys], fontsize=7)
                ax.set_xlabel(io.KNOB_LABEL[xk], fontsize=7.5)
                ax.set_ylabel(io.KNOB_LABEL[yk], fontsize=7.5)
                ax.grid(False)
                if j == 1:
                    ax.set_title(m, fontsize=10, fontweight="bold", pad=8)
        cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015,
                          ticks=[norm.vmin, r0, 1.0])
        cb.set_label(r"rank preservation $R_M$", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(r"Comparability region $\mathcal{A}_M(r_0)$ on the plausible product "
                     r"grid (%s points; slices hold other knobs at $\pi_0$); outlined $R_M < r_0 = %.2f$; "
                     r"+ marks $\pi_0$" % (n_pi, r0), fontsize=10)
        stamp_placeholder(fig, placeholder)
        p = os.path.join(out_dir, "fig3_comparability_region.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        fig.savefig(os.path.splitext(p)[0]+".pdf", bbox_inches="tight")
        plt.close(fig)
        log("fig3: product-grid heat maps for %d metrics (scan_mode=%s, n_pi=%s)"
            % (len(griddable), mode, n_pi))
        return p

    profiles = {m: io.axis_profiles(R[m], pi0) for m in metrics}
    knobs = [k for k in io.KNOBS if any(k in profiles[m] for m in metrics)]
    if not knobs:
        log("fig3 skipped: grid not filled and no knob varies while the others sit at pi0")
        return None
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(1, len(knobs), figsize=(3.2 * len(knobs), 3.2), squeeze=False)
    for ax, knob in zip(axes[0], knobs):
        for k, m in enumerate(metrics):
            pts = profiles[m].get(knob)
            if not pts:
                continue
            ax.plot([q[0] for q in pts], [q[1] for q in pts],
                    marker=markers[k % len(markers)], ms=4.0 - 0.35 * k,
                    linestyle=styles[k % len(styles)], linewidth=2.0 - 0.28 * k,
                    color=PALETTE[k % len(PALETTE)], label=m, alpha=0.9)
        ax.axhline(r0, color="#B4553F", linestyle="--", linewidth=1.0)
        ax.axhspan(r0, 1.02, color=SHADE, alpha=0.55, zorder=0)
        v0 = io.to_float(pi0.get(knob))
        if v0 is not None:
            ax.axvline(v0, color=MUTED, linestyle=":", linewidth=0.9)
        ax.set_xlabel(io.KNOB_LABEL[knob])
        ax.set_ylim(0, 1.05)
        ax.set_title(knob)
    axes[0][0].set_ylabel(r"rank preservation $R_M$")
    axes[0][-1].legend(loc="lower left", fontsize=7)
    fig.suptitle(r"Protocol response: $R_M$ along each knob (%s scan, %s points); "
                 r"shaded $R_M \geq r_0 = %.2f$" % (mode, n_pi, r0), fontsize=10)
    stamp_placeholder(fig, placeholder)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out_dir, "fig3_comparability_region.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(p)[0]+".pdf", bbox_inches="tight")
    plt.close(fig)
    log("fig3: axis profiles (grid not filled)")
    return p


# ------------------------------------------------------------------------------ fig 4

def fig4(plt, protocols, metrics_root, out_dir):
    by_track = {}
    placeholder = False
    for ph, h in protocols:
        for rec in io.load_metrics(metrics_root, ph):
            if io.is_block(rec) or not curve(rec)[1]:
                continue
            by_track.setdefault(rec.get("track", "unknown"), {}).setdefault(ph, []).append((rec, h))
            placeholder = placeholder or io.is_placeholder(rec)
    if not by_track:
        log("fig4 skipped: no non-block S_H curves")
        return None
    tracks = sorted(by_track.keys()) + ["dashcam (absent)"]
    fig, axes = plt.subplots(1, len(tracks), figsize=(4.8 * len(tracks), 3.8), squeeze=False)
    for ax, tr in zip(axes[0], tracks):
        if tr == "dashcam (absent)":
            ax.text(0.5, 0.5, "dashcam track\nnot available this round\n(MM-AU not downloaded)",
                    ha="center", va="center", fontsize=8, color=MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title("dashcam")
            continue
        # one line per system at the reference (middle) H
        phs = sorted(by_track[tr].keys(), key=lambda p: by_track[tr][p][0][1])
        ph = phs[len(phs) // 2]
        items = sorted(by_track[tr][ph], key=lambda t: t[0].get("system_id"))
        if items and items[0][0].get('library_round'):
            from collections import defaultdict
            import numpy as np
            bases=defaultdict(list)
            for rec,h in items:
                if io.is_trivial(rec): continue
                g,s=curve(rec)
                ax.plot(g,s,color='#BBC7D0',alpha=.22,linewidth=.55)
                if rec.get('family') in ['clip','prefix']:
                    key=(rec['library_round'],rec['backbone'],rec['model_kind'])
                    bases[key].append((g,s))
            for k,(key,values) in enumerate(sorted(bases.items())):
                ax.plot(values[0][0],np.mean([x[1] for x in values],axis=0),
                        color=plt.get_cmap('tab10')(k),linewidth=1.25,label='/'.join(key))
            ax.text(.02,.03,'Faint: all 168 systems; color: base classifier 3-seed means',
                    transform=ax.transAxes,fontsize=6)
        else:
            for k, (rec, h) in enumerate(items[:6]):
                g, s = curve(rec)
                ax.plot(g, s, color=PALETTE[k % len(PALETTE)], label="%s; seed %s" % (io.arm_rule_of(rec), io.seed_of(rec)))
        ax.set_title("%s ($H$=%.2f s)" % (tr, items[0][1]))
        ax.set_xlabel(r"$\delta$ (s)")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3, fontsize=6.5)
    axes[0][0].set_ylabel(r"$S_H(\delta)$")
    stamp_placeholder(fig, placeholder)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig4_two_tracks.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(p)[0]+".pdf", bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-root", default=os.path.join(REPO_ROOT, "outputs"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--pi-hash", default="", help="protocol used for figures 1 and 2 (default: mid H)")
    ap.add_argument("--fig1-systems", default="")
    ap.add_argument("--fig2-system", default="")
    args = ap.parse_args()
    from r2c_tables import available_root
    kc_root = available_root(args.outputs_root)
    if kc_root is not None:
        import r2c_figs
        return r2c_figs.render(kc_root,args,sys.modules[__name__])
    args.outputs_root = io.active_outputs_root(args.outputs_root)

    metrics_root = os.path.join(args.outputs_root, "metrics")
    # Figure 3 is a main result, so it reads the plausible product grid; the axis
    # scan is only a fallback when the product grid has not been produced.
    calib_root = os.path.join(args.outputs_root, "calib_plaus")
    if not os.path.isdir(calib_root):
        calib_root = os.path.join(args.outputs_root, "calib")
    out_dir = args.out_dir or os.path.join(args.outputs_root, "figs")
    os.makedirs(out_dir, exist_ok=True)

    protocols = io.reference_protocols(args.outputs_root, metrics_root)
    if not protocols:
        log("no protocols under %s" % metrics_root)
        return
    ref = args.pi_hash or protocols[len(protocols) // 2][0]
    h_ref = dict(protocols).get(ref)
    log("protocols: %s ; reference %s (H=%s)" %
        (", ".join("%s(H=%s)" % (p, h) for p, h in protocols), ref, h_ref))

    recs = io.load_metrics(metrics_root, ref)
    pairs = io.load_pairs(metrics_root, ref)
    placeholder = any(io.is_placeholder(r) for r in recs)

    plt = setup_mpl()
    want = [s.strip() for s in args.fig1_systems.split(",") if s.strip()]
    selection_note = None
    g4 = r2_figs.read(args.outputs_root, 'gates', 'g4')
    if not want:
        gh = next((g for g in g4.get('by_h', []) if g['h_s'] == h_ref), {})
        proofs = sorted((r['system_a'], r['system_b'])
                        for g in gh.get('seed_groups', []) if g['pass']
                        for r in g['seeds'] if r['significant'])
        if proofs:
            want = list(proofs[0])
            selection_note = "first lexical G4 cross-base, same-seed pair in a replicated group"
    made = [
        fig1(plt, recs, pairs, h_ref, ref, out_dir, want or None, selection_note),
        fig2(plt, recs, h_ref, out_dir, args.fig2_system or None),
    ]
    _, calibration = io.load_calibration(calib_root, ref)
    made.append(fig3(plt, calibration, out_dir, placeholder) if calibration else None)
    made.append(fig4(plt, protocols, metrics_root, out_dir))
    r2_figs.render(plt, args.outputs_root, out_dir)
    for p in made:
        if p:
            log("wrote %s" % p)
    log("done: %d of 4 figures produced" % sum(1 for p in made if p))
    if placeholder:
        log("NOTE: stand-in systems present; every figure showing one carries a PLACEHOLDER stamp")


if __name__ == "__main__":
    main()
