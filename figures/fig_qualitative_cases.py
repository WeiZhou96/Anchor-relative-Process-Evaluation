"""Redraw the two qualitative result figures (ACCIDENT A/B trajectories, MM-AU input cadence).

Same 6 cases, same 18 original frames, same stored predictions as the manuscript version
(figure_expansion_20260921/evidence, read-only).  Only the layout and derived read-outs change:
  * each photo is placed above the time axis and linked to its own offset by a leader;
  * v4: the offline stable-correct suffix is drawn as one underline segment per evaluated tile;
  * the predicted class at every evaluated offset is shown as a tile lane (the ACCIDENT figure
    previously showed only correct/wrong markers);
  * the left column gives, per system/cadence, endpoint correctness, stable-correct onset and the
    clip's contribution d_i to RMSCD@H, all recomputed here from the stored predictions.
Frames are shown full-frame at their native aspect ratio (no crop, no enhancement).

Run: APE_QUAL_FRAMES=<folder with the 18 original frames> python -X utf8 fig_qualitative_cases.py
     optional --verify-sources re-reads the full prediction sources (MM-AU arrays from $APE_DATA; ACCIDENT
     answer-set CSVs from $APE_QUAL_SOURCES, see README.md).
Outputs: $APE_FIG_OUT/figures/fig_qual_{accident,mmau}.{pdf,svg,png} and
         $APE_FIG_OUT/internal/qualitative_plot_checks.json
Release note: the original frames are third-party dataset content and are not redistributed; they are read
from $APE_QUAL_FRAMES and checked against the SHA-256 values in qualitative/*_frames.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from paths import FIG, INTERNAL, MANIFEST, MMAU, STYLE  # noqa: E402  (release configuration)

R = Path(__file__).resolve().parent
E = R / "qualitative"  # compact, read-only evidence: selected cases, stored predictions, frame hashes
FRAMES = Path(os.environ["APE_QUAL_FRAMES"]) if os.environ.get("APE_QUAL_FRAMES") else None  # original frames
MI = MMAU / "input_stride_20260921" / "input"                 # read-only MM-AU prediction arrays
OUT = FIG
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(STYLE))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from stylelib import OKABE_ITO, use_style  # noqa: E402
from stylelib.style_base import FONT_STATUS  # noqa: E402

use_style()
EXTRA = {}
if not FONT_STATUS.get("calibri_math") and FONT_STATUS.get("calibri"):
    # Calibri Math is not installed: Calibri for text and mathtext, stixsans as glyph fallback.
    EXTRA = {"mathtext.fontset": "custom", "mathtext.rm": "Calibri", "mathtext.it": "Calibri:italic",
             "mathtext.bf": "Calibri:bold", "mathtext.sf": "Calibri", "mathtext.fallback": "stixsans"}
    matplotlib.rcParams.update(EXTRA)
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

W = 164.6 / 25.4          # CAS single-column text width actually used by the manuscript (6.48 in)
INK, MUTED, FAINT = "#1F2A33", "#5F6B75", "#C9D0D6"
OK_FILL, OK_EDGE = "#D3ECE2", "#8CC2AE"
BAD_FILL, BAD_EDGE = "#FFFFFF", "#CBD2D8"
TRACK = "#E7EAED"
FS = dict(title=7.6, body=6.8, small=6.4, tick=6.4, tile=6.4)

ACC_ABBR = {0: "HO", 1: "RE", 2: "TB", 3: "SS", 4: "SI"}
ACC_NAME = {0: "head-on", 1: "rear-end", 2: "t-bone", 3: "sideswipe", 4: "single"}
# One-sentence readings of the stored prediction sequences (checked against the arrays in check_notes()).
NOTES = {
    "-RrDtLjWsT4_00": "B answers SS until 1 s,\nthen TB; both end on TB.",
    "-NgnSm_oEB4_00": "B is right at 0 s and from\n1.5 to 8 s, then ends on SI.",
    "-PpBteU0p3Q_00": "Both predict SS at\nevery offset.",
    "0d37451c": "q=2 ends on 8; q=4 and q=8\nreturn to 12 from 56 f.",
    "1c08612f": "Only q=2 switches to 43,\nat the last offset.",
    "00786b67": "All cadences change from\n12 to 14 at 32 f.",
}

TITLES = {
    "same_endpoint_different_onset": "Same correct endpoint,\ndifferent onset",
    "early_correct_late_error": "Early correctness\ndoes not persist",
    "both_wrong_at_end": "Neither system is\ncorrect at the endpoint",
    "denser_input_loses": "Denser input loses\nendpoint correctness",
    "denser_input_gains": "Denser input gains\nendpoint correctness",
    "all_cadences_wrong_at_end": "All three cadences fail\nat the endpoint",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stable_suffix(good: np.ndarray) -> np.ndarray:
    return np.logical_and.accumulate(good[::-1])[::-1]


def contribution(stable: np.ndarray, step: float) -> float:
    """Clip contribution d_i to RMSCD@H: area above the interpolated stable indicator (Eq. of Sec. 3)."""
    return float(np.trapezoid(1.0 - stable.astype(float), dx=step))


# ----------------------------------------------------------------------------- verification
checks: dict = {"accident": {}, "mmau": {}}


def load_csv_rows(path: Path, vids: set) -> dict:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["video_id"] in vids:
                rows.setdefault(r["video_id"], {})[int(r["j"])] = r
    return rows


def verify_accident(d: dict) -> None:
    vids = {c["video_id"] for c in d["selected"]}
    man = {}
    with MANIFEST.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["video_id"] in vids:
                man[r["video_id"]] = int(r["class_code"])
    src = {s: E / f"{s}.csv" for s in d["source_answer_sets"]}
    hashes = {s: sha(p) for s, p in src.items()}
    assert all(hashes[s] == d["source_sha256"][s] for s in hashes), "source answer-set hash mismatch"
    assert sha(MANIFEST) == d["manifest_sha256"], "manifest hash mismatch"
    tables = {s: load_csv_rows(p, vids) for s, p in src.items()}
    fine_ema = load_csv_rows(E / "postproc__ema0p7__prefix__gru512__seed20260903.csv", vids)
    offs = np.array(d["offsets"])
    maxdiff, naive = 0.0, {}
    for c in d["selected"]:
        vid, y = c["video_id"], man[c["video_id"]]
        assert y == c["metadata"]["class_code"]
        for s, t in zip(d["source_answer_sets"], c["traces"]):
            rows = tables[s][vid]
            step = float(next(iter(rows.values()))["delta_s"])
            js = np.rint(offs / step).astype(int)
            pred = np.array([int(rows[j]["pred"]) for j in js])
            prob = np.array([[float(rows[j][f"p{k}"]) for k in range(5)] for j in js])
            assert (pred == prob.argmax(1)).all()
            assert pred.tolist() == t["pred"]
            maxdiff = max(maxdiff, float(np.abs(prob[:, y] - np.array(t["p_true"])).max()))
            good = pred == y
            assert good.tolist() == t["correct"] and stable_suffix(good).tolist() == t["stable"]
            assert abs(contribution(stable_suffix(good), 0.5) - t["delay"]) < 1e-12
        # Why B must use the replayed 0.5-s answer set: subsampling the 0.25-s EMA output differs.
        rows = fine_ema[vid]
        sub = np.array([int(rows[j]["pred"]) for j in np.rint(offs / 0.25).astype(int)])
        naive[vid] = int((sub != np.array(c["traces"][1]["pred"])).sum())
    checks["accident"].update(source_sha256=hashes, manifest_sha256=sha(MANIFEST), max_abs_prob_diff_vs_csv=maxdiff,
                              pred_correct_stable_delay_recomputed="identical",
                              ema_fine_subsample_vs_replay_pred_differences=naive)


def verify_mmau(d: dict) -> None:
    cohort = {json.loads(z)["video_id"]: json.loads(z) for z in (MI / "cohort.jsonl").read_text().splitlines()}
    arr = {}
    for q in (2, 4, 8):
        p = MI / f"gru_20260920_stride{q}_predictions.npz"
        assert sha(p) == d["prediction_sha256"][str(q)], f"npz hash mismatch q={q}"
        arr[q] = np.load(p)
    maxdiff = 0.0
    for c in d["selected"]:
        vid = c["video_id"]
        y = cohort[vid]["class_code"]
        assert y == c["metadata"]["class_code"] and cohort[vid]["native_class"] == c["metadata"]["native_class"]
        assert d["native_class_mapping"][str(y)] == c["metadata"]["native_class"]
        for t in c["traces"]:
            a = arr[int(t["q"])]
            ix = a["video_ids"].tolist().index(vid)
            p = a["probabilities"][ix]
            assert a["offsets"].tolist() == d["offsets"] and a["input_offsets"].tolist() == t["input_offsets"]
            assert p.argmax(1).tolist() == t["pred"]
            maxdiff = max(maxdiff, float(np.abs(p[:, y] - np.array(t["p_true"])).max()))
            good = p.argmax(1) == y
            assert good.tolist() == t["correct"] and stable_suffix(good).tolist() == t["stable"]
    checks["mmau"].update(prediction_sha256={q: sha(MI / f"gru_20260920_stride{q}_predictions.npz") for q in (2, 4, 8)},
                          max_abs_prob_diff_vs_npz=maxdiff, pred_correct_stable_recomputed="identical")


def check_notes(acc_d: dict, mm_d: dict) -> dict:
    """Assert the textual readings in NOTES against the stored arrays."""
    t = {c["video_id"]: c["traces"] for c in acc_d["selected"]}
    A, B = [np.array(x["pred"]) for x in t["-RrDtLjWsT4_00"]]
    assert (B[:3] == 3).all() and (B[3:] == 2).all() and A[-1] == 2          # SS at 0-1 s, TB from 1.5 s
    A, B = [np.array(x["pred"]) for x in t["-NgnSm_oEB4_00"]]
    assert B[0] == 1 and (B[1:3] != 1).all() and (B[3:17] == 1).all() and (B[17:] != 1).all() and B[-1] == 4
    A, B = [np.array(x["pred"]) for x in t["-PpBteU0p3Q_00"]]
    assert (A == 3).all() and (B == 3).all()
    nat = {int(k): v for k, v in mm_d["native_class_mapping"].items()}
    m = {c["video_id"]: {int(x["q"]): [nat[p] for p in x["pred"]] for x in c["traces"]} for c in mm_d["selected"]}
    assert m["0d37451c"][2][-1] == 8 and all(m["0d37451c"][q][7:] == [12, 12] for q in (4, 8))
    assert m["1c08612f"][2][-1] == 43 and all(43 not in m["1c08612f"][q] for q in (4, 8)) and 43 not in m["1c08612f"][2][:-1]
    assert all(m["00786b67"][q] == [12] * 4 + [14] * 5 for q in (2, 4, 8))
    return {"notes": NOTES, "verified_against_arrays": True}


def verify_frames(dataset: str, d: dict, offs: list) -> dict:
    meta = json.loads((E / f"{dataset}_frames.json").read_text(encoding="utf-8"))
    known = {m["local"]: m["sha256"] for m in meta}
    out = {}
    for c in d["selected"]:
        for off in offs:
            name = f"{dataset}_{c['video_id']}_{off}.{'png' if dataset == 'accident' else 'jpg'}"
            h = sha(FRAMES / name)
            assert h == known[name], f"frame hash mismatch {name}"
            out[name] = {"sha256": h, "size": list(Image.open(FRAMES / name).size)}
    return out


# ----------------------------------------------------------------------------- drawing helpers
class Canvas:
    """Place axes in inches measured from the top-left corner."""

    def __init__(self, height: float):
        self.H = height
        self.fig = plt.figure(figsize=(W, height))

    def axes(self, x, top, w, h, **kw):
        return self.fig.add_axes([x / W, 1 - (top + h) / self.H, w / W, h / self.H], **kw)

    def text(self, x, top, s, **kw):
        return self.fig.text(x / W, 1 - top / self.H, s, **kw)

    def to_fig(self, x, top):
        return x / W, 1 - top / self.H


def place_photos(centres, width, lo, hi, gap):
    left = [c - width / 2 for c in centres]
    left[0] = max(left[0], lo)
    for i in range(1, len(left)):
        left[i] = max(left[i], left[i - 1] + width + gap)
    left[-1] = min(left[-1], hi - width)
    for i in range(len(left) - 2, -1, -1):
        left[i] = min(left[i], left[i + 1] - width - gap)
    assert left[0] >= lo - 1e-9
    return left


def draw(dataset: str) -> dict:
    d = json.loads((E / f"{dataset}_cases.json").read_text(encoding="utf-8"))
    acc = dataset == "accident"
    x = np.array(d["offsets"], float)
    step = x[1] - x[0]
    H_win = x[-1]
    unit = "s" if acc else "frames"
    ushort = "s" if acc else "f"
    photo_offs = [0, 3, 10] if acc else [0, 32, 64]
    frames_info = verify_frames(dataset, d, photo_offs)
    colours = [OKABE_ITO["blue"], OKABE_ITO["vermillion"]] if acc else \
        [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["purple"]]
    xlim = (-0.3, H_win + 0.3) if acc else (-4.0, H_win + 4.0)
    ylim = (0.0, 1.0) if acc else (0.0, 0.4)
    yticks = [0, 0.5, 1.0] if acc else [0, 0.2, 0.4]
    native = {int(k): v for k, v in d.get("native_class_mapping", {}).items()}

    # geometry (inches)
    PX0, PX1 = 1.56, 6.42          # plot x-range
    LC = 0.04                      # left column start
    PW = 1.50                      # photo width (all frames at native aspect)
    LEAD = 0.15                    # leader band (holds the time label)
    PROB = 0.44 if acc else 0.40   # probability panel
    LANE = 0.165                   # one tile lane incl. its suffix bar
    ROWGAP = 0.15
    n_lane = len(d["selected"][0]["traces"])
    imgs = []
    for c in d["selected"]:
        ims = [Image.open(FRAMES / f"{dataset}_{c['video_id']}_{o}.{'png' if acc else 'jpg'}") for o in photo_offs]
        imgs.append(ims)
    ph = [max(PW * im.size[1] / im.size[0] for im in ims) for ims in imgs]
    row_h = [p + LEAD + PROB + 0.05 + n_lane * LANE for p in ph]
    TOP, BOTTOM = 0.03, 0.58
    Htot = TOP + sum(row_h) + ROWGAP * (len(row_h) - 1) + BOTTOM
    cv = Canvas(Htot)
    fig = cv.fig
    xs = lambda t: PX0 + (t - xlim[0]) / (xlim[1] - xlim[0]) * (PX1 - PX0)  # noqa: E731

    readouts = []
    top = TOP
    for r, (c, ims) in enumerate(zip(d["selected"], imgs)):
        meta = c["metadata"]
        y_ref = int(meta["class_code"])
        ref_txt = f"{ACC_NAME[y_ref]} ({ACC_ABBR[y_ref]})" if acc else f"native class {meta['native_class']}"
        # left column: case header
        cv.text(LC, top + 0.005, f"({chr(97 + r)})", fontsize=FS["title"], fontweight="bold", va="top", color=INK)
        cv.text(LC + 0.20, top + 0.005, TITLES[c["stratum"]], fontsize=FS["title"], fontweight="bold", va="top",
                color=INK, linespacing=1.12)
        cv.text(LC + 0.20, top + 0.30, c["video_id"], fontsize=FS["body"], va="top", color=MUTED)
        cv.text(LC + 0.20, top + 0.43, "reference: " + ref_txt, fontsize=FS["body"], va="top", color=INK)
        cv.text(LC + 0.20, top + 0.60, NOTES[c["video_id"]], fontsize=FS["small"], va="top", color=MUTED,
                style="italic", linespacing=1.1)

        # photos, near their own offsets
        # evenly spaced photo slots; leaders map each photo to its exact offset
        lefts = list(np.linspace(1.30, PX1 - PW, len(photo_offs)))
        ptop = top + ph[r] - max(PW * im.size[1] / im.size[0] for im in ims)
        axis_top = top + ph[r] + LEAD
        for im, t, lx in zip(ims, photo_offs, lefts):
            h = PW * im.size[1] / im.size[0]
            pt = top + ph[r] - h           # bottom-align photos of one row
            ax = cv.axes(lx, pt, PW, h)
            ax.imshow(np.asarray(im), interpolation="antialiased")
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True), sp.set_linewidth(0.4), sp.set_edgecolor("#8A949C")
            # leader from photo bottom centre to the offset on the axis top
            x0, y0 = cv.to_fig(lx + PW / 2, top + ph[r])
            x1, y1 = cv.to_fig(xs(t), axis_top)
            fig.add_artist(Line2D([x0, x1], [y0, y1], color="#9AA4AC", lw=0.5, transform=fig.transFigure))
            lab_x = lx + 0.01
            cv.text(lab_x, top + ph[r] + 0.012, f"+{t} {unit}", fontsize=FS["small"], va="top", color=MUTED)
        del ptop

        # probability panel
        ax = cv.axes(PX0, axis_top, PX1 - PX0, PROB)
        ax.set_xlim(*xlim), ax.set_ylim(*ylim)
        for yt in yticks[1:]:
            ax.axhline(yt, color="#EDEFF1", lw=0.5, zorder=0)
        for t in photo_offs:
            ax.axvline(t, color="#D5DADF", lw=0.5, ls=(0, (1.5, 1.5)), zorder=0)
        for k, tr in enumerate(c["traces"]):
            p = np.array(tr["p_true"])
            good = np.array(tr["correct"])
            col = colours[k]
            ax.plot(x, p, color=col, lw=0.95, zorder=2, clip_on=False)
            ax.scatter(x[~good], p[~good], s=8, facecolors="white", edgecolors=col, lw=0.6, zorder=3, clip_on=False)
            ax.scatter(x[good], p[good], s=9, facecolors=col, edgecolors="white", lw=0.25, zorder=4, clip_on=False)
            assert p.min() >= ylim[0] and p.max() <= ylim[1], "probability outside displayed range"
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:g}" for v in yticks], fontsize=FS["tick"])
        ax.set_xticks([])
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_linewidth(0.55)
        ax.tick_params(axis="y", length=2, pad=1.5, width=0.5)
        cv.text(PX0 - 0.20, axis_top + PROB / 2, "p(reference\nclass)", fontsize=FS["small"], ha="right",
                va="center", color=MUTED, linespacing=1.05)

        # tile lanes: predicted class at every evaluated offset + offline stable suffix
        lane_top = axis_top + PROB + 0.05
        la = cv.axes(PX0, lane_top, PX1 - PX0, n_lane * LANE)
        la.set_xlim(*xlim), la.set_ylim(n_lane, 0)
        la.axis("off")
        tw = step * 0.86
        rows_out = []
        for k, tr in enumerate(c["traces"]):
            col = colours[k]
            good = np.array(tr["correct"])
            stable = np.array(tr["stable"])
            yc = k + 0.36
            for xx, pr, g in zip(x, tr["pred"], good):
                la.add_patch(Rectangle((xx - tw / 2, yc - 0.30), tw, 0.60, facecolor=OK_FILL if g else BAD_FILL,
                                       edgecolor=OK_EDGE if g else BAD_EDGE, lw=0.35))
                lab = ACC_ABBR[pr] if acc else str(native[pr])
                la.text(xx, yc + 0.02, lab, ha="center", va="center", fontsize=FS["tile"],
                        color=INK if g else "#7C8790")
            yb = k + 0.80
            # per-tile underline: one segment per evaluated offset (tile width, gaps kept), coloured when the
            # offset belongs to the offline stable-correct suffix, grey otherwise -- not a continuous interval
            for xx, st in zip(x, stable):
                la.add_patch(Rectangle((xx - tw / 2, yb - (0.07 if st else 0.05)), tw, 0.14 if st else 0.10,
                                       facecolor=col if st else TRACK, edgecolor="none"))
            ix = np.flatnonzero(stable)
            onset = x[ix[0]] if len(ix) else None
            dval = contribution(stable, step)
            rows_out.append({"system": (["A", "B"][k] if acc else f"q={tr['q']}"), "endpoint_correct": bool(good[-1]),
                             "stable_onset": onset, "d_i": dval})
            # read-out row in the left column
            ytop = lane_top + k * LANE + 0.36 * LANE
            name = ["A", "B"][k] if acc else f"q={tr['q']}"
            cv.text(LC, ytop, name, fontsize=FS["body"], fontweight="bold", color=col, va="center")
            mx, my = cv.to_fig(LC + 0.40, ytop)
            fig.add_artist(Line2D([mx], [my], marker="o", ms=3.3, mfc=col if good[-1] else "white", mec=col, mew=0.7,
                                  transform=fig.transFigure))
            on_txt = "none" if onset is None else f"{onset:g} {ushort}"
            cv.text(LC + 0.96, ytop, on_txt, fontsize=FS["body"], color=INK, va="center", ha="right")
            cv.text(LC + 1.27, ytop, f"{dval:g}", fontsize=FS["body"], color=INK, va="center", ha="right")
        hdr_y = lane_top - 0.035
        cv.text(LC + 0.40, hdr_y, "end", fontsize=FS["small"], color=MUTED, ha="center", va="bottom")
        cv.text(LC + 0.96, hdr_y, "stable from", fontsize=FS["small"], color=MUTED, ha="right", va="bottom")
        cv.text(LC + 1.27, hdr_y, f"$d_i$ ({ushort})", fontsize=FS["small"], color=MUTED, ha="right", va="bottom")
        readouts.append({"video_id": c["video_id"], "stratum": c["stratum"], "rows": rows_out})

        # x axis: ticks on every row, labels on the last row only
        bx = cv.axes(PX0, lane_top + n_lane * LANE, PX1 - PX0, 0.001)
        bx.set_xlim(*xlim)
        for s in ("top", "right", "left"):
            bx.spines[s].set_visible(False)
        bx.spines["bottom"].set_linewidth(0.55)
        bx.spines["bottom"].set_bounds(x[0], x[-1])
        bx.set_yticks([])
        ticks = [0, 2, 4, 6, 8, 10] if acc else [0, 16, 32, 48, 64]
        bx.set_xticks(ticks)
        bx.tick_params(axis="x", length=2, width=0.5, pad=1.5, labelsize=FS["tick"])
        if r < len(d["selected"]) - 1:
            bx.set_xticklabels([])
        else:
            bx.set_xlabel(f"Offset after the published anchor ({unit})", fontsize=FS["body"], labelpad=2)
        top += row_h[r] + ROWGAP

    # legend + system key (one line)
    handles = [Line2D([], [], marker="o", ls="", ms=3.2, mfc="#34495E", mec="#34495E", label="correct class"),
               Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec="#34495E", label="wrong class"),
               Rectangle((0, 0), 1, 1, facecolor=OK_FILL, edgecolor=OK_EDGE, lw=0.4, label="tile = predicted class: correct"),
               Rectangle((0, 0), 1, 1, facecolor=BAD_FILL, edgecolor=BAD_EDGE, lw=0.4, label="wrong"),
               Line2D([], [], color="#34495E", lw=2.4, solid_capstyle="butt", label="tile in stable-correct suffix (offline)")]
    lx, ly = cv.to_fig(0.0, Htot - 0.215)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(lx, ly), ncol=5, frameon=False,
               fontsize=FS["small"], handlelength=1.1, handletextpad=0.4, columnspacing=1.0, borderaxespad=0)
    key = ("A: prefix-mean ResNet-18  |  B: GRU-512 with EMA smoothing, replayed at the 0.5-s step  |  seed 20260903  |  H = 10 s, Δ = 0.5 s" if acc else
           "one frozen class-weighted GRU checkpoint, input every q = 2, 4, 8 frames  |  seed 20260920  |  output step 8 frames, H = 64 frames  |  tiles: native class IDs")
    cv.text(0.05, Htot - 0.075, key, fontsize=FS["small"], color=MUTED, ha="left", va="center")

    name = f"fig_qual_{dataset}"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    return {"figure": name, "size_in": [round(W, 4), round(Htot, 4)], "photo_width_in": PW,
            "photo_heights_in": [round(p, 3) for p in ph], "frames": frames_info, "readouts": readouts,
            "cases": [c["video_id"] for c in d["selected"]], "offsets": d["offsets"]}


if __name__ == "__main__":
    if FRAMES is None or not FRAMES.is_dir():
        raise SystemExit("APE_QUAL_FRAMES must point to a folder with the 18 original frames listed in "
                         "qualitative/{accident,mmau}_frames.json (see README.md, extract_qual_frames.py).")
    acc_d = json.loads((E / "accident_cases.json").read_text(encoding="utf-8"))
    mm_d = json.loads((E / "mmau_cases.json").read_text(encoding="utf-8"))
    # Full CSV/NPZ checks are optional; use --verify-sources to repeat them.
    if "--verify-sources" in sys.argv:
        compact_e = E
        if not os.environ.get("APE_QUAL_SOURCES"):
            raise SystemExit("--verify-sources needs APE_QUAL_SOURCES (folder with the ACCIDENT answer-set CSVs)")
        E = Path(os.environ["APE_QUAL_SOURCES"])
        verify_accident(acc_d)
        verify_mmau(mm_d)
        E = compact_e
    else:
        checks["full_sources"] = "not rerun; use --verify-sources to repeat the exact-array checks"
    notes = check_notes(acc_d, mm_d)
    out = {"text_notes": notes, "accident": draw("accident"), "mmau": draw("mmau"), "verification": checks,
           "evidence_json_sha256": {n: sha(E / n) for n in ("accident_cases.json", "mmau_cases.json")},
           "font_status": dict(FONT_STATUS), "extra_rc": EXTRA,
           "resolved_text_font": matplotlib.font_manager.findfont(
               matplotlib.font_manager.FontProperties(family=matplotlib.rcParams["font.sans-serif"])).replace("\\", "/"),
           "font_sizes_pt": FS}
    (INTERNAL / "qualitative_plot_checks.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("exported", [f"fig_qual_{k}" for k in ("accident", "mmau")], "sizes", out["accident"]["size_in"], out["mmau"]["size_in"])
