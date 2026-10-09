"""Write the README media of the repository into docs/media/.

Media (every value is read from stored records; no dataset imagery is used or needed):
  ape_reverse_and.gif / .mp4         offline reverse AND on one stored ACCIDENT trajectory pair (systems A, B)
  ape_tied_pair_rmscd.gif / .mp4     S_H(delta) and the RMSCD area of the window-end-tied pair A/B at H = 10 s
  ape_eligibility_cohort.gif / .mp4  eligibility cohort against the horizon; per-clip-end vs fixed-cohort accounting
  results_at_a_glance.png            three-panel static summary drawn from the same records
  paper_*.png                        unchanged copies of the paper figures, when $APE_FIG_OUT/figures/ holds them

Reads:
  figures/qualitative/accident_cases.json
  $APE_DATA/accident/outputs/r2b/metrics/<pi0 hash>/*.json
  $APE_DATA/accident/manifest/manifest_real.csv
  $APE_DATA/accident/outputs/r2c/gates/g3_v5.json

Run from the repository root:
  python figures/make_media.py [--out DIR] [--no-mp4] [--only NAME ...]
APE_DATA defaults to <repository>/data when unset. MP4 files are written only when ffmpeg is on PATH;
without ffmpeg the GIF files and the PNG files are still written.

GIF palette: one fixed palette per file. It holds pure white and the exact colours of the paper's colour
semantics, followed by median-cut colours of the frames. Near-white pixels (all channels >= 248) are set to
white before mapping, every pixel is mapped to its exact nearest palette colour, and the page background is
read back from the written file and checked to be #FFFFFF.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not os.environ.get("APE_DATA"):
    os.environ["APE_DATA"] = str(HERE.parent / "data")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image, ImageSequence  # noqa: E402

from paths import FIG, HASH, MANIFEST, REMOTE04, ROOT  # noqa: E402
from figstyle import OKABE_ITO, ROLE, style_ax  # noqa: E402  (installs the stylelib rcParams and font fallback)
from stylelib.style_base import FONT_STATUS  # noqa: E402

# Larger text than the print figures: frames are 1100 px wide and shown at about 880 px on GitHub.
matplotlib.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11.5, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9.5, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})

QUAL = HERE / "qualitative"
DPI = 110                 # 10 in x 110 dpi = 1100 px wide frames
W_IN = 10.0
MS_STEP = 100             # every frame duration is a multiple of this (GIF timing and MP4 frame repeats)
FPS = 1000 // MS_STEP
MAX_FRAMES = 120
GIF_LIMIT = 4.0e6
MP4_LIMIT = 2.0e6
GIF_COLORS = 128
SNAP_MIN = 248            # pixels with every channel >= this are set to pure white before palette mapping
TRIM_MARGIN = 14          # blank rows kept above and below the content of an animation
LEGEND_GAP_PX = 6         # gap between the lowest axis label and a legend placed below it
MINUS = "\u2212"

# ----------------------------------------------------------------------------- colours (paper semantics)
C_A, C_B = "#0072B2", "#D55E00"   # system A: Okabe-Ito blue, system B: Okabe-Ito vermillion (fixed hex values)
SYS_COLOR = {"A": C_A, "B": C_B}
SYS_DESC = {"A": "prefix-mean ResNet-18", "B": "GRU-512 + EMA smoothing"}
TEAL = OKABE_ITO["green"]         # correct / stable correct
TEAL_L = "#D6EFE6"                # correct at this offset (before the reverse AND)
TEAL_M = "#97D5BE"                # in the stable-correct suffix
OCHRE = OKABE_ITO["orange"]       # correct now, later wrong; retracted-correct area
OCHRE_F = "#F4D79A"
OCHRE_T = "#9A6A00"               # ochre for text on white
GREY = "#8C8C8C"                  # wrong; error area
GREY_F = "#DCDCDC"
GREY_T = "#555555"
SLATE = "#4F6272"                 # structure: cursors, indicators, cohort
SLATE_BAR = "#6A7F93"
EXCL = "#E4E7EB"
INK = ROLE["ink"]
MUTED = ROLE["muted"]
BUNDLE = "#CDD2D8"
BAND = "#E3ECF5"
TILE_HIDDEN = "#E3E6EA"
ZERO_LINE = "#333333"
KCOL = {"fixed": OKABE_ITO["blue"], "own": OKABE_ITO["orange"]}   # as in the paper's Fig. 3
KLAB = {"fixed": "fixed cohort", "own": "per-clip end"}
MARKERS = "Dsv^"


def tint(color, alpha: float) -> str:
    """Opaque colour equal to `color` drawn with `alpha` on white (exact 8-bit hex value)."""
    c = np.asarray(to_rgb(color), float)
    return to_hex(1.0 - alpha * (1.0 - c))


SYS_FILL = {"A": tint(C_A, 0.16), "B": tint(C_B, 0.16)}

ACC_ABBR = {0: "HO", 1: "RE", 2: "TB", 3: "SS", 4: "SI"}
ACC_NAME = {0: "head-on", 1: "rear-end", 2: "t-bone", 3: "sideswipe", 4: "single"}

PAPER_FIGS = {  # file written by figures/run_all.sh -> copy name (paper numbering)
    "fig3_accounting.png": "paper_fig3_accounting.png",
    "fig2_tied_pairs.png": "paper_fig4_tied_pairs.png",
    "fig5_knobs.png": "paper_fig6_knobs.png",
    "fig4_comparability.png": "paper_figS1_comparability.png",
    "fig6_mmau.png": "paper_figS2_mmau.png",
}


def _rgb255(color) -> tuple:
    return tuple(int(round(v * 255)) for v in to_rgb(color))


def reserved_colors() -> list:
    """Exact palette entries: pure white first, then every colour that carries meaning in the media."""
    out = []
    for c in ["#FFFFFF", "#000000", INK, MUTED, ROLE["grid"], C_A, C_B, SYS_FILL["A"], SYS_FILL["B"],
              TEAL, TEAL_L, TEAL_M, OCHRE, OCHRE_F, OCHRE_T, GREY, GREY_F, GREY_T, SLATE, SLATE_BAR, EXCL,
              BUNDLE, BAND, TILE_HIDDEN, ZERO_LINE, KCOL["fixed"], KCOL["own"]]:
        try:
            rgb = _rgb255(c)
        except (ValueError, TypeError):
            continue
        if rgb not in out:
            out.append(rgb)
    assert out[0] == (255, 255, 255)
    return out


RESERVED_RGB = reserved_colors()


# ----------------------------------------------------------------------------- helpers
def stable_suffix(good: np.ndarray) -> np.ndarray:
    """Offline reverse AND: True where this offset and every later offset are correct."""
    return np.logical_and.accumulate(np.asarray(good, bool)[::-1])[::-1]


def trap(v, step: float) -> float:
    return float(np.trapezoid(np.asarray(v, float), dx=step))


def cum_trap(v, step: float) -> np.ndarray:
    """Cumulative trapezoid area on a uniform grid; element k is the area from grid point 0 to k."""
    v = np.asarray(v, float)
    out = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) * (step / 2.0))])
    assert abs(out[-1] - trap(v, step)) < 1e-9
    return out


def check(label: str, computed, stored, tol: float) -> None:
    diff = abs(float(computed) - float(stored))
    ok = diff <= tol
    print(f"[check] {label}: computed {float(computed):.9g}, stored {float(stored):.9g}, "
          f"|diff| {diff:.2e} (tol {tol:g}) {'OK' if ok else 'MISMATCH'}")
    assert ok, f"{label}: computed {computed} does not match stored {stored}"


def fmt_signed(v, nd: int) -> str:
    """Signed number with a true minus sign; a value that rounds to zero is printed without sign."""
    r = round(float(v), nd)
    if r == 0:
        return f"{0.0:.{nd}f}"
    return ("+" if r > 0 else MINUS) + f"{abs(r):.{nd}f}"


def fmt_range(vals) -> str:
    vals = [float(v) for v in vals]
    nd = 2 if min(abs(v) for v in vals) < 1.0 else 1
    a, b = fmt_signed(min(vals), nd), fmt_signed(max(vals), nd)
    return a if a == b else f"{a} to {b}"


def metrics_dir(h: float) -> Path:
    return REMOTE04 / "r2b" / "metrics" / HASH[h]


def load_record(h: float, sid: str) -> dict:
    p = metrics_dir(h) / f"{sid}.json"
    if not p.is_file():
        raise SystemExit(f"missing per-system record: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def legend_below(fig, axes, x: float, loc: str, handles, gap_px: float = LEGEND_GAP_PX, **kw):
    """Figure legend whose top edge sits gap_px below the lowest tick label or axis label of `axes`."""
    renderer = fig.canvas.get_renderer()
    y0 = min(a.get_tightbbox(renderer).y0 for a in axes)
    return fig.legend(handles=handles, loc=loc, bbox_to_anchor=(x, (y0 - gap_px) / fig.bbox.height),
                      frameon=False, **kw)


def grab(fig) -> np.ndarray:
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    h, w = a.shape[:2]
    return a[: h - h % 2, : w - w % 2].copy()


class Reel:
    def __init__(self, name: str):
        self.name, self.frames, self.durs = name, [], []

    def add(self, fig, ms: int) -> None:
        assert ms % MS_STEP == 0, ms
        self.frames.append(grab(fig))
        self.durs.append(int(ms))


def snap_white(f: np.ndarray) -> np.ndarray:
    f = np.array(f, dtype=np.uint8, copy=True)
    f[(f >= SNAP_MIN).all(axis=2)] = 255
    return f


def trim_rows(frames, margin: int = TRIM_MARGIN):
    """Remove blank rows above and below the content shared by all frames; keep an even height."""
    h = frames[0].shape[0]
    ink = np.zeros(h, bool)
    for f in frames:
        ink |= (f != 255).any(axis=2).any(axis=1)
    rows = np.flatnonzero(ink)
    if rows.size == 0:
        return frames
    top = max(0, int(rows[0]) - margin)
    bot = min(h, int(rows[-1]) + 1 + margin)
    if (bot - top) % 2:
        if bot < h:
            bot += 1
        elif top > 0:
            top -= 1
        else:
            bot -= 1
    return [np.ascontiguousarray(f[top:bot]) for f in frames]


def build_palette(frames, colors: int = GIF_COLORS):
    """Reserved exact colours first (white at index 0), then median-cut colours of a contact sheet."""
    pal = list(RESERVED_RGB)
    assert len(pal) < colors
    picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, min(5, len(frames)))]
    sheet = Image.fromarray(np.concatenate(picks, axis=0))
    q = sheet.quantize(colors=colors - len(pal), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    flat = q.getpalette() or []
    for i in np.unique(np.asarray(q)):
        i = int(i)
        rgb = tuple(int(v) for v in flat[3 * i: 3 * i + 3])
        if len(rgb) == 3 and rgb not in pal and min(rgb) < SNAP_MIN:
            pal.append(rgb)
    return np.asarray(pal[:colors], np.int32)


def map_to_palette(frame: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Exact nearest-colour index of every pixel (squared RGB distance; ties go to the lower index)."""
    flat = frame.reshape(-1, 3).astype(np.int32)
    codes = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    uniq, inv = np.unique(codes, return_inverse=True)
    u = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1)
    idx = np.empty(len(u), np.uint8)
    for s in range(0, len(u), 4096):
        d = ((u[s:s + 4096, None, :] - P[None, :, :]) ** 2).sum(axis=2)
        idx[s:s + 4096] = d.argmin(axis=1)
    return idx[np.asarray(inv).ravel()].reshape(frame.shape[:2])


def write_gif(frames, durs, path: Path, colors: int = GIF_COLORS) -> int:
    P = build_palette(frames, colors)
    flat = P.astype(np.uint8).ravel().tolist()
    ims = []
    for f in frames:
        im = Image.fromarray(map_to_palette(f, P))
        im.putpalette(flat)
        ims.append(im)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=list(durs), loop=0, optimize=False, disposal=1)
    return len(P)


def check_gif_background(path: Path, first: np.ndarray) -> None:
    """Read the GIF back: white pixels of the first source frame and the corners of every frame must be #FFFFFF."""
    src_white = (first == 255).all(axis=2)
    n, corners, first_ok = 0, True, False
    with Image.open(path) as g:
        for i, fr in enumerate(ImageSequence.Iterator(g)):
            rgb = np.asarray(fr.convert("RGB"))
            if i == 0:
                first_ok = rgb.shape == first.shape and bool((rgb[src_white] == 255).all())
            corners = corners and bool((rgb[1, 1] == 255).all() and (rgb[-2, 1] == 255).all())
            n += 1
    ok = first_ok and corners
    print(f"[check] {path.name}: background #FFFFFF read back ({100 * src_white.mean():.1f}% of the first frame "
          f"is pure white; corners pure white in all {n} stored frames) {'OK' if ok else 'MISMATCH'}")
    assert ok, f"{path.name}: page background is not pure white"


def write_mp4(frames, durs, path: Path) -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        return "skipped (ffmpeg not found on PATH)"
    h, w = frames[0].shape[:2]
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{w}x{h}", "-framerate", str(FPS),
           "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-map_metadata", "-1", str(path)]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        try:
            for f, ms in zip(frames, durs):
                data = np.ascontiguousarray(f).tobytes()
                for _ in range(ms // MS_STEP):
                    proc.stdin.write(data)
            proc.stdin.close()
        except OSError:
            pass
        rc = proc.wait()
        err.seek(0)
        msg = err.read().decode("utf-8", "replace").strip()
    if rc != 0:
        if path.exists():
            path.unlink()
        return f"FAILED (ffmpeg exit code {rc}): {msg[-400:]}"
    return "written"


def write_animation(reel: Reel, out: Path, mp4: bool) -> None:
    n = len(reel.frames)
    assert 0 < n <= MAX_FRAMES, f"{reel.name}: {n} frames"
    assert all(f.shape == reel.frames[0].shape for f in reel.frames)
    frames = trim_rows([snap_white(f) for f in reel.frames])
    h, w = frames[0].shape[:2]
    assert h % 2 == 0 and w % 2 == 0
    gif = out / f"{reel.name}.gif"
    ncol = write_gif(frames, reel.durs, gif)
    size = gif.stat().st_size
    print(f"[media] {gif.name}: {n} frames, {w}x{h} px, {sum(reel.durs) / 1000:.1f} s per loop, "
          f"palette {ncol} colours ({len(RESERVED_RGB)} reserved), {size / 1e6:.2f} MB"
          + ("  WARNING: above 4 MB" if size > GIF_LIMIT else ""))
    check_gif_background(gif, frames[0])
    if mp4:
        p = out / f"{reel.name}.mp4"
        status = write_mp4(frames, reel.durs, p)
        extra = ""
        if p.exists():
            s = p.stat().st_size
            extra = f", {s / 1e6:.2f} MB" + ("  WARNING: above 2 MB" if s > MP4_LIMIT else "")
        print(f"[media] {p.name}: {status}{extra}")
    else:
        print(f"[media] {reel.name}.mp4: not written (--no-mp4 or ffmpeg missing)")


# ----------------------------------------------------------------------------- data (read once, checked)
@lru_cache(maxsize=None)
def load_cases() -> dict:
    return json.loads((QUAL / "accident_cases.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def pair() -> dict:
    """Systems A and B at H = 10 s, with the RMSCD area check."""
    ids = list(load_cases()["systems"])
    assert len(ids) == 2
    out = {}
    for tag, sid in zip("AB", ids):
        r = load_record(10.0, sid)
        S = np.asarray(r["S_H"], float)
        step = float(r["delta_s"])
        grid = np.arange(len(S)) * step
        assert abs(grid[-1] - 10.0) < 1e-9, f"{sid}: S_H grid ends at {grid[-1]}, expected 10 s"
        check(f"{tag} RMSCD@10 = trapezoid area above S_H ({sid})", trap(1 - S, step), r["RMSCD"], 1e-6)
        if "S_H_macro" in r and "RMSCD_macro" in r:
            check(f"{tag} RMSCD_macro@10 = trapezoid area above S_H_macro", trap(1 - np.asarray(r["S_H_macro"], float), step),
                  r["RMSCD_macro"], 1e-6)
        bs = r["bootstrap"]
        out[tag] = dict(sid=sid, S=S, grid=grid, step=step, rmscd=float(r["RMSCD"]),
                        rm_ci=(float(bs["RMSCD@H"]["ci_lo"]), float(bs["RMSCD@H"]["ci_hi"])),
                        acc=float(r["end_window_macro_acc"]),
                        acc_ci=(float(bs["end_window_macro_acc"]["ci_lo"]), float(bs["end_window_macro_acc"]["ci_hi"])),
                        N=int(r["N_H"]))
    assert out["A"]["N"] == out["B"]["N"], "A and B must share the eligibility cohort"
    assert np.allclose(out["A"]["grid"], out["B"]["grid"])
    a, b = out["A"]["acc_ci"], out["B"]["acc_ci"]
    out["overlap"] = bool(a[0] <= b[1] and b[0] <= a[1])
    print(f"[info] window-end macro-Acc 95% CI: A {a}, B {b}, overlap = {out['overlap']}")
    return out


@lru_cache(maxsize=None)
def bundle() -> tuple:
    """All non-trivial systems at H = 10 s (every per-system record; families block and trivial excluded)."""
    rows = []
    for p in sorted(metrics_dir(10.0).glob("*.json")):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        fam = r.get("family") or p.stem.split("__")[0]
        if fam in ("block", "trivial"):
            continue
        S = np.asarray(r["S_H"], float)
        assert len(S) == 21, f"{p.name}: S_H has {len(S)} points"
        rows.append((p.stem, S, float(r["end_window_macro_acc"]), float(r["RMSCD"])))
    ids = {r[0] for r in rows}
    assert all(s in ids for s in load_cases()["systems"])
    print(f"[info] non-trivial systems at H = 10 s: {len(rows)} (the paper's Fig. 4 shows 168)")
    return tuple(rows)


@lru_cache(maxsize=None)
def cohort() -> dict:
    man = pd.read_csv(MANIFEST)
    L = np.sort(man.loc[man["split"] == "test", "post_anchor_length_s"].to_numpy(float))[::-1]
    sid = load_cases()["systems"][0]
    counts = {}
    for h in sorted(HASH):
        n_h = int((L >= h - 1e-9).sum())
        check(f"N_H at H = {h:g} s (manifest count vs stored record of {sid})", n_h, int(load_record(h, sid)["N_H"]), 0)
        counts[h] = n_h
    check("N_H at H = 10 s vs cohort_n of accident_cases.json", counts[10.0], int(load_cases()["cohort_n"]), 0)
    print(f"[info] test clips: {len(L)}, N_H: {counts}")
    return {"L": L, "counts": counts}


@lru_cache(maxsize=None)
def g3_groups() -> tuple:
    raw = json.loads((REMOTE04 / "r2c" / "gates" / "g3_v5.json").read_text(encoding="utf-8"))
    groups = []
    for bh in raw["primary"]["v5_22p0"]["by_h"]:
        rows = bh["rows"]
        real = [r for r in rows if r["role"] == "real"]
        rand = [r for r in rows if r["role"] == "random_block"]
        g = {"h": float(bh["h_s"]), "ruler": float(bh["ruler"]), "n_real": len(real),
             "real": {k: np.array([r[k]["gap_long_minus_short"] for r in real], float) for k in KLAB},
             "rand": [{"sid": r["system_id"],
                       **{k: (float(r[k]["gap_long_minus_short"]), float(r[k]["ci_lo"]), float(r[k]["ci_hi"])) for k in KLAB}}
                      for r in rand]}
        groups.append(g)
        print(f"[info] G3 H = {g['h']:g} s: {g['n_real']} non-trivial systems, ruler {g['ruler']:.3f} s, random blocks "
              + "; ".join(f"{rb['sid']} fixed {rb['fixed'][0]:+.2f} / per-clip end {rb['own'][0]:+.2f} s" for rb in g["rand"]))
    return tuple(sorted(groups, key=lambda g: g["h"]))


# ----------------------------------------------------------------------------- animation 1: reverse AND
def anim_reverse_and() -> Reel:
    d = load_cases()
    sel = [s for s in d["selected"] if s["stratum"] == "early_correct_late_error"]
    assert len(sel) == 1
    c = sel[0]
    vid = c["video_id"]
    x = np.asarray(d["offsets"], float)
    n = len(x)
    step = float(x[1] - x[0])
    H = float(x[-1])
    assert np.allclose(np.diff(x), step) and abs(x[0]) < 1e-12
    y = int(c["metadata"]["class_code"])
    assert ACC_NAME[y] == c["metadata"]["class_name"]
    assert len(d["systems"]) == 2 and len(c["traces"]) == 2
    rows = []
    for tag, sid, tr in zip("AB", d["systems"], c["traces"]):
        pred = np.asarray(tr["pred"], int)
        good = pred == y
        assert good.tolist() == [bool(v) for v in tr["correct"]], f"{tag}: stored correctness differs"
        st = stable_suffix(good)
        assert st.tolist() == [bool(v) for v in tr["stable"]], f"{tag}: stored stable suffix differs"
        g, s = good.astype(float), st.astype(float)
        ce, cr, cd = cum_trap(1 - g, step), cum_trap(g - s, step), cum_trap(1 - s, step)
        assert np.allclose(ce + cr, cd, rtol=0, atol=1e-12)
        check(f"{tag} clip contribution d_i ({vid})", cd[-1], float(tr["delay"]), 1e-9)
        onset = float(x[np.flatnonzero(st)[0]]) if st.any() else None
        rows.append(dict(tag=tag, sid=sid, color=SYS_COLOR[tag], desc=SYS_DESC[tag], pred=pred, g=g, s=s,
                         good=good, st=st, ce=ce, cr=cr, cd=cd, onset=onset, stored=float(tr["delay"])))
        print(f"[info] {tag} ({sid}): e = {ce[-1]:.2f} s, r = {cr[-1]:.2f} s, d = {cd[-1]:.2f} s, onset = {onset}")

    fig = plt.figure(figsize=(W_IN, 5.2), dpi=DPI)
    X0, XW, LANE_H, AREA_H = 0.205, 0.775, 0.085, 0.13
    LANE_Y, AREA_Y = (0.705, 0.395), (0.545, 0.235)
    xl = (x[0] - 0.6 * step, x[-1] + 0.6 * step)
    tw = 0.84 * step
    tile = {"hidden": ("white", TILE_HIDDEN, INK), "good": (TEAL_L, TEAL, INK), "bad": (GREY_F, GREY, GREY_T),
            "stable": (TEAL_M, TEAL, INK), "retr": (OCHRE_F, OCHRE, INK)}
    title = "Offline reverse AND: from instantaneous answers to a stable-correct suffix"
    subtitle = (f"Stored ACCIDENT test trajectories of systems A and B  \u00b7  clip {vid}  \u00b7  reference class: "
                f"{ACC_NAME[y]} ({ACC_ABBR[y]})  \u00b7  H = {H:g} s, \u0394 = {step:g} s")
    caps = {0: "1   Answers on causal prefixes, read at each offset after the anchor (green: correct now)",
            1: "1   Answers on causal prefixes, read at each offset after the anchor (green: correct now)",
            2: "2   Reverse AND from the window end: a tile stays in the suffix only if it and all later tiles are correct",
            3: "3   Area above the stable indicator: d = e + r is this clip's contribution to RMSCD@10"}
    handles = [Patch(facecolor=TEAL_L, edgecolor=TEAL, lw=0.7, label="correct at this offset"),
               Patch(facecolor=TEAL_M, edgecolor=TEAL, lw=0.7, label="stable-correct suffix (underlined)"),
               Patch(facecolor=OCHRE_F, edgecolor=OCHRE, lw=0.7, label="correct now, later wrong; area r"),
               Patch(facecolor=GREY_F, edgecolor=GREY, lw=0.7, label="wrong; error area e"),
               Line2D([], [], color=SLATE, lw=1.0, ls=(0, (3, 2)), label="correct indicator"),
               Line2D([], [], color=TEAL, lw=2.0, ls="-", label="stable indicator")]

    def tile_state(phase, m, i, row):
        if phase == 0 or (phase == 1 and i > m):
            return "hidden"
        if phase == 1 or (phase == 2 and i < m):
            return "good" if row["good"][i] else "bad"
        if row["st"][i]:
            return "stable"
        return "retr" if row["good"][i] else "bad"

    def draw(phase, m):
        fig.clf()
        fig.text(0.015, 0.965, title, fontsize=13.5, fontweight="bold", color=INK, va="top")
        fig.text(0.015, 0.912, subtitle, fontsize=10, color=MUTED, va="top")
        fig.text(0.015, 0.862, caps[phase], fontsize=11, fontweight="bold", color=SLATE, va="top")
        lower = None
        for k, row in enumerate(rows):
            lane = fig.add_axes([X0, LANE_Y[k], XW, LANE_H])
            lane.set_xlim(*xl)
            lane.set_ylim(0, 1)
            lane.axis("off")
            for i in range(n):
                stt = tile_state(phase, m, i, row)
                fc, ec, tc = tile[stt]
                lane.add_patch(Rectangle((x[i] - tw / 2, 0.18), tw, 0.78, facecolor=fc, edgecolor=ec, lw=0.7))
                if stt != "hidden":
                    lane.text(x[i], 0.57, ACC_ABBR[int(row["pred"][i])], ha="center", va="center", fontsize=9.5, color=tc)
                if stt == "stable":
                    lane.add_patch(Rectangle((x[i] - tw / 2, 0.0), tw, 0.11, facecolor=TEAL, edgecolor="none"))
            if phase == 2:
                lane.axvline(x[m] - step / 2, color=SLATE, lw=1.6, ls="-")
            ytop = LANE_Y[k] + LANE_H
            fig.text(0.015, ytop + 0.004, row["tag"], fontsize=13, fontweight="bold", color=row["color"], va="top")
            fig.text(0.040, ytop - 0.002, row["desc"], fontsize=9.5, color=MUTED, va="top")
            if phase == 1:
                status = f"correct now: {int(row['good'][:m + 1].sum())} of {m + 1}"
            elif phase == 2:
                status = f"running AND = {int(row['st'][m])}"
            elif phase == 3:
                status = f"stable from {row['onset']:g} s" if row["onset"] is not None else "never stable in the window"
            else:
                status = ""
            fig.text(0.015, LANE_Y[k] + 0.006, status, fontsize=10, color=row["color"], va="bottom")

            ar = fig.add_axes([X0, AREA_Y[k], XW, AREA_H])
            ar.set_xlim(*xl)
            ar.set_ylim(-0.1, 1.18)
            ar.set_yticks([0, 1])
            ar.set_xticks(np.arange(0, H + 1e-9, 2.0))
            style_ax(ar)
            ar.tick_params(length=3.5, width=0.8)
            ar.axhline(1.0, color=ROLE["grid"], lw=0.7, ls="-", zorder=0)
            if k == len(rows) - 1:
                ar.set_xlabel("offset after the anchor  \u03b4 (s)")
                lower = ar
            g, s = row["g"], row["s"]
            if phase >= 1:
                upto = m if phase == 1 else n - 1
                ar.plot(x[:upto + 1], g[:upto + 1], color=SLATE, lw=1.0, ls=(0, (3, 2)), marker="o", ms=2.4, zorder=2)
            if phase == 2:
                ar.plot(x[m:], s[m:], color=TEAL, lw=2.0, ls="-", zorder=3)
            if phase == 3:
                if m >= 1:
                    ar.fill_between(x[:m + 1], g[:m + 1], 1.0, color=GREY_F, lw=0, zorder=1)
                    ar.fill_between(x[:m + 1], s[:m + 1], g[:m + 1], color=OCHRE_F, lw=0, zorder=1)
                ar.plot(x, s, color=TEAL, lw=2.0, ls="-", zorder=3)
                ar.axvline(x[m], color=SLATE, lw=1.0, ls="-", zorder=4)
                vals = [("error area e", row["ce"][m], GREY_T, "normal"),
                        ("retracted-correct r", row["cr"][m], OCHRE_T, "normal"),
                        ("d = e + r", row["cd"][m], INK, "bold")]
                if m == n - 1:
                    vals.append(("stored d_i", row["stored"], MUTED, "normal"))
                for j, (lab, v, col, wt) in enumerate(vals):
                    yy = AREA_Y[k] + AREA_H + 0.004 - j * 0.034
                    fig.text(0.015, yy, lab, fontsize=9.5, color=col, fontweight=wt, va="top")
                    fig.text(0.192, yy, f"{v:.2f} s", fontsize=9.5, color=col, fontweight=wt, va="top", ha="right")
        legend_below(fig, [lower], x=0.015, loc="upper left", handles=handles, ncol=3, fontsize=9.5,
                     handlelength=1.6, columnspacing=1.6)

    states = [(0, -1, 1200)]
    states += [(1, m, 200) for m in range(n)]
    states[-1] = (1, n - 1, 1400)
    states += [(2, m, 200) for m in range(n - 1, -1, -1)]
    states[-1] = (2, 0, 1400)
    states += [(3, m, 200) for m in range(n)]
    states[-1] = (3, n - 1, 5000)
    reel = Reel("ape_reverse_and")
    for phase, m, ms in states:
        draw(phase, m)
        reel.add(fig, ms)
    plt.close(fig)
    return reel


# ----------------------------------------------------------------------------- animation 2: tied pair
def anim_tied_pair() -> Reel:
    P = pair()
    B = bundle()
    grid, step = P["A"]["grid"], P["A"]["step"]
    n, H = len(grid), float(grid[-1])
    cum = {t: cum_trap(1 - P[t]["S"], step) for t in "AB"}
    for t in "AB":
        check(f"{t} accumulated area at delta = H vs stored RMSCD", cum[t][-1], P[t]["rmscd"], 1e-6)
    segs = [np.column_stack([grid, S]) for _, S, _, _ in B]
    fig = plt.figure(figsize=(W_IN, 4.4), dpi=DPI)
    axpos = {"A": [0.075, 0.15, 0.30, 0.62], "B": [0.425, 0.15, 0.30, 0.62]}
    title = "A window-end-tied pair: same end, different course"
    subtitle = (f"ACCIDENT test, H = {H:g} s, one eligibility cohort of $N_H$ = {P['A']['N']} clips for both systems"
                f"  \u00b7  grey: $S_H$ of all {len(B)} non-trivial systems")
    box = dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="none")

    def draw(k):
        fig.clf()
        fig.text(0.015, 0.965, title, fontsize=13.5, fontweight="bold", color=INK, va="top")
        fig.text(0.015, 0.905, subtitle, fontsize=10, color=MUTED, va="top")
        for t in "AB":
            r, col = P[t], SYS_COLOR[t]
            ax = fig.add_axes(axpos[t])
            ax.axhline(1.0, color=ROLE["grid"], lw=0.8, ls="-", zorder=0)
            if k >= 1:
                ax.fill_between(grid[:k + 1], r["S"][:k + 1], 1.0, color=SYS_FILL[t], lw=0, zorder=0.5)
            ax.add_collection(LineCollection(segs, colors=BUNDLE, linewidths=0.6, linestyles="solid", zorder=1))
            ax.plot(grid[:k + 1], r["S"][:k + 1], color=col, ls="-", lw=2.0, marker="o", ms=3.0, mfc=col, mec=col,
                    zorder=3)
            ax.axvline(grid[k], color=SLATE, lw=0.9, ls=(0, (3, 2)), zorder=4)
            txt = (f"area above $S_H$ = RMSCD@10 = {cum[t][k]:.2f} s" if k == n - 1
                   else f"area above $S_H$ up to \u03b4: {cum[t][k]:.2f} s")
            ax.text(0.03, 0.965, txt, transform=ax.transAxes, ha="left", va="top", fontsize=10, color=col, zorder=6,
                    bbox=box)
            ax.set_xlim(0, H + 0.3)
            ax.set_ylim(0, 1.02)
            ax.set_xticks(np.arange(0, H + 1e-9, 2.0))
            ax.set_xlabel("offset after the anchor  \u03b4 (s)")
            if t == "A":
                ax.set_ylabel(r"stable-correct fraction $S_H(\delta)$")
            ax.set_title(f"{t}   {SYS_DESC[t]}", color=col, fontweight="bold", fontsize=11.5)
            style_ax(ax)
            ax.tick_params(length=3.5, width=0.8)

        bx = fig.add_axes([0.80, 0.50, 0.18, 0.20])
        vals = [cum[t][k] for t in "AB"]
        bx.barh([1, 0], vals, color=[C_A, C_B], height=0.55, zorder=2)
        for yy, v in zip([1, 0], vals):
            bx.text(v + 0.2, yy, f"{v:.2f}", va="center", fontsize=10, color=INK)
        if k == n - 1:
            for yy, t in zip([1, 0], "AB"):
                bx.plot([P[t]["rmscd"]] * 2, [yy - 0.36, yy + 0.36], color=INK, lw=1.0, ls="-", zorder=3)
        bx.set_xlim(0, H)
        bx.set_ylim(-0.6, 1.6)
        bx.set_xticks([0, 5, 10])
        bx.set_yticks([1, 0])
        bx.set_yticklabels(["A", "B"])
        for lab, col in zip(bx.get_yticklabels(), [C_A, C_B]):
            lab.set_color(col)
            lab.set_fontweight("bold")
        dtxt = "\u03b4 = H" if k == n - 1 else f"\u03b4 = {grid[k]:.1f} s"
        bx.set_title(f"area above $S_H$ (s)\nup to {dtxt}", fontsize=10.5)
        style_ax(bx)
        bx.tick_params(length=3.5, width=0.8)
        fig.text(0.80, 0.385, "window-end macro-Acc, 95% CI", fontsize=9.5, color=MUTED, va="top")
        for j, t in enumerate("AB"):
            lo, hi = P[t]["acc_ci"]
            fig.text(0.80, 0.335 - j * 0.055, f"{t}  {P[t]['acc']:.3f}  [{lo:.3f}, {hi:.3f}]", fontsize=10,
                     color=SYS_COLOR[t], va="top")
        if P["overlap"]:
            fig.text(0.80, 0.215, "intervals overlap:\ntied at the window end", fontsize=9.5, color=INK, va="top",
                     linespacing=1.15)

    reel = Reel("ape_tied_pair_rmscd")
    for k in range(n):
        draw(k)
        reel.add(fig, 1500 if k == 0 else (5000 if k == n - 1 else 300))
    plt.close(fig)
    return reel


# ----------------------------------------------------------------------------- animation 3: eligibility cohort
def anim_eligibility() -> Reel:
    C = cohort()
    G = g3_groups()
    L, counts = C["L"], C["counts"]
    ntest = len(L)
    reg = sorted(counts)
    hs_grid = np.arange(0, int(round(max(reg) / 0.5)) + 1) * 0.5
    ypos = [gi * 2.7 for gi in range(len(G))]
    allv = [0.0]
    for g in G:
        allv += [g["ruler"], -g["ruler"]]
        for k in KLAB:
            allv += list(g["real"][k]) + [v for rb in g["rand"] for v in rb[k]]
    xlo, xhi = float(np.floor(min(allv)) - 0.5), float(np.ceil(max(allv)) + 0.5)
    xe = np.arange(ntest + 1)
    yy = np.r_[L, L[-1]]
    fig = plt.figure(figsize=(W_IN, 4.8), dpi=DPI)
    title = "The eligibility cohort, and why per-clip-end scoring biases the comparison"
    subtitle = (f"ACCIDENT test split, {ntest} clips  \u00b7  a clip is scored at horizon H only if it is observed for "
                f"at least H s after the anchor")
    left_handles = [Patch(facecolor=SLATE_BAR, label=r"in the cohort ($L^{+} \geq H$)"),
                    Patch(facecolor=EXCL, label="observed for less than H")]
    nreal = G[0]["n_real"]
    right_handles = [Line2D([], [], ls="", marker="|", ms=9, mew=1.0, color=KCOL["fixed"],
                            label=f"{nreal} non-trivial systems, fixed cohort"),
                     Line2D([], [], ls="", marker="|", ms=9, mew=1.0, color=KCOL["own"],
                            label=f"{nreal} non-trivial systems, per-clip end"),
                     Line2D([], [], ls="", marker="D", ms=6, mfc="white", mec=INK, color=INK,
                            label="random-answer blocks (95% CI)"),
                     Patch(facecolor=BAND, label="\u00b1 RMSCD ruler of the horizon")]

    def draw(H):
        fig.clf()
        fig.text(0.015, 0.965, title, fontsize=13.5, fontweight="bold", color=INK, va="top")
        fig.text(0.015, 0.905, subtitle, fontsize=10, color=MUTED, va="top")
        nH = int((L >= H - 1e-9).sum())
        ax = fig.add_axes([0.07, 0.25, 0.30, 0.55])
        ax.fill_between(xe[:nH + 1], yy[:nH + 1], step="post", color=SLATE_BAR, lw=0, zorder=1)
        ax.fill_between(xe[nH:], yy[nH:], step="post", color=EXCL, lw=0, zorder=1)
        for h in reg:
            if h <= H + 1e-9:
                ax.plot([0, counts[h]], [h, h], color=MUTED, lw=0.8, ls=":", zorder=2)
                ax.plot(counts[h], h, "o", ms=4.5, color=INK, mec="white", mew=0.6, zorder=4)
                ax.text(counts[h] + 25, h + 0.3, f"H = {h:g} s: $N_H$ = {counts[h]}", fontsize=9, color=INK,
                        va="bottom", zorder=4)
        ax.axhline(H, color=INK, lw=1.3, ls="-", zorder=3)
        ax.text(0.97, 0.97, f"H = {H:.1f} s\n$N_H$ = {nH} of {ntest} ({100 * nH / ntest:.0f}%)", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color=INK, linespacing=1.3)
        ax.set_xlim(0, ntest)
        ax.set_ylim(0, float(np.ceil(L.max())) + 1)
        ax.set_xlabel("test clips, sorted by post-anchor length")
        ax.set_ylabel(r"post-anchor length $L^{+}$ (s)")
        ax.set_title("Who is in the cohort at horizon H")
        style_ax(ax)
        ax.tick_params(length=3.5, width=0.8)

        bx = fig.add_axes([0.60, 0.25, 0.385, 0.55])
        ticks, labels, colors = [], [], []
        for gi, g in enumerate(G):
            y0 = ypos[gi]
            for j, kind in enumerate(KLAB):
                ticks.append(y0 + j)
                labels.append(f"H = {g['h']:g} s \u00b7 {KLAB[kind]}")
                colors.append(KCOL[kind])
            if g["h"] > H + 1e-9:
                continue
            bx.fill_betweenx([y0 - 0.5, y0 + 1.5], -g["ruler"], g["ruler"], color=BAND, lw=0, zorder=0)
            for j, kind in enumerate(KLAB):
                yk = y0 + j
                gaps = g["real"][kind]
                bx.plot(gaps, np.full(len(gaps), yk), ls="", marker="|", ms=9, mew=0.9, color=KCOL[kind], alpha=0.45,
                        zorder=2)
                for ri, rb in enumerate(g["rand"]):
                    v, lo, hi = rb[kind]
                    bx.errorbar(v, yk, xerr=[[v - lo], [hi - v]], fmt=MARKERS[ri % len(MARKERS)], ms=6, color=INK,
                                mfc="white", mew=1.0, elinewidth=1.0, capsize=2.5, zorder=4)
                bx.text(xhi - 0.2, yk - 0.33, f"random: {fmt_range([rb[kind][0] for rb in g['rand']])} s", ha="right",
                        va="bottom", fontsize=9, color=KCOL[kind] if kind == "fixed" else OCHRE_T, zorder=5)
        bx.axvline(0, color=ZERO_LINE, lw=0.7, ls="-", zorder=1)
        bx.set_yticks(ticks)
        bx.set_yticklabels(labels, fontsize=9.5)
        for lab, col in zip(bx.get_yticklabels(), colors):
            lab.set_color(col)
        bx.set_ylim(ypos[-1] + 1.9, -0.9)
        bx.set_xlim(xlo, xhi)
        bx.set_xlabel("long minus short stratum: mean stable-correct delay (s)")
        bx.set_title("Per-clip-end vs fixed-cohort accounting")
        style_ax(bx)
        bx.tick_params(length=3.5, width=0.8)
        legend_below(fig, [ax, bx], x=0.015, loc="upper left", handles=left_handles, ncol=1)
        legend_below(fig, [ax, bx], x=0.995, loc="upper right", handles=right_handles, ncol=2, columnspacing=1.2)

    reel = Reel("ape_eligibility_cohort")
    for i, H in enumerate(hs_grid):
        H = float(H)
        draw(H)
        if i == len(hs_grid) - 1:
            ms = 5000
        elif any(abs(H - h) < 1e-9 for h in reg):
            ms = 1600
        else:
            ms = 200
        reel.add(fig, ms)
    plt.close(fig)
    return reel


# ----------------------------------------------------------------------------- static summary
def static_glance(out: Path) -> None:
    C = cohort()
    P = pair()
    B = bundle()
    G = g3_groups()
    L, counts = C["L"], C["counts"]
    fig = plt.figure(figsize=(W_IN, 3.7))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.07, right=0.985, bottom=0.17, top=0.86)

    ax = fig.add_subplot(gs[0])
    hh = np.arange(0, np.ceil(L.max()) + 0.05, 0.1)
    ax.plot(hh, [(L >= h - 1e-9).mean() for h in hh], color=SLATE, lw=1.6, ls="-")
    for h in sorted(counts):
        f = counts[h] / len(L)
        ax.plot([h, h], [0, f], color=MUTED, lw=0.8, ls=":")
        ax.plot(h, f, "o", ms=5, color=INK, mec="white", mew=0.6, zorder=4)
        ax.text(h + 0.6, f + 0.03, f"H = {h:g} s\n$N_H$ = {counts[h]}", fontsize=9, color=INK, va="bottom")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(0, float(np.ceil(L.max())))
    ax.set_xlabel("horizon H (s)")
    ax.set_ylabel(r"test clips with $L^{+} \geq H$")
    ax.set_title("(a) Eligibility cohort", fontsize=11, fontweight="bold")
    style_ax(ax)

    ax = fig.add_subplot(gs[1])
    ax.scatter([b[2] for b in B], [b[3] for b in B], s=12, color=BUNDLE, lw=0, zorder=1)
    for t in "AB":
        r, col = P[t], SYS_COLOR[t]
        ax.errorbar(r["acc"], r["rmscd"], xerr=[[r["acc"] - r["acc_ci"][0]], [r["acc_ci"][1] - r["acc"]]],
                    yerr=[[r["rmscd"] - r["rm_ci"][0]], [r["rm_ci"][1] - r["rmscd"]]], fmt="o", ms=5.5, mfc="white",
                    mec=col, ecolor=col, elinewidth=1.1, capsize=2, zorder=4)
        ax.text(r["acc"] + 0.01, r["rmscd"] + 0.15, t, color=col, fontsize=11, fontweight="bold", zorder=5)
    ax.axhline(10.0, color=GREY, lw=0.7, ls=(0, (2, 2)))
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.5, mfc=BUNDLE, mec=BUNDLE, color=BUNDLE,
                              label=f"{len(B)} non-trivial systems"),
                       Line2D([], [], color=GREY, lw=0.7, ls=(0, (2, 2)), label="RMSCD = H")],
              loc="best", fontsize=9, frameon=False, handlelength=1.6)
    ax.set_xlabel("window-end macro-Acc")
    ax.set_ylabel("RMSCD@10 (s)")
    ax.set_title("(b) Window-end tie, H = 10 s", fontsize=11, fontweight="bold")
    style_ax(ax)

    ax = fig.add_subplot(gs[2])
    for gi, g in enumerate(G):
        ax.fill_betweenx([gi - 0.42, gi + 0.42], -g["ruler"], g["ruler"], color=BAND, lw=0, zorder=0)
        for kind, dy in (("fixed", -0.17), ("own", 0.17)):
            nr = len(g["rand"])
            for ri, rb in enumerate(g["rand"]):
                v, lo, hi = rb[kind]
                yk = gi + dy + (ri - (nr - 1) / 2) * 0.1
                ax.errorbar(v, yk, xerr=[[v - lo], [hi - v]], fmt=MARKERS[ri % len(MARKERS)], ms=5, color=KCOL[kind],
                            mfc="white", mew=1.0, elinewidth=1.0, capsize=2, zorder=3)
    ax.axvline(0, color=ZERO_LINE, lw=0.7, ls="-")
    ax.set_yticks(range(len(G)))
    ax.set_yticklabels([f"H = {g['h']:g} s" for g in G])
    ax.set_ylim(len(G) - 0.5, -0.5)
    ax.set_xlabel("long minus short: mean delay (s)")
    ax.set_title("(c) Random answers", fontsize=11, fontweight="bold")
    ax.legend(handles=[Line2D([], [], ls="", marker="D", mfc="white", mec=KCOL[k], color=KCOL[k], label=KLAB[k])
                       for k in KLAB] + [Patch(facecolor=BAND, label="\u00b1 ruler")],
              loc="lower right", fontsize=9, frameon=False)
    style_ax(ax)
    p = out / "results_at_a_glance.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    with Image.open(p) as im:
        print(f"[media] {p.name}: {im.size[0]}x{im.size[1]} px, {p.stat().st_size / 1e6:.2f} MB")


def copy_paper_figures(out: Path) -> None:
    for src_name, dst_name in PAPER_FIGS.items():
        src = FIG / src_name
        if src.is_file():
            shutil.copyfile(src, out / dst_name)
            print(f"[media] {dst_name}: copied unchanged from {src}")
        else:
            print(f"[media] {dst_name}: skipped ({src} not found; run figures/run_all.sh first to include it)")


# ----------------------------------------------------------------------------- main
def main() -> None:
    names = ["reverse_and", "tied_pair", "eligibility", "glance", "copies"]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <repository>/docs/media)")
    ap.add_argument("--no-mp4", action="store_true", help="write GIF files only")
    ap.add_argument("--only", nargs="+", choices=names, default=names, help="subset of media to write")
    args = ap.parse_args()
    out = (args.out or (ROOT / "docs" / "media")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mp4 = (not args.no_mp4) and shutil.which("ffmpeg") is not None
    font = matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(
        family=matplotlib.rcParams["font.sans-serif"]))
    assert fmt_signed(-0.004, 2) == "0.00" and fmt_signed(-0.02, 2) == MINUS + "0.02" and fmt_signed(0.03, 2) == "+0.03"
    print(f"[info] APE_DATA = {os.environ['APE_DATA']}")
    print(f"[info] output = {out}; MP4 = {'yes' if mp4 else 'no'}; text font = {font} (calibri: {FONT_STATUS.get('calibri')})")
    print(f"[info] system colours: A {C_A}, B {C_B} (stylelib OKABE_ITO blue = {OKABE_ITO.get('blue')}, "
          f"vermillion = {OKABE_ITO.get('vermillion')}); reserved GIF palette colours: {len(RESERVED_RGB)}")
    if "reverse_and" in args.only:
        write_animation(anim_reverse_and(), out, mp4)
    if "tied_pair" in args.only:
        write_animation(anim_tied_pair(), out, mp4)
    if "eligibility" in args.only:
        write_animation(anim_eligibility(), out, mp4)
    if "glance" in args.only:
        static_glance(out)
    if "copies" in args.only:
        copy_paper_figures(out)
    print("[done] all checks passed")


if __name__ == "__main__":
    main()
