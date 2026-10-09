"""Write the qualitative replay media of the paper into docs/media/: Fig. 5 (ACCIDENT) and Fig. S3 (MM-AU).

Media (real dataset frames, synchronised with the stored predictions of the six qualitative cases):
  qual_accident_replay.mp4         the three ACCIDENT cases of Fig. 5 in the paper's order (systems A and B)
  qual_mmau_replay.mp4             the three MM-AU development cases of Fig. S3 (one GRU, input cadences q = 2, 4, 8)
  qual_accident_<n>.gif, n = 1..3  one ACCIDENT case per file (README)
  qual_mmau_<n>.gif, n = 1..3      one MM-AU case per file (README)
  qual_media_record.json           per case: source hashes, the decoded frame (or image file) shown at every display
                                   position, the published-frame checks and the read-outs; per file: step and size

Reads (nothing else):
  figures/qualitative/accident_cases.json, mmau_cases.json   stored predictions and selection rules
  figures/qualitative/accident_videos.json                   SHA-256 and size of the three ACCIDENT videos
  figures/qualitative/accident_frames.json                   frame index and timestamp of the published frames
  figures/qualitative/accident_frame_pixels.json             RGB pixel hashes of the published frames
  figures/qualitative/mmau_frames.json                       SHA-256 of the nine published MM-AU frames
  $APE_ACCIDENT/real_videos/<video_id>.mp4                   ACCIDENT dataset root
  $APE_MMAU/extracted/CAP-DATA/.../images/000NNN.jpg         MM-AU dataset root
  optional, --verify-sources: the ACCIDENT answer-set CSVs ($APE_QUAL_SOURCES) and the MM-AU prediction arrays
  ($APE_DATA), with the same checks as fig_qualitative_cases.py --verify-sources.

Frame rule.
  ACCIDENT: sequential OpenCV decoding, as in extract_qual_frames.py; for every display time anchor + t the last
  decoded frame whose timestamp is at or before it is shown. Display grid: t = -1 s .. +10 s, every 0.1 s (MP4) and
  every 0.25 s (GIF; every 0.5 s if the GIF would exceed 6 MB). MM-AU: the image files anchor - 8 .. anchor + 64
  (1-based file names), every image (MP4) and every 2nd image (GIF; every 4th if the GIF would exceed 6 MB).
  Frames are scaled only (area averaging) into a black 512 x 288 letterbox; no crop, no enhancement, no
  interpolation, nothing drawn on the footage.
Prediction rule.
  The read at offset delta (prefix up to anchor + delta) appears when the display reaches anchor + delta; nothing
  changes between reads. The stable-correct suffix, endpoint, onset and d_i are shown only after the window
  (offline), on the held final frame of each case, which carries the read-outs and note of the paper's panel.

Run from the repository root:
  APE_ACCIDENT=/path/to/ACCIDENT APE_MMAU=/path/to/MM-AU python figures/make_qual_media.py
  python figures/make_qual_media.py --cases accident_2 mmau --no-mp4
Options: --out DIR, --no-mp4, --cases {accident,mmau,accident_1..3,mmau_1..3} ..., --verify-sources,
         --allow-decoder-drift (report instead of stop when decoded pixels differ from the published frames).
The published ACCIDENT frames are reproduced bit for bit by opencv-python-headless 4.11.0.86 (see
extract_qual_frames.py). MP4 files are written only when ffmpeg is on PATH. APE_DATA defaults to the repository's
data/ folder, as for make_media.py.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
APE_DATA_GIVEN = bool(os.environ.get("APE_DATA"))
if not APE_DATA_GIVEN:  # same default as make_media.py; only --verify-sources needs the full derived-data package
    os.environ["APE_DATA"] = str(HERE.parent / "data")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import fig_qualitative_cases as fqc  # noqa: E402  read-outs, notes, titles and colours of Figs. 5 and S3
import make_media as mm  # noqa: E402  exact palette mapping, white-background read-back, signed numbers
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from stylelib import OKABE_ITO  # noqa: E402

matplotlib.rcParams.update({"font.size": 10, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
                            "ytick.major.width": 0.8})

QUAL = HERE / "qualitative"
MINUS = "\u2212"
DOT = "  \u00b7  "

# ----------------------------------------------------------------------------- layout (pixels, top-left origin)
DPI = 100
WPX = 960
MARGIN = 16
BOX = (16, 16, 512, 288)          # x, y, width, height of the letterboxed video panel
INFO_X = 548                      # info column to the right of the video
PLOT_X0, PLOT_X1 = 142, 944
PLOT_W = PLOT_X1 - PLOT_X0
PROB_TOP, PROB_H = 326, 100
LANE_TOP, LANE_H = 434, 36

# ----------------------------------------------------------------------------- timing
TICK_MS = 100                     # MP4 time base: 10 frames per second
ACC_PRE_MS = 1000                 # ACCIDENT playback starts 1 s before the anchor
MM_PRE = 8                        # MM-AU playback starts 8 image files before the anchor
MM_MS_PER_FRAME = 100             # MM-AU display pace: 10 image files per second of display
MP4_STEP = {"accident": 100, "mmau": 1}            # ms of video time / image files per display frame
GIF_STEPS = {"accident": (250, 500), "mmau": (2, 4)}  # first choice, then the fallback above GIF_LIMIT
END_HOLD_MS, FINAL_HOLD_MS = 1500, 6000
GIF_LIMIT = 6.0e6
MP4_LIMIT = 10.0e6
GIF_COLORS = 256
GIF_UI_COLORS = 24
SNAP_MIN = mm.SNAP_MIN

# ----------------------------------------------------------------------------- colours (paper semantics)
INK, MUTED = fqc.INK, fqc.MUTED
OK_FILL, OK_EDGE, BAD_FILL, BAD_EDGE, TRACK = fqc.OK_FILL, fqc.OK_EDGE, fqc.BAD_FILL, fqc.BAD_EDGE, fqc.TRACK
WRONG_TXT = "#7C8790"             # tile text of a wrong answer (fig_qualitative_cases.py)
GRID, GUIDE = "#EDEFF1", "#D5DADF"
KEY_INK = "#34495E"               # legend glyphs (fig_qualitative_cases.py)
HIDDEN = "#C9D0D6"                # dashed outline of a tile that is not read yet
PRE_SHADE = "#F3F5F7"             # offsets before the anchor
CURSOR = mm.SLATE                 # display-time cursor
LETTERBOX = (0, 0, 0)

DS = {
    "accident": dict(
        cases="accident_cases.json", fig="Fig. 5", where="ACCIDENT test clip", scale=1000, horizon=10000,
        xlim=(-1.15, 10.3), ylim=(0.0, 1.0), yticks=[0, 0.5, 1.0], ticks=[0, 2, 4, 6, 8, 10],
        xlabel="Offset after the published anchor (s)", ushort="s", hlabel="H = 10 s",
        colours=[OKABE_ITO["blue"], OKABE_ITO["vermillion"]],
        key=("A: prefix-mean ResNet-18  |  B: GRU-512 with EMA smoothing, replayed at the 0.5-s step\n"
             "seed 20260903  |  H = 10 s, \u0394 = 0.5 s"),
        source=("Frames: ACCIDENT dataset (Picek & Hanzl, 2026), Kaggle picekl/accident"
                + DOT + "predictions: stored outputs of this study"),
        env=("APE_ACCIDENT", "ACCIDENT dataset root (the folder that contains real_videos/)")),
    "mmau": dict(
        cases="mmau_cases.json", fig="Fig. S3", where="MM-AU development clip", scale=1, horizon=64,
        xlim=(-9.5, 68.0), ylim=(0.0, 0.4), yticks=[0, 0.2, 0.4], ticks=[0, 16, 32, 48, 64],
        xlabel="Offset after the published anchor (frames)", ushort="f", hlabel="H = 64 frames",
        colours=[OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["purple"]],
        key=("one frozen class-weighted GRU checkpoint, input every q = 2, 4, 8 frames  |  seed 20260920\n"
             "output step 8 frames, H = 64 frames  |  tiles: native class IDs"),
        source=("Frames: MM-AU dataset (Fang et al., 2024), Hugging Face JeffreyChou/MM-AU"
                + DOT + "predictions: stored outputs of this study"),
        env=("APE_MMAU", "MM-AU dataset root (the folder that contains extracted/CAP-DATA/)")),
}


def _rgb255(color) -> tuple:
    return tuple(int(round(v * 255)) for v in to_rgb(color))


def _reserved() -> list:
    out = []
    for c in ["#FFFFFF", "#000000", INK, MUTED, WRONG_TXT, OK_FILL, OK_EDGE, BAD_FILL, BAD_EDGE, TRACK, HIDDEN,
              GRID, GUIDE, PRE_SHADE, CURSOR, KEY_INK, *DS["mmau"]["colours"]]:
        rgb = _rgb255(c)
        if rgb not in out:
            out.append(rgb)
    assert out[0] == (255, 255, 255)
    return out


RESERVED_RGB = _reserved()


# ----------------------------------------------------------------------------- small helpers
def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(name: str):
    return json.loads((QUAL / name).read_text(encoding="utf-8"))


def signed_int(v: int) -> str:
    v = int(v)
    return "0" if v == 0 else ("+" if v > 0 else MINUS) + str(abs(v))


def env_root(var: str, what: str) -> Path:
    v = os.environ.get(var)
    if not v:
        raise SystemExit(f"{var} is not set. Point it to the {what}, e.g. export {var}=/path/to/dataset")
    p = Path(v).expanduser().resolve()
    if not p.is_dir():
        raise SystemExit(f"{var}={p} is not a folder; expected the {what}")
    return p


def fit_frame(rgb: np.ndarray) -> np.ndarray:
    """Scale a frame to fit the video panel at its own aspect ratio (area averaging; nothing else)."""
    h, w = rgb.shape[:2]
    bw, bh = BOX[2], BOX[3]
    s = min(bw / w, bh / h)
    nw, nh = min(bw, max(1, int(round(w * s)))), min(bh, max(1, int(round(h * s))))
    if (nw, nh) == (w, h):
        return np.ascontiguousarray(rgb, dtype=np.uint8).copy()
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize((nw, nh), Image.Resampling.BOX), np.uint8)


# ----------------------------------------------------------------------------- cases and read-outs
@dataclass
class Trace:
    name: str
    colour: str
    labels: list
    p: np.ndarray
    good: np.ndarray
    stable: np.ndarray
    onset: float | None
    d: float


@dataclass
class Case:
    ds: str
    n: int
    vid: str
    stratum: str
    meta: dict
    ref_txt: str
    x: np.ndarray
    xi: np.ndarray
    step: float
    traces: list
    view: dict = field(default_factory=dict)      # display position -> (scaled frame, frame-source text)
    shown: list = field(default_factory=list)
    source: dict = field(default_factory=dict)


def build_cases(ds: str, d: dict) -> list:
    """Recompute every read-out of fig_qualitative_cases.py from the stored predictions and assert agreement."""
    acc = ds == "accident"
    cfg = DS[ds]
    x = np.asarray(d["offsets"], float)
    step = float(x[1] - x[0])
    assert abs(x[0]) < 1e-12 and np.allclose(np.diff(x), step), f"{ds}: offsets are not a uniform grid from 0"
    xi = np.rint(x * cfg["scale"]).astype(int)
    assert np.allclose(xi / cfg["scale"], x, atol=1e-12) and int(xi[-1]) == cfg["horizon"]
    native = {int(k): int(v) for k, v in d.get("native_class_mapping", {}).items()}
    if acc:
        assert len(d["systems"]) == 2 and all("seed20260903" in s for s in d["systems"])
    else:
        assert str(d["seed"]) in cfg["key"]
    out = []
    for n, c in enumerate(d["selected"], 1):
        vid, meta = c["video_id"], c["metadata"]
        assert vid in fqc.NOTES and c["stratum"] in fqc.TITLES, f"{vid}: no note/title in fig_qualitative_cases.py"
        y = int(meta["class_code"])
        if acc:
            assert fqc.ACC_NAME[y] == meta["class_name"]
            ref = f"{fqc.ACC_NAME[y]} ({fqc.ACC_ABBR[y]})"
            assert len(c["traces"]) == 2
        else:
            assert native[y] == int(meta["native_class"])
            ref = f"native class {meta['native_class']}"
            assert len(c["traces"]) == 3
        traces = []
        for k, tr in enumerate(c["traces"]):
            pred = np.asarray(tr["pred"], int)
            p = np.asarray(tr["p_true"], float)
            assert len(pred) == len(p) == len(x), f"{vid}: trace length differs from the offset grid"
            good = pred == y
            assert good.tolist() == [bool(v) for v in tr["correct"]], f"{vid}: stored correctness differs"
            st = fqc.stable_suffix(good)
            assert st.tolist() == [bool(v) for v in tr["stable"]], f"{vid}: stored stable suffix differs"
            dval = fqc.contribution(st, step)
            lo, hi = cfg["ylim"]
            assert p.min() >= lo and p.max() <= hi, "probability outside displayed range"
            if acc:
                assert abs(dval - float(tr["delay"])) < 1e-12, f"{vid}: d_i differs from the stored delay"
                name, labels = "AB"[k], [fqc.ACC_ABBR[int(v)] for v in pred]
            else:
                q = int(tr["q"])
                assert [int(v) for v in tr["input_offsets"]] == list(range(0, cfg["horizon"] + 1, q)), \
                    f"{vid}: input offsets of q={q} are not the prefix grid"
                name, labels = f"q={q}", [str(native[int(v)]) for v in pred]
            onset = float(x[np.flatnonzero(st)[0]]) if st.any() else None
            traces.append(Trace(name, cfg["colours"][k], labels, p, good, st, onset, dval))
            print(f"[check] {cfg['fig']} ({chr(96 + n)}) {vid} {name}: end {'correct' if good[-1] else 'wrong'}, "
                  f"stable from {'none' if onset is None else f'{onset:g} ' + cfg['ushort']}, d_i {dval:g}"
                  + (f" (stored delay {float(tr['delay']):g})" if acc else ""))
        out.append(Case(ds, n, vid, c["stratum"], meta, ref, x, xi, step, traces))
    return out


# ----------------------------------------------------------------------------- source checks
def verify_accident_videos(root: Path, cases: list) -> dict:
    recs = {r["video_id"]: r for r in load_json("accident_videos.json")}
    out = {}
    for c in cases:
        r = recs[c.vid]
        assert r["dataset_relative_path"] == c.meta["path"], f"{c.vid}: video path differs between records"
        p = root / r["dataset_relative_path"]
        if not p.is_file():
            raise SystemExit(f"missing ACCIDENT video {p} (APE_ACCIDENT must be the dataset root)")
        size, h = p.stat().st_size, sha_file(p)
        ok = size == int(r["bytes"]) and h == r["sha256"]
        print(f"[check] {r['dataset_relative_path']}: SHA-256 {h[:16]}..., {size} bytes "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"{p}: SHA-256 {h} / {size} bytes differ from the recorded "
                             f"{r['sha256']} / {r['bytes']} bytes; use the original dataset file")
        out[c.vid] = {"dataset_relative_path": r["dataset_relative_path"], "sha256": h, "bytes": size}
    return out


def verify_mmau_frames(root: Path, cases: list) -> dict:
    by = {c.vid: c for c in cases}
    out, n = {}, 0
    for r in load_json("mmau_frames.json"):
        c = by.get(r["video_id"])
        if c is None:
            continue
        rel = r["dataset_relative_path"]
        num = int(c.meta["anchor_frame"]) + int(r["offset_frames"])
        assert rel == f"{c.meta['relative_image_directory']}/{num:06d}.jpg", f"{rel}: not anchor + offset"
        p = root / rel
        if not p.is_file():
            raise SystemExit(f"missing MM-AU frame {p} (APE_MMAU must be the dataset root)")
        h = sha_file(p)
        ok = h == r["sha256"]
        print(f"[check] {rel}: SHA-256 {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"{p}: SHA-256 {h} differs from the recorded {r['sha256']}")
        out[rel] = h
        n += 1
    assert n == 3 * len(cases), f"expected {3 * len(cases)} recorded MM-AU frames, found {n}"
    return out


def check_probe(rec: dict, pix: dict, fidx: int, pts: float, img, allow_drift: bool) -> dict:
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    meta_ok = fidx == int(rec["frame_index"]) and abs(pts - float(rec["decoded_pts_s"])) < 1e-6
    pix_ok = list(rgb.shape) == list(pix["shape_hwc"]) and sha_bytes(rgb.tobytes()) == pix["rgb_uint8_sha256"]
    print(f"[check] {rec['local']}: frame index {fidx} (recorded {rec['frame_index']}), pts {pts:.6f} s "
          f"(recorded {float(rec['decoded_pts_s']):.6f}), pixels {'identical to' if pix_ok else 'DIFFERENT from'} "
          f"the published frame")
    if not meta_ok:
        raise SystemExit(f"{rec['local']}: decoded frame index/timestamp differ from the published frame")
    if not pix_ok:
        msg = (f"{rec['local']}: decoded pixels differ from the published frame; the recorded frames are reproduced "
               f'by OpenCV 4.11.0 (pip install "opencv-python-headless==4.11.0.86")')
        if not allow_drift:
            raise SystemExit(msg + "; use --allow-decoder-drift to continue with this decoder")
        print("[warn] " + msg)
    return {"frame_index": int(fidx), "pts_s": round(float(pts), 6), "pixels_identical": bool(pix_ok)}


def decode_accident(root: Path, case: Case, positions: list, allow_drift: bool) -> None:
    """Sequential decoding; for each display time the last decoded frame at or before it (extract_qual_frames.py)."""
    try:
        import cv2
    except ImportError as e:
        raise SystemExit('OpenCV is needed to decode the ACCIDENT videos: '
                         'pip install "opencv-python-headless==4.11.0.86"') from e
    pixels = load_json("accident_frame_pixels.json")["frames"]
    anchor = float(case.meta["anchor_s"])
    probes = {}
    for r in load_json("accident_frames.json"):
        if r["video_id"] != case.vid:
            continue
        ms = int(round(float(r["offset_s"]) * 1000))
        assert abs(float(r["target_time_s"]) - (anchor + ms / 1000.0)) < 1e-9, f"{r['local']}: target time"
        probes[ms] = r
    pos = sorted(set(positions))
    assert len(probes) == 3 and set(probes) <= set(pos)
    targets = [anchor + ms / 1000.0 for ms in pos]
    path = root / case.meta["path"]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV cannot open {path}")
    fps, nfr = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scaled, chosen, probe_out, size = {}, {}, {}, {}
    state = {"last": None}

    def take(ms: int) -> None:
        last = state["last"]
        if last is None:
            raise SystemExit(f"{case.vid}: no frame decoded at or before {anchor + ms / 1000.0:.3f} s")
        fidx, pts, img = last
        if ms in probes:
            r = probes[ms]
            probe_out[r["local"]] = check_probe(r, pixels[r["local"]], fidx, pts, img, allow_drift)
        if fidx not in scaled:
            scaled[fidx] = fit_frame(np.ascontiguousarray(img[:, :, ::-1]))
            size["src"] = (img.shape[1], img.shape[0])
        chosen[ms] = (fidx, pts)

    idx, k, ended = -1, 0, False
    while k < len(pos):
        if not cap.grab():
            ended = True
            break
        idx += 1
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        while k < len(pos) and pts > targets[k] + 1e-9:   # these display times end before this frame
            take(pos[k])
            k += 1
        if k < len(pos):
            okr, img = cap.retrieve()
            if not okr:
                ended = True
                break
            state["last"] = (idx, pts, img)
    if ended and k < len(pos):
        print(f"[warn] {case.vid}: video ended before {targets[-1]:.3f} s; the last decoded frame is held")
    while k < len(pos):
        take(pos[k])
        k += 1
    cap.release()
    w0, h0 = size["src"]
    for ms in pos:
        fidx, pts = chosen[ms]
        ph = scaled[fidx]
        case.view[ms] = (ph, f"video frame {fidx}{DOT}pts {pts:.3f} s{DOT}{w0}\u00d7{h0} \u2192 "
                             f"{ph.shape[1]}\u00d7{ph.shape[0]} px")
        case.shown.append([ms / 1000.0, int(fidx), round(float(pts), 6)])
    case.source.update(opencv=cv2.__version__, fps_reported=round(fps, 4), frames_reported=nfr,
                       size_px=[w0, h0], shown_as_px=list(scaled[chosen[pos[0]][0]].shape[1::-1]),
                       distinct_frames_shown=len(scaled), published_frames=probe_out,
                       shown_columns=["display_offset_s", "frame_index", "pts_s"])
    print(f"[info] {case.vid}: OpenCV {cv2.__version__}, {fps:.4f} fps, {nfr} frames, {w0}x{h0}; "
          f"{len(pos)} display times, {len(scaled)} distinct decoded frames")


def load_mmau(root: Path, case: Case, positions: list, published: dict) -> None:
    meta = case.meta
    a = int(meta["anchor_frame"])
    folder = root / meta["relative_image_directory"]
    if not folder.is_dir():
        raise SystemExit(f"missing MM-AU image folder {folder}")
    src = None
    for off in sorted(set(positions)):
        num = a + off
        if not int(meta["first_frame"]) <= num <= int(meta["last_frame"]):
            raise SystemExit(f"{case.vid}: frame {num} is outside {meta['first_frame']}..{meta['last_frame']}")
        p = folder / f"{num:06d}.jpg"
        if not p.is_file():
            raise SystemExit(f"missing MM-AU frame {p}")
        raw = p.read_bytes()
        h = sha_bytes(raw)
        rel = f"{meta['relative_image_directory']}/{p.name}"
        if rel in published:
            assert h == published[rel], f"{rel}: bytes changed since the hash check"
        with Image.open(io.BytesIO(raw)) as im:
            rgb = np.array(im.convert("RGB"))
        src = (rgb.shape[1], rgb.shape[0])
        ph = fit_frame(rgb)
        case.view[off] = (ph, f"image {p.name}{DOT}{src[0]}\u00d7{src[1]} \u2192 {ph.shape[1]}\u00d7{ph.shape[0]} px")
        case.shown.append([int(off), p.name, h])
    case.source.update(relative_image_directory=meta["relative_image_directory"], size_px=list(src),
                       published_frames_checked=sorted(r for r in published if r.startswith(
                           meta["relative_image_directory"] + "/")),
                       shown_columns=["display_frame_offset", "file", "sha256"])
    print(f"[info] {case.vid}: {len(case.view)} image files read (offsets {min(case.view)}..{max(case.view)})")


def verify_full_sources(acc_d: dict, mm_d: dict) -> dict:
    if not APE_DATA_GIVEN:
        raise SystemExit("--verify-sources reads the MM-AU prediction arrays below APE_DATA; set APE_DATA to the "
                         "derived-data package root (the folder containing accident/ and mmau/)")
    src = os.environ.get("APE_QUAL_SOURCES")
    if not src:
        raise SystemExit("--verify-sources needs APE_QUAL_SOURCES (folder with the ACCIDENT answer-set CSVs; "
                         "see fig_qualitative_cases.py)")
    if not (fqc.MI / "cohort.jsonl").is_file():
        raise SystemExit(f"MM-AU prediction arrays not found below {fqc.MI}; APE_DATA must hold mmau/deliverables/")
    compact = fqc.E
    try:
        fqc.E = Path(src)
        fqc.verify_accident(acc_d)
        fqc.verify_mmau(mm_d)
    finally:
        fqc.E = compact
    print(f"[check] full sources: {json.dumps(fqc.checks, default=str)}")
    return json.loads(json.dumps(fqc.checks, default=str))


# ----------------------------------------------------------------------------- rendering
class Layout:
    def __init__(self, n_lane: int):
        self.n = n_lane
        self.lane_end = LANE_TOP + n_lane * LANE_H
        self.legend_y = self.lane_end + 46
        self.key_y = self.legend_y + 44
        self.source_y = self.key_y + 34
        h = self.source_y + 20
        self.H = h + (h % 2)


def caption(case: Case, phase: str, pos: int) -> str:
    cfg = DS[case.ds]
    acc = case.ds == "accident"
    if phase == "final":
        return f"After the window, offline: stable-correct suffix\n(underlines), endpoint, onset and $d_i$, as in {cfg['fig']}."
    if phase == "end":
        return (f"Window end ({cfg['hlabel']}): all {len(case.xi)} reads shown.\n"
                "The offline stable-correct suffix follows.")
    if pos < 0:
        return ("Before the anchor: no read yet.\n"
                + ("Playback starts 1 s before the anchor." if acc else "Playback starts 8 frames before the anchor."))
    if acc:
        return "One read every 0.5 s on the causal prefix; a tile\nappears when the video reaches its offset."
    return "One read every 8 frames, input every q frames;\nthe read at offset \u03b4 uses frames up to anchor + \u03b4."


def read_label(case: Case, j: int) -> str:
    if case.ds == "accident":
        return mm.fmt_signed(case.x[j], 1) + " s"
    return signed_int(int(case.xi[j])) + " f"


class Renderer:
    def __init__(self, n_lane: int):
        self.L = Layout(n_lane)
        self.fig = plt.figure(figsize=(WPX / DPI, self.L.H / DPI), dpi=DPI, facecolor="white")

    def axes(self, x, top, w, h):
        H = self.L.H
        return self.fig.add_axes([x / WPX, 1 - (top + h) / H, w / WPX, h / H])

    def text(self, x, top, s, **kw):
        return self.fig.text(x / WPX, 1 - top / self.L.H, s, **kw)

    def point(self, x, top):
        return x / WPX, 1 - top / self.L.H

    def close(self) -> None:
        plt.close(self.fig)

    def draw(self, case: Case, phase: str, pos: int) -> None:
        fig, L, cfg = self.fig, self.L, DS[case.ds]
        fig.clf()
        acc = case.ds == "accident"
        final = phase == "final"
        xl = cfg["xlim"]
        t = pos / cfg["scale"]
        nread = int((case.xi <= pos).sum())   # reads whose prefix ends at or before the display position
        tw = case.step * 0.86
        tag = chr(96 + case.n)

        # info column: case header, clock, shown frame, phase, latest reads or the paper's read-outs, note
        self.text(INFO_X, 16, f"({tag})", fontsize=13, fontweight="bold", color=INK, va="top")
        self.text(INFO_X + 30, 16, fqc.TITLES[case.stratum], fontsize=13, fontweight="bold", color=INK, va="top",
                  linespacing=1.1)
        self.text(INFO_X + 30, 60, f"{case.vid}{DOT}{cfg['where']}{DOT}{cfg['fig']} ({tag})", fontsize=9,
                  color=MUTED, va="top")
        self.text(INFO_X + 30, 78, "reference: " + case.ref_txt, fontsize=10.5, color=INK, va="top")
        clock = f"t {MINUS} anchor = {mm.fmt_signed(t, 2)} s" if acc else f"frame offset = {signed_int(pos)}"
        self.text(INFO_X, 102, clock, fontsize=16, fontweight="bold", color=INK, va="top")
        self.text(INFO_X, 130, case.view[pos][1], fontsize=8.5, color=MUTED, va="top")
        self.text(INFO_X, 148, caption(case, phase, pos), fontsize=9.5, color=MUTED, va="top", linespacing=1.2)
        if final:
            hdr = [(620, "end", "center"), (760, "stable from", "right"), (860, f"$d_i$ ({cfg['ushort']})", "right")]
        else:
            hdr = [(600, "read at", "left"), (712, "answer", "center"), (770, "p(reference class)", "left")]
        for hx, s, ha in hdr:
            self.text(hx, 190, s, fontsize=9, color=MUTED, va="center", ha=ha)
        for k, tr in enumerate(case.traces):
            yy = 210 + 20 * k
            self.text(INFO_X, yy, tr.name, fontsize=10.5, fontweight="bold", color=tr.colour, va="center")
            if final:
                fx, fy = self.point(620, yy)
                fig.add_artist(Line2D([fx], [fy], ls="", marker="o", ms=7, mfc=tr.colour if tr.good[-1] else "white",
                                      mec=tr.colour, mew=1.2, transform=fig.transFigure))
                on_txt = "none" if tr.onset is None else f"{tr.onset:g} {cfg['ushort']}"
                self.text(760, yy, on_txt, fontsize=10.5, color=INK, va="center", ha="right")
                self.text(860, yy, f"{tr.d:g}", fontsize=10.5, color=INK, va="center", ha="right")
            elif nread == 0:
                for hx, _, ha in hdr:
                    self.text(hx, yy, "\u2013", fontsize=10.5, color=MUTED, va="center", ha=ha)
            else:
                j = nread - 1
                g = bool(tr.good[j])
                self.text(600, yy, read_label(case, j), fontsize=10.5, color=INK, va="center")
                self.text(712, yy, tr.labels[j], fontsize=10, color=INK if g else WRONG_TXT, va="center", ha="center",
                          bbox=dict(boxstyle="square,pad=0.22", facecolor=OK_FILL if g else BAD_FILL,
                                    edgecolor=OK_EDGE if g else BAD_EDGE, linewidth=0.8))
                self.text(770, yy, f"{tr.p[j]:.3f}", fontsize=10.5, color=tr.colour, va="center")
        note = fqc.NOTES[case.vid] if final else \
            "Onset, $d_i$ and the stable-correct suffix are known\nonly after the window end (offline)."
        self.text(INFO_X, 270, note, fontsize=9.5, color=MUTED, va="top", style="italic" if final else "normal",
                  linespacing=1.15)

        # probability panel: p(reference class), point by point
        ax = self.axes(PLOT_X0, PROB_TOP, PLOT_W, PROB_H)
        ax.set_xlim(*xl)
        ax.set_ylim(*cfg["ylim"])
        ax.axvspan(xl[0], 0, color=PRE_SHADE, lw=0, zorder=0)
        ax.text(xl[0] + 0.012 * (xl[1] - xl[0]), cfg["ylim"][1], "before\nanchor", fontsize=7.5, color=MUTED,
                va="top", ha="left", linespacing=1.0)
        for yt in cfg["yticks"][1:]:
            ax.axhline(yt, color=GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=GUIDE, lw=0.8, zorder=0)
        for tr in case.traces:
            if nread:
                xs, p, g = case.x[:nread], tr.p[:nread], tr.good[:nread]
                ax.plot(xs, p, color=tr.colour, lw=1.5, zorder=2, clip_on=False)
                ax.scatter(xs[~g], p[~g], s=22, facecolors="white", edgecolors=tr.colour, linewidths=1.0, zorder=3,
                           clip_on=False)
                ax.scatter(xs[g], p[g], s=24, facecolors=tr.colour, edgecolors="white", linewidths=0.4, zorder=4,
                           clip_on=False)
        if not final:
            ax.axvline(t, color=CURSOR, lw=1.1, zorder=5)
        ax.set_yticks(cfg["yticks"])
        ax.set_yticklabels([f"{v:g}" for v in cfg["yticks"]], fontsize=9)
        ax.set_xticks([])
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.tick_params(axis="y", length=3, pad=2, width=0.8)
        self.text(PLOT_X0 - 32, PROB_TOP + PROB_H / 2, "p(reference\nclass)", fontsize=9, ha="right", va="center",
                  color=MUTED, linespacing=1.05)

        # tile lanes: predicted class per read; the offline suffix only in the final phase
        la = self.axes(PLOT_X0, LANE_TOP, PLOT_W, L.n * LANE_H)
        la.set_xlim(*xl)
        la.set_ylim(L.n, 0)
        la.axis("off")
        if not final:
            la.axvline(t, color=CURSOR, lw=1.1, zorder=0)
        for k, tr in enumerate(case.traces):
            yc = k + 0.40
            for i, xx in enumerate(case.x):
                if i < nread:
                    g = bool(tr.good[i])
                    la.add_patch(Rectangle((xx - tw / 2, yc - 0.28), tw, 0.56, facecolor=OK_FILL if g else BAD_FILL,
                                           edgecolor=OK_EDGE if g else BAD_EDGE, lw=0.8, zorder=2))
                    la.text(xx, yc + 0.02, tr.labels[i], ha="center", va="center", fontsize=9,
                            color=INK if g else WRONG_TXT, zorder=3)
                else:
                    la.add_patch(Rectangle((xx - tw / 2, yc - 0.28), tw, 0.56, facecolor="white", edgecolor=HIDDEN,
                                           lw=0.8, ls=(0, (2, 2)), zorder=2))
            if final:
                yb = k + 0.84
                for xx, st in zip(case.x, tr.stable):
                    la.add_patch(Rectangle((xx - tw / 2, yb - (0.08 if st else 0.055)), tw, 0.16 if st else 0.11,
                                           facecolor=tr.colour if st else TRACK, edgecolor="none", zorder=2))
            self.text(MARGIN, LANE_TOP + yc * LANE_H, tr.name, fontsize=11, fontweight="bold", color=tr.colour,
                      va="center")

        # offset axis
        bx = self.axes(PLOT_X0, L.lane_end, PLOT_W, 0.001)
        bx.set_xlim(*xl)
        for s in ("top", "right", "left"):
            bx.spines[s].set_visible(False)
        bx.spines["bottom"].set_linewidth(0.8)
        bx.spines["bottom"].set_bounds(case.x[0], case.x[-1])
        bx.set_yticks([])
        bx.set_xticks(cfg["ticks"])
        bx.tick_params(axis="x", length=3, width=0.8, pad=2, labelsize=9)
        bx.set_xlabel(cfg["xlabel"], fontsize=10, labelpad=3)

        # legend (paper semantics + unread tile), system key, burned-in source line
        handles = [Line2D([], [], marker="o", ls="", ms=6, mfc=KEY_INK, mec=KEY_INK, label="correct class"),
                   Line2D([], [], marker="o", ls="", ms=6, mfc="white", mec=KEY_INK, label="wrong class"),
                   Rectangle((0, 0), 1, 1, facecolor=OK_FILL, edgecolor=OK_EDGE, lw=0.8,
                             label="tile = predicted class: correct"),
                   Rectangle((0, 0), 1, 1, facecolor=BAD_FILL, edgecolor=BAD_EDGE, lw=0.8, label="wrong"),
                   Line2D([], [], color=KEY_INK, lw=3.5, solid_capstyle="butt",
                          label="tile in stable-correct suffix (offline)"),
                   Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=HIDDEN, lw=0.8, ls=(0, (2, 2)),
                             label="not read yet")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=self.point(MARGIN, L.legend_y), ncol=3,
                   frameon=False, fontsize=9, handlelength=1.4, handletextpad=0.5, columnspacing=1.6,
                   borderaxespad=0)
        self.text(MARGIN, L.key_y, cfg["key"], fontsize=8.8, color=MUTED, va="top", linespacing=1.3)
        self.text(MARGIN, L.source_y, cfg["source"], fontsize=8.5, color=MUTED, va="top")


def grab(fig, H: int) -> np.ndarray:
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    out = np.full((H, WPX, 3), 255, np.uint8)
    h, w = min(H, a.shape[0]), min(WPX, a.shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def compose(R: Renderer, case: Case, phase: str, pos: int) -> np.ndarray:
    """Draw the page, snap page anti-aliasing to white, then paste the scaled real frame pixel-exactly."""
    R.draw(case, phase, pos)
    f = grab(R.fig, R.L.H)
    f[(f >= SNAP_MIN).all(axis=2)] = 255
    x0, y0, w, h = BOX
    f[y0:y0 + h, x0:x0 + w] = LETTERBOX
    photo = case.view[pos][0]
    ph, pw = photo.shape[:2]
    ox, oy = x0 + (w - pw) // 2, y0 + (h - ph) // 2
    f[oy:oy + ph, ox:ox + pw] = photo
    return f


def timeline(case: Case, step: int) -> list:
    """(phase, display position, duration in ms): playback, window-end hold, offline final hold."""
    last = int(case.xi[-1])
    if case.ds == "accident":
        pos, dur = list(range(-ACC_PRE_MS, last + 1, step)), step
    else:
        pos, dur = list(range(-MM_PRE, last + 1, step)), step * MM_MS_PER_FRAME
    assert pos[-1] == last and set(case.xi.tolist()) <= set(pos), "every read offset must be a display position"
    seq = [("play", p, dur) for p in pos]
    return seq + [("end", last, END_HOLD_MS), ("final", last, FINAL_HOLD_MS)]


def step_label(ds: str, step: int) -> str:
    if ds == "accident":
        return f"{step / 1000:g} s of video per display frame ({1000 / step:g} per second, real time)"
    return f"every {step} image file(s), {step * MM_MS_PER_FRAME} ms each"


def all_positions(case: Case) -> list:
    return sorted({p for s in (MP4_STEP[case.ds],) + GIF_STEPS[case.ds] for _, p, _ in timeline(case, s)})


# ----------------------------------------------------------------------------- writers
def gif_palette(frames: list) -> np.ndarray:
    """Reserved exact colours (white first), then median-cut colours of the page and of the video panel."""
    pal = list(RESERVED_RGB)
    x0, y0, w, h = BOX
    picks = sorted({int(round(v)) for v in np.linspace(0, len(frames) - 1, min(8, len(frames)))})
    photo = np.concatenate([frames[i][y0:y0 + h, x0:x0 + w] for i in picks], axis=0)
    page = frames[-1].copy()
    page[y0:y0 + h, x0:x0 + w] = 255
    for sheet, n in ((page, GIF_UI_COLORS), (photo, GIF_COLORS - len(pal) - GIF_UI_COLORS)):
        q = Image.fromarray(np.ascontiguousarray(sheet)).quantize(colors=n, method=Image.Quantize.MEDIANCUT,
                                                                  dither=Image.Dither.NONE)
        flat = q.getpalette() or []
        for i in np.unique(np.asarray(q)):
            i = int(i)
            rgb = tuple(int(v) for v in flat[3 * i: 3 * i + 3])
            if len(rgb) == 3 and rgb not in pal and min(rgb) < SNAP_MIN and len(pal) < GIF_COLORS:
                pal.append(rgb)
    return np.asarray(pal, np.int32)


def write_gif(frames: list, durs: list, path: Path) -> dict:
    P = gif_palette(frames)
    flat = P.astype(np.uint8).ravel().tolist()
    ims = []
    for f in frames:
        im = Image.fromarray(mm.map_to_palette(f, P))
        im.putpalette(flat)
        ims.append(im)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=list(durs), loop=0, optimize=False, disposal=1)
    mm.check_gif_background(path, frames[0])
    return {"palette_colours": int(len(P)), "bytes": int(path.stat().st_size)}


class Mp4Stream:
    """H.264 yuv420p at 10 fps from raw RGB frames; each frame is repeated for its duration."""

    def __init__(self, path: Path, w: int, h: int, exe: str):
        self.path, self.broken, self.n = path, False, 0
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
               "-framerate", str(1000 // TICK_MS), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "slow",
               "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-map_metadata", "-1", str(path)]
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err)

    def add(self, frame: np.ndarray, ms: int) -> None:
        assert ms % TICK_MS == 0, ms
        if self.broken:
            return
        data = np.ascontiguousarray(frame).tobytes()
        try:
            for _ in range(ms // TICK_MS):
                self.proc.stdin.write(data)
                self.n += 1
        except OSError:
            self.broken = True

    def close(self) -> str:
        try:
            self.proc.stdin.close()
        except OSError:
            self.broken = True
        rc = self.proc.wait()
        self.err.seek(0)
        msg = self.err.read().decode("utf-8", "replace").strip()
        self.err.close()
        if rc != 0 or self.broken:
            if self.path.exists():
                self.path.unlink()
            return f"FAILED (ffmpeg exit code {rc}): {msg[-400:]}"
        return "written"


def write_record(out: Path, entry: dict) -> None:
    p = out / "qual_media_record.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    for k in ("files", "cases", "checks"):
        merged = dict(old.get(k, {}))
        merged.update(entry.get(k, {}))
        entry[k] = dict(sorted(merged.items()))
    p.write_text(json.dumps(entry, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[media] {p.name}: written")


# ----------------------------------------------------------------------------- main
def main() -> None:
    keys = [f"{ds}_{n}" for ds in DS for n in (1, 2, 3)]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <repository>/docs/media)")
    ap.add_argument("--no-mp4", action="store_true", help="write GIF files only")
    ap.add_argument("--cases", nargs="+", choices=list(DS) + keys, default=list(DS),
                    help="datasets or single cases (paper order); the MP4 of a dataset needs all three of its cases")
    ap.add_argument("--verify-sources", action="store_true",
                    help="also repeat the exact CSV/NPZ checks of fig_qualitative_cases.py --verify-sources")
    ap.add_argument("--allow-decoder-drift", action="store_true",
                    help="report, instead of stopping, when decoded ACCIDENT pixels differ from the published frames")
    args = ap.parse_args()
    want = set()
    for c in args.cases:
        want |= {f"{c}_{n}" for n in (1, 2, 3)} if c in DS else {c}
    out = (args.out or (mm.ROOT / "docs" / "media")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    mp4 = (not args.no_mp4) and ffmpeg is not None
    print(f"[info] output = {out}; MP4 = {'yes' if mp4 else 'no'}")

    # 1. stored predictions: the read-out checks of fig_qualitative_cases.py
    acc_d, mm_d = load_json(DS["accident"]["cases"]), load_json(DS["mmau"]["cases"])
    fqc.check_notes(acc_d, mm_d)
    print("[check] case notes of fig_qualitative_cases.py verified against the stored arrays")
    cases = {"accident": build_cases("accident", acc_d), "mmau": build_cases("mmau", mm_d)}
    full = verify_full_sources(acc_d, mm_d) if args.verify_sources else \
        "not rerun; use --verify-sources to repeat the exact CSV/NPZ checks"
    datasets = [ds for ds in DS if any(f"{ds}_{c.n}" in want for c in cases[ds])]
    roots = {ds: env_root(*DS[ds]["env"]) for ds in datasets}

    # 2. sources: video hashes and recorded MM-AU frames, all before any frame is read or rendered
    checks = {"case_notes_and_readouts": "identical to the stored arrays", "full_sources": full}
    if "accident" in datasets:
        checks["accident_videos"] = verify_accident_videos(roots["accident"], cases["accident"])
    published_mm = {}
    if "mmau" in datasets:
        published_mm = verify_mmau_frames(roots["mmau"], cases["mmau"])
        checks["mmau_published_frames_sha256"] = "identical (9 of 9)"
    sel = {ds: [c for c in cases[ds] if f"{ds}_{c.n}" in want] for ds in datasets}
    for ds in datasets:
        for c in sel[ds]:
            if ds == "accident":
                decode_accident(roots[ds], c, all_positions(c), args.allow_decoder_drift)
                c.source.update(checks["accident_videos"][c.vid])
            else:
                load_mmau(roots[ds], c, all_positions(c), published_mm)

    # 3. render: one GIF per case, one MP4 per dataset
    record = {"selection_rule": {"accident": acc_d["selection_rule"], "mmau": mm_d["selection_rule"]},
              "files": {}, "cases": {}, "checks": checks}
    for ds in datasets:
        R = Renderer(len(sel[ds][0].traces))
        stream, mp4_name = None, f"qual_{ds}_replay.mp4"
        if mp4 and len(sel[ds]) == len(cases[ds]):
            stream = Mp4Stream(out / mp4_name, WPX, R.L.H, ffmpeg)
        elif not mp4:
            print(f"[media] {mp4_name}: not written (--no-mp4 or ffmpeg missing)")
        else:
            print(f"[media] {mp4_name}: not written (needs all three {ds} cases; --cases selected a subset)")
        for c in sel[ds]:
            cache = {}

            def render(phase: str, pos: int, case: Case = c, store: bool = True) -> np.ndarray:
                key = (phase, pos)
                if key in cache:
                    return cache[key]
                f = compose(R, case, phase, pos)
                if store:
                    cache[key] = f
                return f

            gif = out / f"qual_{ds}_{c.n}.gif"
            steps = GIF_STEPS[ds]
            for step in steps:
                seq = timeline(c, step)
                frames = [render(ph, p) for ph, p, _ in seq]
                durs = [ms for _, _, ms in seq]
                info = write_gif(frames, durs, gif)
                size = info["bytes"]
                print(f"[media] {gif.name}: {len(frames)} frames, {WPX}x{R.L.H} px, {sum(durs) / 1000:.2f} s per loop, "
                      f"{step_label(ds, step)}, palette {info['palette_colours']} colours, {size / 1e6:.2f} MB")
                if size <= GIF_LIMIT:
                    break
                if step != steps[-1]:
                    print(f"[media] {gif.name}: above {GIF_LIMIT / 1e6:g} MB; rewritten with the coarser display step")
                else:
                    print(f"[media] {gif.name}: WARNING: above {GIF_LIMIT / 1e6:g} MB at the coarsest display step")
            record["files"][gif.name] = {"case": c.vid, "panel": f"{DS[ds]['fig']} ({chr(96 + c.n)})",
                                         "display_step": step_label(ds, step), "stored_frames": len(frames),
                                         "loop_s": sum(durs) / 1000, "size_px": [WPX, R.L.H], **info}
            if stream is not None:
                for ph, p, ms in timeline(c, MP4_STEP[ds]):
                    stream.add(render(ph, p, store=False), ms)
            cache.clear()
            record["cases"][c.vid] = {
                "dataset": ds, "panel": f"{DS[ds]['fig']} ({chr(96 + c.n)})", "stratum": c.stratum,
                "reference": c.ref_txt, "note": fqc.NOTES[c.vid].replace("\n", " "),
                "readouts": [{"system": tr.name, "endpoint_correct": bool(tr.good[-1]), "stable_onset": tr.onset,
                              "d_i": tr.d} for tr in c.traces],
                "source": c.source, "shown": c.shown}
            c.view.clear()
        if stream is not None:
            status = stream.close()
            extra = ""
            if (out / mp4_name).exists():
                s = (out / mp4_name).stat().st_size
                extra = f", {s / 1e6:.2f} MB" + (f"  WARNING: above {MP4_LIMIT / 1e6:g} MB" if s > MP4_LIMIT else "")
                record["files"][mp4_name] = {"cases": [c.vid for c in sel[ds]], "fps": 1000 // TICK_MS,
                                             "display_step": step_label(ds, MP4_STEP[ds]),
                                             "frames": stream.n, "size_px": [WPX, R.L.H], "bytes": int(s)}
            print(f"[media] {mp4_name}: {status}{extra}")
        R.close()
    write_record(out, record)
    print("[done] all checks passed")


if __name__ == "__main__":
    main()
