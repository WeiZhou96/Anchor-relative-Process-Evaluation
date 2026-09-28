"""Rebuild the 18 dataset frames shown in the two qualitative figures from your own dataset copies.

The frames are third-party dataset content and are not redistributed with this package. This helper
re-creates them from a local copy of each dataset and checks every result against the SHA-256 values
recorded when the figures were made (qualitative/*_frames.json).

  ACCIDENT (9 PNG files): decodes <accident-root>/real_videos/<video_id>.mp4 sequentially with OpenCV and
      keeps the last decoded frame whose timestamp is at or before the recorded target time
      (anchor + offset); the frame is written unmodified (no crop, no enhancement) with cv2.imwrite
      defaults. The recorded frame index and timestamp are cross-checked, the video file hash is
      checked when available, and the decoded pixels are compared with a pixel hash that does not depend
      on the PNG encoder (qualitative/accident_frame_pixels.json).
  MM-AU (9 JPG files): byte copies of <mmau-root>/<dataset_relative_path> (e.g. extracted/CAP-DATA/...).

Usage:
  python extract_qual_frames.py --accident-root /path/to/ACCIDENT --mmau-root /path/to/MM-AU --out FRAMES_DIR
  python extract_qual_frames.py ... --dry-run      # decode and verify in memory, write nothing
Then:
  APE_QUAL_FRAMES=FRAMES_DIR APE_DATA=... python fig_qualitative_cases.py

Exit status is non-zero if any frame is missing or its pixels differ. A file-hash mismatch with matching
pixels (possible with a different OpenCV/zlib build) is reported as PIXELS-OK-FILE-DIFFERS;
fig_qualitative_cases.py checks file hashes and will then refuse the file.

Decoder version matters: the recorded ACCIDENT frames are reproduced bit for bit by the OpenCV 4.11.0
wheels (opencv-python-headless / opencv-contrib-python 4.11.0.86, bundled FFmpeg avcodec 59.37.100).
Newer wheels (4.13.0, 5.0.0; newer FFmpeg) return the same frame index and timestamp but decode pixels
that differ by a few grey levels (up to 10 of 255), so the hash checks fail. Use
  pip install "opencv-python-headless==4.11.0.86"
MM-AU frames are plain file copies and do not depend on the decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q = HERE / "qualitative"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def accident(root: Path, out: Path | None, report: list) -> bool:
    import cv2
    import numpy as np

    frames = json.loads((Q / "accident_frames.json").read_text(encoding="utf-8"))
    videos = {v["video_id"]: v for v in json.loads((Q / "accident_videos.json").read_text(encoding="utf-8"))}
    pixels = json.loads((Q / "accident_frame_pixels.json").read_text(encoding="utf-8"))["frames"]
    ok_all = True
    by_video: dict[str, list] = {}
    for r in frames:
        by_video.setdefault(r["video_id"], []).append(r)
    for vid, recs in by_video.items():
        vpath = root / videos[vid]["dataset_relative_path"]
        if not vpath.is_file():
            report.append(f"MISSING video {vpath}")
            ok_all = False
            continue
        vhash = sha_file(vpath)
        if vhash != videos[vid]["sha256"]:
            report.append(f"WARNING video hash differs for {vid} (your copy may be a different encode)")
        targets = sorted(recs, key=lambda r: r["target_time_s"])
        cap = cv2.VideoCapture(str(vpath))
        idx, last = -1, None
        chosen: dict[str, tuple] = {}
        k = 0
        while k < len(targets):
            ok = cap.grab()
            if not ok:
                break
            idx += 1
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # all targets that end before this frame take the previous frame
            while k < len(targets) and pts > targets[k]["target_time_s"] + 1e-9:
                chosen[targets[k]["local"]] = last
                k += 1
            if k < len(targets):
                okr, img = cap.retrieve()
                if not okr:
                    break
                last = (idx, pts, img)
        while k < len(targets):  # video ended before the target: keep the last decoded frame
            chosen[targets[k]["local"]] = last
            k += 1
        cap.release()
        for r in recs:
            got = chosen.get(r["local"])
            if got is None:
                report.append(f"FAIL {r['local']}: no frame decoded before {r['target_time_s']} s")
                ok_all = False
                continue
            fidx, pts, img = got
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            pix_ok = (list(rgb.shape) == pixels[r["local"]]["shape_hwc"]
                      and sha_bytes(rgb.tobytes()) == pixels[r["local"]]["rgb_uint8_sha256"])
            okc, enc = cv2.imencode(".png", img)
            file_ok = okc and sha_bytes(enc.tobytes()) == r["sha256"]
            meta_ok = fidx == r["frame_index"] and abs(pts - r["decoded_pts_s"]) < 1e-6
            status = "OK" if (pix_ok and file_ok) else ("PIXELS-OK-FILE-DIFFERS" if pix_ok else "FAIL")
            report.append(f"{status} {r['local']} frame_index={fidx} (recorded {r['frame_index']}) "
                          f"pts={pts:.6f} (recorded {r['decoded_pts_s']:.6f}){'' if meta_ok else ' [index/pts differ]'}")
            ok_all &= pix_ok
            if out is not None and okc:
                (out / r["local"]).write_bytes(enc.tobytes())
    return ok_all


def mmau(root: Path, out: Path | None, report: list) -> bool:
    frames = json.loads((Q / "mmau_frames.json").read_text(encoding="utf-8"))
    ok_all = True
    for r in frames:
        src = root / r["dataset_relative_path"]
        if not src.is_file():
            report.append(f"MISSING {src}")
            ok_all = False
            continue
        good = sha_file(src) == r["sha256"]
        report.append(f"{'OK' if good else 'FAIL'} {r['local']} <- {r['dataset_relative_path']}")
        ok_all &= good
        if out is not None and good:
            shutil.copyfile(src, out / r["local"])
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--accident-root", type=Path, help="ACCIDENT dataset root (contains real_videos/)")
    ap.add_argument("--mmau-root", type=Path, help="MM-AU dataset root (contains extracted/CAP-DATA/)")
    ap.add_argument("--out", type=Path, help="output folder for the 18 frames (use it as APE_QUAL_FRAMES)")
    ap.add_argument("--dry-run", action="store_true", help="verify only; write nothing")
    a = ap.parse_args()
    if not a.dry_run and a.out is None:
        ap.error("--out is required unless --dry-run is given")
    out = None if a.dry_run else a.out
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    report: list[str] = []
    ok = True
    if a.accident_root:
        ok &= accident(a.accident_root, out, report)
    if a.mmau_root:
        ok &= mmau(a.mmau_root, out, report)
    if not (a.accident_root or a.mmau_root):
        ap.error("give --accident-root and/or --mmau-root")
    print("\n".join(report))
    print("ALL PIXELS/BYTES VERIFIED" if ok else "SOME FRAMES FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
