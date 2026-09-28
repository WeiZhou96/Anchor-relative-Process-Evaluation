"""Frame feature cache for the system library.

Videos in ACCIDENT run at 3.9 to 49.7 fps, so frames are picked by *timestamp*
on a fixed 4 fps schedule rather than by frame stride: the first decoded frame
whose timestamp reaches each 0.25 s tick is kept, and its true timestamp is
stored.  Downstream every prefix is formed by ``t_s <= end_s``, which makes the
cache strictly causal and makes coarse-grid sub-sampling identical to a rerun.

Backbone: ImageNet-pretrained torchvision ResNet-18, penultimate 512-d output.
If the weights cannot be fetched the run falls back to a randomly initialised
backbone; that is recorded in the cache meta and must be reported in the cards.

Usage:
    python -X utf8 systems/features.py --limit 50 --splits dev
    python -X utf8 systems/features.py            # everything in the manifest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE_SHORT = 256
CROP = 224


def _preprocess(frame_bgr) -> np.ndarray:
    """BGR HWC uint8 -> RGB CHW uint8, short side 256 then centre crop 224."""
    import cv2

    h, w = frame_bgr.shape[:2]
    if h < w:
        new_h, new_w = RESIZE_SHORT, max(CROP, int(round(w * RESIZE_SHORT / h)))
    else:
        new_w, new_h = RESIZE_SHORT, max(CROP, int(round(h * RESIZE_SHORT / w)))
    img = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (new_h - CROP) // 2
    left = (new_w - CROP) // 2
    img = img[top:top + CROP, left:left + CROP]
    img = img[:, :, ::-1]  # BGR -> RGB
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def decode_clip(abs_path: str, fallback_fps: float, target_fps: float,
                max_seconds: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (frames uint8 [T,3,224,224], t_s float32 [T], fps_used)."""
    import cv2

    cv2.setNumThreads(0)
    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        cap.release()
        return np.zeros((0, 3, CROP, CROP), np.uint8), np.zeros((0,), np.float32), float(fallback_fps)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (fps and 1.0 <= fps <= 240.0):
        fps = float(fallback_fps)

    step = 1.0 / float(target_fps)
    next_tick = 0.0
    idx = 0
    frames: List[np.ndarray] = []
    times: List[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if max_seconds is not None and t > max_seconds:
            break
        if t + 1e-9 >= next_tick:
            frames.append(_preprocess(frame))
            times.append(t)
            while next_tick <= t + 1e-9:
                next_tick += step
        idx += 1
    cap.release()
    if not frames:
        return np.zeros((0, 3, CROP, CROP), np.uint8), np.zeros((0,), np.float32), float(fps)
    return np.stack(frames), np.asarray(times, dtype=np.float32), float(fps)


class ClipDataset(Dataset):
    def __init__(self, rows, data_root: str, target_fps: float, max_seconds: Optional[float]):
        self.rows = rows
        self.data_root = data_root
        self.target_fps = target_fps
        self.max_seconds = max_seconds

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        vid, rel, fps = self.rows[i]
        abs_path = os.path.join(self.data_root, rel)
        frames, times, fps_used = decode_clip(abs_path, fps, self.target_fps, self.max_seconds)
        return vid, torch.from_numpy(frames), torch.from_numpy(times), fps_used


def _collate(batch):
    return batch[0]


def build_backbone(device: torch.device) -> Tuple[nn.Module, bool, str]:
    """ResNet-18 trunk (512-d). Returns (model, pretrained_ok, note)."""
    import torchvision

    pretrained_ok = True
    note = "torchvision ResNet18_Weights.IMAGENET1K_V1"
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=weights)
    except Exception as exc:  # noqa: BLE001 - offline / download failure
        pretrained_ok = False
        note = f"RANDOM INIT fallback ({type(exc).__name__}: {exc})"
        net = torchvision.models.resnet18(weights=None)
    net.fc = nn.Identity()
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net, pretrained_ok, note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--data-root", default=C.DATA_ROOT)
    ap.add_argument("--out-dir", default=C.FEATURES_DIR)
    ap.add_argument("--splits", default="", help="comma separated; empty = all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target-fps", type=float, default=C.FEATURE_FPS)
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="decode cutoff per clip in seconds; 0 = whole clip")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu-batch", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    man = C.load_manifest(args.manifest)
    if args.splits:
        keep = [s.strip() for s in args.splits.split(",") if s.strip()]
        man = man[man["split"].isin(keep)]
    man = man[man["decode_ok"]]
    man = man.sort_values("video_id", kind="stable").reset_index(drop=True)
    if args.limit:
        man = man.head(args.limit)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for r in man.itertuples(index=False):
        if not args.overwrite and os.path.exists(C.feature_path(r.video_id, args.out_dir)):
            continue
        rows.append((r.video_id, r.path, float(r.fps)))
    print(f"manifest rows selected: {len(man)}; to extract: {len(rows)}", flush=True)
    if not rows:
        print("nothing to do")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, pretrained_ok, note = build_backbone(device)
    print(f"backbone: {note}  pretrained_ok={pretrained_ok}  device={device}", flush=True)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    max_seconds = args.max_seconds if args.max_seconds > 0 else None
    ds = ClipDataset(rows, args.data_root, args.target_fps, max_seconds)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                    collate_fn=_collate, prefetch_factor=2 if args.workers else None)

    t0 = time.time()
    n_done = 0
    n_empty = 0
    n_frames_total = 0
    for vid, frames, times, fps_used in dl:
        if frames.numel() == 0:
            n_empty += 1
            print(f"EMPTY {vid}", flush=True)
            continue
        feats = []
        with torch.no_grad():
            for s in range(0, frames.shape[0], args.gpu_batch):
                chunk = frames[s:s + args.gpu_batch].to(device, non_blocking=True).float().div_(255.0)
                chunk = (chunk - mean) / std
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out = net(chunk)
                feats.append(out.float().cpu())
        feat = torch.cat(feats).numpy().astype(np.float16)
        np.savez(C.feature_path(vid, args.out_dir),
                 t_s=times.numpy().astype(np.float32),
                 feat=feat,
                 fps_used=np.float32(fps_used),
                 target_fps=np.float32(args.target_fps),
                 pretrained=np.bool_(pretrained_ok))
        n_done += 1
        n_frames_total += feat.shape[0]
        if n_done % 25 == 0 or n_done == len(rows):
            el = time.time() - t0
            print(f"[{n_done}/{len(rows)}] {el:.1f}s  {n_done / max(el, 1e-9):.2f} clip/s  "
                  f"{n_frames_total / max(el, 1e-9):.1f} feat-frame/s", flush=True)

    meta = {
        "backbone": "resnet18",
        "backbone_dim": 512,
        "pretrained": bool(pretrained_ok),
        "pretrained_note": note,
        "target_fps": args.target_fps,
        "resize_short": RESIZE_SHORT,
        "crop": CROP,
        "max_seconds": max_seconds,
        "n_clips_written": n_done,
        "n_clips_empty": n_empty,
        "n_frames_total": n_frames_total,
        "wall_seconds": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, "_features_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("META " + json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
