"""Produce CONTRACT 5.4 answer matrices on the finest grid.

Grid: ``end_s = anchor_s + j * 0.25`` for j = -11 .. 88, that is anchor - 2.75 s
to anchor + 22 s. The grid extends on both sides of the anchor because the A
track scans anchor perturbations by shifting j; a matrix that started at j = 0
would put every ``eps_sys < 0`` scan point outside the cache. Rows with j < 0
carry the system's raw Top-1 and probabilities -- masking ``delta < 0`` to the
bot code is the protocol layer's job, not the system's.

Every prefix is formed by ``t_s <= end_s`` over the cached 4 fps features, so
both system families are strictly causal functions of the prefix:

  * whole-clip head -- mean of the features up to ``end_s``, then the linear head;
  * prefix GRU     -- the causal hidden state at the last feature step <= ``end_s``.

Because the answer at grid point j depends only on the prefix, evaluating a
coarser grid by keeping the columns with ``j mod k == 0`` returns exactly what a
rerun at that coarser step would return (CONTRACT 5.4, "sub-sampling equals a
rerun").  A clip whose first cached frame lies after ``end_s`` -- which can only
happen at negative j on a clip whose anchor is near its head -- has an empty
prefix, cannot answer, and is written as -1.

Usage:
    python -X utf8 systems/infer_answers.py --ckpt outputs/checkpoints/<id>.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402
from systems import devsel  # noqa: E402
from systems.train_prefix import CausalGRUClassifier  # noqa: E402


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def grid_end_times(anchor_s: float, n_j: int, delta_s: float, j0: int = 0) -> np.ndarray:
    return anchor_s + (np.arange(j0, j0 + n_j, dtype=np.float64) * delta_s)


def clip_logits_on_grid(feat: np.ndarray, t_s: np.ndarray, end_times: np.ndarray,
                        mu: np.ndarray, sd: np.ndarray, w: np.ndarray, b: np.ndarray):
    """Causal prefix mean-pool + linear head at every grid end time."""
    csum = np.cumsum(feat.astype(np.float64), axis=0)
    # number of frames with t_s <= end_s
    counts = np.searchsorted(t_s, end_times + 1e-9, side="right")
    logits = np.full((len(end_times), C.N_CLASSES), np.nan, dtype=np.float64)
    ok = counts > 0
    if ok.any():
        pooled = csum[counts[ok] - 1] / counts[ok][:, None]
        pooled = (pooled - mu) / sd
        logits[ok] = pooled @ w.T + b
    return logits, ok


def prefix_logits_on_grid(step_logits: np.ndarray, t_s: np.ndarray, end_times: np.ndarray):
    """Pick the causal hidden-state output at the last feature step <= end_s."""
    counts = np.searchsorted(t_s, end_times + 1e-9, side="right")
    logits = np.full((len(end_times), C.N_CLASSES), np.nan, dtype=np.float64)
    ok = counts > 0
    if ok.any():
        logits[ok] = step_logits[counts[ok] - 1]
    return logits, ok


def run(ckpt_path: str, manifest: Optional[str], features_dir: str, system_id: Optional[str],
        delta_s: float, j_min: int, j_max: int, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kind = ck["kind"]
    sid = system_id or ck["system_id"]
    mu = np.asarray(ck["feat_mean"], dtype=np.float64)
    sd = np.asarray(ck["feat_std"], dtype=np.float64)

    man = C.load_manifest(manifest)
    have = set(C.available_feature_ids(features_dir))
    man = man[man["video_id"].isin(have)].sort_values("video_id", kind="stable").reset_index(drop=True)

    js = np.arange(j_min, j_max + 1, dtype=np.int64)
    n_j = len(js)
    model = None
    if kind == "prefix_causal_gru":
        hp = ck["hparams"]
        model = CausalGRUClassifier(512, hp["hidden"], hp.get("layers", 1)).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
    elif kind == "clip_mean_linear":
        w = ck["state_dict"]["weight"].numpy().astype(np.float64)
        b = ck["state_dict"]["bias"].numpy().astype(np.float64)
    else:
        raise ValueError(f"unknown checkpoint kind: {kind}")

    vids: List[str] = []
    preds = np.zeros((len(man), n_j), dtype=np.int64)
    probs = np.zeros((len(man), n_j, C.N_CLASSES), dtype=np.float64)
    n_no_frame = 0

    t0 = time.time()
    for i, r in enumerate(man.itertuples(index=False)):
        cf = C.load_features(r.video_id, features_dir)
        vids.append(r.video_id)
        if cf is None or cf.feat.shape[0] == 0:
            preds[i, :] = C.BOT
            probs[i, :, :] = np.nan
            continue
        t_s = cf.t_s.astype(np.float64)
        feat = cf.feat.astype(np.float32)
        anchor = float(r.anchor_s)
        ends = anchor + js.astype(np.float64) * delta_s

        if kind == "clip_mean_linear":
            logits, ok = clip_logits_on_grid(feat, t_s, ends, mu, sd, w, b)
        else:
            keep = t_s <= ends.max() + 1e-6
            t_k, f_k = t_s[keep], feat[keep]
            if f_k.shape[0] == 0:
                logits = np.full((n_j, C.N_CLASSES), np.nan)
                ok = np.zeros(n_j, bool)
            else:
                x = (f_k.astype(np.float64) - mu) / sd
                xt = torch.from_numpy(x.astype(np.float32))[None].to(device)
                with torch.no_grad():
                    step_logits, _ = model(xt)
                logits, ok = prefix_logits_on_grid(
                    step_logits[0].cpu().numpy().astype(np.float64), t_k, ends)

        p = np.full_like(logits, np.nan)
        p[ok] = softmax_np(logits[ok])
        yhat = np.full(n_j, C.BOT, dtype=np.int64)
        yhat[ok] = p[ok].argmax(axis=1)
        # Rows with j < 0 keep the raw Top-1: masking delta < 0 to bot is the
        # protocol layer's job. A row is bot only when the prefix contains no
        # frame at all, which happens when end_s falls before the first frame.
        n_no_frame += int((~ok).sum())
        preds[i] = yhat
        probs[i] = p
        if (i + 1) % 400 == 0:
            print(f"[{i + 1}/{len(man)}] {time.time() - t0:.1f}s", flush=True)

    wall = time.time() - t0
    answers = C.build_answer_frame(vids, preds, probs, delta_s=delta_s, j_offset=int(js[0]))
    return sid, kind, ck, man, js, answers, n_no_frame, wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--features-dir", default=C.FEATURES_DIR)
    ap.add_argument("--system-id", default=None)
    ap.add_argument("--delta-s", type=float, default=C.FINEST_DELTA_S)
    ap.add_argument("--j-min", type=int, default=C.J_MIN)
    ap.add_argument("--j-max", type=int, default=C.J_MAX)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sid, kind, ck, man, js, answers, n_no_frame, wall = run(
        args.ckpt, args.manifest, args.features_dir, args.system_id,
        args.delta_s, args.j_min, args.j_max, device)

    family = "clip" if kind == "clip_mean_linear" else "prefix"
    backbone = ("resnet18-imagenet(4fps) + prefix mean pool + linear"
                if family == "clip"
                else f"resnet18-imagenet(4fps) + causal GRU h={ck['hparams']['hidden']} + linear")
    card = {
        "system_id": sid,
        "family": family,
        "description": ("whole-clip classifier evaluated causally on each prefix"
                        if family == "clip"
                        else "plain multi-prefix classifier, independent per-prefix cross-entropy, "
                             "no class supervision before the anchor, no cross-prefix consistency term"),
        "backbone": backbone,
        "trained_on_split": "train",
        "dev_tuned_params": {"selection": "epoch chosen on split=dev",
                             "best_epoch": ck["train_info"].get("best_epoch"),
                             **{k: v for k, v in ck["hparams"].items()}},
        "train_data_unknown": False,
        "cost_note": (f"one forward pass over cached 4 fps ResNet-18 features; "
                      f"answer matrix built in {wall:.1f}s on 1 GPU"),
        "causal": True,
        "parent_system_id": None,
        "seed": ck.get("seed"),
        "grid": {"delta_s": args.delta_s, "j_min": int(js[0]), "j_max": int(js[-1]),
                 "n_j": int(len(js)),
                 "end_s_range": [float(js[0] * args.delta_s), float(js[-1] * args.delta_s)]},
        "pre_anchor_rows": "raw Top-1 and probabilities; the protocol layer masks delta < 0",
        "bot_rows": "only when the prefix contains no frame at all (end_s before the first frame)",
        "subsampling_equivalent": True,
        "subsampling_note": ("answers depend on the prefix only, via a cached causal feature "
                             "stream, so keeping the columns with j mod k == 0 reproduces a "
                             "rerun at k*delta_s"),
    }
    out = C.write_answers(sid, answers, card)
    print(f"wrote {out}  rows={len(answers)}  j={js[0]}..{js[-1]}  "
          f"no-frame cells={n_no_frame}  wall={wall:.1f}s")

    # smoke numbers using the local dev-selection proxy, post-anchor columns only
    lab = C.labels_map(man)
    vids, js_out, preds, _ = C.answers_to_arrays(answers)
    post = C.post_anchor_slice(js_out)
    idx = {v: k for k, v in enumerate(vids)}
    for split in ("dev", "test"):
        sub = man[man["split"] == split]
        if len(sub) == 0:
            continue
        keep = [v for v in sub["video_id"] if v in idx]
        rows = [idx[v] for v in keep]
        y = np.array([lab[v] for v in keep])
        pl = sub.set_index("video_id").loc[keep, "post_anchor_length_s"].to_numpy()
        summ = devsel.summarize(preds[rows][:, post], y, pl)
        print(f"--- {sid} [{split}] ---")
        print(devsel.fmt(summ, prefix="  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
