"""Whole-clip classifier: mean-pooled ResNet-18 features + a linear head.

An ordinary exam-taker, not a contribution.  Trained on ``split=train`` only;
epoch selection and every hyper-parameter come from ``split=dev``; ``split=test``
is never looked at here.

Usage:
    python -X utf8 systems/train_clip.py --system-id clip__r18mean__seed20260903
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402
from systems.devsel import macro_acc  # noqa: E402


def pooled_matrix(man, features_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Whole-clip mean-pooled features for every manifest row with a cache entry."""
    xs, ys, vids = [], [], []
    for r in man.itertuples(index=False):
        cf = C.load_features(r.video_id, features_dir)
        if cf is None or cf.feat.shape[0] == 0:
            continue
        xs.append(cf.feat.mean(axis=0))
        ys.append(int(r.class_code))
        vids.append(r.video_id)
    if not xs:
        return np.zeros((0, 512), np.float32), np.zeros((0,), np.int64), []
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), vids


def train(x_tr, y_tr, x_dv, y_dv, epochs: int, lr: float, weight_decay: float,
          seed: int, device: torch.device, patience: int) -> Tuple[nn.Module, Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = nn.Linear(x_tr.shape[1], C.N_CLASSES).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()

    xt = torch.from_numpy(x_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device)
    xd = torch.from_numpy(x_dv).to(device)

    best = {"epoch": -1, "dev_macro_acc": -1.0, "state": None}
    hist = []
    since = 0
    for ep in range(1, epochs + 1):
        head.train()
        opt.zero_grad()
        loss = lossf(head(xt), yt)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            dv_pred = head(xd).argmax(dim=1).cpu().numpy()
        m = macro_acc(dv_pred, y_dv)
        hist.append({"epoch": ep, "loss": float(loss.item()), "dev_macro_acc": m})
        if m > best["dev_macro_acc"] + 1e-9:
            best = {"epoch": ep, "dev_macro_acc": m,
                    "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
            since = 0
        else:
            since += 1
            if since >= patience:
                break
    head.load_state_dict(best["state"])
    return head, {"best_epoch": best["epoch"], "best_dev_macro_acc": best["dev_macro_acc"],
                  "epochs_run": len(hist), "history_tail": hist[-5:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--features-dir", default=C.FEATURES_DIR)
    ap.add_argument("--system-id", default="clip__r18mean__seed20260903")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    man = C.load_manifest(args.manifest)
    have = set(C.available_feature_ids(args.features_dir))
    man = man[man["video_id"].isin(have)].reset_index(drop=True)
    tr, dv = C.split_frame(man, "train"), C.split_frame(man, "dev")
    print(f"train clips={len(tr)}  dev clips={len(dv)}", flush=True)

    x_tr, y_tr, _ = pooled_matrix(tr, args.features_dir)
    x_dv, y_dv, _ = pooled_matrix(dv, args.features_dir)
    if len(x_tr) == 0 or len(x_dv) == 0:
        raise SystemExit("no cached features for train or dev; run systems/features.py first")

    mu = x_tr.mean(axis=0, keepdims=True)
    sd = x_tr.std(axis=0, keepdims=True) + 1e-6
    x_tr_n = (x_tr - mu) / sd
    x_dv_n = (x_dv - mu) / sd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    head, info = train(x_tr_n, y_tr, x_dv_n, y_dv, args.epochs, args.lr,
                       args.weight_decay, args.seed, device, args.patience)
    info["wall_seconds"] = time.time() - t0
    info["train_class_counts"] = np.bincount(y_tr, minlength=C.N_CLASSES).tolist()
    info["dev_class_counts"] = np.bincount(y_dv, minlength=C.N_CLASSES).tolist()
    print("TRAIN_INFO " + json.dumps(info), flush=True)

    os.makedirs(C.CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(C.CKPT_DIR, f"{args.system_id}.pt")
    torch.save({
        "kind": "clip_mean_linear",
        "system_id": args.system_id,
        "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
        "feat_mean": mu, "feat_std": sd,
        "seed": args.seed,
        "hparams": {"lr": args.lr, "weight_decay": args.weight_decay,
                    "epochs_max": args.epochs, "patience": args.patience},
        "train_info": info,
    }, ckpt)
    print(f"saved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
