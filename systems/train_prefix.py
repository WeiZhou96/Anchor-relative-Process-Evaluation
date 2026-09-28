"""Plain multi-prefix classifier: causal GRU over 4 fps features, 5-way logits per step.

Deliberately ordinary.  Each prefix gets its own cross-entropy term and nothing
else: no cross-prefix consistency, smoothing, monotonicity or stopping term of
any kind, so the system carries no idea of its own into the audit.  Steps before
the anchor receive no class supervision (they are the protocol's masked region);
they are still fed to the recurrence because the prefix starts at the clip head.

Usage:
    python -X utf8 systems/train_prefix.py --hidden 512 --system-id prefix__gru512__seed20260903
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402
from systems.devsel import macro_acc  # noqa: E402

MAX_STEPS = 520  # 130 s at 4 fps; guards against pathological clips


class CausalGRUClassifier(nn.Module):
    """Unidirectional GRU + per-step linear head. Output at step t depends only on steps <= t."""

    def __init__(self, in_dim: int = 512, hidden: int = 512, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=False, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, C.N_CLASSES)

    def forward(self, x, h0=None):
        out, hn = self.gru(x, h0)
        return self.head(out), hn


class SeqStore:
    """Feature sequences held in RAM as float16, standardised on the fly."""

    def __init__(self, man, features_dir: str, mu: Optional[np.ndarray] = None,
                 sd: Optional[np.ndarray] = None, cap_seconds: Optional[float] = None):
        self.items: List[Tuple[str, np.ndarray, np.ndarray, float, int]] = []
        for r in man.itertuples(index=False):
            cf = C.load_features(r.video_id, features_dir)
            if cf is None or cf.feat.shape[0] == 0:
                continue
            t_s, feat = cf.t_s, cf.feat
            if cap_seconds is not None:
                keep = t_s <= float(r.anchor_s) + cap_seconds + 1e-6
                t_s, feat = t_s[keep], feat[keep]
            if t_s.shape[0] == 0:
                continue
            t_s, feat = t_s[:MAX_STEPS], feat[:MAX_STEPS]
            self.items.append((r.video_id, t_s.astype(np.float32), feat.astype(np.float16),
                               float(r.anchor_s), int(r.class_code)))
        self.mu, self.sd = mu, sd

    def __len__(self):
        return len(self.items)

    def frame_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        allf = np.concatenate([it[2].astype(np.float32) for it in self.items], axis=0)
        return allf.mean(axis=0, keepdims=True), allf.std(axis=0, keepdims=True) + 1e-6

    def batch(self, idxs, device):
        chunk = [self.items[i] for i in idxs]
        lens = [c[2].shape[0] for c in chunk]
        tmax = max(lens)
        b = len(chunk)
        x = np.zeros((b, tmax, chunk[0][2].shape[1]), np.float32)
        sup = np.zeros((b, tmax), bool)   # post-anchor steps: supervised
        valid = np.zeros((b, tmax), bool)
        y = np.zeros((b,), np.int64)
        for k, (_vid, t_s, feat, anchor, cls) in enumerate(chunk):
            n = feat.shape[0]
            f = feat.astype(np.float32)
            if self.mu is not None:
                f = (f - self.mu) / self.sd
            x[k, :n] = f
            valid[k, :n] = True
            sup[k, :n] = t_s >= anchor - 1e-6
            y[k] = cls
        return (torch.from_numpy(x).to(device), torch.from_numpy(sup).to(device),
                torch.from_numpy(valid).to(device), torch.from_numpy(y).to(device))


def evaluate(model, store: SeqStore, device, batch_size: int = 32) -> Dict[str, float]:
    """Pooled macro accuracy over every supervised (clip, post-anchor step) pair."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for s in range(0, len(store), batch_size):
            idxs = list(range(s, min(s + batch_size, len(store))))
            x, sup, _valid, y = store.batch(idxs, device)
            logits, _ = model(x)
            p = logits.argmax(dim=-1)
            m = sup
            preds.append(p[m].cpu().numpy())
            labels.append(y[:, None].expand_as(p)[m].cpu().numpy())
    if not preds:
        return {"pooled_macro_acc": float("nan"), "pooled_acc": float("nan")}
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return {"pooled_macro_acc": macro_acc(preds, labels),
            "pooled_acc": float((preds == labels).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--features-dir", default=C.FEATURES_DIR)
    ap.add_argument("--system-id", default="prefix__gru512__seed20260903")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--train-cap-seconds", type=float, default=C.GRID_MAX_S,
                    help="drop training steps beyond anchor + this many seconds; defaults to "
                         "the answer grid's reach so no prefix is answered outside the "
                         "supervised range")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    man = C.load_manifest(args.manifest)
    have = set(C.available_feature_ids(args.features_dir))
    man = man[man["video_id"].isin(have)].reset_index(drop=True)
    tr_man, dv_man = C.split_frame(man, "train"), C.split_frame(man, "dev")

    tr = SeqStore(tr_man, args.features_dir, cap_seconds=args.train_cap_seconds)
    if len(tr) == 0:
        raise SystemExit("no cached features for train; run systems/features.py first")
    mu, sd = tr.frame_stats()
    tr.mu, tr.sd = mu, sd
    dv = SeqStore(dv_man, args.features_dir, mu=mu, sd=sd, cap_seconds=args.train_cap_seconds)
    print(f"train clips={len(tr)}  dev clips={len(dv)}  hidden={args.hidden}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalGRUClassifier(512, args.hidden, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.CrossEntropyLoss(reduction="none")

    rng = np.random.default_rng(args.seed)
    best = {"epoch": -1, "dev": -1.0, "state": None}
    hist, since = [], 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        for s in range(0, len(order), args.batch_size):
            idxs = order[s:s + args.batch_size].tolist()
            x, sup, _valid, y = tr.batch(idxs, device)
            logits, _ = model(x)
            b, t, k = logits.shape
            # independent cross-entropy per prefix; no term couples adjacent prefixes
            per_step = lossf(logits.reshape(b * t, k), y[:, None].expand(b, t).reshape(b * t))
            per_step = per_step.reshape(b, t) * sup.float()
            denom = sup.float().sum().clamp_min(1.0)
            loss = per_step.sum() / denom
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item())
            nb += 1
        ev = evaluate(model, dv, device, args.batch_size)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), **ev})
        print(f"ep {ep:3d} loss {tot / max(nb, 1):.4f} dev_pooled_macro_acc {ev['pooled_macro_acc']:.4f}",
              flush=True)
        if ev["pooled_macro_acc"] > best["dev"] + 1e-9:
            best = {"epoch": ep, "dev": ev["pooled_macro_acc"],
                    "state": {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}}
            since = 0
        else:
            since += 1
            if since >= args.patience:
                print(f"early stop at epoch {ep} (patience {args.patience})", flush=True)
                break

    model.load_state_dict(best["state"])
    info = {"best_epoch": best["epoch"], "best_dev_pooled_macro_acc": best["dev"],
            "epochs_run": len(hist), "wall_seconds": time.time() - t0,
            "history_tail": hist[-5:]}
    print("TRAIN_INFO " + json.dumps(info), flush=True)

    os.makedirs(C.CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(C.CKPT_DIR, f"{args.system_id}.pt")
    torch.save({
        "kind": "prefix_causal_gru",
        "system_id": args.system_id,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "feat_mean": mu, "feat_std": sd,
        "seed": args.seed,
        "hparams": {"hidden": args.hidden, "layers": args.layers, "lr": args.lr,
                    "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                    "epochs_max": args.epochs, "patience": args.patience,
                    "train_cap_seconds": args.train_cap_seconds},
        "train_info": info,
    }, ckpt)
    print(f"saved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
