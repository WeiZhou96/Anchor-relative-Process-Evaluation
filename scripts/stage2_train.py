"""Fixed-budget source-group OOF development pilot; never fit or score test rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ape.method_analysis import point_weights, trajectory_components  # noqa: E402
from systems.stage2_learning import ARMS, build_model, causal_ema, objective, state_hash  # noqa: E402

SEEDS = [20260930, 20260931, 20260932, 20260933, 20260934]
LRS = [0.0003, 0.001]
EMA_ALPHAS = [0.5, 0.25, 0.1, 0.05]


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, out: Path) -> tuple[pd.DataFrame, list[np.ndarray], np.ndarray, np.ndarray]:
    manifest_path = source / "data/manifest/manifest_real.csv"
    all_rows = pd.read_csv(manifest_path)
    rows = all_rows[all_rows.split.isin(["train", "dev"])].sort_values("video_id").reset_index(drop=True)
    test = all_rows[all_rows.split.eq("test")]
    assert len(rows) == 513
    assert not (set(rows.video_id) & set(test.video_id))
    assert not (set(rows.source_cluster_id) & set(test.source_cluster_id))
    backend = json.loads((source / "outputs/r2_S/feature_backend.json").read_text())
    feature_root = Path(backend["features_dir"])
    if not feature_root.is_absolute():  # release: recorded path is relative to the source root
        feature_root = source / feature_root
    assert backend["backend"] == "clipb16"
    sequences, provenance = [], []
    indices = np.zeros((len(rows), 41), dtype=np.int64)
    weights = np.zeros((len(rows), 41), dtype=np.float32)
    for i, row in enumerate(rows.itertuples()):
        path = feature_root / (row.video_id + ".npz")
        with np.load(path) as data:
            times = data["t_s"].astype(np.float64)
            features = data["feat"].astype(np.float32)
        assert np.isfinite(features).all() and np.all(np.diff(times) > 0)
        start = max(0, float(row.anchor_s) - 4)
        begin = max(0, int(np.searchsorted(times, start + 1e-7, side="right")) - 1)
        end = int(np.searchsorted(times, float(row.anchor_s) + 10 + 1e-7, side="right"))
        t, x = times[begin:end], features[begin:end]
        n = min(41, int(np.floor((float(row.post_anchor_length_s) + 1e-7) / 0.25)) + 1)
        assert n >= 1
        targets = float(row.anchor_s) + np.arange(n) * 0.25
        local = np.searchsorted(t, targets + 1e-7, side="right") - 1
        assert local.min() >= 0 and np.all(t[local] <= targets + 1e-7)
        indices[i, :n] = local
        w = np.ones(n, dtype=np.float32)
        if n > 1:
            w[[0, -1]] = 0.5
        weights[i, :n] = w / w.sum()
        sequences.append(x)
        provenance.append(
            {
                "video_id": row.video_id,
                "file": str(path),
                "sha256": file_hash(path),
                "input_steps": len(t),
                "readout_steps": n,
                "actual_start_s": float(t[0]),
                "max_staleness_s": float(np.max(targets - t[local])),
            }
        )
    rows["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260929)
    for fold, (train, val) in enumerate(splitter.split(rows, rows.class_code, rows.source_cluster_id)):
        assert not (set(rows.iloc[train].source_cluster_id) & set(rows.iloc[val].source_cluster_id))
        assert set(rows.iloc[train].class_code) == set(range(5))
        rows.loc[val, "fold"] = fold
    assert (rows.fold >= 0).all()
    rows.to_csv(out / "development_folds.csv", index=False)
    write_json(
        out / "feature_provenance.json",
        {
            "manifest_sha256": file_hash(manifest_path),
            "backend": backend,
            "test_ids_used": 0,
            "development": provenance,
        },
    )
    return rows, sequences, indices, weights


def evaluate(prob: np.ndarray, rows: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for horizon in [10, 4]:
        mask = rows.post_anchor_length_s.to_numpy() >= horizon - 1e-9
        if not mask.any():
            continue
        y = rows.class_code.to_numpy()[mask]
        p = prob[mask, : int(horizon / 0.25) + 1]
        assert np.isfinite(p).all()
        c = p.argmax(-1) == y[:, None]
        comp = trajectory_components(c, np.arange(c.shape[1]) * 0.25)
        for weighting in ["micro", "macro"]:
            if weighting == "macro" and len(np.unique(y)) != 5:
                continue
            w = point_weights(y, weighting)
            results.append(
                {
                    "horizon": horizon,
                    "weighting": weighting,
                    "n": len(y),
                    **{key: float(w @ v) for key, v in comp.items()},
                }
            )
    return results


def read_probabilities(
    model: torch.nn.Module, x: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor, ema: float | None = None
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        probability = model(x).softmax(-1)
        if ema is not None:
            probability = causal_ema(probability, alpha=ema)
        out = probability.gather(1, indices[:, :, None].expand(-1, -1, 5)).cpu().numpy()
    out[~valid.cpu().numpy()] = np.nan
    return out


def run(source: Path, out: Path, epochs: int, smoke: bool) -> None:
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    (out / "fits").mkdir()
    (out / "oof").mkdir()
    repository = Path(__file__).resolve().parents[1]
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True)
    if git_status.strip():
        raise RuntimeError("Refusing a training run with an uncommitted working tree")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    state = {
        "started_unix": time.time(),
        "code_head": code,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "epochs": epochs,
        "git_status": git_status,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "smoke": smoke,
        "claim": "exploratory OOF on reused development data; not independent test",
        "fits_completed": 0,
        "plan_hashes": {
            name: file_hash(repository / name) for name in ["STAGE2_SCOPE_20260927.md", "STAGE2_TRAINING_20260927.md"]
        },
    }
    write_json(out / "run_state.json", state)
    rows, sequences, indices, weights = prepare(source, out)
    lengths = np.array([len(x) for x in sequences])
    labels = torch.tensor(rows.class_code.to_numpy(), device="cuda", dtype=torch.long)
    index_t = torch.tensor(indices, device="cuda")
    weight_t = torch.tensor(weights, device="cuda")
    seeds = SEEDS[:1] if smoke else SEEDS
    lrs = LRS[:1] if smoke else LRS
    folds = [0] if smoke else list(range(5))
    recipes = ARMS + [f"gru_ema{alpha:g}" for alpha in EMA_ALPHAS]
    oof = {
        (seed, lr, arm): np.full((len(rows), 41, 5), np.nan, np.float32)
        for seed in seeds
        for lr in lrs
        for arm in recipes
    }
    fits = []
    for fold in folds:
        train = np.flatnonzero(rows.fold.to_numpy() != fold)
        val = np.flatnonzero(rows.fold.to_numpy() == fold)
        features = np.concatenate([sequences[i] for i in train])
        mu, sd = features.mean(0), features.std(0) + 1e-6
        del features
        norm_hash = hashlib.sha256(mu.tobytes() + sd.tobytes()).hexdigest()
        padded = np.zeros((len(rows), lengths.max(), 512), dtype=np.float32)
        for i, sequence in enumerate(sequences):
            padded[i, : len(sequence)] = (sequence - mu) / sd
        x = torch.tensor(padded, device="cuda")
        del padded
        counts = np.bincount(rows.iloc[train].class_code, minlength=5)
        class_weights = torch.tensor(len(train) / (5 * counts), device="cuda", dtype=torch.float32)
        write_json(
            out / f"normalization_fold{fold}.json",
            {
                "hash": norm_hash,
                "train_n": len(train),
                "class_counts": counts.tolist(),
                "train_ids": rows.iloc[train].video_id.tolist(),
            },
        )
        for seed in seeds:
            for lr in lrs:
                common_hash = None
                common_batch = None
                for arm in ARMS:
                    started = time.time()
                    fit_id = f"f{fold}_s{seed}_lr{lr:g}_{arm}"
                    torch.manual_seed(seed + fold * 100)
                    torch.cuda.manual_seed_all(seed + fold * 100)
                    model = build_model(arm).cuda()
                    initial_hash = state_hash(model)
                    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                    rng = np.random.default_rng(seed + fold * 100)
                    history = []
                    first_batch = []
                    clipping_count = 0
                    gradient_steps = 0
                    largest_gradient_norm = 0.0
                    for epoch in range(1, epochs + 1):
                        model.train()
                        order = rng.permutation(train)
                        if epoch == 1:
                            first_batch = rows.iloc[order[:64]].video_id.tolist()
                        total = 0.0
                        for start in range(0, len(order), 64):
                            ids = order[start : start + 64]
                            n_steps = int(lengths[ids].max())
                            logits = model(x[ids, :n_steps])
                            readout = logits.gather(1, index_t[ids, :, None].expand(-1, -1, 5))
                            loss = objective(readout, labels[ids], weight_t[ids], class_weights, arm)
                            if not torch.isfinite(loss):
                                raise ValueError(f"Nonfinite loss: {fit_id}, epoch {epoch}")
                            opt.zero_grad(set_to_none=True)
                            loss.backward()
                            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5))
                            if not np.isfinite(gradient_norm):
                                raise ValueError(f"Nonfinite gradient: {fit_id}")
                            clipping_count += int(gradient_norm > 5)
                            gradient_steps += 1
                            largest_gradient_norm = max(largest_gradient_norm, gradient_norm)
                            opt.step()
                            total += float(loss.detach()) * len(ids)
                        record: dict[str, Any] = {"epoch": epoch, "train_loss": total / len(train)}
                        if epoch % 10 == 0 or epoch == epochs:
                            probabilities = read_probabilities(model, x[val], index_t[val], weight_t[val] > 0)
                            record["heldout_fold_metrics"] = evaluate(probabilities, rows.iloc[val])
                        history.append(record)
                    probabilities = read_probabilities(model, x[val], index_t[val], weight_t[val] > 0)
                    oof[seed, lr, arm][val] = probabilities
                    if arm == "gru_ce":
                        for alpha in EMA_ALPHAS:
                            oof[seed, lr, f"gru_ema{alpha:g}"][val] = read_probabilities(
                                model, x[val], index_t[val], weight_t[val] > 0, ema=alpha
                            )
                    if arm.startswith("gru_"):
                        if common_hash is None:
                            common_hash, common_batch = initial_hash, first_batch
                        assert initial_hash == common_hash and first_batch == common_batch
                    checkpoint = {
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "mean": mu,
                        "std": sd,
                        "arm": arm,
                        "fold": fold,
                        "seed": seed,
                        "lr": lr,
                        "epochs": epochs,
                        "code": code,
                    }
                    torch.save(checkpoint, out / "checkpoints" / f"{fit_id}.pt")
                    metadata = {
                        "fit_id": fit_id,
                        "arm": arm,
                        "fold": fold,
                        "seed": seed,
                        "lr": lr,
                        "epochs": epochs,
                        "initial_hash": initial_hash,
                        "first_batch": first_batch,
                        "normalization_hash": norm_hash,
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "seconds": time.time() - started,
                        "history": history,
                        "final_hash": state_hash(model),
                        "gradient_clip_fraction": clipping_count / gradient_steps,
                        "max_gradient_norm": largest_gradient_norm,
                    }
                    if epochs >= 20:
                        previous = float(np.mean([r["train_loss"] for r in history[-20:-10]]))
                        recent = float(np.mean([r["train_loss"] for r in history[-10:]]))
                        metadata["last_window_loss_relative_change"] = (recent - previous) / max(previous, 1e-12)
                        metadata["training_loss_plateau_heuristic"] = (
                            abs(metadata["last_window_loss_relative_change"]) <= 0.01
                        )
                    write_json(out / "fits" / f"{fit_id}.json", metadata)
                    fits.append({k: v for k, v in metadata.items() if k not in ["history", "first_batch"]})
                    state["fits_completed"] = len(fits)
                    state["last_fit"] = fit_id
                    write_json(out / "run_state.json", state)
                    print("FIT", len(fits), fit_id, round(metadata["seconds"], 2), flush=True)
                    del model, opt
        del x
    summaries = []
    for (seed, lr, arm), probabilities in oof.items():
        selected = np.flatnonzero(rows.fold.isin(folds).to_numpy())
        np.savez_compressed(
            out / "oof" / f"s{seed}_lr{lr:g}_{arm}.npz",
            video_id=rows.iloc[selected].video_id.to_numpy(dtype=str),
            probabilities=probabilities[selected],
        )
        for metric in evaluate(probabilities[selected], rows.iloc[selected]):
            summaries.append({"seed": seed, "lr": lr, "arm": arm, **metric})
    pd.DataFrame(summaries).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(fits).to_csv(out / "fits.csv", index=False)
    provenance = json.loads((out / "feature_provenance.json").read_text())
    assert all(file_hash(Path(r["file"])) == r["sha256"] for r in provenance["development"])
    assert file_hash(source / "data/manifest/manifest_real.csv") == provenance["manifest_sha256"]
    state.update(
        complete=True,
        finished_unix=time.time(),
        feature_hashes_rechecked=len(sequences),
        original_manifest_unchanged=True,
    )
    write_json(out / "run_state.json", state)
    print("COMPLETE", json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    run(options.source_root, options.out, options.epochs, options.smoke)
