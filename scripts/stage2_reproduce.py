"""Independently repeat the prespecified first GRU-CE fit with the same fixed recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stage2_train import prepare, write_json  # noqa: E402
from systems.stage2_learning import build_model, objective, state_hash  # noqa: E402


def run(source: Path, reference: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rows, sequences, indices, weights = prepare(source, out)
    train = np.flatnonzero(rows.fold.to_numpy() != 0)
    features = np.concatenate([sequences[i] for i in train])
    mu, sd = features.mean(0), features.std(0) + 1e-6
    lengths = np.array([len(x) for x in sequences])
    padded = np.zeros((len(rows), lengths.max(), 512), np.float32)
    for i, sequence in enumerate(sequences):
        padded[i, : len(sequence)] = (sequence - mu) / sd
    x = torch.tensor(padded, device="cuda")
    index = torch.tensor(indices, device="cuda")
    weight = torch.tensor(weights, device="cuda")
    labels = torch.tensor(rows.class_code.to_numpy(), device="cuda", dtype=torch.long)
    counts = np.bincount(rows.iloc[train].class_code, minlength=5)
    cw = torch.tensor(len(train) / (5 * counts), device="cuda", dtype=torch.float32)
    seed, lr, arm = 20260930, 0.0003, "gru_ce"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(arm).cuda()
    initial = state_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(60):
        model.train()
        order = rng.permutation(train)
        total = 0.0
        for start in range(0, len(order), 64):
            ids = order[start : start + 64]
            logits = model(x[ids, : int(lengths[ids].max())])
            readout = logits.gather(1, index[ids, :, None].expand(-1, -1, 5))
            loss = objective(readout, labels[ids], weight[ids], cw, arm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total += float(loss.detach()) * len(ids)
        losses.append(total / len(train))
    original = json.loads((reference / "fits/f0_s20260930_lr0.0003_gru_ce.json").read_text())
    saved = torch.load(
        reference / "checkpoints/f0_s20260930_lr0.0003_gru_ce.pt", map_location="cpu", weights_only=False
    )
    differences = {
        key: float((value.detach().cpu() - saved["state_dict"][key]).abs().max())
        for key, value in model.state_dict().items()
    }
    result = {
        "fit": "f0_s20260930_lr0.0003_gru_ce",
        "initial_hash_equal": initial == original["initial_hash"],
        "final_hash_equal": state_hash(model) == original["final_hash"],
        "max_parameter_difference": max(differences.values()),
        "max_training_loss_difference": float(
            np.max(np.abs(np.array(losses) - [r["train_loss"] for r in original["history"]]))
        ),
    }
    write_json(out / "REPRODUCTION.json", result)
    assert result["initial_hash_equal"] and result["final_hash_equal"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_root, args.reference, args.out)
