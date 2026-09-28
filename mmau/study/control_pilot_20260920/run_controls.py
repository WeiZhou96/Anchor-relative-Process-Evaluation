"""Fixed causal ablations on the existing frozen MM-AU feature cache."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pixel_pilot_20260920"))
from pixel_pilot import OFFSETS, PrefixModel, dump, macro, sha

ARMS = [
    ("anchor_weighted", "prefix_mean", "anchor", True),
    ("anchor_unweighted", "prefix_mean", "anchor", False),
    ("mean_unweighted", "prefix_mean", "ordered", False),
    ("gru_unweighted", "gru", "ordered", False),
    ("shuffle_weighted", "gru", "shuffle", True),
    ("shuffle_unweighted", "gru", "shuffle", False),
]


def permutations(ids, seed):
    result = np.zeros((len(ids), len(OFFSETS), len(OFFSETS)), dtype=np.int64)
    for i, video in enumerate(ids):
        for k in range(len(OFFSETS)):
            value = int.from_bytes(hashlib.sha256(f"{seed}/{video}/{k}".encode()).digest()[:8], "little")
            result[i, k, :k + 1] = np.random.default_rng(value).permutation(k + 1)
    return result


def forward(model, x, mode, permutation):
    if mode == "ordered":
        return model(x)
    if mode == "anchor":
        return model(x[:, :1]).expand(-1, len(OFFSETS), -1)
    n, t, d = x.shape
    # Each endpoint is recomputed using only that endpoint's observed frame set.
    shuffled = x[torch.arange(n, device=x.device)[:, None, None], permutation]
    hidden, _ = model.gru(shuffled.reshape(n * t, t, d))
    final = hidden[torch.arange(n * t, device=x.device), torch.arange(t, device=x.device).repeat(n)]
    return model.head(final).reshape(n, t, 9)


def checks():
    torch.manual_seed(17)
    x = torch.randn(2, len(OFFSETS), 512)
    perm = torch.from_numpy(permutations(["a", "b"], 17))
    for k in range(len(OFFSETS)):
        for i in range(2):
            assert sorted(perm[i, k, :k + 1].tolist()) == list(range(k + 1))
    for mode, kind in (("anchor", "prefix_mean"), ("shuffle", "gru"), ("ordered", "gru")):
        model = PrefixModel(kind).eval()
        altered = x.clone()
        altered[:, 8:] += 100
        torch.testing.assert_close(forward(model, x, mode, perm)[:, :8], forward(model, altered, mode, perm)[:, :8])
        if mode == "shuffle":
            for k in (0, 7, 18):
                manual = model(x[torch.arange(2)[:, None], perm[:, k, :k + 1]])[:, -1]
                torch.testing.assert_close(forward(model, x, mode, perm)[:, k], manual)
    return {"prefix_sets_exact": True, "future_perturbation_passed": True, "padded_shuffle_matches_unpadded": True}


def main(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=False)
    parent = json.loads((args.parent / "pilot_plan.json").read_text())
    plan = dict(parent, artifact="fixed_native_development_controls", arms=ARMS,
                order_control="fixed per-clip/seed/endpoint permutation of ONLY already observed frames; recompute each endpoint",
                epoch_selection="same dev macro at 67, tie lower unweighted dev CE; also retain epoch40 predictions",
                normalizer="reuse original train-only normalizer in all arms including anchor",
                input_feature_sha256=sha(args.parent / "features.npz"),
                parent_plan_sha256=sha(args.parent / "pilot_plan.json"),
                script_sha256=sha(Path(__file__)))
    dump(args.out / "pilot_plan.json", plan)
    dump(args.out / "causality_checks.json", checks())
    shutil.copy2(args.parent / "cohort.jsonl", args.out / "cohort.jsonl")
    rows = [json.loads(line) for line in (args.parent / "cohort.jsonl").read_text().splitlines()]
    ids = np.array([r["video_id"] for r in rows])
    train = np.array([r["split"] == "train" for r in rows])
    assert set(r["split"] for r in rows) == {"train", "dev"}
    assert train.sum() == 2255 and (~train).sum() == 269
    feature = np.load(args.parent / "features.npz")
    assert np.array_equal(feature["video_ids"], ids)
    assert sha(args.parent / "features.npz") == json.loads((args.parent / "summary.json").read_text())["feature_cache_sha256"]
    norm = np.load(args.parent / "normalizer.npz")
    x = torch.from_numpy((feature["features"] - norm["mean"]) / norm["std"]).cuda()
    y = np.array([r["class_code"] for r in rows])
    labels = torch.from_numpy(y).cuda()
    ti = torch.from_numpy(np.flatnonzero(train)).cuda()
    di = torch.from_numpy(np.flatnonzero(~train)).cuda()
    weights = torch.tensor(train.sum() / (9 * np.bincount(y[train], minlength=9)), dtype=torch.float32, device="cuda")
    runs = []
    for name, kind, mode, weighted in ARMS:
        for seed in plan["seeds"]:
            torch.manual_seed(seed)
            permutation = torch.from_numpy(permutations(ids, seed)).cuda() if mode == "shuffle" else None
            model = PrefixModel(kind).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
            criterion = nn.CrossEntropyLoss(weight=weights if weighted else None)
            best, state, history = (-1, -float("inf")), None, []
            for epoch in range(plan["epochs"]):
                model.train()
                indices = ti[torch.randperm(len(ti), device="cuda")]
                for batch in indices.split(128):
                    logits = forward(model, x[batch], mode, permutation[batch] if permutation is not None else None)
                    loss = criterion(logits.reshape(-1, 9), labels[batch, None].expand(-1, len(OFFSETS)).reshape(-1))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    logits = torch.cat([forward(model, x[b], mode, permutation[b] if permutation is not None else None)[:, -1] for b in di.split(128)])
                    score = macro(logits.argmax(-1).cpu().numpy(), y[~train])
                    ce = float(nn.functional.cross_entropy(logits, labels[di]))
                history.append(dict(epoch=epoch + 1, dev_macro_67=score, dev_ce_67=ce))
                if (score, -ce) > best:
                    best = (score, -ce)
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1
            def predict():
                with torch.inference_mode():
                    return torch.cat([forward(model, x[b], mode, permutation[b] if permutation is not None else None).softmax(-1).cpu() for b in torch.arange(len(rows), device="cuda").split(128)]).numpy()
            final_probs = predict()
            np.savez_compressed(args.out / f"{name}_{seed}_epoch40.npz", probabilities=final_probs, offsets=OFFSETS, video_ids=ids)
            model.load_state_dict(state)
            probs = predict()
            torch.save(state, args.out / f"{name}_{seed}.pt")
            np.savez_compressed(args.out / f"{name}_{seed}_predictions.npz", probabilities=probs, offsets=OFFSETS, video_ids=ids)
            dump(args.out / f"{name}_{seed}_history.json", history)
            # Replay from the saved checkpoint on the execution backend, not an in-memory copy.
            replay = PrefixModel(kind).cuda().eval()
            replay.load_state_dict(torch.load(args.out / f"{name}_{seed}.pt", map_location="cpu", weights_only=True))
            with torch.inference_mode():
                reproduced = torch.cat([forward(replay, x[b], mode, permutation[b] if permutation is not None else None).softmax(-1).cpu() for b in torch.arange(len(rows), device="cuda").split(128)]).numpy()
            np.testing.assert_array_equal(reproduced, probs)
            run = dict(system=f"{name}_{seed}", arm=name, seed=seed, selected_epoch=best_epoch, replay_exact=True, splits={})
            for split, mask in (("train", train), ("dev", ~train)):
                run["splits"][split] = {}
                for offset in (0, 39, 67):
                    pred = probs[mask, OFFSETS.index(offset)].argmax(-1)
                    confusion = np.zeros((9, 9), dtype=int)
                    np.add.at(confusion, (y[mask], pred), 1)
                    run["splits"][split][str(offset)] = dict(macro_accuracy=macro(pred, y[mask]), micro_accuracy=float((pred == y[mask]).mean()), confusion=confusion.tolist())
            runs.append(run)
            dump(args.out / "partial_runs.json", runs)
            print(json.dumps({"system": run["system"], "best_epoch": best_epoch, "dev_macro_67": best[0]}), flush=True)
    dump(args.out / "summary.json", dict(plan=plan, runs=runs, elapsed_seconds=time.time() - started, all_replays_exact=True))
    print("CONTROLS_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
