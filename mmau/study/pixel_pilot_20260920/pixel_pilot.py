"""Frozen-image causal baselines for provisional MM-AU train/dev diagnostics."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

OFFSETS = sorted(set(range(0, 68, 4)) | {39, 67})


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Frames(Dataset):
    def __init__(self, rows, root):
        self.items = []
        self.transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        for row in rows:
            directory = root / row["relative_image_directory"]
            files = {int(p.stem): p for p in directory.iterdir() if p.stem.isdecimal()}
            for offset in OFFSETS:
                frame = row["anchor_frame"] + offset
                assert row["first_frame"] <= frame <= row["last_frame"]
                self.items.append(files[frame])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path = self.items[index]
        raw = path.read_bytes()
        import io
        with Image.open(io.BytesIO(raw)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, hashlib.sha256(raw).hexdigest()


class PrefixModel(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.gru = nn.GRU(512, 128, batch_first=True) if kind == "gru" else None
        self.head = nn.Linear(128 if kind == "gru" else 512, 9)

    def forward(self, x):
        if self.gru is None:
            x = x.cumsum(1) / torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]
        else:
            x, _ = self.gru(x)
        return self.head(x)


def macro(pred, y):
    return float(np.mean([(pred[y == k] == k).mean() for k in range(9)]))


def checks():
    torch.manual_seed(1)
    x = torch.randn(3, len(OFFSETS), 512)
    for kind in ("prefix_mean", "gru"):
        model = PrefixModel(kind).eval()
        altered = x.clone()
        altered[:, 8:] = torch.randn_like(altered[:, 8:]) * 20
        torch.testing.assert_close(model(x)[:, :8], model(altered)[:, :8], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model(x)[:, :8], model(x[:, :8]), atol=1e-6, rtol=1e-5)
    held = np.searchsorted(OFFSETS, np.arange(68), side="right") - 1
    assert all(OFFSETS[held[t]] <= t for t in range(68))
    assert OFFSETS[held[39]] == 39 and OFFSETS[held[67]] == 67
    return {"future_perturbation_and_prefix_truncation": "passed_both_models", "dense_hold_causal": True}


def main(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    started = time.time()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    manifest = args.development / "development_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows = sorted([r for r in rows if r["post_anchor_frames"] >= 67], key=lambda r: (r["split"], r["video_id"]))
    assert {r["split"] for r in rows} == {"train", "dev"}
    train_mask = np.array([r["split"] == "train" for r in rows])
    assert train_mask.sum() == 2255 and (~train_mask).sum() == 269
    assert not ({r["source_cluster_id"] for r in rows if r["split"] == "train"} & {r["source_cluster_id"] for r in rows if r["split"] == "dev"})
    y = np.array([r["class_code"] for r in rows])
    plan = dict(artifact="exploratory_native_pixel_pilot", offsets_frames=OFFSETS, horizons=[39, 67],
                seeds=[20260920, 20260921, 20260922], epochs=40, batch_size=128,
                optimizer="Adam", learning_rate=0.001, weight_decay=0.0001,
                loss="train-class-balanced CE, equal weight across sampled prefixes",
                checkpoint="best dev macro accuracy at 67, tie by lower unweighted dev CE",
                features="frozen ImageNet ResNet18, torchvision V1 resize256 crop224 normalization",
                normalizer="per-feature mean/std fitted on train clips and all sampled train frames only",
                dense_predictions="causal forward hold of latest observed sample", formal=False,
                source_identity_certified=False, dev_reused_for_selection=True,
                train_n=int(train_mask.sum()), dev_n=int((~train_mask).sum()),
                manifest_sha256=sha(manifest), vocabulary_sha256=sha(args.development / "vocabulary.json"),
                weights_sha256=sha(args.weights), script_sha256=sha(Path(__file__)),
                torch_version=torch.__version__, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    dump(out / "pilot_plan.json", plan)
    dump(out / "causality_checks.json", checks())
    (out / "cohort.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    dataset = Frames(rows, args.root)
    loader = DataLoader(dataset, batch_size=128, num_workers=8, pin_memory=True, shuffle=False)
    backbone = models.resnet18(weights=None)
    backbone.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    backbone.fc = nn.Identity()
    backbone = backbone.cuda().eval()
    features, digests = [], []
    with torch.inference_mode():
        for batch, (images, hashes) in enumerate(loader):
            features.append(backbone(images.cuda(non_blocking=True)).cpu().numpy())
            digests.extend(hashes)
            if batch % 25 == 0:
                print(f"features {min((batch + 1) * 128, len(dataset))}/{len(dataset)}", flush=True)
    x = np.concatenate(features).reshape(len(rows), len(OFFSETS), 512)
    np.savez_compressed(out / "features.npz", features=x, video_ids=np.array([r["video_id"] for r in rows]))
    with (out / "image_inputs.jsonl").open("w") as stream:
        for path, digest in zip(dataset.items, digests):
            stream.write(json.dumps({"path": str(path.relative_to(args.root)), "sha256": digest}) + "\n")
    del backbone, features
    mean = x[train_mask].mean(axis=(0, 1), keepdims=True)
    std = np.maximum(x[train_mask].std(axis=(0, 1), keepdims=True), 1e-5)
    np.savez(out / "normalizer.npz", mean=mean, std=std)
    x = torch.from_numpy((x - mean) / std).cuda()
    train_indices = torch.from_numpy(np.flatnonzero(train_mask)).cuda()
    dev_indices = torch.from_numpy(np.flatnonzero(~train_mask)).cuda()
    labels = torch.from_numpy(y).cuda()
    counts = np.bincount(y[train_mask], minlength=9)
    weights = torch.tensor(train_mask.sum() / (9 * counts), dtype=torch.float32, device="cuda")
    criterion = nn.CrossEntropyLoss(weight=weights)
    runs = []
    for kind in ("prefix_mean", "gru"):
        for seed in plan["seeds"]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = PrefixModel(kind).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
            best_key, best_state, history = (-1, -float("inf")), None, []
            for epoch in range(plan["epochs"]):
                model.train()
                permutation = train_indices[torch.randperm(len(train_indices), device="cuda")]
                for indices in permutation.split(plan["batch_size"]):
                    logits = model(x[indices])
                    loss = criterion(logits.flatten(0, 1), labels[indices, None].expand(-1, len(OFFSETS)).reshape(-1))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    dev_logits = model(x[dev_indices])[:, -1]
                    dev_macro = macro(dev_logits.argmax(-1).cpu().numpy(), y[~train_mask])
                    dev_ce = float(nn.functional.cross_entropy(dev_logits, labels[dev_indices]))
                history.append(dict(epoch=epoch + 1, dev_macro_67=dev_macro, dev_ce_67=dev_ce))
                key = (dev_macro, -dev_ce)
                if key > best_key:
                    best_key = key
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1
            model.load_state_dict(best_state)
            torch.save(best_state, out / f"{kind}_{seed}.pt")
            with torch.inference_mode():
                probs = torch.cat([model(part).softmax(-1).cpu() for part in x.split(128)]).numpy()
            np.savez_compressed(out / f"{kind}_{seed}_predictions.npz", probabilities=probs, offsets=OFFSETS,
                                video_ids=np.array([r["video_id"] for r in rows]))
            dump(out / f"{kind}_{seed}_history.json", history)
            run = dict(model=kind, seed=seed, selected_epoch=best_epoch, splits={})
            for split, mask in (("train", train_mask), ("dev", ~train_mask)):
                run["splits"][split] = {}
                for offset in (0, 39, 67):
                    pred = probs[mask, OFFSETS.index(offset)].argmax(-1)
                    confusion = np.zeros((9, 9), dtype=int)
                    np.add.at(confusion, (y[mask], pred), 1)
                    run["splits"][split][str(offset)] = dict(macro_accuracy=macro(pred, y[mask]),
                        micro_accuracy=float((pred == y[mask]).mean()), confusion=confusion.tolist())
            runs.append(run)
            print(json.dumps({"model": kind, "seed": seed, "epoch": best_epoch, "dev_macro_67": best_key[0]}), flush=True)
            dump(out / "partial_runs.json", runs)
    dump(out / "summary.json", dict(plan=plan, runs=runs, elapsed_seconds=time.time() - started,
         train_majority_class=int(counts.argmax()), class_counts_train=counts.tolist(),
         class_counts_dev=np.bincount(y[~train_mask], minlength=9).tolist(),
         feature_shape=list(x.shape), feature_cache_sha256=sha(out / "features.npz")))
    print("PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
