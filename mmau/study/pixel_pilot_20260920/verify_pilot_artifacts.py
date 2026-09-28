"""Replay saved checkpoints and verify train-only statistics and input order."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pixel_pilot import OFFSETS, PrefixModel, dump, sha


def main(directory):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    rows = [json.loads(line) for line in (directory / "cohort.jsonl").read_text().splitlines()]
    ids = np.array([row["video_id"] for row in rows])
    mask = np.array([row["split"] == "train" for row in rows])
    assert len(set(ids)) == len(ids)
    data = np.load(directory / "features.npz")
    x = data["features"]
    assert np.array_equal(data["video_ids"], ids)
    norm = np.load(directory / "normalizer.npz")
    np.testing.assert_array_equal(norm["mean"], x[mask].mean((0, 1), keepdims=True))
    np.testing.assert_array_equal(norm["std"], np.maximum(x[mask].std((0, 1), keepdims=True), 1e-5))
    entries = [json.loads(line) for line in (directory / "image_inputs.jsonl").read_text().splitlines()]
    assert len(entries) == len(rows) * len(OFFSETS)
    for index, row in enumerate(rows):
        for j, offset in enumerate(OFFSETS):
            item = entries[index * len(OFFSETS) + j]
            path = Path(item["path"])
            assert str(path.parent) == row["relative_image_directory"]
            assert int(path.stem) == row["anchor_frame"] + offset <= row["last_frame"]
            assert len(item["sha256"]) == 64
    x = torch.from_numpy((x - norm["mean"]) / norm["std"])
    replay = []
    for path in sorted(directory.glob("*_predictions.npz")):
        name = path.stem.removesuffix("_predictions")
        kind = name.rsplit("_", 1)[0]
        model = PrefixModel(kind).eval()
        checkpoint = directory / (name + ".pt")
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        with torch.inference_mode():
            actual = torch.cat([model(batch).softmax(-1) for batch in x.split(128)]).numpy()
        saved = np.load(path)["probabilities"]
        # CPU/cuDNN kernels may differ numerically; the acceptance check replays the original CUDA backend.
        model = model.cuda()
        with torch.inference_mode():
            gpu = torch.cat([model(batch.cuda()).softmax(-1).cpu() for batch in x.split(128)]).numpy()
        np.testing.assert_allclose(gpu, saved, atol=1e-7, rtol=1e-6)
        assert np.array_equal(gpu.argmax(-1), saved.argmax(-1))
        replay.append({"system": name, "cpu_replay_max_abs_error": float(np.max(np.abs(actual - saved))),
                       "cpu_prediction_disagreements": int((actual.argmax(-1) != saved.argmax(-1)).sum()),
                       "cuda_replay_max_abs_error": float(np.max(np.abs(gpu - saved))),
                       "checkpoint_sha256": sha(checkpoint), "predictions_sha256": sha(path)})
    result = {"all_passed": True, "image_positions_checked": len(entries), "normalizer_train_only_exact": True,
              "replayed_checkpoints": replay, "script_sha256": sha(Path(__file__)),
              "acceptance_backend": "original CUDA/cuDNN; CPU discrepancy separately disclosed",
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    dump(directory / "artifact_verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    main(parser.parse_args().directory)
