"""Evaluate held causal predictions through APE and a direct numerical check."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ape.frame_axis import FrameAnswerTable, FrameProtocolVector, evaluate_system_frames, validate_frame_manifest
from ape.vocabulary import TaskVocabulary


def main(directory: Path, vocabulary_path: Path) -> None:
    vocabulary = TaskVocabulary.from_dict(json.loads(vocabulary_path.read_text()))
    rows = [json.loads(line) for line in (directory / "cohort.jsonl").read_text().splitlines()]
    manifest = validate_frame_manifest(pd.DataFrame(rows), vocabulary)
    plan = json.loads((directory / "pilot_plan.json").read_text())
    offsets = np.array(plan["offsets_frames"])
    held = np.searchsorted(offsets, np.arange(68), side="right") - 1
    assert np.all(offsets[held] <= np.arange(68))
    ids = manifest["video_id"].to_numpy()
    y = manifest["class_code"].to_numpy()
    train_mask = manifest["split"].to_numpy() == "train"
    majority = int(np.bincount(y[train_mask], minlength=9).argmax())
    runs = []
    sources = list(sorted(directory.glob("*_predictions.npz")))
    for path in [None] + sources:
        if path is None:
            name = "train_majority"
            predictions = np.full((len(manifest), 68), majority)
        else:
            name = path.stem.removesuffix("_predictions")
            cache = np.load(path)
            assert np.array_equal(cache["video_ids"], ids)
            assert np.array_equal(cache["offsets"], offsets)
            probabilities = cache["probabilities"]
            assert probabilities.shape == (len(ids), len(offsets), vocabulary.n_classes)
            assert np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0)
            np.testing.assert_allclose(probabilities.sum(-1), 1, atol=1e-6)
            predictions = probabilities.argmax(-1)[:, held]
        for split in ("train", "dev"):
            mask = manifest["split"].to_numpy() == split
            cohort = manifest.loc[mask].reset_index(drop=True)
            for horizon in (39, 67):
                pred = predictions[mask, : horizon + 1]
                pv = FrameProtocolVector(
                    h_f=horizon,
                    vocabulary_hash=vocabulary.vocabulary_hash,
                    protocol_version="native-pixel-development-unfrozen-v1",
                )
                answers = FrameAnswerTable.from_frame(
                    pd.DataFrame(
                        dict(
                            video_id=np.repeat(ids[mask], horizon + 1),
                            j=np.tile(np.arange(horizon + 1), mask.sum()),
                            delta_f=1,
                            pred=pred.reshape(-1),
                            vocabulary_hash=vocabulary.vocabulary_hash,
                        )
                    ),
                    name,
                    vocabulary=vocabulary,
                )
                result = evaluate_system_frames(cohort, answers, pv, vocabulary=vocabulary)
                assert result.n_missing_lookup == result.n_offgrid_lookup == 0
                metrics = result.metrics_on()
                truth = y[mask]
                correct = pred == truth[:, None]
                stable = np.logical_and.accumulate(correct[:, ::-1], axis=1)[:, ::-1]
                area_per_clip = (1 - stable[:, 0]) * 0.5 + (1 - stable[:, -1]) * 0.5 + (1 - stable[:, 1:-1]).sum(1)
                direct_area = float(area_per_clip.mean())
                direct_macro_area = float(np.mean([area_per_clip[truth == k].mean() for k in range(9)]))
                assert abs(metrics["RMSCD@H_frames"] - direct_area) < 1e-10
                assert abs(metrics["RMSCD@H_frames_macro"] - direct_macro_area) < 1e-10
                flips = (pred[:, 1:] != pred[:, :-1]).sum(1)
                assert abs(metrics["mean_flips"] - flips.mean()) < 1e-10
                # Successful early decisions that subsequently become wrong are explicitly counted.
                ever_correct = correct.any(1)
                late_wrong = ever_correct & ~correct[:, -1]
                row = dict(
                    system=name,
                    split=split,
                    h_f=horizon,
                    n=int(mask.sum()),
                    metrics=metrics,
                    native_class_recall={
                        str(vocabulary.decode(k)): float(correct[truth == k, -1].mean()) for k in range(9)
                    },
                    ever_correct_then_wrong_at_end=int(late_wrong.sum()),
                    stable_at_anchor_macro=float(np.mean([stable[truth == k, 0].mean() for k in range(9)])),
                    pi_hash=pv.pi_hash,
                    numerical_check_passed=True,
                )
                runs.append(row)
    report = dict(
        formal_verdict="not_evaluable",
        source_identity_certified=False,
        predictions_from_pixels=True,
        no_new_annotation=True,
        horizons_frozen=False,
        model_checkpoint_selected_on_dev=True,
        dense_hold_causal=True,
        numerical_checks_passed=True,
        runs=runs,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (directory / "ape_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"evaluations": len(runs), "checks_passed": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    args = parser.parse_args()
    main(args.directory, args.vocabulary)
