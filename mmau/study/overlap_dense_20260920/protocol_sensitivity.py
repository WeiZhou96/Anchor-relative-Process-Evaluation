"""Development-only horizon and output-query-grid sensitivity on fixed traces."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ape.frame_axis import FrameAnswerTable, FrameProtocolVector, evaluate_system_frames, validate_frame_manifest
from ape.vocabulary import TaskVocabulary


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    vocabulary = TaskVocabulary.from_dict(json.loads(args.vocabulary.read_text()))
    records = [json.loads(line) for line in (args.first / "cohort.jsonl").read_text().splitlines()]
    assert (args.first / "cohort.jsonl").read_bytes() == (args.controls / "cohort.jsonl").read_bytes()
    mask = np.array([r["split"] == "dev" for r in records])
    manifest = validate_frame_manifest(pd.DataFrame([r for r in records if r["split"] == "dev"]), vocabulary)
    assert len(manifest) == 269
    ids = np.array([r["video_id"] for r in records])
    y = manifest.class_code.to_numpy()
    majority = int(np.bincount([r["class_code"] for r in records if r["split"] == "train"], minlength=9).argmax())
    sources = sorted(args.first.glob("*_predictions.npz")) + sorted(args.controls.glob("*_predictions.npz"))
    assert len(sources) == 24
    plan = dict(
        horizons_frames=[32, 48, 64],
        output_query_steps_frames=[1, 2, 4, 8],
        n=269,
        cohort="same span>=67 development cohort for ALL combinations",
        operation="rescore fixed causal predictions with previous-value hold; input frames, model state and checkpoints unchanged",
        fine_grid_caveat="delta=1/2 reuses held predictions, not new image inference; original model input cadence ~4 frames plus 39/67",
        anchor_shift_run=False,
        anchor_limitation="model starts at original anchor; negative offsets and changed starts require new frame/features/inference",
        formal=False,
        source_identity_certified=False,
        thresholds_frozen=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2))
    runs = []
    for path in [None] + sources:
        if path is None:
            system = "train_majority"
            dense = np.full((269, 68), majority)
        else:
            cache = np.load(path)
            assert np.array_equal(cache["video_ids"], ids)
            offsets = cache["offsets"]
            held = np.searchsorted(offsets, np.arange(68), side="right") - 1
            assert np.all(offsets[held] <= np.arange(68))
            dense = cache["probabilities"][mask].argmax(-1)[:, held]
            system = path.stem.removesuffix("_predictions")
        for horizon in plan["horizons_frames"]:
            endpoint = None
            for delta in plan["output_query_steps_frames"]:
                pred = dense[:, : horizon + 1 : delta]
                pv = FrameProtocolVector(
                    h_f=horizon,
                    delta_f=delta,
                    vocabulary_hash=vocabulary.vocabulary_hash,
                    protocol_version="native-development-output-grid-sensitivity-v1",
                )
                answers = FrameAnswerTable.from_frame(
                    pd.DataFrame(
                        dict(
                            video_id=np.repeat(manifest.video_id, pred.shape[1]),
                            j=np.tile(np.arange(pred.shape[1]), len(manifest)),
                            delta_f=delta,
                            pred=pred.reshape(-1),
                            vocabulary_hash=vocabulary.vocabulary_hash,
                        )
                    ),
                    system,
                    vocabulary=vocabulary,
                )
                result = evaluate_system_frames(manifest, answers, pv, vocabulary=vocabulary)
                assert result.n_missing_lookup == result.n_offgrid_lookup == 0
                metrics = result.metrics_on()
                stable = np.logical_and.accumulate((pred == y[:, None])[:, ::-1], axis=1)[:, ::-1]
                areas = ((1 - stable[:, 0]) * 0.5 + (1 - stable[:, -1]) * 0.5 + (1 - stable[:, 1:-1]).sum(1)) * delta
                assert abs(areas.mean() - metrics["RMSCD@H_frames"]) < 1e-10
                assert abs(np.mean([areas[y == k].mean() for k in range(9)]) - metrics["RMSCD@H_frames_macro"]) < 1e-10
                assert abs((pred[:, 1:] != pred[:, :-1]).sum(1).mean() - metrics["mean_flips"]) < 1e-10
                if endpoint is None:
                    endpoint = metrics["end_window_macro_acc"]
                assert endpoint == metrics["end_window_macro_acc"]
                runs.append(
                    dict(system=system, horizon=horizon, delta=delta, n=result.n, metrics=metrics, pi_hash=pv.pi_hash)
                )
        print(f"evaluated {system}", flush=True)
    report = dict(
        runs=runs,
        evaluations=len(runs),
        all_numeric_checks_passed=True,
        common_cohort=True,
        endpoint_accuracy_invariant_to_output_query_step=True,
        formal_verdict="not_evaluable",
    )
    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"evaluations": len(runs), "checks_passed": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
