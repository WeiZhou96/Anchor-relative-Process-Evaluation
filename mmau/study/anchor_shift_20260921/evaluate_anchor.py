"""Evaluate actual shifted-input predictions against derived experimental anchors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ape.frame_axis import FrameAnswerTable, FrameProtocolVector, evaluate_system_frames, validate_frame_manifest
from ape.vocabulary import TaskVocabulary


def main(directory, vocabulary_path):
    vocabulary = TaskVocabulary.from_dict(json.loads(vocabulary_path.read_text()))
    rows = [json.loads(line) for line in (directory / "cohort.jsonl").read_text().splitlines()]
    original = validate_frame_manifest(pd.DataFrame(rows), vocabulary)
    summary = json.loads((directory / "summary.json").read_text())
    results = []
    for run in summary["runs"]:
        cache = np.load(directory / run["predictions_file"])
        assert np.array_equal(cache["video_ids"], original.video_id)
        assert int(cache["shift"]) == run["shift"]
        p = cache["probabilities"]
        assert p.shape == (269, 17, 9) and np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(-1), 1, atol=1e-6)
        offsets = cache["offsets"]
        held = np.searchsorted(offsets, np.arange(61), side="right") - 1
        assert np.all(offsets[held] <= np.arange(61))
        pred = p.argmax(-1)[:, held]
        cohort = original.copy()
        cohort["original_anchor_frame"] = cohort.anchor_frame
        cohort["anchor_frame"] = cohort.anchor_frame + run["shift"]
        cohort["post_anchor_frames"] = cohort.last_frame - cohort.anchor_frame
        cohort = validate_frame_manifest(cohort, vocabulary)
        assert (cohort.anchor_frame >= cohort.first_frame).all() and (cohort.post_anchor_frames >= 60).all()
        pv = FrameProtocolVector(
            h_f=60,
            delta_f=1,
            vocabulary_hash=vocabulary.vocabulary_hash,
            protocol_version=f"native-actual-input-shift-{run['shift']:+d}-unfrozen-v1",
        )
        answers = FrameAnswerTable.from_frame(
            pd.DataFrame(
                dict(
                    video_id=np.repeat(cohort.video_id, 61),
                    j=np.tile(np.arange(61), len(cohort)),
                    delta_f=1,
                    pred=pred.reshape(-1),
                    vocabulary_hash=vocabulary.vocabulary_hash,
                )
            ),
            run["system"],
            vocabulary=vocabulary,
        )
        result = evaluate_system_frames(cohort, answers, pv, vocabulary=vocabulary)
        assert result.n_missing_lookup == result.n_offgrid_lookup == 0
        metrics = result.metrics_on()
        y = cohort.class_code.to_numpy()
        stable = np.logical_and.accumulate((pred == y[:, None])[:, ::-1], axis=1)[:, ::-1]
        area = (1 - stable[:, 0]) * 0.5 + (1 - stable[:, -1]) * 0.5 + (1 - stable[:, 1:-1]).sum(1)
        assert abs(area.mean() - metrics["RMSCD@H_frames"]) < 1e-10
        assert abs(np.mean([area[y == k].mean() for k in range(9)]) - metrics["RMSCD@H_frames_macro"]) < 1e-10
        assert abs((pred[:, 1:] != pred[:, :-1]).sum(1).mean() - metrics["mean_flips"]) < 1e-10
        results.append(
            dict(
                **run,
                metrics=metrics,
                pi_hash=pv.pi_hash,
                manifest_sha256=hashlib.sha256(cohort.to_json(orient="records").encode()).hexdigest(),
                native_class_recall={
                    str(vocabulary.decode(k)): float((pred[y == k, -1] == k).mean()) for k in range(9)
                },
            )
        )
    report = dict(
        runs=results,
        evaluations=len(results),
        numeric_checks_passed=True,
        n=269,
        horizon=60,
        semantics="actual input and observation endpoint move with the experimental anchor; eps_sys_f=0 within each derived manifest, condition bound in protocol_version",
        source_annotations_unchanged=True,
        formal_verdict="not_evaluable",
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (directory / "ape_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"evaluations": len(results), "checks_passed": True}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--vocabulary", type=Path, required=True)
    a = p.parse_args()
    main(a.directory, a.vocabulary)
