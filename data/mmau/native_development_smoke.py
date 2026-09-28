"""Exercise native-task instruments on provisional development metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ape.frame_axis import (
    FrameAnswerTable,
    FrameProtocolVector,
    check_no_future_reads,
    evaluate_system_frames,
    frame_horizon_candidates,
    make_frame_prefixes,
    validate_frame_manifest,
)
from ape.vocabulary import TaskVocabulary


def run(directory: Path, output: Path) -> dict[str, Any]:
    vocabulary = TaskVocabulary.from_dict(json.loads((directory / "vocabulary.json").read_text(encoding="utf-8")))
    data = pd.read_json(directory / "development_manifest.jsonl", lines=True, dtype={"video_id": str})
    data = validate_frame_manifest(data, vocabulary)
    if not set(data["split"]).issubset({"train", "dev"}):
        raise ValueError("This diagnostic refuses audit/test rows")
    train = data.loc[data["split"] == "train"]
    candidates = frame_horizon_candidates(train, vocabulary=vocabulary)
    common = data.loc[data["post_anchor_frames"] >= 67]
    train_common = common.loc[common["split"] == "train"]
    majority = int(train_common["class_code"].value_counts().sort_index().idxmax())
    runs = []
    for part in ("train", "dev"):
        cohort = common.loc[common["split"] == part].sort_values("video_id").reset_index(drop=True)
        truth = cohort["class_code"].to_numpy()
        for horizon in (39, 67):
            pv = FrameProtocolVector(
                h_f=horizon,
                vocabulary_hash=vocabulary.vocabulary_hash,
                protocol_version="native-development-instrument-v1",
            )
            prefixes = make_frame_prefixes(cohort, pv, grid_max_f=horizon, h_list_f=[horizon], vocabulary=vocabulary)
            assert check_no_future_reads(prefixes, cohort, horizon).empty
            width = horizon + 1
            for kind in ("oracle", "constant_wrong", "step7", "train_majority"):
                j = np.tile(np.arange(width), len(cohort))
                y = np.repeat(truth, width)
                if kind == "oracle":
                    pred = y
                    expected = 0.0
                elif kind == "constant_wrong":
                    pred = (y + 1) % vocabulary.n_classes
                    expected = float(horizon)
                elif kind == "step7":
                    pred = np.where(j >= 7, y, (y + 1) % vocabulary.n_classes)
                    expected = 6.5
                else:
                    pred = np.full(len(y), majority)
                    expected = horizon * float(np.mean(truth != majority))
                answers = FrameAnswerTable.from_frame(
                    pd.DataFrame(
                        {
                            "video_id": np.repeat(cohort["video_id"].to_numpy(), width),
                            "j": j,
                            "delta_f": 1,
                            "pred": pred,
                            "vocabulary_hash": vocabulary.vocabulary_hash,
                        }
                    ),
                    kind,
                    vocabulary=vocabulary,
                )
                result = evaluate_system_frames(cohort, answers, pv, vocabulary=vocabulary)
                metrics = result.metrics_on()
                assert abs(metrics["RMSCD@H_frames"] - expected) < 1e-9
                assert result.n_missing_lookup == result.n_offgrid_lookup == 0
                runs.append(
                    {
                        "split": part,
                        "h_f": horizon,
                        "system": kind,
                        "n": result.n,
                        "metrics": metrics,
                        "expected_rmscd_frames": expected,
                        "passed": True,
                        "pi_hash": pv.pi_hash,
                        "vocabulary_hash": vocabulary.vocabulary_hash,
                    }
                )
    report = {
        "artifact": "native_development_instrument_check",
        "is_formal_experiment": False,
        "learned_pixel_model_trained": False,
        "source_independence_certified": False,
        "confidence_intervals_computed": False,
        "horizon_frozen": False,
        "train_horizon_candidates_after_guard": candidates,
        "horizons_exercised_for_comparison_to_prior_proposal": [39, 67],
        "common_cohort_min_span": 67,
        "majority_class_from_train_only": vocabulary.decode(majority),
        "all_checks_passed": True,
        "future_read_violations": 0,
        "runs": runs,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.directory, args.output)
    print(
        json.dumps(
            {
                "checks": len(result["runs"]),
                "passed": result["all_checks_passed"],
                "train_horizon_candidates": result["train_horizon_candidates_after_guard"],
            },
            indent=2,
        )
    )
