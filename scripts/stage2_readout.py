"""Descriptive paired OOF readouts; overlapping training folds are not independent trials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ape.method_analysis import CONTRAST_NAMES, paired_components, point_weights, trajectory_components  # noqa: E402
from scripts.stage2_train import EMA_ALPHAS, LRS, SEEDS  # noqa: E402


def run(root: Path) -> None:
    state = json.loads((root / "run_state.json").read_text())
    assert state.get("complete") and state["fits_completed"] == 250 and not state["smoke"]
    manifest = pd.read_csv(root / "development_folds.csv")
    metrics = pd.read_csv(root / "metrics.csv")
    arms = metrics.arm.unique().tolist()
    assert len(arms) == 9
    pairs = []
    class_rows = []
    strata_rows = []
    for seed in SEEDS:
        for lr in LRS:
            components = {}
            for arm in arms:
                with np.load(root / "oof" / f"s{seed}_lr{lr:g}_{arm}.npz") as data:
                    assert np.array_equal(data["video_id"], manifest.video_id.to_numpy())
                    probability = data["probabilities"]
                for horizon in [10, 4]:
                    mask = manifest.post_anchor_length_s.to_numpy() >= horizon - 1e-9
                    p = probability[mask, : int(horizon / 0.25) + 1]
                    assert np.isfinite(p).all() and np.allclose(p.sum(-1), 1, atol=1e-5)
                    correct = p.argmax(-1) == manifest.class_code.to_numpy()[mask, None]
                    components[arm, horizon] = trajectory_components(correct, np.arange(correct.shape[1]) * 0.25)
                    eligible = manifest.loc[mask]
                    for category in range(5):
                        sub = eligible.class_code.to_numpy() == category
                        class_rows.append(
                            {
                                "seed": seed,
                                "lr": lr,
                                "arm": arm,
                                "horizon": horizon,
                                "class_code": category,
                                "n": int(sub.sum()),
                                **{k: float(v[sub].mean()) for k, v in components[arm, horizon].items()},
                            }
                        )
                    for truncated in [True, False]:
                        sub = (eligible.anchor_s.to_numpy() < 4) == truncated
                        for weighting in ["micro", "macro"]:
                            ys = eligible.class_code.to_numpy()[sub]
                            if len(ys) == 0 or (weighting == "macro" and len(np.unique(ys)) != 5):
                                continue
                            sw = point_weights(ys, weighting)
                            strata_rows.append(
                                {
                                    "seed": seed,
                                    "lr": lr,
                                    "arm": arm,
                                    "horizon": horizon,
                                    "pre_anchor_shorter_than_4s": truncated,
                                    "weighting": weighting,
                                    "n": len(ys),
                                    **{k: float(sw @ v[sub]) for k, v in components[arm, horizon].items()},
                                }
                            )
                    for weighting in ["macro", "micro"]:
                        weights = point_weights(manifest.class_code.to_numpy()[mask], weighting)
                        stored = metrics[
                            (metrics.seed == seed)
                            & np.isclose(metrics.lr, lr)
                            & (metrics.arm == arm)
                            & (metrics.horizon == horizon)
                            & (metrics.weighting == weighting)
                        ].iloc[0]
                        for key, value in components[arm, horizon].items():
                            assert np.isclose(weights @ value, stored[key], atol=1e-10)
            for horizon in [10, 4]:
                mask = manifest.post_anchor_length_s.to_numpy() >= horizon - 1e-9
                for other in [
                    "gru_ce",
                    "gru_time",
                    *[f"gru_ema{a:g}" for a in EMA_ALPHAS],
                    "mean_ce",
                    "transformer_ce",
                ]:
                    a, b = components["gru_suffix", horizon], components[other, horizon]
                    values = paired_components(a, b, horizon)
                    for mode in ["macro", "micro"]:
                        w = point_weights(manifest.class_code.to_numpy()[mask], mode)
                        means = w @ values
                        row = {
                            "seed": seed,
                            "lr": lr,
                            "horizon": horizon,
                            "weighting": mode,
                            "arm_a": "gru_suffix",
                            "arm_b": other,
                            **{key: float(value) for key, value in zip(CONTRAST_NAMES, means)},
                            "endpoint_accuracy_advantage": float(means[-1] / horizon),
                            "n": int(mask.sum()),
                            "common_success_mass": float(w @ (a["endpoint"] & b["endpoint"])),
                        }
                        source = manifest.loc[mask, "source_cluster_id"].to_numpy()
                        for j, key in enumerate(CONTRAST_NAMES):
                            row[key + "_nonzero_clusters"] = int(len(np.unique(source[np.abs(values[:, j]) > 1e-12])))
                        pairs.append(row)
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(root / "paired_oof.csv", index=False)
    pd.DataFrame(class_rows).to_csv(root / "per_class.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(root / "input_start_strata.csv", index=False)
    primary = metrics[(metrics.weighting == "macro") & (metrics.horizon == 10)]
    group = primary.groupby(["lr", "arm"])[["delay", "error", "retracted", "endpoint"]].agg(["mean", "min", "max"])
    group.columns = ["_".join(c) for c in group.columns]
    group.to_csv(root / "primary_summary.csv")
    dominance = []
    for lr in LRS:
        average = primary[np.isclose(primary.lr, lr)].groupby("arm")[["delay", "retracted", "endpoint"]].mean()
        candidate = average.loc["gru_suffix"]
        for arm in arms:
            if arm == "gru_suffix":
                continue
            other = average.loc[arm]
            dominance.append(
                {
                    "lr": lr,
                    "comparator": arm,
                    "dominates_suffix_endpoint_delay": bool(
                        other.endpoint >= candidate.endpoint and other.delay <= candidate.delay
                    ),
                    "dominates_suffix_endpoint_retraction": bool(
                        other.endpoint >= candidate.endpoint and other.retracted <= candidate.retracted
                    ),
                }
            )
    pd.DataFrame(dominance).to_csv(root / "dominance.csv", index=False)
    selected = pair_frame[(pair_frame.weighting == "macro") & (pair_frame.horizon == 10)]
    direction = []
    for (lr, arm), rows in selected.groupby(["lr", "arm_b"]):
        direction.append(
            {
                "lr": float(lr),
                "comparator": arm,
                "seeds": len(rows),
                "delay_better_seeds": int((rows.advantage > 0).sum()),
                "retraction_better_seeds": int((rows.retracted_area > 0).sum()),
                "endpoint_better_seeds": int((rows.endpoint_accuracy_advantage > 0).sum()),
                "mean_delay_advantage": float(rows.advantage.mean()),
                "mean_retraction_advantage": float(rows.retracted_area.mean()),
                "mean_endpoint_accuracy_advantage": float(rows.endpoint_accuracy_advantage.mean()),
            }
        )
    summary = {
        "primary": "H10 macro; means and min/max across five seeds, both LRs separately",
        "direction": direction,
        "development_clips": len(manifest),
        "h10_clips": int((manifest.post_anchor_length_s >= 10 - 1e-9).sum()),
        "h4_clips": int((manifest.post_anchor_length_s >= 4 - 1e-9).sum()),
        "all_recomputations_passed": True,
        "class_counts_h10": manifest.loc[manifest.post_anchor_length_s >= 10 - 1e-9, "class_code"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "pre_anchor_shorter_than_4s_fraction": float((manifest.anchor_s < 4).mean()),
        "dominance": dominance,
        "original_test_evaluated": False,
        "uncertainty": (
            "No population confidence intervals from fixed OOF predictions; "
            "training folds overlap and dev was previously used."
        ),
    }
    (root / "READOUT.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(group.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
