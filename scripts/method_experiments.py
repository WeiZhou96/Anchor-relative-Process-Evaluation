"""Run the retrospective CPU experiments specified in EXPERIMENT_SCOPE_20260927.md."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ape.method_analysis import (  # noqa: E402
    CONTRAST_NAMES,
    cluster_weights,
    delay_bounds,
    paired_components,
    point_weights,
    simultaneous_band,
    trajectory_components,
)
from ape.metrics import AnswerTable, evaluate_system  # noqa: E402
from ape.r2b_identity import group_key  # noqa: E402
from ape.r2c import Context  # noqa: E402

LOGGER = logging.getLogger(__name__)
STEP = 0.25
SEED = 20260927


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def pairs_for(ids: list[str], metadata: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    return [(a, b) for a, b in itertools.combinations(ids, 2) if metadata[a]["seed"] == metadata[b]["seed"]]


def inference_df(cohort: pd.DataFrame, mode: str) -> int:
    """Conservative t reference based on the least-supported class for macro."""
    if mode == "micro":
        return int(cohort.source_cluster_id.nunique()) - 1
    return int(cohort.groupby("class_code").source_cluster_id.nunique().min()) - 1


def load_inputs(ctx: Context, out: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Read one prediction file at a time; never load probabilities or weights."""
    correct = {}
    provenance = []
    columns = np.arange(-2, 89, dtype=int)
    grid = np.broadcast_to(columns, (len(ctx.test), len(columns)))
    labels = ctx.test.class_code.to_numpy()
    for i, sid in enumerate(ctx.nontrivial):
        path = ctx.answers / sid / "answers.csv"
        digest = sha256(path)
        frame = pd.read_csv(path, usecols=["video_id", "j", "delta_s", "pred"])
        if frame.duplicated(["video_id", "j"]).any():
            raise ValueError(f"Duplicate cached prediction keys: {sid}")
        table = AnswerTable.from_frame(frame, sid, ctx.cards[sid])
        if table.delta_s != STEP:
            raise ValueError("Unexpected finest input grid")
        pred, present, _, _ = table.lookup_2d(ctx.test.video_id.tolist(), grid)
        raw_supported = (ctx.test.anchor_s.to_numpy()[:, None] + columns * STEP >= -1e-9) & (
            ctx.test.post_anchor_length_s.to_numpy()[:, None] >= columns * STEP - 1e-9
        )
        if np.any(raw_supported & ~present):
            raise ValueError(f"Unobserved cached cells in video support: {sid}")
        correct[sid] = pred == labels[:, None]
        provenance.append(
            {
                "system_id": sid,
                "answers": str(path),
                "sha256": digest,
                "size": path.stat().st_size,
                "card_sha256": sha256(path.parent / "system_card.yaml"),
            }
        )
        if (i + 1) % 20 == 0:
            LOGGER.info("Read %d/%d systems", i + 1, len(ctx.nontrivial))
    write_json(out / "prediction_provenance.json", provenance)
    return correct, provenance


def select_correct(
    correct: np.ndarray, mask: np.ndarray, horizon: float, shift: float = 0, step: float = STEP
) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(int(round(horizon / step)) + 1) * step
    if not np.isclose(times[-1], horizon):
        raise ValueError("This protocol grid must include the exact endpoint")
    indices = np.rint((times + shift) / STEP).astype(int) + 2
    return correct[mask][:, indices], times


def describe_horizon(
    ctx: Context, cache: dict[str, np.ndarray], horizon: float, out: Path, n_boot: int
) -> dict[str, Any]:
    mask = ctx.test.post_anchor_length_s.to_numpy() >= horizon - 1e-9
    cohort = ctx.test.loc[mask]
    labels = cohort.class_code.to_numpy()
    if set(labels) != set(range(5)):
        raise ValueError("The planned class-balanced target requires all five classes")
    components = {}
    raw = {}
    records = []
    natural = [sid for sid in ctx.nontrivial if ctx.cards[sid]["family"] != "commit"]
    base = [sid for sid in natural if ctx.cards[sid]["family"] in ["clip", "prefix"]]
    for sid in ctx.nontrivial:
        c, times = select_correct(cache[sid], mask, horizon)
        raw[sid] = c
        comp = trajectory_components(c, times)
        components[sid] = comp
        for mode in ["micro", "macro"]:
            w = point_weights(labels, mode)
            records.append(
                {
                    "system_id": sid,
                    "family": ctx.cards[sid]["family"],
                    **ctx.meta[sid],
                    "horizon": horizon,
                    "weighting": mode,
                    "n": len(labels),
                    **{k: float(w @ v) for k, v in comp.items()},
                    "anchor_accuracy": float(w @ c[:, 0]),
                    "anchor_wrong_end_correct": float(w @ (~c[:, 0] & c[:, -1])),
                    "anchor_correct_end_wrong": float(w @ (c[:, 0] & ~c[:, -1])),
                    "ever_correct_then_wrong": float(w @ (comp["retracted"] > 0)),
                }
            )
    pd.DataFrame(records).to_csv(out / f"systems_H{horizon:g}.csv", index=False)
    boot, rejected = cluster_weights(cohort.source_cluster_id.to_numpy(), labels, n_boot, SEED)
    pair_list = pairs_for(natural, ctx.meta)
    primary = np.array([a in base and b in base for a, b in pair_list])
    if primary.sum() != 108 or len(pair_list) != 630:
        raise AssertionError("Card-derived comparison family differs from the plan")
    values = np.stack([paired_components(components[a], components[b], horizon) for a, b in pair_list], axis=1)
    flat = values.reshape(len(labels), -1)
    result = {
        "horizon": horizon,
        "n": len(labels),
        "n_clusters": int(cohort.source_cluster_id.nunique()),
        "class_counts": cohort.class_code.value_counts().sort_index().to_dict(),
        "redrawn_bootstrap": rejected,
        "families": {},
    }
    for mode in ["micro", "macro"]:
        point_w = point_weights(labels, mode)
        points = (point_w @ flat).reshape(len(pair_list), -1)
        samples = (boot[mode] @ flat).reshape(n_boot, len(pair_list), -1)
        all_rows = []
        for family, indices in [
            ("base_primary", np.flatnonzero(primary)),
            ("noncommit_secondary", np.arange(len(pair_list))),
        ]:
            band = simultaneous_band(points[indices], samples[:, indices], degrees_freedom=inference_df(cohort, mode))
            counts = {}
            for k, contrast in enumerate(CONTRAST_NAMES):
                significant = (band["lower"][:, k] > 0) | (band["upper"][:, k] < 0)
                counts[contrast] = int(significant.sum())
            for z, pair_index in enumerate(indices):
                a, b = pair_list[pair_index]
                ca, cb = components[a]["endpoint"], components[b]["endpoint"]
                strata = {
                    "p11": float(point_w @ (ca & cb)),
                    "p10": float(point_w @ (ca & ~cb)),
                    "p01": float(point_w @ (~ca & cb)),
                    "p00": float(point_w @ (~ca & ~cb)),
                }
                for k, contrast in enumerate(CONTRAST_NAMES):
                    all_rows.append(
                        {
                            "system_a": a,
                            "system_b": b,
                            "family": family,
                            "seed": ctx.meta[a]["seed"],
                            "group_a": group_key(ctx.meta[a]),
                            "group_b": group_key(ctx.meta[b]),
                            "contrast": contrast,
                            "value": points[pair_index, k],
                            "lo": band["lower"][z, k],
                            "hi": band["upper"][z, k],
                            "se": band["se"][z, k],
                            "zero_variance": bool(band["zero_variance"][z, k]),
                            **strata,
                        }
                    )
            result["families"][f"{family}_{mode}"] = {
                "pairs": len(indices),
                "simultaneous_critical": band["critical"],
                "significant_contrasts": counts,
            }
        pd.DataFrame(all_rows).to_csv(out / f"pairs_H{horizon:g}_{mode}.csv", index=False)
    bound_rows = []
    sign_rows = []
    base_pairs = pairs_for(base, ctx.meta)
    for stride in [2, 4]:
        for phase in range(stride):
            observed = np.arange(len(times)) % stride == phase
            observed[[0, -1]] = True
            per_system = {}
            for sid in ctx.nontrivial:
                c = raw[sid]
                lo, hi = delay_bounds(c, observed, times)
                fine = components[sid]["delay"]
                coarse = trajectory_components(c[:, observed], times[observed])["delay"]
                if np.any(lo > fine + 1e-10) or np.any(hi < fine - 1e-10):
                    raise AssertionError("Finite-grid bounds failed to contain the known dense result")
                per_system[sid] = (lo, hi)
                for mode in ["micro", "macro"]:
                    w = point_weights(labels, mode)
                    bound_rows.append(
                        {
                            "system_id": sid,
                            "family": ctx.cards[sid]["family"],
                            "step": stride * STEP,
                            "phase": phase * STEP,
                            "weighting": mode,
                            "fine": float(w @ fine),
                            "naive_coarse": float(w @ coarse),
                            "lower": float(w @ lo),
                            "upper": float(w @ hi),
                            "width": float(w @ (hi - lo)),
                            "coarse_under_fraction": float(w @ (coarse < fine - 1e-10)),
                            "coarse_over_fraction": float(w @ (coarse > fine + 1e-10)),
                        }
                    )
            for mode in ["micro", "macro"]:
                w = point_weights(labels, mode)
                for a, b in base_pairs:
                    lower = float(w @ (per_system[b][0] - per_system[a][1]))
                    upper = float(w @ (per_system[b][1] - per_system[a][0]))
                    true = float(w @ (components[b]["delay"] - components[a]["delay"]))
                    identified = 1 if lower > 1e-10 else -1 if upper < -1e-10 else 0
                    sign_rows.append(
                        {
                            "system_a": a,
                            "system_b": b,
                            "weighting": mode,
                            "step": stride * STEP,
                            "phase": phase * STEP,
                            "lower": lower,
                            "upper": upper,
                            "dense_advantage": true,
                            "identified_sign": identified,
                        }
                    )
    pd.DataFrame(bound_rows).to_csv(out / f"bounds_H{horizon:g}.csv", index=False)
    pd.DataFrame(sign_rows).to_csv(out / f"bound_pairs_H{horizon:g}.csv", index=False)
    return result


def robust_comparison(
    ctx: Context, cache: dict[str, np.ndarray], horizon: float, out: Path, n_boot: int
) -> dict[str, Any]:
    """Compare base classifiers on one raw-video cohort across nine read protocols."""
    mask = (ctx.test.anchor_s.to_numpy() >= 0.5 - 1e-9) & (
        ctx.test.post_anchor_length_s.to_numpy() >= horizon + 0.5 - 1e-9
    )
    cohort = ctx.test.loc[mask]
    labels = cohort.class_code.to_numpy()
    ids = [sid for sid in ctx.nontrivial if ctx.cards[sid]["family"] in ["clip", "prefix"]]
    pair_list = pairs_for(ids, ctx.meta)
    protocols = list(itertools.product([-0.5, 0.0, 0.5], [0.25, 0.5, 1.0]))
    values = np.empty((len(labels), len(pair_list), len(protocols)))
    for k, (shift, step) in enumerate(protocols):
        delays = {}
        for sid in ids:
            c, times = select_correct(cache[sid], mask, horizon, shift, step)
            delays[sid] = trajectory_components(c, times)["delay"]
        for p, (a, b) in enumerate(pair_list):
            values[:, p, k] = delays[b] - delays[a]
    flat = values.reshape(len(labels), -1)
    weights, rejected = cluster_weights(cohort.source_cluster_id.to_numpy(), labels, n_boot, SEED)
    result = {
        "horizon": horizon,
        "n": len(cohort),
        "n_clusters": int(cohort.source_cluster_id.nunique()),
        "redrawn_bootstrap": rejected,
        "protocols": protocols,
        "weightings": {},
    }
    for mode in ["micro", "macro"]:
        points = (point_weights(labels, mode) @ flat).reshape(len(pair_list), -1)
        reps = (weights[mode] @ flat).reshape(n_boot, len(pair_list), -1)
        band = simultaneous_band(points, reps, degrees_freedom=inference_df(cohort, mode))
        rows = []
        details = []
        for i, (a, b) in enumerate(pair_list):
            lo, hi = band["lower"][i], band["upper"][i]
            for epsilon in [0.0, 0.125, 0.25]:
                status = (
                    "robust_a"
                    if lo.min() > epsilon
                    else (
                        "robust_b"
                        if hi.max() < -epsilon
                        else (
                            "supported_reversal"
                            if np.any(lo > epsilon) and np.any(hi < -epsilon)
                            else (
                                "protocol_specific"
                                if np.any(lo > epsilon) or np.any(hi < -epsilon)
                                else "inconclusive"
                            )
                        )
                    )
                )
                rows.append(
                    {
                        "system_a": a,
                        "system_b": b,
                        "seed": ctx.meta[a]["seed"],
                        "epsilon": epsilon,
                        "status": status,
                        "min_lower": float(lo.min()),
                        "max_upper": float(hi.max()),
                        "min_advantage": float(points[i].min()),
                        "max_advantage": float(points[i].max()),
                        "point_sign_reversal": bool(points[i].min() < -1e-10 and points[i].max() > 1e-10),
                    }
                )
            for k, (shift, step) in enumerate(protocols):
                details.append(
                    {
                        "system_a": a,
                        "system_b": b,
                        "shift": shift,
                        "step": step,
                        "advantage": points[i, k],
                        "lo": lo[k],
                        "hi": hi[k],
                        "zero_variance": bool(band["zero_variance"][i, k]),
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f"robust_H{horizon:g}_{mode}.csv", index=False)
        pd.DataFrame(details).to_csv(out / f"robust_detail_H{horizon:g}_{mode}.csv", index=False)
        result["weightings"][mode] = {
            "critical": band["critical"],
            "statuses": {str(e): g.status.value_counts().to_dict() for e, g in frame.groupby("epsilon")},
        }
    return result


def nested_cohort(ctx: Context, cache: dict[str, np.ndarray], out: Path) -> None:
    records = []
    labels_all = ctx.test.class_code.to_numpy()
    for sid in ctx.nontrivial:
        for horizon in [4.0, 10.0]:
            for eligibility in [horizon, 21.5]:
                mask = ctx.test.post_anchor_length_s.to_numpy() >= eligibility - 1e-9
                c, times = select_correct(cache[sid], mask, horizon)
                parts = trajectory_components(c, times)
                for mode in ["micro", "macro"]:
                    w = point_weights(labels_all[mask], mode)
                    records.append(
                        {
                            "system_id": sid,
                            "family": ctx.cards[sid]["family"],
                            "horizon": horizon,
                            "cohort_eligibility": eligibility,
                            "n": int(mask.sum()),
                            "weighting": mode,
                            **{k: float(w @ v) for k, v in parts.items()},
                        }
                    )
    pd.DataFrame(records).to_csv(out / "nested_cohort.csv", index=False)


def verify_legacy(ctx: Context, cache: dict[str, np.ndarray], out: Path) -> None:
    rows = []
    for sid in ctx.nontrivial:
        if ctx.cards[sid]["family"] not in ["clip", "prefix"]:
            continue
        for h in [4.0, 10.0, 21.5]:
            pv = ctx.cfg.pi0(h)
            # Existing evaluator uses the same untouched finest cache for base models.
            frame = pd.read_csv(ctx.answers / sid / "answers.csv", usecols=["video_id", "j", "delta_s", "pred"])
            table = AnswerTable.from_frame(frame, sid, ctx.cards[sid])
            old = evaluate_system(ctx.test, table, pv, ctx.cfg.grid_max_s)
            mask = ctx.test.post_anchor_length_s.to_numpy() >= h - 1e-9
            c, times = select_correct(cache[sid], mask, h, step=0.5)
            new = trajectory_components(c, times)["delay"].mean()
            old_value = old.frozen_family(ctx.cfg.s_report_delta_s)["RMSCD@H"]
            record = json.loads((ctx.root / "outputs/r2c/metrics" / pv.pi_hash / f"{sid}.json").read_text())
            published = record["RMSCD"]
            rows.append(
                {
                    "system_id": sid,
                    "horizon": h,
                    "existing_evaluator": old_value,
                    "new_evaluator": new,
                    "abs_difference": abs(old_value - new),
                    "saved_record_rmscd": published,
                }
            )
            if not np.isclose(old_value, new, atol=1e-10):
                raise AssertionError("New extraction does not reproduce the original evaluator")
            if not np.isclose(published, new, atol=1e-10):
                raise AssertionError("New extraction does not reproduce the saved historical result")
    pd.DataFrame(rows).to_csv(out / "legacy_reproduction.csv", index=False)


def coverage_simulation(out: Path) -> dict[str, Any]:
    """A finite Monte Carlo check, not a proof of coverage for traffic data."""
    rng = np.random.default_rng(20260928)
    truth = np.linspace(-0.2, 0.2, 9)
    covered = 0
    trials = 1000
    for trial in range(trials):
        shared = rng.normal(size=(80, 1))
        clusters = shared + rng.normal(size=(80, 9)) * 0.7
        observations = np.repeat(clusters, 2, axis=0) + rng.normal(size=(160, 9)) * 0.3 + truth
        weights, _ = cluster_weights(np.repeat(np.arange(80), 2), np.tile([0, 1], 80), 1000, 20270928 + trial)
        point = observations.mean(0)
        band = simultaneous_band(point, weights["micro"] @ observations, degrees_freedom=79)
        covered += bool(np.all(band["lower"] <= truth) and np.all(band["upper"] >= truth))
    rate = covered / trials
    result = {
        "kind": "synthetic_clustered_gaussian_only",
        "trials": trials,
        "bootstrap_per_trial": 1000,
        "simulation_seed": 20260928,
        "calibration": "max of empirical max-t and Bonferroni t guard; independent simulation seed",
        "simultaneous_coordinates": 9,
        "coverage": rate,
        "mc_standard_error": float(np.sqrt(rate * (1 - rate) / trials)),
        "not_a_universal_guarantee": True,
    }
    write_json(out / "coverage_simulation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=False)
    own = Path(__file__).resolve().parents[1]
    frozen_files = [
        "data/manifest/manifest_real.csv",
        "protocol/pi0.yaml",
        "protocol/pi0.frozen.sha256",
        "outputs/r2b/report_index.json",
        "outputs/r2c/final_conclusions.json",
    ]
    before = {f: sha256(args.source_root / f) for f in frozen_files}
    code_head = subprocess.run(
        ["git", "-C", str(own), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = {
        "started": datetime.now(timezone.utc).isoformat(),
        "code_head": code_head,
        "plan_sha256": sha256(own / "EXPERIMENT_SCOPE_20260927.md"),
        "amendment_sha256": sha256(own / "EXPERIMENT_AMENDMENT_20260927.md"),
        "source_root": str(args.source_root),
        "inputs_before": before,
        "bootstrap": args.bootstrap,
        "pid": os.getpid(),
        "claim_status": "retrospective exploratory analysis; not independent test or new training",
    }
    write_json(args.out / "run_state.json", status)
    ctx = Context(args.source_root, False)
    cache, provenance = load_inputs(ctx, args.out)
    LOGGER.info("Checking exact agreement with legacy base-model evaluation")
    verify_legacy(ctx, cache, args.out)
    summaries = []
    for horizon in [10.0, 4.0, 21.5]:
        LOGGER.info("Decomposition and observation bounds H=%s", horizon)
        summaries.append(describe_horizon(ctx, cache, horizon, args.out, args.bootstrap))
        write_json(args.out / "decomposition_summary.json", summaries)
    robust = []
    for horizon in [10.0, 4.0]:
        LOGGER.info("Shared-cohort protocol comparison H=%s", horizon)
        robust.append(robust_comparison(ctx, cache, horizon, args.out, args.bootstrap))
        write_json(args.out / "robust_summary.json", robust)
    nested_cohort(ctx, cache, args.out)
    LOGGER.info("Checking bootstrap coverage in a declared synthetic setting")
    coverage_simulation(args.out)
    after = {f: sha256(args.source_root / f) for f in frozen_files}
    if after != before:
        raise AssertionError("An immutable input changed during the run")
    # Source predictions are hash-checked a second time; do not rely only on mtimes.
    if any(sha256(Path(item["answers"])) != item["sha256"] for item in provenance):
        raise AssertionError("A source prediction file changed during the run")
    status.update(
        finished=datetime.now(timezone.utc).isoformat(),
        complete=True,
        inputs_unchanged=True,
        prediction_hashes_rechecked=len(provenance),
    )
    write_json(args.out / "run_state.json", status)
    LOGGER.info("Finished without modifying source predictions or frozen records")


if __name__ == "__main__":
    main()
