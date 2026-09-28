"""Diagnostic G6 pipeline from per-video scores to seed-paired family comparisons.

Only synthetic or development inputs are accepted. This implementation does not
register an experiment or issue a scientific pass/fail. It propagates cluster
resampling within each track, conditional on the supplied trained seed ensemble.
The bootstrap/normal approximation for a median of seed differences still needs
scientific validation before a separately registered audit implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .g6 import (
    Evaluability,
    G6Config,
    PairOutcome,
    TrackReading,
    cross_track_family_key,
    g6_report,
    load_g6_config,
    match_families,
    pair_direction,
)
from .stats import cluster_bootstrap_indices, holm


def rmscd_contributions(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-video normalised trapezoidal RMSCD for a complete uniform H grid.

    Predictions include anchor and window endpoint. Abstention (-1) remains
    incorrect and never removes a row. Missing/off-grid entries must already be
    represented as abstentions by the corresponding native-axis evaluator.
    """
    pred = np.asarray(predictions)
    y = np.asarray(labels)
    if pred.ndim != 2 or pred.shape[1] < 2 or y.shape != (pred.shape[0],):
        raise ValueError("predictions must be N x (J+1), J>=1, with one label per row")
    if not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError("non-finite predictions or labels")
    if np.any(pred != np.floor(pred)) or np.any(y != np.floor(y)) or np.any(y < 0) or np.any(pred < -1):
        raise ValueError("integer class predictions and nonnegative labels required")
    correct = pred == y[:, None]
    stable = np.logical_and.accumulate(correct[:, ::-1], axis=1)[:, ::-1]
    integrand = 1.0 - stable.astype(float)
    return (integrand[:, :-1] + integrand[:, 1:]).sum(axis=1) / (2 * (pred.shape[1] - 1))


def _token(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty stripped string")
    return value


@dataclass
class TrackData:
    """Validated common cohort and all system scores for one track."""

    name: str
    role: str
    ids: list[str]
    clusters: np.ndarray
    classes: list[str]
    cards: dict[str, dict[str, Any]]
    scores: dict[str, np.ndarray]
    families: dict[str, dict[str, str]]
    provenance: dict[str, Any]


def validate_track(payload: dict[str, Any], cfg: G6Config) -> TrackData:
    """Reject incomplete, non-paired or audit data before computing any score."""
    role = payload.get("split_role")
    if role not in {"synthetic", "development"}:
        raise ValueError("diagnostic pipeline accepts only synthetic or development data")
    if payload.get("metric") != "RMSCD@H_norm" or cfg.primary_metric != "RMSCD@H_norm":
        raise ValueError("only per-video additive RMSCD@H_norm is implemented")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("cohort_sha256") or not provenance.get("protocol_id"):
        raise ValueError("explicit cohort and protocol provenance required")
    cards = payload.get("systems", {})
    if not isinstance(cards, dict) or not cards:
        raise ValueError("nonempty system cards required")
    families: dict[str, dict[str, str]] = {}
    family_rounds: dict[str, str] = {}
    for sid, card in cards.items():
        _token(sid, "system_id")
        if not isinstance(card, dict):
            raise ValueError("system card must be an object")
        for field in cfg.family_fields:
            if field == "commit_threshold":
                value = card.get(field)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                ):
                    raise ValueError("invalid commitment threshold")
            elif "|" in _token(card.get(field), field):
                raise ValueError("family field contains reserved separator")
        seed = _token(card.get("seed"), "seed")
        family = cross_track_family_key(card, cfg.family_fields)
        library_round = _token(card.get("library_round"), "library_round")
        if family in family_rounds and family_rounds[family] != library_round:
            raise ValueError("a within-track family mixes library rounds; select one explicit training recipe")
        family_rounds[family] = library_round
        group = families.setdefault(family, {})
        if seed in group:
            raise ValueError("duplicate recipe/seed; do not silently pool library rounds")
        group[seed] = sid
    observations: dict[str, dict[str, float]] = {sid: {} for sid in cards}
    cohort: dict[str, tuple[str, str]] = {}
    for row in payload.get("observations", []):
        sid = row["system_id"]
        if sid not in cards:
            raise ValueError("observation lacks a system card")
        vid = _token(row.get("video_id"), "video_id")
        metadata = (
            _token(row.get("source_cluster_id"), "source_cluster_id"),
            _token(row.get("class_code"), "class_code"),
        )
        if vid in observations[sid]:
            raise ValueError("duplicate system/video observation")
        if vid in cohort and cohort[vid] != metadata:
            raise ValueError("cluster or class differs between systems")
        value = row.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("normalised RMSCD contributions must be finite in [0,1]")
        cohort[vid] = metadata
        observations[sid][vid] = float(value)
    ids = sorted(cohort)
    if not ids or any(set(rows) != set(ids) for rows in observations.values()):
        raise ValueError("all systems must cover the identical nonempty cohort")
    if len({cohort[vid][0] for vid in ids}) < 2:
        raise ValueError("at least two source clusters required for resampling")
    return TrackData(
        _token(payload.get("name"), "track name"),
        role,
        ids,
        np.array([cohort[v][0] for v in ids]),
        [cohort[v][1] for v in ids],
        cards,
        {sid: np.array([rows[v] for v in ids]) for sid, rows in observations.items()},
        families,
        provenance,
    )


def system_resamples(track: TrackData, n_boot: int, seed: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Resample clusters as blocks; retain video weights when cluster sizes differ."""
    replicates = cluster_bootstrap_indices(track.clusters, n_boot, seed)
    point = {sid: float(values.mean()) for sid, values in track.scores.items()}
    samples = {sid: np.array([values[idx].mean() for idx in replicates]) for sid, values in track.scores.items()}
    return point, samples


def compare_family_pair(
    track: TrackData,
    family_a: str,
    family_b: str,
    point: dict[str, float],
    samples: dict[str, np.ndarray],
    min_seeds: int,
    alpha: float,
) -> dict[str, Any]:
    """Median of paired-seed differences, recomputed inside each cluster draw."""
    group_a, group_b = track.families.get(family_a, {}), track.families.get(family_b, {})
    shared = sorted(set(group_a) & set(group_b))
    base = {
        "family_a": family_a,
        "family_b": family_b,
        "shared_seeds": shared,
        "seeds_only_a": sorted(set(group_a) - set(group_b)),
        "seeds_only_b": sorted(set(group_b) - set(group_a)),
    }
    if len(shared) < min_seeds:
        return {**base, "usable": False, "reason": "insufficient paired seeds", "p_value": 1.0}
    difference = float(np.median([point[group_a[s]] - point[group_b[s]] for s in shared]))
    boot = np.median(np.stack([samples[group_a[s]] - samples[group_b[s]] for s in shared]), axis=0)
    se = float(np.std(boot, ddof=1))
    p = math.erfc(abs(difference) / (se * math.sqrt(2))) if se > 0 else (0.0 if difference else 1.0)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        **base,
        "usable": True,
        "diff": difference,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "bootstrap_se": se,
        "degenerate_resampling": se == 0,
        "p_value": p,
        "p_method": "diagnostic_normal_from_cluster_bootstrap_se",
        "interval_kind": "cluster_percentile_conditional_on_supplied_seed_ensemble",
    }


def run_pipeline(cfg: G6Config, input_a: dict[str, Any], input_b: dict[str, Any]) -> dict[str, Any]:
    """Run the complete diagnostic primary-metric path, with no audit verdict."""
    for value, floor in (
        (cfg.raw.get("direction", {}).get("n_boot", 1000), 20),
        (cfg.raw.get("family_matching", {}).get("min_seeds_per_family", 1), 2),
        (cfg.raw.get("direction", {}).get("seed", 20260920), 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < floor:
            raise ValueError("resample count, minimum seeds and RNG seed must be valid integers")
    if not 0 < cfg.alpha < 1:
        raise ValueError("require valid alpha, >=20 resamples and >=2 paired seeds")
    selection = cfg.raw.get("direction", {}).get("selection", {})
    if (
        selection.get("track", "a") != "a"
        or selection.get("correction", "holm") != "holm"
        or selection.get("require_significant", True) is not True
    ):
        raise ValueError("diagnostic selector implements track A, Holm and significant-only pairs")
    a = validate_track(input_a, cfg)
    b = validate_track(input_b, cfg)
    point_a, samples_a = system_resamples(a, cfg.n_boot, cfg.seed)
    # Selection and Holm's multiplicity family are determined solely by track A.
    tests_a = [
        compare_family_pair(a, x, y, point_a, samples_a, cfg.min_seeds_per_family, cfg.alpha)
        for x, y in combinations(sorted(a.families), 2)
    ]
    corrected = holm([t["p_value"] for t in tests_a], cfg.alpha)
    selected = []
    for test, adj, rejected in zip(tests_a, corrected["adjusted"], corrected["reject"]):
        test["p_adjusted"] = float(adj)
        test["selected_on_a"] = bool(test["usable"] and rejected and test["ci_lo"] * test["ci_hi"] > 0)
        if test["selected_on_a"]:
            selected.append(test)
    point_b, samples_b = system_resamples(b, cfg.n_boot, cfg.seed + 1)
    tests_b = []
    outcomes = []
    for test_a in selected:
        x, y = test_a["family_a"], test_a["family_b"]
        test_b = compare_family_pair(b, x, y, point_b, samples_b, cfg.min_seeds_per_family, cfg.alpha)
        tests_b.append(test_b)
        reading_a = TrackReading(test_a["diff"], test_a["ci_lo"], test_a["ci_hi"])
        reading_b = TrackReading(test_b["diff"], test_b["ci_lo"], test_b["ci_hi"]) if test_b["usable"] else None
        outcome, reason = pair_direction(reading_a, reading_b, tie_below_ruler=False)
        outcomes.append(
            PairOutcome(x, y, cfg.primary_metric, test_a["diff"], test_b.get("diff", float("nan")), outcome, reason)
        )
    matching = match_families(a.cards, b.cards, cfg.family_fields, cfg.min_seeds_per_family)
    gate = Evaluability(
        False,
        [
            "diagnostic-only pipeline; no registered audit",
            "source-cluster provenance and class mapping not certified by this program",
            "MRD rule and inferential calibration remain unregistered",
        ],
    )
    report = g6_report(cfg, outcomes, gate, matching)
    report.update(
        {
            "pipeline_version": "0.1-diagnostic",
            "execution_mode": "diagnostic_only",
            "formal_verdict": "not_evaluable",
            "is_experiment": False,
            "input_roles": {"a": a.role, "b": b.role},
            "pair_tests_a": tests_a,
            "pair_tests_b": tests_b,
            "selected_pairs_sha256": hashlib.sha256(
                json.dumps([(t["family_a"], t["family_b"]) for t in selected]).encode()
            ).hexdigest(),
            "track_a": {"videos": len(a.ids), "clusters": len(set(a.clusters)), "provenance": a.provenance},
            "track_b": {"videos": len(b.ids), "clusters": len(set(b.clusters)), "provenance": b.provenance},
            "system_cards": {"a": a.cards, "b": b.cards},
            "provenance_verification": "input hashes recorded; external provenance assertions not certified",
            "resampling": {
                "unit": "source_cluster_id",
                "estimand_weighting": "video-weighted",
                "seed_aggregation": "median of within-track paired-seed differences",
                "training_seed_uncertainty": "not resampled; conditional on supplied seeds",
                "n_boot": cfg.n_boot,
                "alpha": cfg.alpha,
                "mrd_used": False,
            },
        }
    )
    return report


def strict_json(value: Any) -> Any:
    """Write missing numeric values as null, never nonstandard JSON NaN."""
    if isinstance(value, dict):
        return {k: strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    """Read explicitly labelled diagnostic inputs and write a non-overwriting report."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "track-a", "track-b", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = load_g6_config(str(args.config))
    inputs = [json.loads(path.read_text(encoding="utf-8")) for path in (args.track_a, args.track_b)]
    report = run_pipeline(cfg, *inputs)
    report["input_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("a", args.track_a), ("b", args.track_b))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(strict_json(report), stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
