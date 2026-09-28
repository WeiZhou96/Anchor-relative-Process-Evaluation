"""G6 -- cross-track direction agreement: family matching, direction statistic, gates.

G6 asks one question: does a process difference the protocol reports between two
system families on the ACCIDENT track reproduce, with the same sign, on the MM-AU
track? It is a transfer check on the *sign* of a within-track difference, never a
comparison of the two tracks' metric levels -- the tracks have different label
spaces, different anchor semantics, and on MM-AU no known frame rate.

This module is the executable half of ``prereg/G6_draft_2026-09-20.yaml``. Three
properties are enforced in code rather than left to discipline:

**A verdict requires a registration.** While the config says ``registered:
false``, :func:`g6_report` computes and returns the whole direction statistic but
:func:`g6_verdict` refuses to convert it into a pass or a fail. There is no
default pass threshold anywhere in this file;
a universal sign-agreement null is not available; a pass threshold is a
pre-registration decision.

**Preconditions are structural, not advisory.** :func:`evaluability` returns the
list of unmet preconditions, and ``not evaluable`` is a distinct outcome from
``not passed``. A missing human mapping, an undefined cluster unit or an
unverified decode each block the gate on their own.

**Track B is never used to choose the pairs it is asked to confirm.** Pair
selection runs entirely on track A; :func:`select_pairs_on_track_a` takes no
track-B argument at all.

Ties are first-class. A pair whose track-B interval contains zero, or whose
track-B difference is smaller than that track's minimum resolvable difference, is
a tie: a difference too small to read, which is a failure to reproduce and not
evidence of sameness. Ties count in the registered primary denominator; the
tie-excluded rate is reported beside it and is explicitly secondary, because
reporting only that rate would let an unreadable difference look like a success.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml as _yaml
except Exception:  # pragma: no cover - environment guard
    _yaml = None

#: Per-pair outcomes. ``uncertain`` means the pair could not be evaluated on
#: track B at all, which is different from a tie.
OUTCOMES = ("agree", "reverse", "tie", "uncertain")

#: Default cross-track family fields: ``group_key`` minus ``library_round``.
DEFAULT_FAMILY_FIELDS = ("backbone", "model_kind", "arm_rule", "commit_threshold")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass
class G6Config:
    """The parsed G6 configuration plus the digest of the file it came from."""

    raw: Dict[str, object]
    path: Optional[str] = None
    sha256: Optional[str] = None
    registration_verified: bool = False

    @property
    def registered(self) -> bool:
        return self.raw.get("registered", False) is True

    @property
    def family_fields(self) -> Tuple[str, ...]:
        fields = dict(self.raw.get("family_matching", {})).get("fields")
        return tuple(fields) if fields else DEFAULT_FAMILY_FIELDS

    @property
    def min_seeds_per_family(self) -> int:
        return int(dict(self.raw.get("family_matching", {})).get("min_seeds_per_family", 1))

    @property
    def primary_metric(self) -> str:
        return str(dict(self.raw.get("metrics", {})).get("primary", "RMSCD@H_norm"))

    @property
    def alpha(self) -> float:
        sel = dict(dict(self.raw.get("direction", {})).get("selection", {}))
        return float(sel.get("alpha", 0.05))

    @property
    def pass_threshold(self) -> Optional[float]:
        value = dict(self.raw.get("direction", {})).get("pass_threshold")
        return None if value is None else float(value)

    @property
    def null_reference(self) -> Optional[float]:
        return None  # No universal coin-flip null is assumed for dependent system pairs.

    @property
    def n_boot(self) -> int:
        return int(dict(self.raw.get("direction", {})).get("n_boot", 1000))

    @property
    def seed(self) -> int:
        return int(dict(self.raw.get("direction", {})).get("seed", 20260920))

    @property
    def fallback_mode(self) -> str:
        return str(dict(self.raw.get("mapping_fallback", {})).get("primary", "restricted_closed_set"))

    @property
    def evaluability_cfg(self) -> Dict[str, object]:
        return dict(self.raw.get("evaluability", {}))


def load_g6_config(path: str, registration_path: Optional[str] = None) -> G6Config:
    """Read the G6 config and record the digest of the exact bytes read.

    The digest is what a later registration must pin. Recording it at load time
    means a report can always say which text of the protocol produced it, even if
    the file is edited afterwards.
    """
    if _yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read the G6 config")
    with open(path, "rb") as fh:
        blob = fh.read()
    raw = _yaml.safe_load(blob.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: G6 config must be a mapping")
    digest = hashlib.sha256(blob).hexdigest()
    verified = False
    if registration_path is not None:
        with open(registration_path, encoding="utf-8") as receipt_file:
            receipt = json.load(receipt_file)
        verified = (
            receipt.get("config_sha256") == digest
            and receipt.get("development_split") in {"train", "dev", "val", "validation"}
            and bool(receipt.get("registered_at"))
            and receipt.get("registered_before_first_audit") is True
        )
        if not verified:
            raise ValueError("registration receipt does not pin this config and development provenance")
    return G6Config(raw=raw, path=path, sha256=digest, registration_verified=verified)


# --------------------------------------------------------------------------
# family matching
# --------------------------------------------------------------------------
def cross_track_family_key(meta: Dict[str, object], fields: Sequence[str] = DEFAULT_FAMILY_FIELDS) -> str:
    """Family key of one system, from its card metadata.

    Deliberately excludes the seed -- seeds are replicates inside a family -- and
    excludes ``library_round``, which is within-track provenance of the ACCIDENT
    library and would make every cross-track family fail to match. Both exclusions
    are recorded in the config's ``recorded_not_matched`` list so that a reader
    sees what was set aside.
    """
    missing = [f for f in fields if f not in meta]
    if missing:
        raise ValueError(f"system metadata is missing family fields {missing}")
    return "|".join(str(meta[f]) for f in fields)


@dataclass
class FamilyMatch:
    """One family present on both tracks, with its per-track seed counts."""

    family_key: str
    systems_a: List[str]
    systems_b: List[str]
    seeds_a: List[str]
    seeds_b: List[str]

    @property
    def n_seeds_a(self) -> int:
        return len({s for s in self.seeds_a if s not in {"None", "nan", ""}})

    @property
    def n_seeds_b(self) -> int:
        return len({s for s in self.seeds_b if s not in {"None", "nan", ""}})


def match_families(
    meta_a: Dict[str, Dict[str, object]],
    meta_b: Dict[str, Dict[str, object]],
    fields: Sequence[str] = DEFAULT_FAMILY_FIELDS,
    min_seeds_per_family: int = 1,
) -> Dict[str, object]:
    """Match system families across two tracks, reporting both sides' misses.

    The report is symmetric on purpose: listing only the track-A families that
    failed to find a partner would hide a track-B library that trained recipes
    the audit never asked for.
    """
    groups_a: Dict[str, List[str]] = {}
    groups_b: Dict[str, List[str]] = {}
    for meta, groups in ((meta_a, groups_a), (meta_b, groups_b)):
        for sid, m in meta.items():
            groups.setdefault(cross_track_family_key(m, fields), []).append(sid)

    matched: List[FamilyMatch] = []
    underpowered: List[Dict[str, object]] = []
    for key in sorted(set(groups_a) & set(groups_b)):
        fm = FamilyMatch(
            family_key=key,
            systems_a=sorted(groups_a[key]),
            systems_b=sorted(groups_b[key]),
            seeds_a=[str(meta_a[s].get("seed")) for s in sorted(groups_a[key])],
            seeds_b=[str(meta_b[s].get("seed")) for s in sorted(groups_b[key])],
        )
        if fm.n_seeds_a >= int(min_seeds_per_family) and fm.n_seeds_b >= int(min_seeds_per_family):
            matched.append(fm)
        else:
            underpowered.append(
                {
                    "family_key": key,
                    "n_seeds_a": fm.n_seeds_a,
                    "n_seeds_b": fm.n_seeds_b,
                    "min_required": int(min_seeds_per_family),
                }
            )
    return {
        "family_fields": list(fields),
        "matched": matched,
        "n_matched": len(matched),
        "underpowered": underpowered,
        "only_on_track_a": sorted(set(groups_a) - set(groups_b)),
        "only_on_track_b": sorted(set(groups_b) - set(groups_a)),
        "n_families_a": len(groups_a),
        "n_families_b": len(groups_b),
    }


# --------------------------------------------------------------------------
# per-pair direction
# --------------------------------------------------------------------------
@dataclass
class TrackReading:
    """One family's reading of one metric on one track."""

    diff: float
    ci_lo: float
    ci_hi: float
    ruler: float = float("nan")  # minimum resolvable difference on this track

    @property
    def ci_contains_zero(self) -> bool:
        if not np.isfinite(self.ci_lo) or not np.isfinite(self.ci_hi):
            return True
        return bool(self.ci_lo <= 0.0 <= self.ci_hi)

    @property
    def below_ruler(self) -> bool:
        if not np.isfinite(self.ruler):
            return False
        return bool(abs(self.diff) < self.ruler)


@dataclass
class PairOutcome:
    family_a: str
    family_b: str
    metric: str
    diff_a: float
    diff_b: float
    outcome: str
    reason: str


def pair_direction(
    reading_a: Optional[TrackReading],
    reading_b: Optional[TrackReading],
    tie_on_ci: bool = True,
    tie_below_ruler: bool = True,
) -> Tuple[str, str]:
    """Outcome of one selected pair, as ``(outcome, reason)``.

    ``uncertain`` is returned when track B has no usable reading -- a family that
    was never evaluated there, or a non-finite difference. It is kept separate
    from ``tie`` because the two have different causes: a tie is a real reading
    that is too small to resolve, while ``uncertain`` means there is no reading.
    """
    if reading_a is None or not np.isfinite(reading_a.diff) or reading_a.diff == 0.0:
        return "uncertain", "track A difference missing, non-finite or exactly zero"
    if reading_b is None or not np.isfinite(reading_b.diff):
        return "uncertain", "no usable track B reading"
    if tie_on_ci and (
        not np.isfinite(reading_b.ci_lo) or not np.isfinite(reading_b.ci_hi) or reading_b.ci_lo > reading_b.ci_hi
    ):
        return "uncertain", "track B uncertainty interval unavailable or invalid"
    if tie_below_ruler and np.isfinite(reading_b.ruler) and reading_b.ruler < 0:
        raise ValueError("ruler must be nonnegative")
    if reading_b.diff == 0:
        return "tie", "track B difference exactly zero"
    if tie_on_ci and reading_b.ci_contains_zero:
        return "tie", "track B interval contains zero"
    if tie_below_ruler and reading_b.below_ruler:
        return "tie", "track B difference is below that track's minimum resolvable difference"
    same_sign = (reading_a.diff > 0.0) == (reading_b.diff > 0.0)
    return ("agree", "same sign on both tracks") if same_sign else ("reverse", "opposite sign on track B")


def select_pairs_on_track_a(pair_tests: Sequence[object], require_significant: bool = True) -> List[Tuple[str, str]]:
    """Pairs entering G6, chosen from track A alone.

    Accepts the ``PairTest`` records the calibration layer already produces. This
    function takes no track-B argument: track B must not participate in choosing
    which differences it is then asked to reproduce.
    """
    out: List[Tuple[str, str]] = []
    for test in pair_tests:
        significant = bool(getattr(test, "significant", False))
        if require_significant and not significant:
            continue
        out.append((str(getattr(test, "system_a")), str(getattr(test, "system_b"))))
    return sorted(set(out))


# --------------------------------------------------------------------------
# aggregate direction statistic
# --------------------------------------------------------------------------
def _families_of(outcomes: Sequence[PairOutcome]) -> List[str]:
    keys = set()
    for o in outcomes:
        keys.add(o.family_a)
        keys.add(o.family_b)
    return sorted(keys)


def agreement_rates(outcomes: Sequence[PairOutcome]) -> Dict[str, object]:
    """Counts and both registered agreement rates.

    ``primary`` keeps ties in the denominator: a difference the protocol cannot
    read on track B has not reproduced. ``secondary`` drops them, and is reported
    beside the primary rather than instead of it.
    """
    counts = {name: 0 for name in OUTCOMES}
    for o in outcomes:
        counts[o.outcome] += 1
    evaluated = counts["agree"] + counts["reverse"] + counts["tie"]
    decided = counts["agree"] + counts["reverse"]
    return {
        "counts": counts,
        "n_selected": len(outcomes),
        "n_evaluated_excluding_uncertain": evaluated,
        "n_decided_excluding_ties": decided,
        "primary_rate": (float(counts["agree"]) / evaluated if evaluated else float("nan")),
        "primary_denominator": "selected_pairs_excluding_uncertain",
        "evaluation_coverage": evaluated / len(outcomes) if outcomes else float("nan"),
        "agreement_over_all_selected": counts["agree"] / len(outcomes) if outcomes else float("nan"),
        "secondary_rate": (float(counts["agree"]) / decided if decided else float("nan")),
        "secondary_denominator": "selected_pairs_excluding_uncertain_and_ties",
    }


def bootstrap_agreement_ci(
    outcomes: Sequence[PairOutcome],
    n_boot: int = 1000,
    seed: int = 20260920,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Exploratory family-weight sensitivity interval, not a calibrated CI.

    Positive family weights induce dependent weights on shared pairs. This is a
    diagnostic over the observed system-family graph; nominal frequentist
    coverage is unvalidated and it does not propagate video/cluster sampling
    uncertainty. A hub common to every pair cancels from the ratio. No claim
    that shared-family graphs must yield wider intervals is made.
    """
    if n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("n_boot must be positive and alpha must be in (0, 1)")
    families = _families_of(outcomes)
    usable = [o for o in outcomes if o.outcome != "uncertain"]
    if not families or not usable:
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0, "resample_unit": "family"}
    index: Dict[str, int] = {f: i for i, f in enumerate(families)}
    rows_a = np.array([index[o.family_a] for o in usable], dtype=np.int64)
    rows_b = np.array([index[o.family_b] for o in usable], dtype=np.int64)
    is_agree = np.array([o.outcome == "agree" for o in usable], dtype=float)

    rng = np.random.default_rng(int(seed))
    # Dirichlet(1,...,1) weights, scaled to mean 1 so that a replicate's total
    # weight matches the observed pair count.
    draws = rng.gamma(shape=1.0, scale=1.0, size=(int(n_boot), len(families)))
    draws /= draws.sum(axis=1, keepdims=True)
    draws *= len(families)

    pair_weights = draws[:, rows_a] * draws[:, rows_b]
    evaluated = pair_weights.sum(axis=1)
    agree = (pair_weights * is_agree[None, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(evaluated > 0.0, agree / evaluated, np.nan)
    rates = rates[np.isfinite(rates)]
    if rates.size == 0:
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0, "resample_unit": "family"}
    return {
        "ci_lo": float(np.percentile(rates, 100.0 * alpha / 2.0)),
        "ci_hi": float(np.percentile(rates, 100.0 * (1.0 - alpha / 2.0))),
        "n_boot": int(rates.size),
        "n_families": len(families),
        "n_pairs_used": len(usable),
        "resample_unit": "family",
        "interval_kind": "exploratory_family_reweighting",
        "nominal_coverage_validated": False,
        "weight_scheme": "dirichlet_multiplier",
    }


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
@dataclass
class Evaluability:
    """Whether G6 can be computed at all, and what is missing if not."""

    evaluable: bool
    unmet: List[str] = field(default_factory=list)
    checked: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def evaluability(
    cfg: G6Config,
    human_mapping_returned: bool,
    source_clusters_defined: bool,
    decode_verified: bool,
    n_populated_classes_track_b: int,
    n_matched_families: int,
    n_selected_pairs: int,
    n_clusters_track_b: Optional[int] = None,
    class_counts_track_b: Optional[Dict[str, int]] = None,
) -> Evaluability:
    """Check the structural preconditions of G6.

    ``not evaluable`` is not ``not passed``. A caller that turns an unmet
    precondition into a failed gate would be reporting a scientific result where
    there is only a missing input, so the two are returned as different things.
    """
    cfgv = cfg.evaluability_cfg
    unmet: List[str] = []
    checked: List[str] = []

    def _need(flag_name: str, ok: bool, message: str) -> None:
        if bool(cfgv.get(flag_name, False)):
            checked.append(flag_name)
            if not ok:
                unmet.append(message)

    _need("require_human_mapping", human_mapping_returned, "B8 two-person human mapping not returned")
    _need("require_source_clusters", source_clusters_defined, "MM-AU source-video clusters not established")
    _need("require_decode_verified", decode_verified, "MM-AU cohort not decode-verified")
    _need(
        "require_registered_config",
        cfg.registered and cfg.registration_verified,
        "G6 config is a draft or has no verified external registration receipt",
    )

    min_classes = cfgv.get("min_populated_classes_track_b")
    if min_classes is not None:
        checked.append("min_populated_classes_track_b")
        if int(n_populated_classes_track_b) < int(min_classes):
            unmet.append(
                f"track B has {n_populated_classes_track_b} populated classes, "
                f"fewer than the required {int(min_classes)}"
            )

    for name, value in (
        ("min_matched_families", n_matched_families),
        ("min_selected_pairs", n_selected_pairs),
        ("min_clusters_track_b", n_clusters_track_b),
    ):
        floor = cfgv.get(name)
        if floor is None:
            unmet.append(f"{name} is unregistered; a sample-size floor cannot be chosen after seeing the audit")
            continue
        checked.append(name)
        if value is None or int(value) < int(floor):
            unmet.append(f"{name}: got {value}, need at least {int(floor)}")

    floor = cfgv.get("min_videos_per_class_track_b")
    track_b = dict(dict(cfg.raw.get("tracks", {})).get("b", {}))
    classes = track_b.get("closed_set")
    for name in ("h_f", "delta_f", "grid_max_f", "closed_set", "split_source"):
        if track_b.get(name) in (None, "", []):
            unmet.append(f"track B {name} is unregistered")
    if floor is None:
        unmet.append("min_videos_per_class_track_b is unregistered")
    elif class_counts_track_b is None or not classes:
        unmet.append("per-class cohort counts or closed set missing")
    else:
        checked.append("min_videos_per_class_track_b")
        if int(floor) != floor or floor < 1:
            unmet.append("invalid per-class floor")
        elif any(class_counts_track_b.get(c, 0) < floor for c in classes):
            unmet.append("at least one registered class is below its sample floor")

    return Evaluability(evaluable=not unmet, unmet=unmet, checked=checked)


class G6NotRegisteredError(RuntimeError):
    """Raised when a verdict is requested from an unregistered G6 configuration."""


def configuration_issues(cfg: G6Config) -> List[str]:
    """Required static protocol fields, checked again at the verdict boundary."""
    issues = []
    b = dict(dict(cfg.raw.get("tracks", {})).get("b", {}))
    for key in ("h_f", "delta_f", "grid_max_f"):
        value = b.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or int(value) != value or value < 1:
            issues.append(f"track B {key} must be a registered positive integer")
    if not issues and (b["h_f"] % b["delta_f"] or b["h_f"] > b["grid_max_f"]):
        issues.append("track B grid does not include its common horizon")
    classes = b.get("closed_set")
    if not isinstance(classes, list) or len(set(classes)) != len(classes) or len(classes) < 2:
        issues.append("track B closed set must contain at least two distinct classes")
    if not b.get("split_source"):
        issues.append("track B split provenance missing")
    for key in ("min_matched_families", "min_selected_pairs", "min_clusters_track_b", "min_videos_per_class_track_b"):
        value = cfg.evaluability_cfg.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or int(value) != value or value < 1:
            issues.append(f"{key} must be a registered positive integer")
    return issues


def g6_verdict(cfg: G6Config, rates: Dict[str, object], gate: Evaluability) -> Dict[str, object]:
    """Convert the direction statistic into a verdict, or refuse to.

    Refuses in two situations, and they are reported differently: an unmet
    precondition yields ``not_evaluable``, while a missing pass threshold raises,
    because computing a pass against a threshold invented at report time is the
    one failure mode this function exists to prevent.
    """
    if not gate.evaluable:
        return {
            "verdict": "not_evaluable",
            "is_failure": False,
            "unmet_preconditions": gate.unmet,
            "note": "not evaluable is not the same as not passed; this is a missing input, not a result",
        }
    if not cfg.registered:
        raise G6NotRegisteredError("G6 configuration is not registered")
    if not cfg.registration_verified:
        return {
            "verdict": "not_evaluable",
            "is_failure": False,
            "unmet_preconditions": ["external registration receipt not verified"],
        }
    config_issues = configuration_issues(cfg)
    if config_issues:
        return {"verdict": "not_evaluable", "is_failure": False, "unmet_preconditions": config_issues}
    threshold = cfg.pass_threshold
    if threshold is None:
        raise G6NotRegisteredError("G6 has no registered pass threshold; no universal 0.5 null is assumed.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("pass threshold must be finite and in [0, 1]")
    rate = float(rates["primary_rate"])
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        return {
            "verdict": "not_evaluable",
            "is_failure": False,
            "unmet_preconditions": ["finite agreement rate unavailable"],
        }
    return {
        "verdict": "pass" if rate >= threshold else "fail",
        "is_failure": rate < threshold,
        "primary_rate": rate,
        "pass_threshold": threshold,
        "null_reference": cfg.null_reference,
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def g6_report(
    cfg: G6Config,
    outcomes: Sequence[PairOutcome],
    gate: Evaluability,
    family_report: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """The full G6 report: statistic, interval, gate state and provenance.

    Always computes and returns the statistic, including when the gate is unmet,
    so that a draft run is informative; the gate state travels with the numbers
    so they cannot be quoted as a finished result.
    """
    rates = agreement_rates(outcomes)
    ci = bootstrap_agreement_ci(outcomes, n_boot=cfg.n_boot, seed=cfg.seed, alpha=cfg.alpha)
    payload: Dict[str, object] = {
        "g6_version": str(cfg.raw.get("g6_version", "unknown")),
        "config_path": cfg.path,
        "config_sha256": cfg.sha256,
        "registered": cfg.registered,
        "registration_receipt_verified": cfg.registration_verified,
        "primary_metric": cfg.primary_metric,
        "fallback_mode": cfg.fallback_mode,
        "family_fields": list(cfg.family_fields),
        "rates": rates,
        "exploratory_agreement_interval": ci,
        "null_reference": cfg.null_reference,
        "pass_threshold": cfg.pass_threshold,
        "evaluability": gate.as_dict(),
        "outcomes": [asdict(o) for o in outcomes],
    }
    if family_report is not None:
        payload["family_matching"] = {k: v for k, v in family_report.items() if k != "matched"}
        payload["family_matching"]["matched_keys"] = [m.family_key for m in family_report.get("matched", [])]
    if not cfg.registered:
        payload["status_note"] = (
            "Draft configuration: the statistic below is a dry run. No pass or fail is "
            "derived from it, and it must not be quoted as a G6 result."
        )
    return payload


def write_g6_report(payload: Dict[str, object], path: str) -> str:
    """Write the report as JSON, with the draft note preserved."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return path
