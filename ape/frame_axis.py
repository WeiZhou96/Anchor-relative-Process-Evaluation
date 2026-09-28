"""Native frame-axis adaptation of the APE protocol layer.

Why a separate axis instead of a unit conversion
------------------------------------------------
The MM-AU release carries no FPS. Every temporal field it publishes -- ``t_co``,
``t_ai``, ``t_ae``, ``total_frames`` -- is a frame index, and the official
metadata has no field from which seconds could be recovered (checked on
``official_metadata/video_metadata.json``: the 18 fields are frame counts,
indices and free text). Two shortcuts are therefore ruled out:

* writing frame counts into the seconds columns ``anchor_s`` / ``duration_s`` /
  ``post_anchor_length_s``, which would make every downstream number a
  seconds-labelled quantity that is not in seconds; and
* assuming a nominal FPS to convert, which would smuggle an unverified constant
  into the cohort definition and into ``RMSCD``.

This module is the third option: the protocol runs on the frame axis in its own
units, with its own protocol vector and its own hash namespace, and the seconds
axis is left exactly as it was. Nothing here imports or mutates the frozen
seconds artefacts; :mod:`ape.protocol` and :mod:`ape.cohort` are untouched.

What is shared and what is not
------------------------------
*Shared.* The metric primitives in :mod:`ape.metrics` -- ``correctness``,
``stable_correct``, ``s_curve``, ``trapezoid``, ``flip_counts``,
``commit_stats`` -- are dimensionless: they consume an indicator matrix and a
scalar step. They are reused verbatim, so a frame-axis ``S_H`` is computed by the
same code path as a seconds-axis one and cannot drift from it.

*Not shared.* The protocol vector, the eligibility cohort, the prefix grid and
the answer lookup are reimplemented in integer arithmetic. Three behavioural
differences are consequences of the axis, not choices:

1. **Eligibility is exact.** The seconds axis needs a ``TOL`` of 1e-9 s because
   ``duration_s - anchor_s`` is binary floating point. Post-anchor length in
   frames is ``last_frame - anchor_frame``, an integer, so ``L+ >= H`` is decided
   exactly and no tolerance is used or wanted here.
2. **Lookup misses are detected, not rounded.** The seconds axis resolves a
   perturbed end time to a cached column with ``rint``, which silently absorbs
   an off-grid request. On the frame axis the division is exact or it is not:
   an off-grid request is reported in ``n_offgrid_lookup`` and scored as
   ``BOT``/wrong, never rounded to a neighbour.
3. **Sub-frame perturbation is not representable.** A common-mode shift must be a
   whole number of frames; a jitter standard deviation may be fractional but the
   realised shift is rounded to whole frames, and the rounding is reported in
   ``jitter_quantization_frames``. A jitter with ``sd << 1`` frame is therefore
   mostly invisible, which is the frame-axis form of the cached-grid limitation
   already documented for the seconds axis.

Discipline carried over unchanged
---------------------------------
Prefixes are strictly causal and never read past ``end_frame``; the cohort is
fixed before any metric is computed and is identical for every system and grid
point; videos that are never correct and systems that never commit stay in the
denominator; ``BOT`` at ``delta >= 0`` is wrong rather than an abstention.

Normalisation for display divides by the **common** horizon ``H`` of the track.
It never divides by each clip's own remaining length, which would replace one
fixed observation window by a per-clip one and make the curve uncomparable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import (
    AnswerTable,
    commit_stats,
    correctness,
    first_stable_index,
    flip_counts,
    macro_accuracy,
    micro_accuracy,
    s_curve,
    s_curve_macro,
    stable_correct,
    trapezoid,
)
from .vocabulary import ACCIDENT_VOCABULARY, TaskVocabulary

#: Columns a frame-axis manifest must provide.
FRAME_MANIFEST_REQUIRED = [
    "video_id",
    "source_cluster_id",
    "first_frame",
    "anchor_frame",
    "last_frame",
    "post_anchor_frames",
    "class_code",
]

LENGTH_COL = "post_anchor_frames"
LENGTH_COL_EFF = "post_anchor_frames_eff"

#: The only split a frame-axis audit number may be computed on.
AUDIT_SPLIT = "test"


def _integer(value: object, name: str, minimum: Optional[int] = None) -> int:
    """Accept finite whole values only; never truncate a protocol input."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a whole number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a whole number") from exc
    if not np.isfinite(number) or number != np.floor(number):
        raise ValueError(f"{name} must be a finite whole number")
    result = int(number)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _integer_column(values: pd.Series, name: str, minimum: Optional[int] = None) -> np.ndarray:
    return np.asarray([_integer(v, name, minimum) for v in values], dtype=np.int64)


def h_label_f(h_f: int) -> str:
    """Column-name label for a frame horizon: ``40 -> 'H40f'``."""
    return f"H{_integer(h_f, 'h_f', 1)}f"


def eligible_col_f(h_f: int) -> str:
    return f"eligible_{h_label_f(h_f)}"


def n_grid_points_f(span_f: int, delta_f: int) -> int:
    """Number of whole steps of ``delta_f`` fitting in ``span_f``; grid is ``0..J``."""
    return _integer(span_f, "span_f", 0) // _integer(delta_f, "delta_f", 1)


# --------------------------------------------------------------------------
# protocol vector
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameProtocolVector:
    """One point of the protocol space on the frame axis.

    ``axis_unit`` is part of the hashed payload, so a frame-axis vector can never
    collide with a seconds-axis vector that happens to carry the same numbers.
    That matters because both axes write artefacts keyed by ``pi_hash``.

    Attributes
    ----------
    eps_sys_f : common-mode anchor shift, a whole number of frames. A fractional
        common-mode shift is rejected rather than rounded: it is not a property
        the frame axis can represent without resampling the video.
    eps_jit_sd_f : standard deviation, in frames, of the per-video independent
        jitter. May be fractional; the realised shift is rounded to whole frames.
    delta_f : prefix step, a positive whole number of frames.
    h_f : observation horizon in frames; fixes both the eligibility cohort and
        the censoring bound.
    """

    eps_sys_f: int = 0
    eps_jit_sd_f: float = 0.0
    delta_f: int = 1
    h_f: int = 30
    jit_seed: int = 20260920
    pre_anchor_outputs_bot: bool = True
    protocol_version: str = "0.1-frame-unfrozen"
    axis_unit: str = "frame"
    vocabulary_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.vocabulary_hash, str) or (
            self.vocabulary_hash
            and (len(self.vocabulary_hash) != 64 or any(c not in "0123456789abcdef" for c in self.vocabulary_hash))
        ):
            raise ValueError("vocabulary_hash must be empty or a SHA-256 hex digest")
        if self.axis_unit != "frame":
            raise ValueError(f"axis_unit must be 'frame', got {self.axis_unit!r}")
        for name, minimum in (("eps_sys_f", None), ("delta_f", 1), ("h_f", 1), ("jit_seed", 0)):
            _integer(getattr(self, name), name, minimum)
        if not np.isfinite(float(self.eps_jit_sd_f)) or float(self.eps_jit_sd_f) < 0:
            raise ValueError("eps_jit_sd_f must be finite and >= 0")
        if self.h_f % self.delta_f:
            raise ValueError("h_f must be divisible by delta_f so every grid includes the common horizon")

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        if not d["vocabulary_hash"]:
            del d["vocabulary_hash"]  # Preserve legacy frame protocol hashes.
        d["eps_sys_f"] = int(d["eps_sys_f"])
        d["delta_f"] = int(d["delta_f"])
        d["h_f"] = int(d["h_f"])
        d["eps_jit_sd_f"] = float(round(float(d["eps_jit_sd_f"]), 9)) + 0.0
        return d

    @property
    def pi_hash(self) -> str:
        """Stable 12-hex-char hash of the frame-axis protocol vector."""
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=6).hexdigest()

    def replace(self, **kw) -> "FrameProtocolVector":
        d = asdict(self)
        d.update(kw)
        return FrameProtocolVector(**d)

    def label(self) -> str:
        return (
            f"eps{int(self.eps_sys_f):+d}f_jit{float(self.eps_jit_sd_f):g}f"
            f"_d{int(self.delta_f)}f_H{int(self.h_f)}f"
        )


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def _check_vocabulary_metadata(df: pd.DataFrame, vocabulary: TaskVocabulary) -> None:
    if "vocabulary_hash" in df:
        if df["vocabulary_hash"].isna().any() or not df["vocabulary_hash"].eq(vocabulary.vocabulary_hash).all():
            raise ValueError("table vocabulary hash mismatch")
    elif vocabulary != ACCIDENT_VOCABULARY:
        raise ValueError("custom task tables require a vocabulary_hash column")


def _check_protocol_vocabulary(pv: FrameProtocolVector, vocabulary: TaskVocabulary) -> None:
    if pv.vocabulary_hash != vocabulary.vocabulary_hash:
        if vocabulary != ACCIDENT_VOCABULARY or pv.vocabulary_hash:
            raise ValueError("protocol vocabulary hash mismatch")


def validate_frame_manifest(df: pd.DataFrame, vocabulary: TaskVocabulary = ACCIDENT_VOCABULARY) -> pd.DataFrame:
    """Type-check and self-consistency-check a frame-axis manifest.

    Refuses a manifest that carries seconds columns for the temporal fields, so
    that a seconds manifest cannot be fed to the frame axis by accident, and
    refuses one whose ``post_anchor_frames`` disagrees with its own endpoints.
    """
    missing = [c for c in FRAME_MANIFEST_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"frame manifest is missing columns {missing}")
    for forbidden in ("anchor_s", "duration_s", "post_anchor_length_s"):
        if forbidden in df.columns:
            raise ValueError(
                f"frame manifest carries the seconds column {forbidden!r}. The frame axis "
                "does not consume seconds, and a frame count stored in a seconds column is "
                "the specific mistake this axis exists to prevent."
            )
    _check_vocabulary_metadata(df, vocabulary)
    out = df.copy()
    for col in ("video_id", "source_cluster_id"):
        if (
            out[col].isna().any()
            or out[col].astype(str).str.strip().str.lower().isin(["", "nan", "none", "null"]).any()
        ):
            raise ValueError(f"{col} must contain nonempty defined identifiers")
        out[col] = out[col].astype(str)
    for col in ("first_frame", "anchor_frame", "last_frame", "post_anchor_frames"):
        out[col] = _integer_column(out[col], col, 0)
    out["class_code"] = _integer_column(out["class_code"], "class_code", 0)
    if (out["class_code"] >= vocabulary.n_classes).any():
        raise ValueError("class_code outside the closed set")
    if "native_class" in out.columns and vocabulary != ACCIDENT_VOCABULARY:
        native = _integer_column(out["native_class"], "native_class", 0)
        encoded = np.asarray([vocabulary.encode(int(code)) for code in native])
        if not np.array_equal(encoded, out["class_code"].to_numpy()):
            raise ValueError("native_class and class_code disagree under vocabulary")
    if out["video_id"].duplicated().any():
        dup = out.loc[out["video_id"].duplicated(), "video_id"].tolist()[:5]
        raise ValueError(f"duplicated video_id, e.g. {dup}")

    derived = out["last_frame"] - out["anchor_frame"]
    bad = derived.to_numpy() != out[LENGTH_COL].to_numpy()
    if bool(bad.any()):
        raise ValueError(f"post_anchor_frames != last_frame - anchor_frame for {int(bad.sum())} rows")
    off = (out["anchor_frame"] < out["first_frame"]) | (out["anchor_frame"] > out["last_frame"])
    if bool(off.any()):
        raise ValueError(f"anchor_frame outside [first_frame, last_frame] for {int(off.sum())} rows")
    return out.reset_index(drop=True)


def load_frame_manifest(path: str, vocabulary: TaskVocabulary = ACCIDENT_VOCABULARY) -> pd.DataFrame:
    """Read and validate a frame-axis manifest CSV."""
    return validate_frame_manifest(pd.read_csv(path), vocabulary)


def select_split(df: pd.DataFrame, split: str = AUDIT_SPLIT, split_col: str = "split") -> pd.DataFrame:
    """Restrict to one split before any cohort is formed (see ``ape.cohort``)."""
    if split in (None, "all"):
        return df.reset_index(drop=True)
    if split_col not in df.columns:
        raise ValueError(
            f"frame manifest has no {split_col!r} column; cannot restrict to split {split!r}. "
            "An audit must not silently run on every clip."
        )
    sub = df.loc[df[split_col].astype(str) == str(split)]
    if len(sub) == 0:
        available = sorted(df[split_col].astype(str).unique())
        raise ValueError(f"split {split!r} is empty; available: {available}")
    return sub.reset_index(drop=True)


# --------------------------------------------------------------------------
# perturbation
# --------------------------------------------------------------------------
def jitter_draws_frames(video_ids: Sequence[str], sd_f: float, jit_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-video independent jitter, as ``(raw_draw, whole_frames)``.

    Keyed by ``(video_id, jit_seed)`` through a hash, exactly as on the seconds
    axis, so the same video receives the same jitter in every cohort regardless
    of row order. Both the continuous draw and its rounding to whole frames are
    returned so that the quantisation can be reported instead of assumed away.
    """
    sd = float(sd_f)
    raw = np.zeros(len(video_ids), dtype=float)
    if sd > 0.0:
        for k, vid in enumerate(video_ids):
            key = f"{jit_seed}|{vid}".encode("utf-8")
            digest = hashlib.blake2b(key, digest_size=8).digest()
            seed = int.from_bytes(digest, "little", signed=False)
            raw[k] = np.random.default_rng(seed).normal(0.0, sd)
    return raw, np.rint(raw).astype(np.int64)


def perturb_frame_manifest(manifest: pd.DataFrame, pv: FrameProtocolVector) -> pd.DataFrame:
    """Apply ``pi``'s anchor perturbations on the frame axis.

    Adds ``jitter_raw_f``, ``jitter_f``, ``anchor_frame_eff``, ``anchor_clipped``
    and ``post_anchor_frames_eff``. As on the seconds axis the cohort is then
    recomputed from the *effective* length, which is the second-order mechanism
    a shift acts through: it moves both the measured delay and the cohort
    membership.
    """
    df = manifest.copy()
    raw, jit = jitter_draws_frames(df["video_id"].tolist(), pv.eps_jit_sd_f, pv.jit_seed)
    anchor = df["anchor_frame"].to_numpy(dtype=np.int64)
    first = df["first_frame"].to_numpy(dtype=np.int64)
    last = df["last_frame"].to_numpy(dtype=np.int64)
    requested = anchor + int(pv.eps_sys_f) + jit
    eff = np.clip(requested, first, last)
    df["jitter_raw_f"] = raw
    df["jitter_f"] = jit
    df["anchor_frame_eff"] = eff
    df["anchor_clipped"] = eff != requested
    df[LENGTH_COL_EFF] = last - eff
    return df


# --------------------------------------------------------------------------
# eligibility cohort
# --------------------------------------------------------------------------
def _length_column(manifest: pd.DataFrame) -> str:
    return LENGTH_COL_EFF if LENGTH_COL_EFF in manifest.columns else LENGTH_COL


def frame_eligible_mask(manifest: pd.DataFrame, h_f: int) -> np.ndarray:
    """Boolean mask of ``E_H`` on the frame axis.

    Exact integer comparison. The seconds axis carries a 1e-9 s tolerance to stop
    the cohort from depending on binary floating point; here both sides are
    integers, so a tolerance would have nothing to protect against and would only
    widen the cohort by an amount nobody registered.
    """
    col = _length_column(manifest)
    return manifest[col].to_numpy(dtype=np.int64) >= _integer(h_f, "h_f", 1)


def frame_cohort(manifest: pd.DataFrame, h_f: int) -> pd.DataFrame:
    """Rows of ``manifest`` forming ``E_H``, ordered by ``video_id``."""
    sub = manifest.loc[frame_eligible_mask(manifest, h_f)]
    return sub.sort_values("video_id").reset_index(drop=True)


def frame_cohort_summary(
    manifest: pd.DataFrame, h_f: int, vocabulary: TaskVocabulary = ACCIDENT_VOCABULARY
) -> Dict[str, object]:
    """``N_H``, class distribution and cluster count for one frame horizon."""
    manifest = validate_frame_manifest(manifest, vocabulary)
    sub = frame_cohort(manifest, h_f)
    counts = np.zeros(vocabulary.n_classes, dtype=int)
    if len(sub):
        for code, cnt in sub["class_code"].value_counts().items():
            code = int(code)
            if 0 <= code < vocabulary.n_classes:
                counts[code] = int(cnt)
    total = int(counts.sum())
    return {
        "h_f": int(h_f),
        "axis_unit": "frame",
        "vocabulary_hash": vocabulary.vocabulary_hash,
        "macro_class_policy": "mean_over_classes_present_in_scored_rows",
        "N_H": int(len(sub)),
        "n_clusters": int(sub["source_cluster_id"].nunique()) if len(sub) else 0,
        "class_counts": {vocabulary.class_names[c]: int(counts[c]) for c in range(vocabulary.n_classes)},
        "class_fractions": {
            vocabulary.class_names[c]: (float(counts[c]) / total if total else float("nan"))
            for c in range(vocabulary.n_classes)
        },
        "attrition_from_all": int(len(manifest) - len(sub)),
    }


def frame_cohort_table(
    manifest: pd.DataFrame, h_list_f: Sequence[int], vocabulary: TaskVocabulary = ACCIDENT_VOCABULARY
) -> pd.DataFrame:
    """One row per frame horizon: the cohort sizes every table must report."""
    return pd.DataFrame([frame_cohort_summary(manifest, int(h), vocabulary) for h in h_list_f])


# --------------------------------------------------------------------------
# prefix list
# --------------------------------------------------------------------------
def make_frame_prefixes(
    manifest: pd.DataFrame,
    pv: FrameProtocolVector,
    grid_max_f: int,
    h_list_f: Iterable[int],
    delta_fine_f: int = 1,
    vocabulary: TaskVocabulary = ACCIDENT_VOCABULARY,
) -> pd.DataFrame:
    """Build the frame-axis prefix list for one protocol vector.

    Every prefix ends at ``anchor_eff + j*delta_f`` and carries no information
    about the clip end, so nothing downstream can read the future. ``base_j`` is
    the exact column index into a cached finest-step answer matrix, together with
    ``base_on_grid`` marking the requests that do not land on a cached column;
    those are scored as missing rather than rounded to a neighbour.
    """
    delta_fine_f = _integer(delta_fine_f, "delta_fine_f", 1)
    grid_max_f = _integer(grid_max_f, "grid_max_f", 0)
    if int(delta_fine_f) < 1:
        raise ValueError(f"delta_fine_f must be >= 1 frame, got {delta_fine_f!r}")
    if int(pv.delta_f) % int(delta_fine_f) != 0:
        raise ValueError(
            f"delta_f={pv.delta_f} is not a whole multiple of delta_fine_f={delta_fine_f}; "
            "a coarse step must be obtainable by sub-sampling the cached grid"
        )
    _check_protocol_vocabulary(pv, vocabulary)
    manifest = validate_frame_manifest(manifest, vocabulary)
    eff = perturb_frame_manifest(manifest, pv)
    J = n_grid_points_f(grid_max_f, pv.delta_f)
    js = np.arange(0, J + 1, dtype=np.int64)
    n = len(eff)

    vid = np.repeat(eff["video_id"].to_numpy(), J + 1)
    anchor_eff = np.repeat(eff["anchor_frame_eff"].to_numpy(dtype=np.int64), J + 1)
    anchor_base = np.repeat(eff["anchor_frame"].to_numpy(dtype=np.int64), J + 1)
    jj = np.tile(js, n)

    offset = jj * int(pv.delta_f)
    end_frame = anchor_eff + offset
    base_offset = end_frame - anchor_base
    base_j, remainder = np.divmod(base_offset, int(delta_fine_f))

    out = pd.DataFrame(
        {
            "video_id": vid,
            "j": jj,
            "delta_f": int(pv.delta_f),
            "offset_f": offset,
            "end_frame": end_frame,
            "is_post_anchor": np.ones(len(jj), dtype=bool),
            "anchor_frame_eff": anchor_eff,
            "base_j": base_j,
            "base_on_grid": remainder == 0,
        }
    )
    plen = eff.set_index("video_id")[LENGTH_COL_EFF]
    plen_rep = plen.reindex(out["video_id"]).to_numpy(dtype=np.int64)
    for h in h_list_f:
        out[eligible_col_f(int(h))] = plen_rep >= int(h)
    return out


def check_no_future_reads(prefixes: pd.DataFrame, manifest: pd.DataFrame, h_f: int) -> pd.DataFrame:
    """Rows of the eligible window whose ``end_frame`` exceeds the clip's last frame.

    Must be empty. For a clip in ``E_H`` the grid stops at
    ``anchor_eff + floor(H/delta)*delta <= anchor_eff + H <= last_frame``, so a
    non-empty result means the cohort and the grid have come apart -- which is
    exactly the failure a strictly causal protocol must not be able to hide.
    """
    last = manifest.set_index("video_id")["last_frame"]
    col = eligible_col_f(int(h_f))
    sub = prefixes.loc[prefixes[col]] if col in prefixes.columns else prefixes
    sub = sub.loc[sub["offset_f"] <= int(h_f)]
    last_rep = last.reindex(sub["video_id"]).to_numpy(dtype=np.int64)
    bad = sub["end_frame"].to_numpy(dtype=np.int64) > last_rep
    return pd.DataFrame(
        {
            "video_id": sub.loc[bad, "video_id"].to_numpy(),
            "j": sub.loc[bad, "j"].to_numpy(),
            "end_frame": sub.loc[bad, "end_frame"].to_numpy(),
            "last_frame": last_rep[bad],
        }
    )


# --------------------------------------------------------------------------
# answer table
# --------------------------------------------------------------------------
class FrameAnswerTable:
    """A cached answer matrix on the frame axis, indexed by ``(video_id, j)``.

    Composed around :class:`ape.metrics.AnswerTable` rather than reimplementing
    it: that class's indexing, gather and sub-sampling logic is pure integer
    column arithmetic and carries no seconds semantics. The wrapped instance is
    constructed with its step field holding the step **in frames**; that field is
    an implementation detail here and no seconds value is ever read from it or
    derived from it. Only ``delta_f`` is exposed.
    """

    def __init__(self, inner: AnswerTable, delta_f: int, vocabulary: TaskVocabulary = ACCIDENT_VOCABULARY) -> None:
        if inner.n_classes != vocabulary.n_classes:
            raise ValueError("answer dimension differs from vocabulary")
        if "vocabulary_hash" in inner.card and inner.card["vocabulary_hash"] != vocabulary.vocabulary_hash:
            raise ValueError("answer card vocabulary hash mismatch")
        if vocabulary != ACCIDENT_VOCABULARY and inner.card.get("vocabulary_hash") != vocabulary.vocabulary_hash:
            raise ValueError("custom answer card requires vocabulary hash")
        self.vocabulary = vocabulary
        self._inner = inner
        self.delta_f = _integer(delta_f, "delta_f", 1)

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        system_id: str,
        card: Optional[Dict[str, object]] = None,
        vocabulary: TaskVocabulary = ACCIDENT_VOCABULARY,
    ) -> "FrameAnswerTable":
        """Build from a long table with columns ``video_id, j, delta_f, pred``."""
        need = {"video_id", "j", "delta_f", "pred"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"frame answers for {system_id}: missing columns {sorted(missing)}")
        if "delta_s" in df.columns:
            raise ValueError(
                f"frame answers for {system_id}: carries a delta_s column; the frame axis " "does not consume seconds"
            )
        _check_vocabulary_metadata(df, vocabulary)
        card = dict(card or {})
        if "vocabulary_hash" in card and card["vocabulary_hash"] != vocabulary.vocabulary_hash:
            raise ValueError("answer card vocabulary hash mismatch")
        if vocabulary != ACCIDENT_VOCABULARY:
            card["vocabulary_hash"] = vocabulary.vocabulary_hash
        actual_probs = {c for c in df.columns if isinstance(c, str) and c.startswith("p") and c[1:].isdigit()}
        expected_probs = {f"p{i}" for i in range(vocabulary.n_classes)}
        if actual_probs and actual_probs != expected_probs:
            raise ValueError("probability columns do not match vocabulary dimension")
        df = df.copy()
        if df["video_id"].isna().any() or df["video_id"].astype(str).str.strip().eq("").any():
            raise ValueError("frame answers require nonempty video_id")
        df["video_id"] = df["video_id"].astype(str)
        for col in ("j", "delta_f", "pred"):
            df[col] = _integer_column(df[col], col)
        if ((df["pred"] < -1) | (df["pred"] >= vocabulary.n_classes)).any():
            raise ValueError("prediction outside BOT and the closed set")
        if df.duplicated(["video_id", "j"]).any():
            raise ValueError("duplicate answer key (video_id, j)")
        deltas = np.unique(df["delta_f"].to_numpy(dtype=np.int64))
        if len(deltas) != 1:
            raise ValueError(
                f"frame answers for {system_id}: expected one delta_f (the finest cached "
                f"step), got {deltas.tolist()}"
            )
        delta_f = int(deltas[0])
        if delta_f < 1:
            raise ValueError(f"frame answers for {system_id}: delta_f must be >= 1")
        renamed = df.rename(columns={"delta_f": "delta_s"}).copy()
        renamed["delta_s"] = float(delta_f)
        inner = AnswerTable.from_frame(renamed, system_id, card, n_classes=vocabulary.n_classes)
        return cls(inner, delta_f, vocabulary)

    # ---- delegation ----
    @property
    def system_id(self) -> str:
        return self._inner.system_id

    @property
    def card(self) -> Dict[str, object]:
        return self._inner.card

    @property
    def video_ids(self) -> pd.Index:
        return self._inner.video_ids

    @property
    def j_min(self) -> int:
        return self._inner.j_min

    @property
    def j_max(self) -> int:
        return self._inner.j_max

    def lookup_2d(self, video_ids: Sequence[str], base_j: np.ndarray):
        return self._inner.lookup_2d(video_ids, base_j)

    def lookup(self, video_ids: Sequence[str], base_j: np.ndarray):
        return self._inner.lookup(video_ids, base_j)

    def with_identity(self, system_id: str, card: Optional[Dict[str, object]] = None) -> "FrameAnswerTable":
        return FrameAnswerTable(self._inner.with_identity(system_id, card), self.delta_f, self.vocabulary)

    def subsample(self, factor: int) -> "FrameAnswerTable":
        """Keep every ``factor``-th cached column, giving step ``factor*delta_f``."""
        factor = _integer(factor, "factor", 1)
        return FrameAnswerTable(self._inner.subsample(factor), self.delta_f * factor, self.vocabulary)

    def to_frame(self) -> pd.DataFrame:
        out = self._inner.to_frame().rename(columns={"delta_s": "delta_f"})
        out["delta_f"] = self.delta_f
        if self.vocabulary != ACCIDENT_VOCABULARY:
            out["vocabulary_hash"] = self.vocabulary.vocabulary_hash
        return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
@dataclass
class FrameEvalResult:
    """Everything one ``(system, pi)`` cell produces on the frame axis."""

    system_id: str
    pi: FrameProtocolVector
    video_ids: np.ndarray
    source_cluster_id: np.ndarray
    y: np.ndarray
    offsets_f: np.ndarray
    pred: np.ndarray
    present: np.ndarray
    committed: np.ndarray
    cached_tail_pred: np.ndarray
    cohort_info: Dict[str, object]
    n_missing_lookup: int
    n_offgrid_lookup: int
    jitter_quantization_frames: float
    card: Dict[str, object] = field(default_factory=dict)
    vocabulary: TaskVocabulary = ACCIDENT_VOCABULARY

    @property
    def h_f(self) -> int:
        return int(self.pi.h_f)

    @property
    def delta_f(self) -> int:
        return int(self.pi.delta_f)

    @property
    def effective_h_f(self) -> int:
        """``floor(H/delta)*delta``: the window actually integrated.

        The protocol requires H to be divisible by the step. Therefore this is
        exactly the registered common horizon, independent of sampling stride.
        """
        return int(self.delta_f) * (self.pred.shape[1] - 1)

    @property
    def n(self) -> int:
        return int(self.pred.shape[0])

    @property
    def correct(self) -> np.ndarray:
        return correctness(self.pred, self.y)

    @property
    def indicator(self) -> np.ndarray:
        return stable_correct(self.correct)

    @property
    def missing_lookup_rate(self) -> float:
        total = self.pred.size
        return float(self.n_missing_lookup) / total if total else float("nan")

    def metrics_on(self, idx: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Frame-axis metric family plus companions, on a row subset.

        ``RMSCD@H_frames`` is in frames. ``RMSCD@H_norm`` divides it by the
        **common** horizon ``H`` of the track, giving a dimensionless relative-window reading. It does not establish
        equal physical observation durations across tracks with unknown frame rates. It is
        never normalised by a clip's own remaining length.
        """
        if idx is None:
            idx = np.arange(self.n)
        idx = np.asarray(idx, dtype=np.int64)
        pred = self.pred[idx]
        y = self.y[idx]
        ind = stable_correct(correctness(pred, y))
        curve = s_curve(ind)
        curve_macro = s_curve_macro(ind, y, self.vocabulary.n_classes)
        flips = flip_counts(pred)
        j_end = pred.shape[1] - 1
        rmscd = float(trapezoid(1.0 - np.asarray(curve, dtype=float), float(self.delta_f)))
        rmscd_macro = float(trapezoid(1.0 - np.asarray(curve_macro, dtype=float), float(self.delta_f)))
        eff_h = float(self.effective_h_f) or float("nan")
        return {
            "RMSCD@H_frames": rmscd,
            "RMSCD@H_frames_macro": rmscd_macro,
            "RMSCD@H_norm": rmscd / eff_h,
            "RMSCD@H_norm_macro": rmscd_macro / eff_h,
            "end_window_macro_acc": macro_accuracy(pred[:, j_end], y, self.vocabulary.n_classes),
            "end_window_micro_acc": micro_accuracy(pred[:, j_end], y),
            "median_flips": float(np.median(flips)) if len(flips) else float("nan"),
            "mean_flips": float(np.mean(flips)) if len(flips) else float("nan"),
            "flip_rate": float((flips > 0).mean()) if len(flips) else float("nan"),
            "cached_tail_macro_acc": macro_accuracy(self.cached_tail_pred[idx], y, self.vocabulary.n_classes),
        }

    def s_at_f(self, delta_k_f: int, macro: bool = False) -> float:
        """``S_H`` at a grid offset given in frames; NaN outside the window.

        The offset must land exactly on the grid. On the seconds axis the
        analogous helper rounds to the nearest grid index; here an off-grid
        request is a mis-specified read-out, not something to round.
        """
        k = _integer(delta_k_f, "delta_k_f")
        if k < 0 or k > self.effective_h_f or k % self.delta_f != 0:
            return float("nan")
        j = k // self.delta_f
        if j < 0 or j >= self.pred.shape[1]:
            return float("nan")
        ind = self.indicator
        curve = s_curve_macro(ind, self.y, self.vocabulary.n_classes) if macro else s_curve(ind)
        return float(curve[j])

    def frozen_family(self, s_report_delta_f: Sequence[int], idx: Optional[np.ndarray] = None) -> Dict[str, float]:
        """The frame-axis metric family, from a single stable-correctness pass.

        ``s_report_delta_f`` has no default. The seconds axis reports ``S_H`` at
        1 s and 3 s because those were registered for that track; the equivalent
        frame offsets for a new track have to be chosen on its own development
        subset and registered before the audit, so this function will not invent
        them.
        """
        if not list(s_report_delta_f):
            raise ValueError(
                "s_report_delta_f is empty: the read-out offsets of the metric family must "
                "be registered on the development subset before the audit, not defaulted here"
            )
        if idx is None:
            idx = np.arange(self.n)
        idx = np.asarray(idx, dtype=np.int64)
        pred = self.pred[idx]
        y = self.y[idx]
        curve = s_curve(stable_correct(correctness(pred, y)))
        flips = flip_counts(pred)
        j_end = pred.shape[1] - 1
        rmscd = float(trapezoid(1.0 - np.asarray(curve, dtype=float), float(self.delta_f)))
        fam: Dict[str, float] = {
            "RMSCD@H_frames": rmscd,
            "RMSCD@H_norm": rmscd / (float(self.effective_h_f) or float("nan")),
            "end_window_macro_acc": macro_accuracy(pred[:, j_end], y, self.vocabulary.n_classes),
            "median_flips": float(np.median(flips)) if len(flips) else float("nan"),
        }
        for k in s_report_delta_f:
            k = _integer(k, "s_report_delta_f")
            on_grid = 0 <= k <= self.effective_h_f and k % self.delta_f == 0
            j = k // self.delta_f if on_grid else -1
            fam[f"S_H@{k}f"] = float(curve[j]) if (on_grid and 0 <= j < len(curve)) else float("nan")
        return fam

    def commit(self) -> Dict[str, object]:
        """Commitment triple with ``tau_c`` in frames; never-committing clips kept."""
        stats = commit_stats(self.committed, self.pred, self.y, float(self.delta_f))
        out = {k: v for k, v in stats.items() if k not in ("h_s", "tau_c_mean", "tau_c_median")}
        out["h_f"] = int(self.effective_h_f)
        out["tau_c_mean_frames"] = stats["tau_c_mean"]
        out["tau_c_median_frames"] = stats["tau_c_median"]
        return out

    def first_stable_offset_f(self) -> np.ndarray:
        """First grid offset in frames at which a clip becomes stably correct.

        ``-1`` marks a clip that never stabilises inside the window; such clips
        are kept everywhere else too, so this is a diagnostic and not a filter.
        """
        idx = first_stable_index(self.indicator)
        return np.where(idx < 0, -1, idx * self.delta_f)


def evaluate_system_frames(
    manifest: pd.DataFrame,
    answers: FrameAnswerTable,
    pv: FrameProtocolVector,
    grid_max_f: Optional[int] = None,
    vocabulary: TaskVocabulary = ACCIDENT_VOCABULARY,
) -> FrameEvalResult:
    """Score one system under one frame-axis protocol vector on the fixed cohort.

    Same order of operations as the seconds axis: perturb the anchors, fix the
    cohort, build the causal grid, look the cached answers up by absolute frame
    index, then compute. No metric may change the cohort.
    """
    _check_protocol_vocabulary(pv, vocabulary)
    if answers.vocabulary != vocabulary:
        raise ValueError("answers and evaluation vocabularies differ")
    manifest = validate_frame_manifest(manifest, vocabulary)
    eff = perturb_frame_manifest(manifest, pv)
    h_f = int(pv.h_f)
    if grid_max_f is not None and h_f > int(grid_max_f):
        raise ValueError(f"h_f={h_f} exceeds grid_max_f={grid_max_f}")
    sub = eff.loc[eff[LENGTH_COL_EFF].to_numpy(dtype=np.int64) >= h_f].sort_values("video_id")
    sub = sub.reset_index(drop=True)

    J_H = n_grid_points_f(h_f, pv.delta_f)
    offsets = np.arange(J_H + 1, dtype=np.int64) * int(pv.delta_f)

    vid = sub["video_id"].to_numpy()
    anchor_eff = sub["anchor_frame_eff"].to_numpy(dtype=np.int64)
    anchor_base = sub["anchor_frame"].to_numpy(dtype=np.int64)
    y = sub["class_code"].to_numpy(dtype=np.int64)

    end_frame = anchor_eff[:, None] + offsets[None, :]
    base_offset = end_frame - anchor_base[:, None]
    base_j, remainder = np.divmod(base_offset, int(answers.delta_f))
    on_grid = remainder == 0

    pred, present, committed, _ = answers.lookup_2d(vid, base_j)
    # An off-grid request has no cached answer. Treat it as missing -- wrong, in
    # the denominator, and counted -- rather than rounding to a neighbouring
    # column, which would answer a question that was not asked.
    present = present & on_grid
    pred = np.where(present, pred, -1)
    committed = committed & on_grid
    if committed.size:
        committed = np.logical_or.accumulate(committed, axis=1)

    last = sub["last_frame"].to_numpy(dtype=np.int64)
    max_j_in_clip = (last - anchor_base) // int(answers.delta_f)
    fc_j = np.clip(max_j_in_clip, answers.j_min, answers.j_max)
    full_pred, fc_present, _, _ = answers.lookup(vid, fc_j)
    full_pred = np.where(fc_present, full_pred, -1)

    return FrameEvalResult(
        system_id=answers.system_id,
        pi=pv,
        video_ids=vid,
        source_cluster_id=sub["source_cluster_id"].to_numpy(),
        y=y,
        offsets_f=offsets,
        pred=pred,
        present=present,
        committed=committed,
        cached_tail_pred=full_pred,
        cohort_info=frame_cohort_summary(eff, h_f, vocabulary),
        n_missing_lookup=int((~present).sum()),
        n_offgrid_lookup=int((~on_grid).sum()),
        jitter_quantization_frames=0.5 if float(pv.eps_jit_sd_f) > 0.0 else 0.0,
        card=answers.card,
        vocabulary=vocabulary,
    )


# --------------------------------------------------------------------------
# horizon selection on a development subset
# --------------------------------------------------------------------------
def frame_horizon_candidates(
    dev_manifest: pd.DataFrame,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    vocabulary: TaskVocabulary = ACCIDENT_VOCABULARY,
) -> Dict[str, object]:
    """Post-anchor length quantiles of a development subset, in frames.

    Returns candidates only. Choosing which of them to freeze as ``H``, and
    registering that choice, is a separate act that must happen before the audit
    split is touched -- this function deliberately does not freeze anything and
    must never be called on the audit split.
    """
    if "split" not in dev_manifest or not set(dev_manifest["split"].astype(str)).issubset(
        {"train", "dev", "val", "validation"}
    ):
        raise ValueError("horizon candidates require an explicitly identified development split, never test/audit")
    dev_manifest = validate_frame_manifest(dev_manifest, vocabulary)
    lengths = np.sort(dev_manifest[LENGTH_COL].to_numpy(dtype=np.int64))
    if len(lengths) == 0:
        raise ValueError("development subset is empty; no horizon candidate can be derived")
    out: Dict[str, object] = {
        "n_dev": int(len(lengths)),
        "min_f": int(lengths[0]),
        "max_f": int(lengths[-1]),
        "quantiles": {},
        "status": "candidates_only_not_frozen",
    }
    for q in quantiles:
        value = int(np.floor(np.quantile(lengths, float(q))))
        out["quantiles"][f"q{q:g}"] = {
            "h_f": value,
            "n_eligible_dev": int((lengths >= value).sum()),
        }
    return out
