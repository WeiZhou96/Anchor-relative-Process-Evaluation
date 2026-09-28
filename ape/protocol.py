"""Protocol vector, stable protocol hash, and prefix-list generation.

Implements idea section 3.1, 3.4 and contract sections 5.2, 5.3, 8.

Design notes that matter for the rest of the package
----------------------------------------------------
**Perturbations live in the prefix list, not in the model.** A protocol vector
``pi = (eps_sys, eps_jit, Delta, H)`` moves the *recorded* anchor; the video and
the system are untouched. The effective anchor of video ``i`` is

    anchor_eff_i = clip(anchor_i + eps_sys + jit_i, 0, duration_i)

and the evaluation grid is ``end_s = anchor_eff_i + j * Delta`` for
``j = 0..floor(grid_max_s / Delta)``.

**Answer lookup on an absolute time axis.** Answer matrices (contract 5.4) are
produced once on the finest step ``Delta_fine`` relative to the *unperturbed*
anchor, so row ``(video_id, j)`` corresponds to the absolute end time
``anchor_i + j * Delta_fine``. Under a perturbed ``pi`` we therefore look the
answer up by absolute time:

    base_j = round((end_s - anchor_i) / Delta_fine)

This is what makes "perturb and recompute without re-running the model"
(contract section 8) well defined. Two consequences are documented rather than
hidden, and are reported by :mod:`ape.metrics` as ``lookup_quantization_s`` and
``missing_lookup_rate``:

1. Anchor perturbations are quantised to ``Delta_fine`` (so ``|eps| < Delta_fine/2``
   is invisible to a cached matrix). Blocks can be regenerated exactly on any
   prefix list via :func:`ape.blocks.answers_for_prefixes`, which is how the
   sub-``Delta_fine`` jitter behaviour is tested.
2. A negative ``eps_sys`` asks the system for answers slightly *before* its
   cached grid starts. Cached rows with ``j < 0`` are allowed (idea section 3.1
   permits recording raw pre-anchor outputs for diagnostics; the ``BOT`` mask is
   a protocol-layer decision applied at evaluation time). Rows that are missing
   are scored as ``BOT``, i.e. wrong, and counted in ``missing_lookup_rate``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

try:  # PyYAML is present in the target env; keep the import failure legible.
    import yaml as _yaml
except Exception:  # pragma: no cover - environment guard
    _yaml = None


TOL = 1e-9

#: Columns the manifest must provide for the protocol layer to work.
MANIFEST_REQUIRED = [
    "video_id",
    "source_cluster_id",
    "anchor_s",
    "fps",
    "duration_s",
    "post_anchor_length_s",
    "class_code",
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def h_label(h: float) -> str:
    """Column-name label for an observation horizon (contract 5.3).

    ``3.0 -> 'H3'``, ``2.5 -> 'H25'`` (integers as integers, otherwise the
    decimal point is dropped).
    """
    h = float(h)
    if abs(h - round(h)) < TOL:
        return f"H{int(round(h))}"
    return "H" + ("%g" % h).replace(".", "").replace("-", "m")


def eligible_col(h: float) -> str:
    return f"eligible_{h_label(h)}"


def n_grid_points(span_s: float, delta_s: float) -> int:
    """Number of grid *steps* fitting in ``span_s`` (grid is ``0..J``)."""
    return int(math.floor(float(span_s) / float(delta_s) + TOL))


def _round_for_hash(x: float) -> float:
    return float(round(float(x), 9)) + 0.0  # normalise -0.0


# --------------------------------------------------------------------------
# protocol vector
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProtocolVector:
    """A single point of the protocol space (idea section 3.4).

    Attributes
    ----------
    eps_sys_s : common-mode anchor shift applied to every video (seconds).
    eps_jit_sd_s : standard deviation of the per-video independent anchor jitter.
    delta_s : prefix step.
    h_s : observation horizon; it fixes both the eligibility cohort and the
        censoring bound (idea section 3.4 item 4).
    jit_seed : seed of the jitter realisation; part of the identity of ``pi``
        because a different realisation is a different prefix list.
    pre_anchor_outputs_bot : whether ``delta < 0`` is masked to ``BOT``.
    protocol_version : provenance string carried into the hash.
    """

    eps_sys_s: float = 0.0
    eps_jit_sd_s: float = 0.0
    delta_s: float = 0.5
    h_s: float = 6.0
    jit_seed: int = 20260903
    pre_anchor_outputs_bot: bool = True
    protocol_version: str = "0.1-unfrozen"

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        for k in ("eps_sys_s", "eps_jit_sd_s", "delta_s", "h_s"):
            d[k] = _round_for_hash(d[k])
        return d

    @property
    def pi_hash(self) -> str:
        """Stable 12-hex-char hash of the protocol vector (contract 5.3)."""
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=6).hexdigest()

    def replace(self, **kw) -> "ProtocolVector":
        d = asdict(self)
        d.update(kw)
        return ProtocolVector(**d)

    def label(self) -> str:
        return (
            f"eps{self.eps_sys_s:+g}_jit{self.eps_jit_sd_s:g}"
            f"_d{self.delta_s:g}_H{self.h_s:g}"
        )


# --------------------------------------------------------------------------
# protocol configuration file (contract 5.2)
# --------------------------------------------------------------------------
@dataclass
class ProtocolConfig:
    raw: Dict[str, object]
    path: Optional[str] = None

    # ---- accessors ----
    @property
    def protocol_version(self) -> str:
        return str(self.raw.get("protocol_version", "0.1-unfrozen"))

    @property
    def frozen(self) -> bool:
        return bool(self.raw.get("frozen", False))

    @property
    def delta_s(self) -> float:
        return float(self.raw["delta_s"])

    @property
    def h_list_s(self) -> List[float]:
        return [float(h) for h in self.raw["h_list_s"]]

    @property
    def h_ref_s(self) -> float:
        """Reference horizon of ``pi0``: the middle entry (idea section 3.4)."""
        hs = sorted(self.h_list_s)
        return hs[len(hs) // 2]

    @property
    def grid_max_s(self) -> float:
        return float(self.raw.get("grid_max_s", max(self.h_list_s)))

    @property
    def r0(self) -> float:
        return float(self.raw.get("r0", 0.9))

    @property
    def bootstrap_n(self) -> int:
        return int(self.raw.get("bootstrap_n", 1000))

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 20260903))

    @property
    def alpha_pairs(self) -> float:
        return float(self.raw.get("alpha_pairs", 0.05))

    @property
    def pre_anchor_outputs_bot(self) -> bool:
        return bool(self.raw.get("pre_anchor_outputs_bot", True))

    @property
    def perturb(self) -> Dict[str, object]:
        return dict(self.raw.get("perturb", {}))

    @property
    def commit_rho_levels(self) -> List[float]:
        return [float(x) for x in dict(self.raw.get("commit", {})).get("rho_levels", [])]

    @property
    def answers_cfg(self) -> Dict[str, object]:
        return dict(self.raw.get("answers", {}))

    @property
    def delta_fine_s(self) -> float:
        """Finest step; answer matrices must be produced on it (contract 5.4)."""
        explicit = self.answers_cfg.get("delta_fine_s")
        cands = [self.delta_s] + [float(d) for d in self.perturb.get("delta_s", [])]
        if explicit is not None:
            fine = float(explicit)
            if fine > min(cands) + TOL:
                raise ValueError(
                    f"answers.delta_fine_s={fine} is coarser than the finest "
                    f"perturbation step {min(cands)}; coarse steps must be "
                    "obtainable by sub-sampling"
                )
            return fine
        return min(cands)

    @property
    def answers_pad_s(self) -> float:
        """How far the cached answer columns extend beyond the evaluation grid.

        Must cover the largest anchor perturbation, otherwise a shifted grid
        would fall off the cached range and be scored as ``BOT``.
        """
        explicit = self.answers_cfg.get("pad_s")
        p = self.perturb
        need = max([abs(float(x)) for x in p.get("eps_sys_s", [0.0])] + [0.0])
        need += 3.0 * max([float(x) for x in p.get("eps_jit_sd_s", [0.0])] + [0.0])
        need += float(self.delta_fine_s)
        if explicit is None:
            return float(need)
        return max(float(explicit), float(need))

    @property
    def s_report_delta_s(self) -> List[float]:
        """The two pre-registered ``delta_k`` of the frozen metric family."""
        fam = dict(self.raw.get("metric_family", {}))
        return [float(x) for x in fam.get("s_report_delta_s", [1.0, 3.0])]

    @property
    def block_factors(self) -> List[Dict[str, object]]:
        return [dict(b) for b in self.raw.get("blocks", [])]

    @property
    def phenomena_cfg(self) -> Dict[str, object]:
        return dict(self.raw.get("phenomena", {}))

    # ---- protocol vectors ----
    def pi0(self, h_s: Optional[float] = None) -> ProtocolVector:
        """Reference protocol: both perturbations zero, reference step, mid ``H``."""
        return ProtocolVector(
            eps_sys_s=0.0,
            eps_jit_sd_s=0.0,
            delta_s=self.delta_s,
            h_s=self.h_ref_s if h_s is None else float(h_s),
            jit_seed=self.seed,
            pre_anchor_outputs_bot=self.pre_anchor_outputs_bot,
            protocol_version=self.protocol_version,
        )

    def _axis_grid(self, plaus: bool) -> Dict[str, List[float]]:
        p = self.perturb
        if plaus:
            sub = dict(p.get("plaus", {}))
            return {
                "eps_sys_s": sorted(set([0.0] + [float(x) for x in sub.get("eps_sys_s", [])])),
                "eps_jit_sd_s": sorted(set([0.0] + [float(x) for x in sub.get("eps_jit_sd_s", [])])),
                "delta_s": sorted(set([self.delta_s] + [float(x) for x in sub.get("delta_s", [])])),
                "h_s": sorted(set([self.h_ref_s] + [float(x) for x in sub.get("h_s", [])])),
            }
        return {
            "eps_sys_s": sorted(set([0.0] + [float(x) for x in p.get("eps_sys_s", [])])),
            "eps_jit_sd_s": sorted(set([0.0] + [float(x) for x in p.get("eps_jit_sd_s", [])])),
            "delta_s": sorted(set([self.delta_s] + [float(x) for x in p.get("delta_s", [])])),
            "h_s": sorted(set([self.h_ref_s] + [float(x) for x in p.get("h_s", [])])),
        }

    def scan_grid(self, plaus: bool = False, mode: str = "axis") -> List[ProtocolVector]:
        """Scan grid ``Pi`` (``mode='axis'``) or its full product (``mode='full'``).

        ``axis`` walks one knob at a time away from ``pi0`` (this is what E4's
        three propagation curves need and it keeps the grid affordable);
        ``full`` is the Cartesian product used for the joint comparability
        region of idea section 3.5 / figure 3.
        """
        axes = self._axis_grid(plaus)
        pi0 = self.pi0()
        out: List[ProtocolVector] = [pi0]
        if mode == "axis":
            for name, values in axes.items():
                for v in values:
                    pv = pi0.replace(**{name: float(v)})
                    if pv != pi0:
                        out.append(pv)
        elif mode == "full":
            for e in axes["eps_sys_s"]:
                for jsd in axes["eps_jit_sd_s"]:
                    for d in axes["delta_s"]:
                        for h in axes["h_s"]:
                            pv = pi0.replace(
                                eps_sys_s=float(e),
                                eps_jit_sd_s=float(jsd),
                                delta_s=float(d),
                                h_s=float(h),
                            )
                            if pv != pi0:
                                out.append(pv)
        else:
            raise ValueError(f"unknown scan mode {mode!r}")
        # de-duplicate, keep pi0 first
        seen, uniq = set(), []
        for pv in out:
            if pv.pi_hash not in seen:
                seen.add(pv.pi_hash)
                uniq.append(pv)
        return uniq


def load_protocol(path: str) -> ProtocolConfig:
    if _yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read the protocol file")
    with open(path, "r", encoding="utf-8") as fh:
        raw = _yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: protocol file must be a mapping")
    return ProtocolConfig(raw=raw, path=os.path.abspath(path))


# --------------------------------------------------------------------------
# S1 freeze guard
# --------------------------------------------------------------------------
class ProtocolFrozenError(RuntimeError):
    """Raised when a frozen protocol file no longer matches its recorded hash."""


def _strip_ext(path: str) -> str:
    base, ext = os.path.splitext(path)
    return base if ext in (".yaml", ".yml") else path


def frozen_hash_path(pi_path: str) -> str:
    return _strip_ext(pi_path) + ".frozen.sha256"


def frozen_snapshot_path(pi_path: str) -> str:
    return _strip_ext(pi_path) + ".frozen.json"


def canonical_bytes(path: str) -> bytes:
    """File bytes with line endings normalised, so CRLF alone is not a change."""
    with open(path, "rb") as fh:
        return fh.read().replace(b"\r\n", b"\n")


def content_sha256(path: str) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def read_frozen_hash(pi_path: str) -> Optional[str]:
    p = frozen_hash_path(pi_path)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        first = fh.readline().strip()
    return first.split()[0] if first else None


def write_freeze(pi_path: str) -> Dict[str, object]:
    """Record the frozen digest and a snapshot for field-level diffing.

    Writes ``<pi>.frozen.sha256`` (``sha256sum`` format, so it can also be
    checked outside this package) and ``<pi>.frozen.json`` (the parsed config, so
    a later mismatch can say *which field* moved rather than only that something
    did).
    """
    digest = content_sha256(pi_path)
    with open(frozen_hash_path(pi_path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{digest}  {os.path.basename(pi_path)}\n")
    cfg = load_protocol(pi_path)
    with open(frozen_snapshot_path(pi_path), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(cfg.raw, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return {"sha256": digest, "hash_file": frozen_hash_path(pi_path),
            "snapshot_file": frozen_snapshot_path(pi_path)}


def _flatten(obj, prefix: str = "") -> Dict[str, object]:
    out: Dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, sort_keys=True, default=str)
    else:
        out[prefix] = obj
    return out


def diff_against_snapshot(pi_path: str) -> List[str]:
    """Human-readable list of fields that differ from the frozen snapshot."""
    snap_path = frozen_snapshot_path(pi_path)
    if not os.path.exists(snap_path):
        return ["(no snapshot recorded; cannot show a field-level diff)"]
    with open(snap_path, "r", encoding="utf-8") as fh:
        snap = json.load(fh)
    cur = load_protocol(pi_path).raw
    a, b = _flatten(snap), _flatten(cur)
    lines: List[str] = []
    for key in sorted(set(a) | set(b)):
        if key not in a:
            lines.append(f"  + {key} = {b[key]!r}   (added since freeze)")
        elif key not in b:
            lines.append(f"  - {key} = {a[key]!r}   (removed since freeze)")
        elif a[key] != b[key]:
            lines.append(f"  ~ {key}: frozen {a[key]!r} -> now {b[key]!r}")
    return lines or ["  (parsed values identical; only formatting or comments changed)"]


def verify_frozen(cfg: ProtocolConfig) -> Dict[str, object]:
    """Refuse to run a frozen protocol whose file has drifted from its digest.

    S1 froze the protocol; contract discipline is that nothing may change
    afterwards without re-running everything and saying so. A silent edit would
    invalidate every ``pi_hash`` and every cached artefact while the file still
    claimed to be the frozen one, so this is a hard stop rather than a warning.
    Re-freezing is possible but has to be an explicit act: ``ape.cli freeze``.
    """
    if not cfg.frozen:
        return {"frozen": False, "checked": False}
    if cfg.path is None:  # pragma: no cover - defensive
        raise ProtocolFrozenError("frozen protocol has no file path to verify")
    recorded = read_frozen_hash(cfg.path)
    actual = content_sha256(cfg.path)
    if recorded is None:
        raise ProtocolFrozenError(
            f"{cfg.path} declares frozen: true but {frozen_hash_path(cfg.path)} is "
            "missing.\nIf this is the S1 freeze, record it once with:\n"
            f"  python -m ape.cli freeze --pi {cfg.path}"
        )
    if recorded != actual:
        diff = "\n".join(diff_against_snapshot(cfg.path))
        raise ProtocolFrozenError(
            f"FROZEN PROTOCOL MODIFIED: {cfg.path}\n"
            f"  recorded sha256 : {recorded}\n"
            f"  actual   sha256 : {actual}\n"
            f"changes since the freeze:\n{diff}\n"
            "Refusing to run: results computed now would not be the frozen "
            "protocol's results.\nEither restore the file, or re-freeze "
            "deliberately and re-run everything:\n"
            f"  python -m ape.cli freeze --pi {cfg.path} --force"
        )
    return {"frozen": True, "checked": True, "sha256": actual,
            "hash_file": frozen_hash_path(cfg.path)}


def load_protocol_checked(path: str) -> ProtocolConfig:
    """Load a protocol file and enforce the freeze guard."""
    cfg = load_protocol(path)
    verify_frozen(cfg)
    return cfg


# --------------------------------------------------------------------------
# manifest handling and anchor perturbation
# --------------------------------------------------------------------------
def load_manifest(path: str) -> pd.DataFrame:
    """Read a contract 5.1 manifest and validate the columns the protocol uses."""
    df = pd.read_csv(path)
    missing = [c for c in MANIFEST_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: manifest is missing columns {missing}")
    df = df.copy()
    df["video_id"] = df["video_id"].astype(str)
    df["source_cluster_id"] = df["source_cluster_id"].astype(str)
    for c in ("anchor_s", "fps", "duration_s", "post_anchor_length_s"):
        df[c] = df[c].astype(float)
    df["class_code"] = df["class_code"].astype(int)
    if df["video_id"].duplicated().any():
        dup = df.loc[df["video_id"].duplicated(), "video_id"].tolist()[:5]
        raise ValueError(f"{path}: duplicated video_id, e.g. {dup}")
    # self-consistency of the derived column; tolerate rounding in the CSV
    derived = df["duration_s"] - df["anchor_s"]
    bad = (derived - df["post_anchor_length_s"]).abs() > 1e-3
    if bool(bad.any()):
        raise ValueError(
            f"{path}: post_anchor_length_s != duration_s - anchor_s for "
            f"{int(bad.sum())} rows"
        )
    if "decode_ok" in df.columns:
        df = df.loc[df["decode_ok"].astype(str).str.lower().isin(["true", "1", "yes"])]
    return df.reset_index(drop=True)


def jitter_draws(video_ids: Sequence[str], sd_s: float, jit_seed: int) -> np.ndarray:
    """Per-video independent Gaussian anchor jitter (idea section 3.4 item 2).

    The draw is keyed by ``(video_id, jit_seed)`` through a hash, so it does not
    depend on row order or on which subset of the manifest is being evaluated:
    the same video receives the same jitter in every cohort.
    """
    sd = float(sd_s)
    out = np.zeros(len(video_ids), dtype=float)
    if sd <= 0.0:
        return out
    for k, vid in enumerate(video_ids):
        key = f"{jit_seed}|{vid}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        seed = int.from_bytes(digest, "little", signed=False)
        out[k] = np.random.default_rng(seed).normal(0.0, sd)
    return out


def perturb_manifest(manifest: pd.DataFrame, pv: ProtocolVector) -> pd.DataFrame:
    """Apply ``pi``'s anchor perturbations, returning the effective manifest.

    Adds ``anchor_s_eff``, ``jitter_s``, ``post_anchor_length_s_eff`` and
    ``anchor_clipped``. The eligibility cohort is recomputed from
    ``post_anchor_length_s_eff``, which is exactly the second-order mechanism
    that P-a and P-b are about (idea section 3.5): a shift changes both the
    measured delay and who is in the cohort.
    """
    df = manifest.copy()
    jit = jitter_draws(df["video_id"].tolist(), pv.eps_jit_sd_s, pv.jit_seed)
    raw_anchor = df["anchor_s"].to_numpy(dtype=float) + float(pv.eps_sys_s) + jit
    dur = df["duration_s"].to_numpy(dtype=float)
    eff = np.clip(raw_anchor, 0.0, dur)
    df["jitter_s"] = jit
    df["anchor_s_eff"] = eff
    df["anchor_clipped"] = (np.abs(eff - raw_anchor) > TOL)
    df["post_anchor_length_s_eff"] = dur - eff
    return df


# --------------------------------------------------------------------------
# prefix list (contract 5.3)
# --------------------------------------------------------------------------
def make_prefixes(
    manifest: pd.DataFrame,
    pv: ProtocolVector,
    grid_max_s: float,
    h_list_s: Iterable[float],
    delta_fine_s: Optional[float] = None,
) -> pd.DataFrame:
    """Build the prefix list for one protocol vector.

    Every prefix is strictly causal: it ends at ``anchor_eff + j*Delta`` and no
    downstream code is given the clip duration, the distance to the clip end, or
    any future padding (contract section 8). ``end_frame`` is
    ``floor(end_s * fps)``, i.e. the last frame whose timestamp is inside the
    prefix.

    Returns a long table with one row per ``(video_id, j)``, the eligibility
    columns for every ``H`` in ``h_list_s``, and ``base_j`` -- the index into a
    cached finest-step answer matrix (see the module docstring).
    """
    if delta_fine_s is None:
        delta_fine_s = pv.delta_s
    eff = perturb_manifest(manifest, pv)
    J = n_grid_points(grid_max_s, pv.delta_s)
    js = np.arange(0, J + 1, dtype=int)

    n = len(eff)
    vid = np.repeat(eff["video_id"].to_numpy(), J + 1)
    anchor_eff = np.repeat(eff["anchor_s_eff"].to_numpy(dtype=float), J + 1)
    anchor_base = np.repeat(eff["anchor_s"].to_numpy(dtype=float), J + 1)
    fps = np.repeat(eff["fps"].to_numpy(dtype=float), J + 1)
    jj = np.tile(js, n)

    offset = jj * float(pv.delta_s)
    end_s = anchor_eff + offset
    base_offset = end_s - anchor_base
    base_j = np.rint(base_offset / float(delta_fine_s)).astype(np.int64)

    out = pd.DataFrame(
        {
            "video_id": vid,
            "j": jj,
            "delta_s": float(pv.delta_s),
            "offset_s": offset,
            "end_s": end_s,
            "end_frame": np.floor(end_s * fps + TOL).astype(np.int64),
            "is_post_anchor": np.ones(len(jj), dtype=bool),
            "anchor_s_eff": anchor_eff,
            "base_j": base_j,
        }
    )
    plen = eff.set_index("video_id")["post_anchor_length_s_eff"]
    plen_rep = plen.reindex(out["video_id"]).to_numpy(dtype=float)
    for h in h_list_s:
        out[eligible_col(h)] = plen_rep >= (float(h) - TOL)
    return out


def write_prefix_list(prefixes: pd.DataFrame, out_dir: str, pv: ProtocolVector) -> str:
    """Write ``outputs/prefix_lists/<pi_hash>.csv`` and a sidecar description."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{pv.pi_hash}.csv")
    prefixes.to_csv(path, index=False)
    with open(os.path.join(out_dir, f"{pv.pi_hash}.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"pi_hash": pv.pi_hash, "pi": pv.as_dict(), "n_rows": int(len(prefixes))},
            fh,
            indent=2,
            sort_keys=True,
        )
    return path
