"""Shared constants and helpers for the B (system library) track of ape_protocol.

Nothing here implements protocol metrics -- those belong to the A track (``ape/``).
Only the pieces every system in ``systems/`` needs: paths, the class code table,
manifest loading and the answer-matrix / system-card writers of CONTRACT 5.4.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("APE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("APE_DATA_ROOT", os.environ.get("APE_ACCIDENT", os.path.join(PROJECT_ROOT, "external", "ACCIDENT_2026")))
METADATA_REAL = os.path.join(DATA_ROOT, "metadata-real.csv")

OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")
FEATURES_DIR = os.path.join(OUTPUTS, "features")
CKPT_DIR = os.path.join(OUTPUTS, "checkpoints")
ANSWERS_DIR = os.path.join(OUTPUTS, "answers")
TMP_MANIFEST_B = os.path.join(OUTPUTS, "tmp_manifest_b.csv")
# The C track's authoritative manifest; same columns (CONTRACT 5.1), so every
# entry point below accepts either path.
MANIFEST_C = os.path.join(PROJECT_ROOT, "data", "manifest", "manifest_real.csv")

# ---------------------------------------------------------------------------
# Protocol-side constants that B must honour (CONTRACT 4, 5.2, 5.4)
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = ["head-on", "rear-end", "t-bone", "sideswipe", "single"]
CLASS_TO_CODE: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
N_CLASSES = len(CLASS_NAMES)
BOT = -1  # the protocol's "no answer" code

SEED = 20260903
DEV_FRACTION = 0.20
FINEST_DELTA_S = 0.25   # smallest delta_s candidate in CONTRACT 5.2
FEATURE_FPS = 4.0       # fixed decode rate for the feature cache

# Answer grid, in units of j at the finest step. The A track scans anchor
# perturbations by shifting j, so the matrix must extend on both sides of the
# anchor: every eps_sys < 0 scan point would otherwise fall outside the cache and
# be charged as an error. j = -11 .. 88 covers end_s = anchor - 2.75 s .. + 22 s.
J_MIN = -11
J_MAX = 88
GRID_MIN_S = J_MIN * FINEST_DELTA_S   # -2.75
GRID_MAX_S = J_MAX * FINEST_DELTA_S   # 22.0
# Rows with j < 0 carry the system's raw Top-1 and probabilities. Masking them to
# bot is the protocol layer's job, not the system's (A track, 2026-09-03).
PRE_ANCHOR_RAW = True
# Observation windows reported in the summary tables. H_DEV_SELECT is the window
# every dev readout and every arm's parameter selection runs on; it was raised to
# the frozen value 10.0 at S1.
H_PLACEHOLDERS = [3.0, 6.0, 10.0]
H_DEV_SELECT = 10.0

TRACK_ID = "roadside"
DATASET_ID = "ACCIDENT_real"
LICENSE_NOTE = "academic-noncommercial; see paper vs kaggle"

ANSWER_COLUMNS = ["video_id", "j", "delta_s", "pred", "p0", "p1", "p2", "p3", "p4", "committed"]


def grid_js(j_min: int = J_MIN, j_max: int = J_MAX) -> np.ndarray:
    """The full answer grid in units of j at the finest step, pre-anchor included."""
    return np.arange(j_min, j_max + 1, dtype=np.int64)


def post_anchor_slice(js: np.ndarray) -> slice:
    """Column slice of an answer array covering j >= 0."""
    first = int(np.searchsorted(js, 0))
    return slice(first, len(js))


_SEGMENT_SUFFIX = re.compile(r"_\d{1,3}$")


def source_cluster_id(video_id: str) -> str:
    """Strip trailing numeric segment suffixes to recover the source video id.

    Clip ids are mostly ``<youtube_id>_<segment>``, but nested forms such as
    ``3Xd6PxEvbNk_11_00`` occur; stripping only one suffix splits 42 source videos
    across what should be a single cluster (C track, 2026-09-03). Suffixes of one
    to three digits are removed repeatedly. A source id whose own last underscore
    group is purely numeric would be over-stripped; none is known in this release,
    and the rule is kept identical to the C track's so both manifests cluster alike.
    """
    prev = None
    out = video_id
    while prev != out:
        prev = out
        out = _SEGMENT_SUFFIX.sub("", out)
    return out or video_id


def md5_of_file(path: str, chunk: int = 4 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def default_manifest_path() -> str:
    """Prefer the C track's manifest once it exists; fall back to B's temporary one."""
    if os.path.exists(MANIFEST_C):
        return MANIFEST_C
    return TMP_MANIFEST_B


def load_manifest(path: Optional[str] = None) -> pd.DataFrame:
    path = path or default_manifest_path()
    df = pd.read_csv(path)
    required = [
        "video_id", "path", "source_cluster_id", "anchor_s", "fps", "duration_s",
        "post_anchor_length_s", "class_code", "split",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"manifest {path} is missing columns: {missing}")
    df["video_id"] = df["video_id"].astype(str)
    df["class_code"] = df["class_code"].astype(int)
    if "decode_ok" in df.columns:
        df["decode_ok"] = df["decode_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["decode_ok"] = True
    return df


def split_frame(man: pd.DataFrame, split: str) -> pd.DataFrame:
    return man[man["split"] == split].reset_index(drop=True)


def labels_map(man: pd.DataFrame) -> Dict[str, int]:
    return dict(zip(man["video_id"].tolist(), man["class_code"].astype(int).tolist()))


# ---------------------------------------------------------------------------
# Answer matrix (CONTRACT 5.4)
# ---------------------------------------------------------------------------
def build_answer_frame(
    video_ids: Sequence[str],
    preds: np.ndarray,          # (n_videos, n_j) int
    probs: Optional[np.ndarray],  # (n_videos, n_j, 5) float or None
    delta_s: float = FINEST_DELTA_S,
    committed: Optional[np.ndarray] = None,  # (n_videos, n_j) bool or None
    j_offset: int = 0,
) -> pd.DataFrame:
    n_v, n_j = preds.shape
    assert n_v == len(video_ids)
    js = np.arange(j_offset, j_offset + n_j, dtype=np.int64)
    frame = {
        "video_id": np.repeat(np.asarray(video_ids, dtype=object), n_j),
        "j": np.tile(js, n_v),
        "delta_s": np.full(n_v * n_j, float(delta_s), dtype=np.float64),
        "pred": preds.reshape(-1).astype(np.int64),
    }
    if probs is None:
        flat = np.full((n_v * n_j, N_CLASSES), np.nan, dtype=np.float64)
    else:
        flat = probs.reshape(-1, N_CLASSES).astype(np.float64)
    for k in range(N_CLASSES):
        frame[f"p{k}"] = flat[:, k]
    if committed is None:
        frame["committed"] = np.zeros(n_v * n_j, dtype=bool)
    else:
        frame["committed"] = committed.reshape(-1).astype(bool)
    return pd.DataFrame(frame, columns=ANSWER_COLUMNS)


def answers_dir(system_id: str) -> str:
    return os.path.join(ANSWERS_DIR, system_id)


def write_answers(system_id: str, answers: pd.DataFrame, card: Dict[str, Any]) -> str:
    out = answers_dir(system_id)
    os.makedirs(out, exist_ok=True)
    answers = answers[ANSWER_COLUMNS]
    answers.to_csv(os.path.join(out, "answers.csv"), index=False)
    write_card(out, card)
    return out


def write_card(out_dir: str, card: Dict[str, Any]) -> None:
    import yaml

    full = {
        "system_id": None,
        "family": None,
        "description": "",
        "backbone": None,
        "trained_on_split": None,
        "dev_tuned_params": {},
        "train_data_unknown": False,
        "cost_note": "",
        "causal": True,
        "parent_system_id": None,
    }
    full.update(card)
    with open(os.path.join(out_dir, "system_card.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(full, fh, allow_unicode=True, sort_keys=False)


def read_answers(system_id_or_path: str) -> pd.DataFrame:
    path = system_id_or_path
    if not path.endswith(".csv"):
        path = os.path.join(answers_dir(system_id_or_path), "answers.csv")
    df = pd.read_csv(path)
    df["video_id"] = df["video_id"].astype(str)
    df["pred"] = df["pred"].astype(int)
    return df


def answers_to_arrays(df: pd.DataFrame):
    """Long answer table -> (video_ids, js[J], preds[n,J], probs[n,J,5]).

    Requires a rectangular table (every video has the same j grid), which every
    system produced by this package satisfies. ``js`` is returned because the grid
    starts before the anchor, so column 0 is not j = 0.
    """
    df = df.sort_values(["video_id", "j"], kind="stable")
    vids = df["video_id"].drop_duplicates().tolist()
    n_j = int(df.groupby("video_id", observed=True)["j"].size().iloc[0])
    js = df["j"].to_numpy()[:n_j].astype(np.int64)
    preds = df["pred"].to_numpy().reshape(len(vids), n_j)
    probs = df[[f"p{k}" for k in range(N_CLASSES)]].to_numpy().reshape(len(vids), n_j, N_CLASSES)
    return vids, js, preds, probs


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------
@dataclass
class ClipFeatures:
    video_id: str
    t_s: np.ndarray   # (T,) float32, actual frame timestamps in seconds
    feat: np.ndarray  # (T, 512) float32


def feature_path(video_id: str, features_dir: str = FEATURES_DIR) -> str:
    return os.path.join(features_dir, f"{video_id}.npz")


def load_features(video_id: str, features_dir: str = FEATURES_DIR) -> Optional[ClipFeatures]:
    path = feature_path(video_id, features_dir)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return ClipFeatures(video_id, z["t_s"].astype(np.float32), z["feat"].astype(np.float32))


def available_feature_ids(features_dir: str = FEATURES_DIR) -> List[str]:
    if not os.path.isdir(features_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(features_dir) if f.endswith(".npz"))
