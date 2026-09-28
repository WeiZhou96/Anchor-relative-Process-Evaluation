"""Shared fixtures and hand-built answer matrices for the APE test suite."""

from __future__ import annotations

import os
import sys
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ape.classes import N_CLASSES  # noqa: E402
from ape.metrics import PROB_COLS, AnswerTable  # noqa: E402
from ape.protocol import load_manifest, load_protocol  # noqa: E402

FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")
MANIFEST_MIN = os.path.join(FIXTURE_DIR, "manifest_min.csv")
PI0_PATH = os.path.join(REPO_ROOT, "protocol", "pi0.yaml")


@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture(scope="session")
def cfg():
    return load_protocol(PI0_PATH)


@pytest.fixture(scope="session")
def manifest() -> pd.DataFrame:
    return load_manifest(MANIFEST_MIN)


def make_manifest(rows: Sequence[dict]) -> pd.DataFrame:
    """Build a minimal in-memory manifest from ``(video_id, anchor_s, post_s, y)``."""
    out = []
    for i, r in enumerate(rows):
        anchor = float(r["anchor_s"])
        post = float(r["post_s"])
        out.append(
            {
                "video_id": str(r["video_id"]),
                "source_cluster_id": str(r.get("cluster", r["video_id"])),
                "anchor_s": anchor,
                "fps": float(r.get("fps", 25.0)),
                "duration_s": anchor + post,
                "post_anchor_length_s": post,
                "class_code": int(r.get("y", 0)),
                "class_name": "synthetic",
            }
        )
    return pd.DataFrame(out)


def make_answers(
    preds: Dict[str, Sequence[int]],
    delta_s: float = 0.5,
    j_min: int = 0,
    system_id: str = "handmade",
    committed: Dict[str, Sequence[bool]] = None,
    confidence: float = None,
    family: str = "trivial",
) -> AnswerTable:
    """Build an :class:`AnswerTable` straight from explicit prediction paths."""
    frames = []
    for vid, path in preds.items():
        path = list(path)
        js = np.arange(j_min, j_min + len(path), dtype=np.int64)
        piece = pd.DataFrame(
            {"video_id": str(vid), "j": js, "delta_s": float(delta_s), "pred": path}
        )
        if confidence is None:
            for c in PROB_COLS:
                piece[c] = np.nan
        else:
            probs = np.full((len(path), N_CLASSES), (1.0 - confidence) / (N_CLASSES - 1))
            for k, p in enumerate(path):
                if 0 <= int(p) < N_CLASSES:
                    probs[k, int(p)] = confidence
            for c, name in enumerate(PROB_COLS):
                piece[name] = probs[:, c]
        if committed is not None and vid in committed:
            piece["committed"] = list(committed[vid])
        else:
            piece["committed"] = False
        frames.append(piece)
    df = pd.concat(frames, ignore_index=True)
    return AnswerTable.from_frame(df, system_id, {"system_id": system_id, "family": family})


# --------------------------------------------------------------------------
# the four hand-written trajectories of contract section 6 / idea S2
# --------------------------------------------------------------------------
#: H = 3.0 s with Delta = 0.5 s gives seven grid points, offsets 0.0 .. 3.0.
HANDMADE_H = 3.0
HANDMADE_DELTA = 0.5
HANDMADE_PATHS = {
    # stable and correct from the anchor on
    "V_stable": [0, 0, 0, 0, 0, 0, 0],
    # wrong first, then correct and staying correct
    "V_wrong_then_right": [1, 1, 0, 0, 0, 0, 0],
    # correct first, then wrong and staying wrong
    "V_right_then_wrong": [0, 0, 0, 0, 1, 1, 1],
    # repeated flipping, settling correct only at the very end
    "V_flipping": [1, 0, 1, 0, 1, 0, 0],
}
#: Ground truth worked out by hand; see tests/test_metrics_handmade.py.
HANDMADE_TRUTH = {
    "V_stable": {"rmscd": 0.0, "flips": 0, "first_stable_j": 0},
    "V_wrong_then_right": {"rmscd": 0.75, "flips": 1, "first_stable_j": 2},
    "V_right_then_wrong": {"rmscd": 3.0, "flips": 1, "first_stable_j": -1},
    "V_flipping": {"rmscd": 2.25, "flips": 5, "first_stable_j": 5},
}
HANDMADE_S_CURVE = [0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 0.75]
HANDMADE_RMSCD = 1.5


@pytest.fixture
def handmade():
    """Manifest plus answer table for the four hand-written trajectories."""
    mf = make_manifest(
        [
            {"video_id": v, "anchor_s": 5.0, "post_s": 4.0, "y": 0, "cluster": v}
            for v in HANDMADE_PATHS
        ]
    )
    at = make_answers(HANDMADE_PATHS, delta_s=HANDMADE_DELTA, system_id="handmade")
    return mf, at
