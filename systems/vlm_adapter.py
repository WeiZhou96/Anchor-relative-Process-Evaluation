"""Read-only VLM adapter: interface and fixed closed-set prompt. STUB THIS ROUND.

No model is downloaded and no answer matrix is produced here.  What is fixed now
is the part that must not be tuned later: the prompt template, the decoding
contract, and the mapping from free text to the five class codes.  A VLM is an
exam-taker like any other system in this package; prompt wording is frozen on
``split=dev`` before the audit split is touched and is never presented as a
contribution.

Causality: a VLM answers on the frames of one prefix only.  The adapter is
therefore handed a list of frame timestamps already filtered by ``t_s <= end_s``;
it must never receive the clip duration, the distance to the clip end, or any
frame past the prefix end.

To implement a backend later, subclass ``VLMBackend`` and pass it to
``VLMSystem.run``; the answer-matrix writing path is already shared with the
trained systems via ``common.build_answer_frame`` / ``common.write_answers``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from systems import common as C

# ---------------------------------------------------------------------------
# Frozen prompt (S1 freeze candidate; wording tuned on split=dev only)
# ---------------------------------------------------------------------------
CLOSED_SET_PROMPT = """You are shown consecutive frames from a single fixed roadside camera.
A collision occurs during this footage. Frames are given in time order and stop at the
current moment; you cannot see anything that happens after them.

Name the configuration of the collision. Answer with exactly one of these five labels
and nothing else:

head-on
rear-end
t-bone
sideswipe
single

Answer:"""

# The only accepted surface forms; anything else is an unparsable answer and is
# scored as wrong, never as an abstention (idea 3.1: no implicit abstention).
LABEL_ALIASES: Dict[str, int] = {
    "head-on": 0, "head on": 0, "headon": 0, "frontal": 0,
    "rear-end": 1, "rear end": 1, "rearend": 1, "rear-ended": 1,
    "t-bone": 2, "t bone": 2, "tbone": 2, "side-impact": 2, "broadside": 2,
    "sideswipe": 3, "side-swipe": 3, "side swipe": 3,
    "single": 4, "single-vehicle": 4, "single vehicle": 4, "solo": 4,
}


def parse_label(text: str) -> int:
    """Map raw generated text to a class code; return C.BOT only if nothing matches."""
    s = (text or "").strip().lower()
    if s in LABEL_ALIASES:
        return LABEL_ALIASES[s]
    hits = [(s.find(k), v) for k, v in LABEL_ALIASES.items() if k in s]
    if not hits:
        return C.BOT
    return min(hits)[1]


@dataclass
class PrefixRequest:
    """Everything a backend is allowed to see for one (video, prefix) pair."""
    video_id: str
    j: int
    end_s: float
    frame_times_s: Sequence[float]   # already filtered to t <= end_s
    frame_paths: Optional[Sequence[str]] = None
    prompt: str = CLOSED_SET_PROMPT


class VLMBackend:
    """Interface a concrete video-language model must satisfy. Not implemented this round."""

    name = "abstract"
    train_data_unknown = True

    def answer(self, req: PrefixRequest) -> str:
        raise NotImplementedError("no VLM backend is wired up in this round")

    def logprobs(self, req: PrefixRequest) -> Optional[np.ndarray]:
        """Optional 5-vector over the label tokens; None means the card records no probabilities."""
        return None


class VLMSystem:
    """Driver that would turn a backend into a CONTRACT 5.4 answer matrix."""

    def __init__(self, backend: VLMBackend, system_id: str, max_frames_per_prefix: int = 8):
        self.backend = backend
        self.system_id = system_id
        self.max_frames_per_prefix = max_frames_per_prefix

    def card(self) -> Dict:
        return {
            "system_id": self.system_id,
            "family": "vlm",
            "description": "read-only video-language model queried with a frozen closed-set prompt",
            "backbone": self.backend.name,
            "trained_on_split": "none (not trained by us)",
            "dev_tuned_params": {"prompt": "frozen after split=dev",
                                 "max_frames_per_prefix": self.max_frames_per_prefix},
            "train_data_unknown": bool(self.backend.train_data_unknown),
            "cost_note": "one generation per (clip, prefix); the dominant cost in the library",
            "causal": True,
            "parent_system_id": None,
            "subsampling_equivalent": True,
            "subsampling_note": "each prefix is answered independently from its own frames",
            "status": "NOT RUN in this round; interface only",
        }

    def run(self, *args, **kwargs):
        raise NotImplementedError(
            "VLM systems are declared but not executed in this round: no large model is "
            "downloaded. See systems/README_vlm.md for what a backend must provide."
        )
