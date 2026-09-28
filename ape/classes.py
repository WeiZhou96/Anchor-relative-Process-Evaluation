"""Closed-set collision-type class codes for the APE protocol.

Contract section 4 fixes the code table for the whole repository. ``BOT`` (-1)
marks the protocol-masked pre-anchor output and any refusal to answer; per
contract section 8 a ``BOT`` answer at ``delta >= 0`` is scored as wrong, it is
never dropped from the denominator.

The string table below is the *expected* ACCIDENT ``type`` vocabulary. Track C
builds the manifest from the real CSV; if the CSV vocabulary differs, extend
``ALIASES`` here (never renumber ``CLASS_NAMES``) and record it in REPORT.md.
"""

from __future__ import annotations

from typing import Iterable, List

# Contract section 4. Index == class_code.
CLASS_NAMES: List[str] = ["head-on", "rear-end", "t-bone", "sideswipe", "single"]
N_CLASSES: int = len(CLASS_NAMES)

#: Protocol-masked / no-answer symbol. Scored as wrong for ``delta >= 0``.
BOT: int = -1

NAME_TO_CODE = {name: code for code, name in enumerate(CLASS_NAMES)}

#: Spelling variants seen in the wild -> canonical name. Extend, do not renumber.
ALIASES = {
    "head_on": "head-on",
    "headon": "head-on",
    "rear_end": "rear-end",
    "rearend": "rear-end",
    "t_bone": "t-bone",
    "tbone": "t-bone",
    "t-bone/side": "t-bone",
    "side_swipe": "sideswipe",
    "side-swipe": "sideswipe",
    "single_vehicle": "single",
    "single-vehicle": "single",
}


def normalize_type_string(raw: str) -> str:
    """Canonicalise a raw ``type`` string to a name in :data:`CLASS_NAMES`."""
    key = str(raw).strip().lower().replace(" ", "")
    key = ALIASES.get(key, key)
    if key in NAME_TO_CODE:
        return key
    raise KeyError(
        f"unknown collision type {raw!r}; add it to ape.classes.ALIASES "
        f"(known: {sorted(NAME_TO_CODE)})"
    )


def to_code(raw: str) -> int:
    """Map a raw ``type`` string to its integer class code."""
    return NAME_TO_CODE[normalize_type_string(raw)]


def to_codes(raws: Iterable[str]) -> List[int]:
    return [to_code(r) for r in raws]


def code_to_name(code: int) -> str:
    """Inverse of :func:`to_code`; ``BOT`` maps to ``"bot"``."""
    if int(code) == BOT:
        return "bot"
    return CLASS_NAMES[int(code)]


def wrong_class(code: int, n_classes: int = N_CLASSES) -> int:
    """A deterministic *wrong* class for a given truth code.

    Used by the synthetic blocks so that "not yet stabilised" is a concrete,
    reproducible closed-set answer rather than ``BOT``: a block that answered
    ``BOT`` would conflate "wrong answer" with "no answer".
    """
    return (int(code) + 1) % int(n_classes)
