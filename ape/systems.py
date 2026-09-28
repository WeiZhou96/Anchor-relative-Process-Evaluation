"""System identity helpers: family, arm rule, seed, parent.

The audit library names a system by what it *is* -- family, arm, tuned
parameter, seed -- all inside one ``system_id`` string. Reporting needs to group
those rows by the **rule** rather than by the tuned parameter: ``ema0p2`` and
``ema0p7`` are the same post-processing rule with the value that the development
subset chose for that seed, so grouping them by arm string would report one seed
per group and make the seed spread uncomputable.

``arm_rule`` therefore strips the tuned value and keeps the rule identity:

    postproc__ema0p2__prefix__gru512__seed20260904   -> ema
    postproc__patience6__prefix__gru512__seed20260905 -> patience
    commit__msp0p9__prefix__gru512__seed20260903     -> commit_msp
    commit__margin0p4__prefix__gru512__seed20260904  -> commit_margin
    prefix__gru512__seed20260903                     -> prefix_gru512
    clip__r18mean__seed20260905                      -> clip
    trivial__constbot                                -> trivial_constbot
    block__osc__d0-4_p-1                             -> block_osc

Parsing is by naming convention, so it is a reporting convenience and never a
source of truth: the system card remains authoritative for what a system is.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

#: leading alphabetic part of an arm token: "ema0p2" -> "ema", "majority7" -> "majority"
_ARM_RULE = re.compile(r"^([A-Za-z]+)")
_SEED = re.compile(r"seed(\d+)")

UNKNOWN = "unknown"


def _rule_of(token: str) -> str:
    m = _ARM_RULE.match(str(token))
    return m.group(1).lower() if m else UNKNOWN


def seed_of(system_id: str) -> Optional[str]:
    m = _SEED.search(str(system_id))
    return m.group(1) if m else None


def family_of(system_id: str) -> str:
    head = str(system_id).split("__", 1)[0]
    return head.lower() if head else UNKNOWN


def arm_rule(system_id: str) -> str:
    """Rule-level identity of a system, with any dev-tuned value stripped."""
    parts = str(system_id).split("__")
    fam = parts[0].lower() if parts else UNKNOWN
    arm = parts[1] if len(parts) > 1 else ""

    if fam == "trivial":
        return f"trivial_{_rule_of(arm)}" if arm else "trivial"
    if fam == "clip":
        return "clip"
    if fam == "prefix":
        return f"prefix_{arm.lower()}" if arm else "prefix"
    if fam == "postproc":
        return _rule_of(arm)
    if fam == "commit":
        return f"commit_{_rule_of(arm)}"
    if fam == "block":
        return f"block_{_rule_of(arm)}"
    if fam == "standin":
        return "standin"
    return fam or UNKNOWN


def arm_value(system_id: str) -> Optional[str]:
    """The dev-tuned value carried by the arm token, e.g. ``ema0p2`` -> ``0p2``."""
    parts = str(system_id).split("__")
    if len(parts) < 2 or parts[0].lower() not in ("postproc", "commit"):
        return None
    arm = parts[1]
    rule = _rule_of(arm)
    tail = arm[len(rule):]
    return tail or None


def parent_of(system_id: str) -> Optional[str]:
    """The base system a post-processing or commitment arm sits on."""
    parts = str(system_id).split("__")
    if len(parts) < 3 or parts[0].lower() not in ("postproc", "commit"):
        return None
    return "__".join(parts[2:])


def describe(system_id: str) -> Dict[str, Optional[str]]:
    """All identity fields at once, for a report row."""
    return {
        "system_id": str(system_id),
        "family": family_of(system_id),
        "arm_rule": arm_rule(system_id),
        "arm_value": arm_value(system_id),
        "seed": seed_of(system_id),
        "parent": parent_of(system_id),
    }
