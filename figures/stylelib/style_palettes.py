# -*- coding: utf-8 -*-
"""Colour-blind-safe palettes distilled from extracted quantitative figures.

Default categorical palette is Okabe-Ito (Coordination nnstyle, RAL
make_figures). Historical palettes from crack / SynCrash / SCOPE / RAL-NPG
are kept for same-paper reproduction and are not the default for new figures.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

# Okabe & Ito (2008). Safe under deuteranopia, protanopia and tritanopia.
# Sources: earlier figure-style modules of the authors (not included).
OKABE_ITO: Dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

# Semantic roles used across Coordination / RAL (not paper-specific sites).
ROLE: Dict[str, str] = {
    "ours": OKABE_ITO["vermillion"],
    "baseline": "#4D4D4D",
    "oracle": OKABE_ITO["green"],
    "blue": OKABE_ITO["blue"],
    "sky": OKABE_ITO["sky"],
    "orange": OKABE_ITO["orange"],
    "purple": OKABE_ITO["purple"],
    "gray": "#9E9E9E",
    "safe": OKABE_ITO["blue"],
    "fail": OKABE_ITO["vermillion"],
    "ambiguous": "#BDBDBD",
    "grid": "#D9D9D9",
    "grid_alt": "#C9D0DA",
    "annot": "#3A3A3A",
    "spine": "#333333",
    "ink": "#111111",
    "muted": "#6B7280",
    "rule": "#9A9A9A",
    "infeasible": "#EFEFEF",
    "mist": "#F4F6F8",
}

CATEGORICAL_8: List[str] = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["green"],
    OKABE_ITO["orange"],
    OKABE_ITO["sky"],
    OKABE_ITO["purple"],
    OKABE_ITO["yellow"],
    "#4D4D4D",
]

SEQUENTIAL = "cividis"
DIVERGING = "RdBu_r"

# --- historical, same-paper reproduction only --------------------------------

# Road-crack figures (fig06 / fig09 / fig10). Grey-blue family, hero red.
CRACK: Dict[str, str] = {
    "blue": "#3F6CA8",
    "purple": "#6E5BAA",
    "teal": "#2E7F86",
    "green": "#4F8E58",
    "red": "#C13B33",
    "orange": "#D27A22",
    "ink": "#1F2937",
    "muted": "#6B7280",
    "rule": "#9CA3AF",
    "soft": "#D5DCE6",
    "mist": "#F4F6F8",
}

# SynCrash t-SNE / other figures. ColorBrewer-like saturated set.
SYNCRASH: Dict[str, str] = {
    "blue": "#2171B5",
    "red": "#CB181D",
    "green": "#238B45",
    "orange": "#D94801",
    "purple": "#6A51A3",
    "teal": "#006D75",
    "gray": "#636363",
    "pink": "#DD3497",
    "gold": "#B8860B",
    "cyan": "#0097A7",
    "brown": "#8B4513",
}

SYNCRASH_CATEGORICAL: List[str] = [
    SYNCRASH["blue"],
    SYNCRASH["red"],
    SYNCRASH["green"],
    SYNCRASH["orange"],
    SYNCRASH["purple"],
    SYNCRASH["teal"],
    SYNCRASH["pink"],
    SYNCRASH["gold"],
    SYNCRASH["cyan"],
    SYNCRASH["brown"],
    SYNCRASH["gray"],
]

# SCOPE pair-curve. Okabe-inspired, hero is a brick red rather than vermillion.
SCOPE: Dict[str, str] = {
    "ours": "#A6383C",
    "naive": "#B07B5F",
    "base": "#7A7D82",
    "grid": "#1A1A1A",
    "spine": "#2B2D31",
}

# RAL composite figures. Nature Publishing Group print set (not Okabe).
NPG: Dict[str, str] = {
    "red": "#E64B35",
    "cyan": "#4DBBD5",
    "green": "#00A087",
    "navy": "#3C5488",
    "coral": "#F39B7F",
    "gray": "#8491B4",
    "teal": "#91D1C2",
    "dkred": "#DC0000",
    "brown": "#7E6148",
    "sand": "#B09C85",
}


def categorical(n: int) -> List[str]:
    """Return n categorical colours, cycling the Okabe-derived eight."""
    if n <= 0:
        return []
    return [CATEGORICAL_8[i % len(CATEGORICAL_8)] for i in range(n)]


def role_colors(names: Sequence[str]) -> List[str]:
    """Map role names to ROLE; unknown names fall back to categorical order."""
    out: List[str] = []
    for i, name in enumerate(names):
        key = str(name).lower().strip()
        out.append(ROLE.get(key, CATEGORICAL_8[i % len(CATEGORICAL_8)]))
    return out
