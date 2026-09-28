# -*- coding: utf-8 -*-
"""rcParams, geometry, fonts and export for journal quantitative figures.

Vendored copy of the authors' matplotlib style package (release note: only this docstring, one
comment below and one comment in style_palettes.py were edited to remove local paths/names).
Skeleton: Coordination nnstyle (Nature / Science / TPAMI hairline plate).
Font contract: Calibri Math first, then Calibri / Arial; on hosts without these fonts matplotlib
falls back to DejaVu Sans (see FONT_STATUS).
Default column is 6.93 in; IEEE 7.16 in is an explicit alias.
Dependencies: matplotlib + numpy only. No seaborn.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt

# Nature / cas-sc text width (Coordination nnstyle; ~6.9 in).
TEXT_WIDTH_IN = 6.93
HALF_WIDTH_IN = 3.35
IEEE_TEXT_WIDTH_IN = 7.16
IEEE_COL_WIDTH_IN = 3.487
IEEE_HALF_WIDTH_IN = IEEE_COL_WIDTH_IN
NATURE_TEXT_WIDTH_IN = TEXT_WIDTH_IN
NATURE_HALF_WIDTH_IN = HALF_WIDTH_IN

GRID_COLOR = "#D9D9D9"
EDGE_COLOR = "#333333"
LABEL_COLOR = "#111111"

FONT_STACK = ["Calibri Math", "Calibri", "Arial", "Helvetica", "DejaVu Sans"]
FONT_STATUS: Dict[str, Union[str, bool]] = {
    "calibri_math": False,
    "calibri": False,
    "mathtext": "dejavusans",
    "note": "not yet registered",
}

_FONT_CANDIDATES = {
    "calibri_math": (
        "C:/Windows/Fonts/Calibri Math.ttf",
        "C:/Windows/Fonts/calibri math.ttf",
        "C:/Windows/Fonts/CalibriMath.ttf",
        "C:/Windows/Fonts/CalibriMath.otf",
    ),
    "calibri": (
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/Calibri.ttf",
    ),
}


def register_preferred_fonts() -> Dict[str, Union[str, bool]]:
    """Register Calibri Math / Calibri if the files exist. Idempotent."""
    from matplotlib import font_manager as fm

    notes = []
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    extra = {
        "calibri_math": (
            os.path.join(fonts_dir, "CalibriMath.ttf"),
            os.path.join(fonts_dir, "Calibri Math.ttf"),
        ),
        "calibri": (os.path.join(fonts_dir, "calibri.ttf"),),
    }
    for key, paths in _FONT_CANDIDATES.items():
        candidates = tuple(paths) + extra.get(key, ())
        hit = next((p for p in candidates if os.path.isfile(p)), None)
        if hit is None:
            FONT_STATUS[key] = False
            notes.append("%s not found on disk" % key)
            continue
        try:
            fm.fontManager.addfont(hit)
            FONT_STATUS[key] = True
            notes.append("%s <- %s" % (key, hit.replace("\\", "/")))
        except (OSError, RuntimeError) as exc:
            FONT_STATUS[key] = False
            notes.append("%s register failed: %s" % (key, exc))

    if FONT_STATUS["calibri_math"]:
        FONT_STATUS["mathtext"] = "custom"
    else:
        FONT_STATUS["mathtext"] = "dejavusans"
    FONT_STATUS["note"] = "; ".join(notes)
    return dict(FONT_STATUS)


def _font_rc() -> dict:
    params = {
        "font.family": "sans-serif",
        "font.sans-serif": list(FONT_STACK),
        "mathtext.fontset": "dejavusans",
        "mathtext.fallback": "stixsans",
    }
    if FONT_STATUS.get("calibri_math"):
        params.update({
            "mathtext.fontset": "custom",
            "mathtext.rm": "Calibri Math",
            "mathtext.it": "Calibri Math:italic",
            "mathtext.bf": "Calibri Math:bold",
            "mathtext.sf": "Calibri" if FONT_STATUS.get("calibri") else "Calibri Math",
            "mathtext.fallback": "stixsans",
        })
    return params


DEFAULT_RCPARAMS = {
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "figure.titlesize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 6,
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "axes.edgecolor": EDGE_COLOR,
    "axes.labelcolor": LABEL_COLOR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "axes.titlepad": 4.0,
    "axes.labelpad": 2.5,
    "axes.axisbelow": True,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.minor.size": 1.4,
    "ytick.minor.size": 1.4,
    "xtick.color": EDGE_COLOR,
    "ytick.color": EDGE_COLOR,
    "xtick.top": False,
    "ytick.right": False,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.9,
    "grid.linestyle": ":",
    "lines.linewidth": 1.1,
    "lines.markersize": 3.6,
    "lines.markeredgewidth": 0.5,
    "errorbar.capsize": 2.0,
    "scatter.edgecolors": "none",
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.32,
    "legend.borderaxespad": 0.2,
    "legend.columnspacing": 1.0,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "savefig.transparent": False,
}

_WIDTHS = {
    "text": TEXT_WIDTH_IN,
    "single": TEXT_WIDTH_IN,
    "double": TEXT_WIDTH_IN,
    "full": TEXT_WIDTH_IN,
    "nature": TEXT_WIDTH_IN,
    "half": HALF_WIDTH_IN,
    "nature_half": HALF_WIDTH_IN,
    "ieee": IEEE_TEXT_WIDTH_IN,
    "ieee_double": IEEE_TEXT_WIDTH_IN,
    "ieee_single": IEEE_COL_WIDTH_IN,
    "ieee_half": IEEE_COL_WIDTH_IN,
}


def use_style(extra: Optional[dict] = None) -> None:
    """Install rcParams. Call once before building figures."""
    register_preferred_fonts()
    params = dict(DEFAULT_RCPARAMS)
    params.update(_font_rc())
    if extra:
        params.update(extra)
    mpl.rcParams.update(params)


apply_style = use_style


def figsize(width: str = "text", height: float = 2.55) -> Tuple[float, float]:
    """Return (w, h) in inches. Default ``text``/``single`` is 6.93 in."""
    return (_WIDTHS.get(width, TEXT_WIDTH_IN), height)


def save_figure(
    fig,
    path: Union[str, os.PathLike],
    *,
    dpi: int = 300,
    also_png: bool = True,
    close: bool = True,
) -> str:
    """Write vector PDF (preferred) and optional PNG backup. Returns the PDF path."""
    path = str(path).replace("\\", "/")
    root, ext = os.path.splitext(path)
    parent = os.path.dirname(path) or "."
    if parent and parent not in {".", ""}:
        os.makedirs(parent, exist_ok=True)
    if ext.lower() in {".pdf", ".png", ".svg", ".eps"}:
        fig.savefig(path, dpi=dpi)
        out = path
        if also_png and ext.lower() == ".pdf":
            fig.savefig(root + ".png", dpi=dpi)
    else:
        out = root + ".pdf"
        fig.savefig(out, dpi=dpi)
        if also_png:
            fig.savefig(root + ".png", dpi=dpi)
    if close:
        plt.close(fig)
    return out.replace("\\", "/")


def savefig_both(
    fig,
    path: Union[str, os.PathLike],
    *,
    dpi: int = 300,
    png: bool = True,
    png_subdir: Optional[str] = None,
    close: bool = True,
) -> Tuple[str, Optional[str]]:
    """Write vector PDF + 300 dpi PNG. Returns ``(pdf_path, png_path)``."""
    path = str(path).replace("\\", "/")
    root, ext = os.path.splitext(path)
    stem = root if ext.lower() in {".pdf", ".png", ".svg", ".eps"} else path.rstrip("/")
    parent = os.path.dirname(stem) or "."
    os.makedirs(parent, exist_ok=True)
    pdf_path = stem + ".pdf"
    fig.savefig(pdf_path, dpi=dpi)
    png_path = None
    if png:
        if png_subdir:
            png_dir = os.path.join(parent, png_subdir)
            os.makedirs(png_dir, exist_ok=True)
            png_path = os.path.join(png_dir, os.path.basename(stem) + ".png")
        else:
            png_path = stem + ".png"
        fig.savefig(png_path, dpi=dpi)
        png_path = str(png_path).replace("\\", "/")
    if close:
        plt.close(fig)
    return pdf_path.replace("\\", "/"), png_path


def save(fig, stem, outdir=None, png=True, dpi=300):
    """Coordination ``nnstyle.save`` compatible alias. PNG goes to ``<outdir>/png/``."""
    if outdir is None:
        target = stem
    else:
        os.makedirs(str(outdir), exist_ok=True)
        target = os.path.join(str(outdir), os.path.basename(str(stem)))
    return savefig_both(
        fig, target, dpi=dpi, png=png, png_subdir="png" if png else None,
    )


def panel_tag(ax, text, dx=-0.08, dy=1.04, weight="bold", size=9.5):
    """Bold (a)/(b)/(c) tag in axes-fraction coordinates (top-left)."""
    ax.text(
        dx, dy, text, transform=ax.transAxes, ha="left", va="top",
        fontsize=size, fontweight=weight, color="#000000", clip_on=False,
    )


def hairline_grid(ax, axis="y", which="major"):
    """Light grid under data; opt-in (do not force on every axes)."""
    ax.grid(
        True, axis=axis, which=which, color=GRID_COLOR,
        linewidth=0.4, alpha=0.9, linestyle=":", zorder=0,
    )
    ax.set_axisbelow(True)


def fallback_mathtext_note() -> Optional[str]:
    """Delivery note when Calibri Math is missing on this host."""
    register_preferred_fonts()
    if FONT_STATUS.get("calibri_math"):
        return None
    return str(FONT_STATUS.get("note") or (
        "Calibri Math not found; body Calibri if present, mathtext dejavusans."
    ))


__all__ = [
    "TEXT_WIDTH_IN",
    "HALF_WIDTH_IN",
    "IEEE_TEXT_WIDTH_IN",
    "IEEE_COL_WIDTH_IN",
    "IEEE_HALF_WIDTH_IN",
    "NATURE_TEXT_WIDTH_IN",
    "NATURE_HALF_WIDTH_IN",
    "GRID_COLOR",
    "EDGE_COLOR",
    "LABEL_COLOR",
    "FONT_STACK",
    "FONT_STATUS",
    "DEFAULT_RCPARAMS",
    "register_preferred_fonts",
    "use_style",
    "apply_style",
    "figsize",
    "save_figure",
    "savefig_both",
    "save",
    "panel_tag",
    "hairline_grid",
    "fallback_mathtext_note",
]
