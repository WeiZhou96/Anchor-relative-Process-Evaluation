# -*- coding: utf-8 -*-
"""Axis, legend, annotation and uncertainty helpers.

Spines, hairline y-grid, frameless legend and (a)(b) tags follow
Coordination nnstyle. Halo text / chip labels follow the crack figures.
fill_between bands follow SynCrash (alpha ~ 0.13). Minor ticks are optional
(SCOPE pair-curve).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patheffects import withStroke

from .style_palettes import ROLE
from .style_base import save_figure, use_style

GRID_LW = 0.4
GRID_ALPHA = 0.9
BAND_ALPHA = 0.13
HALO_LW = 2.4
PANEL_SIZE = 9.5


def apply_spines(
    ax: Axes,
    top: bool = False,
    right: bool = False,
    left: bool = True,
    bottom: bool = True,
    color: Optional[str] = None,
    linewidth: float = 0.6,
) -> Axes:
    """Show only the requested spines (default: left + bottom)."""
    vis = {"top": top, "right": right, "left": left, "bottom": bottom}
    edge = color or ROLE["spine"]
    for name, sp in ax.spines.items():
        if vis.get(name, False):
            sp.set_visible(True)
            sp.set_color(edge)
            sp.set_linewidth(linewidth)
        else:
            sp.set_visible(False)
    return ax


def light_grid(
    ax: Axes,
    axis: str = "y",
    which: str = "major",
    *,
    color: Optional[str] = None,
    linestyle: Union[str, tuple] = "-",
    alpha: float = GRID_ALPHA,
    linewidth: float = GRID_LW,
) -> Axes:
    """Hairline grid under the data. Prefer y-only on cartesian plots."""
    ax.grid(
        True,
        axis=axis,
        which=which,
        color=color or ROLE["grid"],
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=0,
    )
    ax.set_axisbelow(True)
    return ax


hairline_grid = light_grid


def style_ax(
    ax: Axes,
    grid: Optional[str] = None,
    *,
    top: bool = False,
    right: bool = False,
) -> Axes:
    """Finish an axis: outward ticks, no top/right, optional light grid.

    Grid is off by default (Coordination). Pass grid='y' for bars/lines.
    """
    apply_spines(ax, top=top, right=right)
    ax.tick_params(length=2.6, pad=2.0, direction="out")
    if grid:
        light_grid(ax, axis=grid)
    return ax


def enable_minor_ticks(ax: Axes, x: int = 2, y: int = 2) -> Axes:
    """SCOPE-style AutoMinorLocator. x/y are subdivisions of each major interval."""
    from matplotlib.ticker import AutoMinorLocator

    if x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(x))
    if y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(y))
    return ax


def panel_tag(
    ax: Axes,
    text: str,
    dx: float = -0.085,
    dy: float = 1.045,
    size: float = PANEL_SIZE,
    weight: str = "bold",
) -> None:
    """Place a bold (a)/(b)/(c) tag in axes-fraction coordinates."""
    ax.text(
        dx, dy, text,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=size, fontweight=weight, color="#000000",
    )


def legend_out(ax: Axes, loc: str = "best", ncol: int = 1, **kwargs):
    """Frameless legend with journal handle lengths. No-op if nothing is labelled."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    kw = dict(frameon=False, loc=loc, ncol=ncol, handlelength=1.4)
    kw.update(kwargs)
    return ax.legend(handles, labels, **kw)


def legend_outside(ax: Axes, loc: str = "center left", **kwargs):
    """RAL stacked-bar pattern: legend to the right of the axes."""
    kw = dict(
        frameon=False,
        loc=loc,
        bbox_to_anchor=(1.01, 0.5),
        handlelength=1.1,
        labelspacing=0.6,
    )
    kw.update(kwargs)
    return ax.legend(**kw)


def despine_all(axes: Union[Axes, Sequence[Axes]]) -> None:
    if isinstance(axes, Axes):
        axes = [axes]
    for ax in axes:
        apply_spines(ax)


def halo_text(ax: Axes, x, y, text, halo: str = "white", halo_lw: float = HALO_LW, **kw):
    """White-stroke label so text stays readable on top of marks (crack)."""
    pe = [withStroke(linewidth=halo_lw, foreground=halo)]
    return ax.text(x, y, text, path_effects=pe, **kw)


def chip_label(
    ax: Axes,
    x,
    y,
    text,
    *,
    fc: str = "white",
    ec: Optional[str] = None,
    tc=None,
    fontsize: float = 7.4,
    pad: float = 0.28,
    zorder: int = 9,
    ha: str = "center",
    va: str = "center",
    **kw,
):
    """Rounded chip used for Δ callouts on the crack figures."""
    ec = ec or ROLE["ours"]
    if tc is None:
        tc = ec
    return ax.text(
        x, y, text, ha=ha, va=va, fontsize=fontsize, color=tc,
        fontweight="bold", zorder=zorder,
        bbox=dict(
            boxstyle="round,pad=%.3f" % pad,
            fc=fc, ec=ec, lw=0.65, alpha=0.97,
        ),
        **kw,
    )


def errorband(ax: Axes, x, y, yerr, color=None, alpha: float = BAND_ALPHA, **plot_kw):
    """Mean curve plus SynCrash-style fill_between band (alpha ~ 0.13)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    color = color or ROLE["ours"]
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha, linewidth=0, zorder=2)
    plot_kw.setdefault("color", color)
    plot_kw.setdefault("lw", 1.4)
    plot_kw.setdefault("zorder", 3)
    return ax.plot(x, y, **plot_kw)


def errorbar_ci(ax: Axes, x, y, lo, hi, color=None, **kwargs):
    """Point estimate with [lo, hi] interval (RAL bootstrap / Coord forest)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    color = color or ROLE["ours"]
    yerr = np.vstack([y - lo, hi - y])
    kw = dict(
        fmt="-o", color=color, lw=1.5, ms=3.6,
        ecolor=color, elinewidth=0.9, capsize=2.0, capthick=0.9,
        markerfacecolor="white", markeredgewidth=0.7, zorder=3,
    )
    kw.update(kwargs)
    return ax.errorbar(x, y, yerr=yerr, **kw)


def style_colorbar(cb, labelsize: float = 6.5, label: Optional[str] = None):
    """Thin colorbar outline used on Coordination heatmaps."""
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=labelsize, length=1.6, width=0.4, pad=1.2)
    if label:
        cb.set_label(label, fontsize=labelsize, labelpad=2)
    return cb


def luminance(rgba) -> float:
    """Rec. 601 luminance of an RGBA tuple in [0, 1]."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    return 0.299 * r + 0.587 * g + 0.114 * b


__all__ = [
    "apply_spines",
    "light_grid",
    "hairline_grid",
    "style_ax",
    "enable_minor_ticks",
    "panel_tag",
    "legend_out",
    "legend_outside",
    "despine_all",
    "halo_text",
    "chip_label",
    "errorband",
    "errorbar_ci",
    "style_colorbar",
    "luminance",
    "use_style",
    "save_figure",
    "BAND_ALPHA",
]
