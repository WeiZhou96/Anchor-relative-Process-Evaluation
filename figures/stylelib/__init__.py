# -*- coding: utf-8 -*-
"""Quantitative figure style package (implementation)."""
from .style_base import DEFAULT_RCPARAMS, figsize, save_figure, use_style
from .style_palettes import CATEGORICAL_8, OKABE_ITO, ROLE, categorical, role_colors
from .style_helpers import apply_spines, legend_out, light_grid, panel_tag, style_ax

__all__ = [
    "DEFAULT_RCPARAMS", "figsize", "save_figure", "use_style",
    "OKABE_ITO", "ROLE", "CATEGORICAL_8", "categorical", "role_colors",
    "apply_spines", "light_grid", "style_ax", "panel_tag", "legend_out",
]
