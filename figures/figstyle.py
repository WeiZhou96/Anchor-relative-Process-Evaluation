"""Shared figure settings: stylelib + font fallback + export (PDF, SVG, 300-dpi PNG)."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

from paths import FIG, INTERNAL, STYLE  # noqa: E402

sys.path.insert(0, str(STYLE))  # vendored stylelib/ next to this file
from stylelib import ROLE, OKABE_ITO, panel_tag, save_figure, style_ax, use_style  # noqa: E402,F401
from stylelib.style_base import FONT_STATUS  # noqa: E402

# CAS single-column text width: 192 mm paper - 2 x 13.7 mm margins = 164.6 mm.
FULL_W = 164.6 / 25.4
HALF_W = FULL_W / 2 - 0.1

EXTRA = {}
use_style()
if not FONT_STATUS.get("calibri_math") and FONT_STATUS.get("calibri"):
    # Calibri Math is not installed on this machine; use Calibri for text and mathtext.
    EXTRA = {"mathtext.fontset": "custom", "mathtext.rm": "Calibri", "mathtext.it": "Calibri:italic",
             "mathtext.bf": "Calibri:bold", "mathtext.sf": "Calibri", "mathtext.fallback": "stixsans"}
    matplotlib.rcParams.update(EXTRA)

FAMILY_COLOR = {
    "clip": OKABE_ITO["blue"],        # prefix-mean (whole-clip) classifiers
    "prefix": OKABE_ITO["sky"],       # causal GRU prefix classifiers
    "postproc": OKABE_ITO["green"],   # smoothing / hysteresis / patience arms
    "commit": OKABE_ITO["orange"],    # commitment rules (MSP, margin, Ringel, TEASER)
    "trivial": "#7F7F7F",
    "block": "#000000",
}
FAMILY_LABEL = {"clip": "prefix-mean classifier", "prefix": "GRU classifier", "postproc": "post-processing arm",
                "commit": "commitment rule", "trivial": "trivial system", "block": "gauge block"}
EMPH = OKABE_ITO["vermillion"]
INK = ROLE["ink"]
MUTED = ROLE["muted"]


def export(fig, name):
    """Write $APE_FIG_OUT/figures/<name>.pdf, .svg and .png (300 dpi); log resolved fonts."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.svg")
    out = save_figure(fig, FIG / f"{name}.pdf", dpi=300, also_png=True, close=True)
    log = INTERNAL / "figure_font_log.json"
    data = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {}
    fam = matplotlib.rcParams["font.sans-serif"]
    resolved = fm.findfont(fm.FontProperties(family=fam))
    data[name] = {"font_status": dict(FONT_STATUS), "extra_rc": EXTRA, "resolved_text_font": resolved.replace("\\", "/")}
    log.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return out
