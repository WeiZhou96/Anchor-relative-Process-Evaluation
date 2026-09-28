"""Shared paths for the evidence-extraction, table and plotting scripts (release configuration).

All locations come from environment variables; nothing is machine specific.

  APE_DATA         root of the derived-data package (the folder that contains accident/ and mmau/).
                   Required; run_all.sh sets it to ../../data when that folder exists.
  APE_ROOT         root of the code repository (contains ape/ and report/text/g0_length_shift_stats.json).
                   Default: the parent folder of this figures/ folder.
  APE_FIG_OUT      output root; figures/, tables/ and internal/ are created below it.
                   Default: <this folder>/out.
  APE_QUAL_FRAMES  only for fig_qualitative_cases.py: folder with the 18 original dataset frames
                   (not redistributed; see README.md and extract_qual_frames.py).

The names REMOTE04, R2C_TABLES, MANIFEST, MMAU, REPO, FIG, TAB, INTERNAL and STYLE are those used by
the original authoring scripts; only their targets changed.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("APE_ROOT") or HERE.parent).resolve()
if not os.environ.get("APE_DATA"):
    raise SystemExit("APE_DATA is not set. Point it to the derived-data package root "
                     "(the folder containing accident/ and mmau/), e.g. export APE_DATA=/path/to/data")
DATA = Path(os.environ["APE_DATA"]).resolve()
if not (DATA / "accident").is_dir():
    raise SystemExit(f"APE_DATA={DATA} does not contain accident/; is it the derived-data package root?")
OUT = Path(os.environ.get("APE_FIG_OUT") or HERE / "out").resolve()

# Frozen ACCIDENT audit records (formerly a local copy of the 04-server outputs/).
REMOTE04 = DATA / "accident" / "outputs"
# Code repository: REPO / "report/text/g0_length_shift_stats.json".
REPO = ROOT
R2C_TABLES = REMOTE04 / "r2c" / "tables"
MANIFEST = DATA / "accident" / "manifest" / "manifest_real.csv"
# MM-AU development records (same layout as the original "deliverables" folder).
MMAU = DATA / "mmau" / "deliverables"
FIG = OUT / "figures"
TAB = OUT / "tables"
INTERNAL = OUT / "internal"
STYLE = HERE  # the vendored stylelib/ package lives next to this file
for _d in (FIG, TAB, INTERNAL):
    _d.mkdir(parents=True, exist_ok=True)

HASH = {4.0: "77200b351bf1", 10.0: "8ac32aae418b", 21.5: "3fb251dc210c"}
METRICS = ["RMSCD@H", "S_H@1", "S_H@3", "end_window_macro_acc", "median_flips"]
METRIC_FILE = {"RMSCD@H": "RMSCDatH", "S_H@1": "S_Hat1", "S_H@3": "S_Hat3",
               "end_window_macro_acc": "end_window_macro_acc", "median_flips": "median_flips"}


def placeholder(path) -> str:
    """Replace the configured roots in a path string by <APE_FIG_OUT>, <APE_DATA>, <APE_ROOT>."""
    s = str(path)
    for root, tag in ((OUT, "<APE_FIG_OUT>"), (DATA, "<APE_DATA>"), (ROOT, "<APE_ROOT>")):
        s = s.replace(str(root), tag)
    return s.replace("\\", "/")
