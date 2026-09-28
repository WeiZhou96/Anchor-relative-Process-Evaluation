#!/usr/bin/env bash
# Regenerate the paper's derived intermediates, numeric tables and quantitative figures.
#
#   APE_DATA=/path/to/data  [APE_FIG_OUT=/path/to/out]  [PYTHON=python3]  bash run_all.sh
#
# APE_DATA defaults to ../../data (the derived-data package next to the code repository) when that
# folder exists. APE_FIG_OUT defaults to ./out. The qualitative figures need the original dataset
# frames and run only when APE_QUAL_FRAMES is set (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ -z "${APE_DATA:-}" ] && [ -d ../../data/accident ]; then
  APE_DATA="$(cd ../../data && pwd)"
fi
: "${APE_DATA:?set APE_DATA to the derived-data package root (folder containing accident/ and mmau/)}"
export APE_DATA
export APE_FIG_OUT="${APE_FIG_OUT:-$(pwd)/out}"
export MPLBACKEND=Agg

step() {
  local t0=$SECONDS
  echo "== $*"
  "$PY" -X utf8 "$@"
  echo "   ($(( SECONDS - t0 )) s)"
}

# 1. per-system reference table (level/delay decomposition of RMSCD@H)
step build_system_table.py
# 2. post hoc decompositions (regroupings of stored values)
step tied_pairs_decomposition.py
step knob_decomposition.py
step g4_composition.py
# 3. every manuscript number with its source
step extract_evidence.py
# 4. LaTeX tables from the frozen records and 5. re-analysis table numbers
step make_tables.py
step paper_numbers.py
# 6. consistency check of the stored characterization (recomputes MRD and min R)
step check_reproduce.py
# 7. quantitative figures
step fig_accounting.py
step fig_tied_pairs.py
step fig_comparability.py
step fig_knobs.py
step fig_mmau.py
# 8. qualitative figures (need the original frames; not redistributed)
if [ -n "${APE_QUAL_FRAMES:-}" ]; then
  step fig_qualitative_cases.py
else
  echo "== fig_qualitative_cases.py skipped (APE_QUAL_FRAMES not set; see README.md)"
fi
echo "outputs in $APE_FIG_OUT/{internal,tables,figures}"
