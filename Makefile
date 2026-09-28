# APE replication package.
#
#   make test            unit and integration tests
#   make install-data    verify and install the derived-data package (APE_DATA=/path/to/ape-derived-data)
#   make reanalysis      rerun the re-analyses from the released answers into outputs/*_rerun
#   make report          audit tables and internal audit figures in outputs/r2c/ (from released results)
#   make audit-check     k_c_verify: consistency assertions over the installed audit results
#   make paper           article and ESM figures and tables (figures/run_all.sh)
#
#   make manifest ...    data-track targets of the original pipeline (need the raw ACCIDENT videos)
#
# Every variable can be overridden: make PY=/path/to/python APE_DATA=... <target>

PY       ?= $(or $(APE_PY),python)
ROOT     ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TMP      ?= $(or $(APE_TMP),$(ROOT)/tmp)
APE_DATA ?=
export PYTHONPATH := $(ROOT)

.PHONY: help test install-data verify-data reanalysis run002 m3 stage2 report tables figs audit-check paper \
        smoke full manifest manifest-smoke manifest-synthetic clusters stats deps

help:
	@echo "targets: test install-data verify-data reanalysis run002 m3 stage2 report audit-check paper"
	@echo "         manifest manifest-smoke manifest-synthetic clusters stats smoke full deps"

test:
	CUBLAS_WORKSPACE_CONFIG=:4096:8 $(PY) -m pytest -q tests data/mmau

verify-data:
	$(PY) $(ROOT)/scripts/release/install_data.py verify --data $(APE_DATA)

install-data: verify-data
	$(PY) $(ROOT)/scripts/release/install_data.py install --data $(APE_DATA) --root $(ROOT)

# ---- re-analyses (CPU; start from outputs/answers restored by install-data)
reanalysis: run002 m3 stage2

run002:
	$(PY) $(ROOT)/scripts/method_experiments.py --source-root $(ROOT) --out $(ROOT)/outputs/run002_rerun

m3:
	$(PY) $(ROOT)/scripts/m3_reanalysis.py --source-root $(ROOT) --run002 $(ROOT)/outputs/run002 \
	      --out $(ROOT)/outputs/m3_rerun --bootstrap 2000
	$(PY) $(ROOT)/scripts/m3_compare_runs.py $(ROOT)/outputs/m3_20260928 $(ROOT)/outputs/m3_rerun

stage2:
	$(PY) $(ROOT)/scripts/stage2_calibration.py --out $(ROOT)/outputs/stage2_calibration_rerun
	$(PY) $(ROOT)/scripts/stage2_plasmode.py --source-root $(ROOT) --out $(ROOT)/outputs/stage2_plasmode_rerun
	$(PY) $(ROOT)/scripts/stage2_final_calibration.py --source-root $(ROOT) --out $(ROOT)/outputs/stage2_final_calibration_rerun
	$(PY) $(ROOT)/scripts/stage2_final_calibration.py --source-root $(ROOT) --out $(ROOT)/outputs/stage2_exact_bootstrap_rerun \
	      --only-h10-decomposition --bootstrap 2000

# ---- audit tables and checks (CPU; K-c results)
report: tables figs

tables:
	$(PY) -X utf8 $(ROOT)/report/make_tables.py

figs:
	$(PY) -X utf8 $(ROOT)/report/make_figs.py

audit-check:
	$(PY) $(ROOT)/scripts/k_c_verify.py

# ---- article and ESM figures/tables
paper:
	APE_DATA=$(APE_DATA) PYTHON=$(PY) bash $(ROOT)/figures/run_all.sh

# ---- original data-track targets (raw ACCIDENT videos in $APE_ACCIDENT)
smoke: manifest-smoke report
	@echo "smoke done"

full: manifest manifest-synthetic clusters stats report
	@echo "full pass done"

manifest-smoke:
	mkdir -p $(TMP)
	$(PY) -X utf8 $(ROOT)/data/build_manifest.py --kind real --limit 50 --out $(TMP)/manifest_real_smoke.csv

manifest:
	$(PY) -X utf8 $(ROOT)/data/build_manifest.py --kind real

manifest-synthetic:
	$(PY) -X utf8 $(ROOT)/data/build_manifest.py --kind synthetic

clusters:
	$(PY) -X utf8 $(ROOT)/data/clusters.py --pairs 100

stats:
	$(PY) -X utf8 $(ROOT)/data/stats.py

PIP_INDEX ?= https://pypi.org/simple

deps:
	$(PY) -m pip install --index-url $(PIP_INDEX) -r $(ROOT)/requirements.txt
