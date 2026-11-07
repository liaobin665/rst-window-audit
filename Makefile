PY ?= python
CITES_DB ?= data/cites.duckdb
CITES_CSV ?= /path/to/cites_trade.csv

.PHONY: help install test reproduce figures checks data pipeline clean

help:
	@echo "rst-window-audit"
	@echo "  make install        install the package and test dependencies"
	@echo "  make test           run the test suite (no raw data)"
	@echo "  make reproduce      redraw figures and run all 66 numeric checks"
	@echo "  make figures        redraw figures only"
	@echo "  make checks         run numeric checks only"
	@echo "  make data           convert CITES_CSV into CITES_DB"
	@echo "  make pipeline       rebuild everything from CITES_DB"
	@echo "  make clean          remove regenerated outputs (keeps figures/)"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

reproduce:
	$(PY) -m rst_audit.cli reproduce

figures:
	$(PY) -m rst_audit.cli figures

checks:
	$(PY) -m rst_audit.cli checks

data:
	$(PY) scripts/prepare_cites_db.py --csv "$(CITES_CSV)" --out "$(CITES_DB)"

pipeline:
	CITES_DB=$(CITES_DB) $(PY) -m rst_audit.cli pipeline

clean:
	rm -rf output .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
