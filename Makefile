PY      ?= python3
export PYTHONPATH := src
CSV     ?= data/raw/paysim_like.csv
PROV    ?= synthetic

.PHONY: help setup data run test lint verify readme readme-check clean

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup: ## create .venv and install everything
	$(PY) -m venv .venv
	.venv/bin/pip install -r requirements.txt

data: ## generate the labelled synthetic PaySim-layout CSV (skip if you placed the real PaySim file)
	$(PY) -m fraudlab.cli generate --out-dir data/raw --seed 42

run: ## validate, featurise, evaluate out of time, write reports/metrics.json and figures
	$(PY) -m fraudlab.cli run --csv $(CSV) --provenance $(PROV)

test: ## unit tests (stdlib unittest + scikit-learn)
	$(PY) -m unittest discover -s tests -v

lint: ## static analysis
	ruff check src tests

readme: ## regenerate the results block in README.md from reports/metrics.json
	$(PY) -m fraudlab.cli sync-readme

readme-check: ## fail if the README results block is stale
	$(PY) -m fraudlab.cli sync-readme --check

verify: lint test readme-check ## everything CI runs

clean:
	rm -rf .ruff_cache data/raw/*.csv
