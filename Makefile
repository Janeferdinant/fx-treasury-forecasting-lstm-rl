.PHONY: install lint test test-fast fetch forecast trade report smoke explorer

PY ?= python
JOBS ?= 4

install:
	$(PY) -m pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e . --no-deps

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest -q

# what CI runs: no network, no long experiments
test-fast:
	$(PY) -m pytest -q -m "not slow"

fetch:
	$(PY) -m stocklab.cli fetch

forecast:
	$(PY) -m stocklab.cli forecast --jobs $(JOBS)

trade:
	$(PY) -m stocklab.cli trade --jobs $(JOBS)

report:
	$(PY) -m stocklab.cli report

# tiny end-to-end check (US_MARKET, last two test blocks, one seed, few epochs): about a minute
smoke:
	$(PY) -m stocklab.cli forecast --smoke --jobs 2 --raw-dir results/smoke_raw
	$(PY) -m stocklab.cli trade --smoke --jobs 2 --raw-dir results/smoke_raw
	$(PY) -m stocklab.cli report --raw-dir results/smoke_raw --out results/smoke_summary.json --explorer results/smoke_data.json

explorer:
	$(PY) -m http.server 8000 --directory docs
