PY ?= .venv/bin/python

.PHONY: setup test validate dev freeze heldout repair-sim all clean

setup:
	uv sync --extra figures --extra live

test:
	$(PY) -m pytest -q

validate:
	$(PY) -m marketdebug validate

dev:            ## dev-only experiments (never touches held-out)
	$(PY) -m marketdebug dev

freeze:         ## select the market config on dev and write configs/frozen.json
	$(PY) -m marketdebug freeze

heldout:        ## every experiment with the frozen config (refuses if code/data changed)
	$(PY) -m marketdebug heldout

repair-sim:     ## SYNTHETIC repair-ledger study
	$(PY) -m marketdebug repair-sim

all: test validate heldout

clean:
	rm -rf .pytest_cache **/__pycache__
