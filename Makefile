.PHONY: help setup sync lint fmt type test run
.DEFAULT_GOAL := help

help:   ## show this help
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | sort | awk -F ':.*## ' '{printf "  \033[36m%-7s\033[0m %s\n", $$1, $$2}'

setup:  ## bootstrap dev env: install uv, sync deps, wire git hooks
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	$(MAKE) sync
	git config core.hooksPath .githooks

sync:   ## install/sync env from the lock
	uv sync

lint:   ## lint + format check
	uv run ruff check .
	uv run ruff format --check .

fmt:    ## auto-format
	uv run ruff format .
	uv run ruff check --fix .

type:   ## strict type check (src + tests)
	uv run mypy

test:   ## run tests (pass ARGS=... for a single test)
	uv run pytest $(ARGS)

run:    ## run the worker
	PYTHONPATH=src uv run uvicorn api.app:app
