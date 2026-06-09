.PHONY: help sync lint fmt type test run
.DEFAULT_GOAL := help

help:   ## show this help
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | sort | awk -F ':.*## ' '{printf "  \033[36m%-7s\033[0m %s\n", $$1, $$2}'

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
