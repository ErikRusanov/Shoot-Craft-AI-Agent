.PHONY: help setup sync lint fmt type test run infra infra-down models presets-dev presets-build
.DEFAULT_GOAL := help

# Sibling checkout of the private preset library (override: make X PRESETS=...).
PRESETS ?= ../presets

help:   ## show this help
	@grep -E '^[a-z][a-z-]*:.*## ' $(MAKEFILE_LIST) | sort | awk -F ':.*## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## bootstrap dev env: install uv, sync deps, wire git hooks
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	$(MAKE) sync
	git config core.hooksPath .githooks
	@test -f .env || { cp .env.example .env && echo "created .env from .env.example"; }

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
	PYTHONPATH=src uv run python -m main

models: ## download InsightFace weights into INSIGHTFACE_ROOT (never committed)
	PYTHONPATH=src uv run python scripts/download_models.py

infra:  ## start local backing services (redis) via docker compose
	docker compose up -d --wait

infra-down: ## stop local backing services
	docker compose down

presets-dev:   ## local dev: editable-install the private library (PRESETS=../presets) for PRESET_SOURCE=package
	uv pip install -e $(PRESETS)
	@echo "installed photocore-presets (editable). NOTE: 'make sync' prunes it — re-run after sync."
	@echo "then set PRESET_SOURCE=package in .env (no PRESET_LIBRARY_PATH needed)."

presets-build: ## build the private library wheel (PRESETS=../presets) into its dist/
	uv build --wheel $(PRESETS)
	@echo "wheel written to $(PRESETS)/dist — feed it to the private Docker image."
