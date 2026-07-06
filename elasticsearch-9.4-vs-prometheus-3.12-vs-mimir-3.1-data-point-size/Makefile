# ── Load .env ─────────────────────────────────────────────────────────────────
include .env
export

BIN_DIR = .bin
export PATH := $(abspath $(BIN_DIR)):$(PATH)
export UID := $(shell id -u)
export GID := $(shell id -g)

BENCHMARK ?= duration_270m-query_240m-scale_100

# Tells docker compose where to find the compose file without requiring -f on every call
export COMPOSE_FILE ?= deploy/docker/docker-compose.yml

.PHONY: help run elasticsearch prometheus mimir report setup clean lint

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Data point storage benchmark — Elasticsearch vs Prometheus vs Mimir"
	@echo "===================================================================="
	@echo ""
	@echo "Usage:  make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Create Python venv + install metricsgenreceiver and vegeta"
	@echo ""
	@echo "Benchmark:"
	@echo "  make run            Run all three engines sequentially (~45 min)"
	@echo "  make elasticsearch  Ingest + query + measure Elasticsearch (~15 min)"
	@echo "  make prometheus     Ingest + query + measure Prometheus (~15 min)"
	@echo "  make mimir          Ingest + query + measure Mimir (~15 min)"
	@echo ""
	@echo "Results:"
	@echo "  make report         Print bytes-per-data-point comparison table"
	@echo ""
	@echo "Standalone (requires the target engine's container to still be running):"
	@echo "  uv run query --engine <name>       Re-run query benchmarks for one engine"
	@echo "  uv run disk-usage (with ENGINE=<name> exported) Re-measure storage for one engine"
	@echo ""
	@echo "Dev:"
	@echo "  make lint           Run ruff linter"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Stop containers, remove volumes, wipe data/, results/, and .bin/"
	@echo ""
	@echo "Configuration: edit .env (ES_VERSION, ES_HEAP, ES_DATA_STREAM); select a scenario"
	@echo "  with BENCHMARK=<name> (default: $(BENCHMARK)) — see scenarios/*.yml"

run: elasticsearch prometheus mimir report

# ── Elasticsearch ──────────────────────────────────────────────────────────────
elasticsearch:
	@echo "\n━━━ Elasticsearch $(ES_VERSION) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	docker compose up -d --wait elasticsearch
	@sleep 2
	@echo "✓ Elasticsearch ready"
	uv run run-engine elasticsearch --benchmark $(BENCHMARK)
	docker compose down elasticsearch

# ── Prometheus ─────────────────────────────────────────────────────────────────
prometheus:
	@echo "\n━━━ Prometheus $(PROMETHEUS_VERSION) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@rm -rf data/prometheus && mkdir -p data/prometheus
	docker compose up -d --wait prometheus
	@echo "✓ Prometheus ready"
	uv run run-engine prometheus --benchmark $(BENCHMARK)
	docker compose down prometheus

# ── Mimir ──────────────────────────────────────────────────────────────────────
mimir:
	@echo "\n━━━ Mimir $(MIMIR_VERSION) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@rm -rf data/mimir && mkdir -p data/mimir
	docker compose up -d mimir
	@uv run wait-for http://localhost:8080/ready
	uv run run-engine mimir --benchmark $(BENCHMARK)
	docker compose down mimir

# ── Dev ───────────────────────────────────────────────────────────────────────
lint:
	uv run ruff format --check .
	uv run ruff check .

# ── Results ────────────────────────────────────────────────────────────────────
report:
	@uv run report

# ── Python environment ─────────────────────────────────────────────────────────
.venv:
	uv sync

$(BIN_DIR)/metricsgenreceiver:
	@bash scripts/install_binaries.sh metricsgenreceiver $(METRICSGENRECEIVER_VERSION)

$(BIN_DIR)/vegeta:
	@bash scripts/install_binaries.sh vegeta $(VEGETA_VERSION)

setup: .venv $(BIN_DIR)/metricsgenreceiver $(BIN_DIR)/vegeta
	@echo "✓ Setup complete"

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	docker compose down -v
	@rm -rf data/ results/ $(BIN_DIR)/
	@echo "✓ Cleaned"
