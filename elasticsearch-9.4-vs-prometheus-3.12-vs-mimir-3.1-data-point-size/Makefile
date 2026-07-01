# ── Load .env ─────────────────────────────────────────────────────────────────
include .env
export

# ES_HEAP defaults to 50% of CONTAINER_MEMORY_LIMIT unless set explicitly in
# .env. CONTAINER_MEMORY_LIMIT itself mirrors the default in docker-compose.yml.
CONTAINER_MEMORY_LIMIT ?= 12g
ES_HEAP ?= $(shell echo $(CONTAINER_MEMORY_LIMIT) | awk -F'g' '{printf "%dg", $$1/2}')
export ES_HEAP

BIN_DIR = .bin
export PATH := $(abspath $(BIN_DIR)):$(PATH)
export UID := $(shell id -u)
export GID := $(shell id -g)

# docker-compose.linux.yml sets Linux-only container tuning (e.g.
# vm.max_map_count) that Docker Desktop's VM rejects on macOS/Windows.
UNAME_S := $(shell uname -s)
COMPOSE_FILES := -f docker-compose.yml
ifeq ($(UNAME_S),Linux)
  COMPOSE_FILES += -f docker-compose.linux.yml
endif
DOCKER_COMPOSE := docker compose $(COMPOSE_FILES)

.PHONY: help run elasticsearch prometheus mimir report setup clean query query-es query-mimir query-prometheus disk-usage disk-usage-es disk-usage-mimir disk-usage-prometheus lint

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
	@echo "  make elasticsearch  Ingest OTel data into Elasticsearch (~15 min)"
	@echo "  make prometheus     Ingest OTel data into Prometheus (~15 min)"
	@echo "  make mimir          Ingest OTel data into Mimir (~15 min)"
	@echo ""
	@echo "Query performance:"
	@echo "  make query          Run query benchmarks against all three engines"
	@echo "  make query-es       Run query benchmarks against Elasticsearch only"
	@echo "  make query-prometheus  Run query benchmarks against Prometheus only"
	@echo "  make query-mimir    Run query benchmarks against Mimir only"
	@echo ""
	@echo "Results:"
	@echo "  make report         Print bytes-per-data-point comparison table"
	@echo "  make disk-usage     Re-measure on-disk storage for all engines without re-ingesting"
	@echo "  make disk-usage-es  Re-measure Elasticsearch storage only"
	@echo "  make disk-usage-prometheus  Re-measure Prometheus storage only"
	@echo "  make disk-usage-mimir  Re-measure Mimir storage only"
	@echo ""
	@echo "Dev:"
	@echo "  make lint           Run ruff linter"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Stop containers, remove volumes, wipe data/, results/, and .bin/"
	@echo ""
	@echo "Configuration: edit .env (ES_VERSION, SCALE, INTERVAL, START_NOW_MINUS, ES_HEAP)"

run: elasticsearch prometheus mimir report

# ── Elasticsearch ──────────────────────────────────────────────────────────────
elasticsearch: export ELASTICSEARCH_URL  = http://localhost:9200
elasticsearch: export RESULTS_FILE       = results/elasticsearch.json
elasticsearch: export QUERY_ENGINE       = elasticsearch
elasticsearch:
	@echo "\n━━━ Elasticsearch $(ES_VERSION) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	$(DOCKER_COMPOSE) up -d --wait elasticsearch
	@sleep 2
	@echo "✓ Elasticsearch ready"
	@# Trial license — free 30-day, required for synthetic _source in TSDB (idempotent)
	@curl -s -X POST "http://localhost:9200/_license/start_trial?acknowledge=true" | \
	  python3 -c "import sys,json; r=json.load(sys.stdin); print('✓ Trial license:', r.get('trial_was_started','already active'))"
	@# Apply OTel custom component template (0 replicas, 3 shards, 7d look_back)
	@curl -sf -X PUT http://localhost:9200/_component_template/metrics-otel@custom \
	  -H "Content-Type: application/json" \
	  -d @config/elasticsearch/metrics-otel@custom.json >/dev/null \
	  && echo "✓ Component template applied"
	@# Apply ILM policy
	@curl -sf -X PUT http://localhost:9200/_ilm/policy/metrics-policy \
	  -H "Content-Type: application/json" \
	  -d @config/elasticsearch/metrics-policy.json >/dev/null \
	  && echo "✓ ILM policy applied"
	@# Delete existing data stream so it picks up the new template settings
	@curl -s -X DELETE http://localhost:9200/_data_stream/metrics-demo.otel-default >/dev/null || true
	@curl -sf -X PUT http://localhost:9200/_data_stream/metrics-demo.otel-default >/dev/null \
	  && echo "✓ Data stream created"
	@echo "Ingesting data..."
	uv run load --wait-for-merges
	@echo "Running queries..."
	@uv run query
	@echo "Calculating disk usage..."
	@uv run disk-usage
	$(DOCKER_COMPOSE) down elasticsearch

# ── Prometheus ─────────────────────────────────────────────────────────────────
prometheus: export PROMETHEUS_URL = http://localhost:9090
prometheus: export RESULTS_FILE   = results/prometheus.json
prometheus: export QUERY_ENGINE   = prometheus
prometheus:
	@echo "\n━━━ Prometheus $(PROMETHEUS_VERSION) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@rm -rf data/prometheus && mkdir -p data/prometheus
	$(DOCKER_COMPOSE) up -d --wait prometheus
	@echo "✓ Prometheus ready"
	@echo "Ingesting data..."
	uv run load
	@echo "Running queries..."
	@uv run query
	@echo "Calculating disk usage..."
	@uv run disk-usage
	$(DOCKER_COMPOSE) down prometheus

# ── Mimir ──────────────────────────────────────────────────────────────────────
mimir: export MIMIR_URL     = http://localhost:8080
mimir: export RESULTS_FILE  = results/mimir.json
mimir: export QUERY_ENGINE  = mimir
mimir:
	@echo "\n━━━ Mimir $(MIMIR_VERSION) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@rm -rf data/mimir && mkdir -p data/mimir
	$(DOCKER_COMPOSE) up -d mimir
	@uv run wait-for http://localhost:8080/ready
	@echo "Ingesting data..."
	uv run load
	@echo "Running queries..."
	@uv run query
	@echo "Calculating disk usage..."
	@uv run disk-usage
	$(DOCKER_COMPOSE) down mimir

# ── Storage measurement ─────────────────────────────────────────────────────────
disk-usage-es:
	@ELASTICSEARCH_URL=http://localhost:9200 RESULTS_FILE=results/elasticsearch.json uv run disk-usage

disk-usage-prometheus:
	@PROMETHEUS_URL=http://localhost:9090 RESULTS_FILE=results/prometheus.json uv run disk-usage

disk-usage-mimir:
	@MIMIR_URL=http://localhost:8080 RESULTS_FILE=results/mimir.json uv run disk-usage

disk-usage: disk-usage-es disk-usage-prometheus disk-usage-mimir

# ── Query performance ──────────────────────────────────────────────────────────
query-es:
	@RESULTS_FILE=results/elasticsearch.json QUERY_ENGINE=elasticsearch uv run query

query-prometheus:
	@RESULTS_FILE=results/prometheus.json QUERY_ENGINE=prometheus uv run query

query-mimir:
	@RESULTS_FILE=results/mimir.json QUERY_ENGINE=mimir uv run query

query: query-es query-prometheus query-mimir

# ── Dev ───────────────────────────────────────────────────────────────────────
lint:
	uv run ruff format --check .
	uv run ruff check .

# ── Results ────────────────────────────────────────────────────────────────────
report:
	@uv run python report.py

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
	$(DOCKER_COMPOSE) down -v
	@rm -rf data/ results/ $(BIN_DIR)/
	@echo "✓ Cleaned"
