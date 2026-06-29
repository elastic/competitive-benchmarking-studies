# ── Load .env ─────────────────────────────────────────────────────────────────
include .env
export

.PHONY: help run elasticsearch prometheus mimir report setup clean

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Data point storage benchmark — Elasticsearch vs Prometheus vs Mimir"
	@echo "===================================================================="
	@echo ""
	@echo "Usage:  make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Create Python virtual environment via uv"
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
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Stop containers, remove volumes, wipe data/ and results/"
	@echo ""
	@echo "Configuration: edit .env (ES_VERSION, SCALE, INTERVAL, START_NOW_MINUS, ES_HEAP)"

run: elasticsearch prometheus mimir report

# ── Elasticsearch ──────────────────────────────────────────────────────────────
elasticsearch:
	@echo "\n━━━ Elasticsearch $(ES_VERSION) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	docker compose up -d --wait elasticsearch
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
	ELASTICSEARCH_URL=http://localhost:9200 \
	  RESULTS_FILE=results/elasticsearch.json \
	  uv run python generate_data.py

# ── Prometheus ─────────────────────────────────────────────────────────────────
prometheus:
	@echo "\n━━━ Prometheus $(PROMETHEUS_VERSION) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@rm -rf data/prometheus && mkdir -p data/prometheus
	@chmod a+w data/prometheus
	docker compose up -d --wait prometheus
	@echo "✓ Prometheus ready"
	PROMETHEUS_URL=http://localhost:9090 \
	  RESULTS_FILE=results/prometheus.json \
	  uv run python generate_data.py

# ── Mimir ──────────────────────────────────────────────────────────────────────
mimir:
	@echo "\n━━━ Mimir $(MIMIR_VERSION) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@rm -rf data/mimir && mkdir -p data/mimir
	docker compose up -d mimir
	@uv run python wait_for.py http://localhost:8080/ready
	MIMIR_URL=http://localhost:8080 \
	  RESULTS_FILE=results/mimir.json \
	  uv run python generate_data.py

# ── Results ────────────────────────────────────────────────────────────────────
report:
	@uv run python report.py

# ── Python environment ─────────────────────────────────────────────────────────
.venv:
	uv sync

setup: .venv ensure_vegeta
	@echo "✓ Python environment ready (.venv)"

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	docker compose down -v
	@rm -rf data/ results/
	@echo "✓ Cleaned"
