SHELL := /bin/bash

REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ANALYZE_DIR := $(REPO_ROOT)analyze
ANALYZE_OUTPUT_DIR := $(ANALYZE_DIR)/output

include $(REPO_ROOT)shared/make/run-id.mk

ENGINES := elasticsearch qdrant

STACK_TARGETS := \
	terraform-plan terraform-apply terraform-destroy \
	k8s-apply k8s-delete \
	jingra-load jingra-load-stop jingra-load-logs \
	jingra-eval jingra-eval-stop jingra-eval-logs copy-query-dumps \
	status metrics logs-engine connect-ui logs-ui

# Spell stacks literally for shell tab completion (zsh/bash grep this file; they do not expand $(ENGINES)).
# When you add engines/*/engine.mk, add the name here too (alongside $(ENGINES), duplication is harmless).
.PHONY: help analyze analyze-clean elasticsearch qdrant $(STACK_TARGETS)

.DEFAULT_GOAL := help

-include $(REPO_ROOT)shared/secrets/.secrets.env

define require_env
	@bash -c 'if [ -z "$${$1:-}" ]; then echo "ERROR: $1 not set" >&2; exit 1; fi'
endef

help:
	@echo "Vector search benchmark — run from this directory (repo root)"
	@echo "============================================================="
	@echo ""
	@echo "Pattern:  make <stack> <target>"
	@echo ""
	@echo "Stacks: any directory under engines/ that contains engine.mk"
	@echo "Discovered now: $(ENGINES)"
	@echo ""
	@echo "Example:  make elasticsearch k8s-apply"
	@echo "          make qdrant jingra-eval JINGRA_IMAGE=registry.example.com/jingra:36"
	@echo ""
	@echo "Terraform (init runs automatically before plan, apply, destroy):"
	@echo "  terraform-plan"
	@echo "  terraform-apply"
	@echo "  terraform-destroy"
	@echo "    Uses shared/secrets/terraform.tfvars and shared/variables/terraform.tfvars; init copies"
	@echo "    shared/infra/terraform/modules/gke-benchmark/gke-engine-root-variables.tf"
	@echo "    to engines/<stack>/terraform/gke-engine-root-variables.tf."
	@echo ""
	@echo "Kubernetes cluster:"
	@echo "  k8s-apply"
	@echo "  k8s-delete"
	@echo ""
	@echo "Jingra — load / eval (needs JINGRA_IMAGE; merges evaluation.run_id from shared/variables/run_id.yml via yq):"
	@echo "  jingra-load"
	@echo "  jingra-load-stop"
	@echo "  jingra-load-logs"
	@echo "  jingra-eval"
	@echo "  jingra-eval-stop"
	@echo "  jingra-eval-logs"
	@echo "  copy-query-dumps"
	@echo ""
	@echo "Cluster status and observability:"
	@echo "  status"
	@echo "  metrics"
	@echo "  logs-engine"
	@echo "  connect-ui"
	@echo "  logs-ui"
	@echo ""
	@echo "Variables (pass on the make command line with any stack target):"
	@echo "  JINGRA_IMAGE=<ref>        required for jingra-load / jingra-eval"
	@echo "  JINGRA_VERSION=<text>     optional; defaults to image tag after last ':'"
	@echo "  LINES=<n>                 optional; tail limit for jingra-load-logs / jingra-eval-logs"
	@echo "  NAMESPACE=<ns>            Kubernetes namespace (default: default)"
	@echo "  K8S_VARS=<path>           optional; shared sizing YAML for k8s-apply (default: shared/variables/k8s.yml; needs envsubst + yq). Stack version: engines/<stack>/k8s/.version"
	@echo "  QUERY_DUMP_REMOTE=<path>  path in pod for copy-query-dumps"
	@echo "  QUERY_DUMP_LOCAL=<path>   host directory for copy-query-dumps"
	@echo "  EVAL_POD=<name>           pod override for copy-query-dumps"
	@echo ""
	@echo "Post-run analysis (not stack-prefixed):"
	@echo "  make analyze JINGRA_IMAGE=<ref>"
	@echo "    Clears analyze/output/, runs Jingra analyze in Docker."
	@echo "    Injects analysis.run_id from shared/variables/run_id.yml (needs mikefarah yq v4)."
	@echo "    Needs RESULTS_ES_URL, RESULTS_ES_USER, RESULTS_ES_PASSWORD (env or shared/secrets/.secrets.env)."
	@echo "    Writes CSVs under analyze/output/ (recall, summary, throughput comparisons)."
	@echo "    ANALYZE_CONFIG=<path>  optional; override config (default: analyze/jingra.yml)"
	@echo "  make analyze-clean"
	@echo "    Removes analyze/output/."
	@echo ""

analyze-clean:
	@rm -rf "$(ANALYZE_OUTPUT_DIR)"

$(ANALYZE_OUTPUT_DIR):
	@mkdir -p "$@"

ANALYZE_CONFIG ?= $(ANALYZE_DIR)/jingra.yml

analyze: analyze-clean | $(ANALYZE_OUTPUT_DIR)
	@set -euo pipefail; \
	if [ -f "$(REPO_ROOT)shared/secrets/.secrets.env" ]; then \
		set -a; source "$(REPO_ROOT)shared/secrets/.secrets.env"; set +a; \
	fi; \
	for var in RESULTS_ES_URL RESULTS_ES_USER RESULTS_ES_PASSWORD JINGRA_IMAGE; do \
		if [ -z "$${!var:-}" ]; then echo "ERROR: $$var not set" >&2; exit 1; fi; \
	done; \
	TMP="$$(mktemp)"; \
	trap 'rm -f "$$TMP"' EXIT; \
	$(call MERGE_RUN_ID_TO_FILE,$(ANALYZE_CONFIG),$$TMP,.analysis.run_id); \
	VERSIONS_DIR="$(REPO_ROOT)shared/variables/versions"; \
	while IFS= read -r engine; do \
		vfile="$$VERSIONS_DIR/.$$engine"; \
		if [[ ! -f "$$vfile" ]]; then \
			echo >&2 "ERROR: version file not found for engine '$$engine': $$vfile"; \
			exit 1; \
		fi; \
		version="$$(head -n1 "$$vfile" | tr -d '[:space:]')"; \
		TMP2="$$(mktemp)"; \
		yq eval ".analysis.engine_versions.$$engine = \"$$version\"" "$$TMP" > "$$TMP2" && mv "$$TMP2" "$$TMP"; \
	done < <(yq eval '.analysis.engines[]' "$$TMP"); \
	docker run --rm \
		-v "$$TMP:/config/jingra.yml:ro" \
		-v "$(ANALYZE_OUTPUT_DIR):/output" \
		-e RESULTS_ES_URL \
		-e RESULTS_ES_USER \
		-e RESULTS_ES_PASSWORD \
		"$(JINGRA_IMAGE)" \
		analyze /config/jingra.yml

# Literal stack names = same idea as .PHONY line (tab completion grep); add new stacks here when you add engines/*/engine.mk.
elasticsearch qdrant:
	@targets="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$targets" ]; then \
		$(MAKE) help; \
		exit 0; \
	fi; \
	if echo "$$targets" | tr ' ' '\n' | grep -qx 'help'; then \
		$(MAKE) help; \
	fi; \
	fwd="$$(echo "$$targets" | tr ' ' '\n' | grep -vx 'help' | tr '\n' ' ')"; \
	if [ -n "$$fwd" ]; then \
		$(MAKE) -f "$(REPO_ROOT)shared/make/stack.mk" STACK=$@ $$fwd; \
	fi

$(STACK_TARGETS):
	@:
