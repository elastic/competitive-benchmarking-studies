.PHONY: jingra-load jingra-load-stop jingra-eval jingra-eval-stop

include $(REPO_ROOT)/shared/make/run-id.mk

JINGRA_CONFIG ?= $(STACK_DIR)jingra.yml
JINGRA_SCHEMAS_DIR ?= $(STACK_DIR)schemas
JINGRA_QUERIES_DIR ?= $(STACK_DIR)queries

define APPLY_JINGRA_CONFIG
	@NS="$(or $(NAMESPACE),default)"; \
	TMP="$$(mktemp)"; \
	trap 'rm -f "$$TMP"' EXIT; \
	$(call MERGE_RUN_ID_TO_FILE,$(JINGRA_CONFIG),$$TMP,.evaluation.run_id); \
	kubectl create configmap jingra-config --from-file=jingra.yaml="$$TMP" -n "$$NS" --dry-run=client -o yaml | kubectl apply -f -; \
	kubectl create configmap jingra-schemas --from-file="$(JINGRA_SCHEMAS_DIR)" -n "$$NS" --dry-run=client -o yaml | kubectl apply -f -; \
	kubectl create configmap jingra-queries --from-file="$(JINGRA_QUERIES_DIR)" -n "$$NS" --dry-run=client -o yaml | kubectl apply -f -; \
	kubectl apply -f $(REPO_ROOT)/shared/infra/k8s/jingra-datasets-pvc.yml
endef

jingra-load: secrets-create connect-k8s jingra-load-stop
	@if [ -z "$(JINGRA_IMAGE)" ]; then \
		echo >&2 "ERROR: JINGRA_IMAGE is required but not set"; \
		echo >&2 "  Usage: make jingra-load JINGRA_IMAGE=<registry>/<repo>:<tag>"; \
		exit 1; \
	fi
	$(APPLY_JINGRA_CONFIG)
	@export JINGRA_IMAGE="$(JINGRA_IMAGE)"; \
	JINGRA_VER="$(strip $(JINGRA_VERSION))"; \
	if [ -z "$$JINGRA_VER" ]; then JINGRA_VER=$$(echo "$$JINGRA_IMAGE" | sed 's/^.*://'); fi; \
	if [ -z "$$JINGRA_VER" ]; then JINGRA_VER=unknown; fi; \
	export JINGRA_VERSION="$$JINGRA_VER"; \
	cat $(REPO_ROOT)/shared/infra/k8s/jingra-load-job.yml | envsubst '$${JINGRA_IMAGE} $${JINGRA_VERSION}' | kubectl apply -f -

jingra-load-stop: connect-k8s
	kubectl delete job jingra-load --ignore-not-found

jingra-eval: secrets-create connect-k8s jingra-eval-stop
	@if [ -z "$(JINGRA_IMAGE)" ]; then \
		echo >&2 "ERROR: JINGRA_IMAGE is required but not set"; \
		echo >&2 "  Usage: make jingra-eval JINGRA_IMAGE=<registry>/<repo>:<tag>"; \
		exit 1; \
	fi
	$(APPLY_JINGRA_CONFIG)
	@export JINGRA_IMAGE="$(JINGRA_IMAGE)"; \
	JINGRA_VER="$(strip $(JINGRA_VERSION))"; \
	if [ -z "$$JINGRA_VER" ]; then JINGRA_VER=$$(echo "$$JINGRA_IMAGE" | sed 's/^.*://'); fi; \
	if [ -z "$$JINGRA_VER" ]; then JINGRA_VER=unknown; fi; \
	export JINGRA_VERSION="$$JINGRA_VER"; \
	cat $(REPO_ROOT)/shared/infra/k8s/jingra-eval-job.yml | envsubst '$${JINGRA_IMAGE} $${JINGRA_VERSION}' | kubectl apply -f -

jingra-eval-stop: connect-k8s
	kubectl delete job jingra-eval --ignore-not-found
