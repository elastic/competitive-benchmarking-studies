# Shared `secrets-create` target for engine stacks.
# Requires: REPO_ROOT, STACK_DIR; optional NAMESPACE (default: default)
#
# Engine-specific hooks (set by stack Makefile):
# - ENGINE_SECRET_PREREQS (targets): extra prerequisites (e.g. ensure-qdrant-api-key-in-env)
# - ENGINE_SOURCE_ENV (shell): source engine-local env files (must end lines with `; \`)
# - ENGINE_EXTRA_SECRETS (shell): create engine-specific k8s secrets (must end lines with `; \`)
# - ENGINE_CREDENTIALS_UPSERT (shell): upsert jingra-credentials (must end lines with `; \`)

.PHONY: secrets-create

ENGINE_SECRET_PREREQS ?=
ENGINE_SOURCE_ENV ?=
ENGINE_EXTRA_SECRETS ?=
ENGINE_CREDENTIALS_UPSERT ?=

secrets-create: connect-k8s $(ENGINE_SECRET_PREREQS)
	@set -euo pipefail; \
	NS="$(or $(NAMESPACE),default)"; \
	SECRETS_ENV="$(REPO_ROOT)/shared/secrets/.secrets.env"; \
	if [[ -f "$$SECRETS_ENV" ]]; then \
		set -a; source "$$SECRETS_ENV"; set +a; \
	else \
		echo >&2 "Note: $$SECRETS_ENV not found; using environment only. Copy shared/secrets/.secrets.env.example to shared/secrets/.secrets.env to use a file."; \
	fi; \
	$(ENGINE_SOURCE_ENV) \
	for var in DOCKER_USERNAME DOCKER_PASSWORD DOCKER_EMAIL; do \
		if [[ -z "$${!var:-}" ]]; then \
			echo >&2 "ERROR: $$var not set"; exit 1; \
		fi; \
	done; \
	kubectl delete secret regcred --namespace="$$NS" --ignore-not-found; \
	kubectl create secret docker-registry regcred \
		--docker-server="$$DOCKER_REGISTRY" \
		--docker-username="$$DOCKER_USERNAME" \
		--docker-password="$$DOCKER_PASSWORD" \
		--docker-email="$$DOCKER_EMAIL" \
		--namespace="$$NS"; \
	if [[ -n "$${RESULTS_ES_PASSWORD:-}" ]]; then \
		kubectl delete secret jingra-results-cluster --namespace="$$NS" --ignore-not-found; \
		kubectl create secret generic jingra-results-cluster \
			--from-literal=RESULTS_ES_URL="$$RESULTS_ES_URL" \
			--from-literal=RESULTS_ES_USER="$$RESULTS_ES_USER" \
			--from-literal=RESULTS_ES_PASSWORD="$$RESULTS_ES_PASSWORD" \
			--namespace="$$NS"; \
	fi; \
	$(ENGINE_EXTRA_SECRETS) \
	$(ENGINE_CREDENTIALS_UPSERT) \
	kubectl delete secret jingra-dataset-urls --namespace="$$NS" --ignore-not-found; \
	kubectl create secret generic jingra-dataset-urls \
		--from-literal=DATASET_DATA_URL="$$DATASET_DATA_URL" \
		--from-literal=DATASET_QUERIES_URL="$$DATASET_QUERIES_URL" \
		--namespace="$$NS"

