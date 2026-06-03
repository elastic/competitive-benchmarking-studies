# Operational settings (gke-connect, stack-status, stack-access)
GKE_CLUSTER_NAME := elasticsearch-benchmark
ECK_VERSION := 3.1.0

# ECK: sets ES_PASS from es-cluster-es-elastic-user or empty if missing (used by ENGINE_CREDENTIALS_UPSERT + k8s-post-apply)
define JINGRA_ELASTICSEARCH_FETCH_ES_PASS
	if kubectl get secret es-cluster-es-elastic-user --namespace="$$NS" >/dev/null 2>&1; then \
		ES_PASS=$$(kubectl get secret es-cluster-es-elastic-user --namespace="$$NS" -o jsonpath='{.data.elastic}' | base64 -d); \
	else \
		ES_PASS=; \
	fi; \

endef

# Timeout configuration for k8s-wait-ready
MAX_WAIT_ATTEMPTS ?= 200
WAIT_INTERVAL_SECS ?= 3
KUBECTL_WAIT_TIMEOUT ?= 600s

STACK_ENGINE_LABEL := Elasticsearch

ENGINE_POD_SELECTOR := common.k8s.elastic.co/type=elasticsearch
UI_POD_SELECTOR := common.k8s.elastic.co/type=kibana
SERVICE_LABEL_SELECTOR := common.k8s.elastic.co/type

ENGINE_LOG_POD_SELECTOR := common.k8s.elastic.co/type=elasticsearch
UI_SERVICE_NAME := es-cluster-kb-http
UI_URL_TEMPLATE := URL:      https://$${EXTERNAL_IP}:5601

define ENGINE_CREDENTIALS_UPSERT
	$(JINGRA_CREDENTIALS_SHELL_FUNCTIONS) \
	$(JINGRA_ELASTICSEARCH_FETCH_ES_PASS) \
	if [ -n "$$ES_PASS" ]; then \
		jingra_credentials_apply_once "$$ES_PASS"; \
	fi; \

endef

define UI_CREDENTIAL_LINES
	ENGINE_PASSWORD=$$(kubectl get secret jingra-credentials -n $(NAMESPACE) -o jsonpath='{.data.ENGINE_PASSWORD}' 2>/dev/null | base64 -d); \
	if [ -z "$$ENGINE_PASSWORD" ]; then \
		ENGINE_PASSWORD=$$(kubectl get secret es-cluster-es-elastic-user -n $(NAMESPACE) -o go-template='{{.data.elastic | base64decode}}{{"\\n"}}' 2>/dev/null); \
	fi; \
	echo "Username: elastic"; \
	echo "Password: $$ENGINE_PASSWORD"; \

endef

.PHONY: k8s-apply k8s-apply-manifests _k8s-wait-ready k8s-post-apply k8s-delete-impl logs-ui

k8s-apply: secrets-create k8s-apply-manifests _k8s-wait-ready k8s-post-apply

k8s-post-apply:
	@NS="$(or $(NAMESPACE),default)"; \
	$(JINGRA_CREDENTIALS_SHELL_FUNCTIONS) \
	$(JINGRA_ELASTICSEARCH_FETCH_ES_PASS) \
	if [ -n "$$ES_PASS" ]; then \
		jingra_credentials_apply_once "$$ES_PASS"; \
		ES_POD=$$(kubectl get pod -n "$$NS" -l common.k8s.elastic.co/type=elasticsearch -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
		if [ -n "$$ES_POD" ]; then \
			TRIAL_RESPONSE=$$(kubectl exec -n "$$NS" "$$ES_POD" -- \
				curl -sk -u "elastic:$$ES_PASS" -X POST "https://localhost:9200/_license/start_trial?acknowledge=true" 2>&1); \
			if echo "$$TRIAL_RESPONSE" | grep -q '"error_message"'; then \
				echo >&2 "WARNING: trial activation failed: $$TRIAL_RESPONSE"; \
			fi; \
		fi; \
	fi

k8s-apply-manifests:
	@CRD_EXISTS=$$(kubectl get crd elasticsearches.elasticsearch.k8s.elastic.co --ignore-not-found); \
	if [ -z "$$CRD_EXISTS" ]; then \
		kubectl create -f https://download.elastic.co/downloads/eck/$(ECK_VERSION)/crds.yaml; \
	fi
	@OPERATOR_EXISTS=$$(kubectl get po -n elastic-system --no-headers 2>/dev/null | grep '^elastic-operator' || true); \
	if [ -z "$$OPERATOR_EXISTS" ]; then \
		kubectl apply -f https://download.elastic.co/downloads/eck/$(ECK_VERSION)/operator.yaml; \
		kubectl wait --for=condition=ready pod -l control-plane=elastic-operator -n elastic-system --timeout=120s || true; \
	fi
	$(KUBECTL_APPLY_K8S_FROM_VARS)

_k8s-wait-ready:
	@set -euo pipefail; \
	NS="$(or $(NAMESPACE),default)"; \
	i=0; \
	until kubectl get pods -n "$$NS" -l common.k8s.elastic.co/type=elasticsearch -o name 2>/dev/null | grep -q .; do \
		i=$$((i+1)); \
		if [ "$$i" -ge $(MAX_WAIT_ATTEMPTS) ]; then echo >&2 "ERROR: Timed out waiting for Elasticsearch pods (try: kubectl describe elasticsearch es-cluster -n $$NS)"; exit 1; fi; \
		sleep $(WAIT_INTERVAL_SECS); \
	done; \
	kubectl wait --for=condition=ready pod -n "$$NS" -l common.k8s.elastic.co/type=elasticsearch --timeout=$(KUBECTL_WAIT_TIMEOUT); \
	i=0; \
	until kubectl get pods -n "$$NS" -l common.k8s.elastic.co/type=kibana -o name 2>/dev/null | grep -q .; do \
		i=$$((i+1)); \
		if [ "$$i" -ge $(MAX_WAIT_ATTEMPTS) ]; then echo >&2 "ERROR: Timed out waiting for Kibana pods"; exit 1; fi; \
		sleep $(WAIT_INTERVAL_SECS); \
	done; \
	kubectl wait --for=condition=ready pod -n "$$NS" -l common.k8s.elastic.co/type=kibana --timeout=$(KUBECTL_WAIT_TIMEOUT)

k8s-delete-impl:
	@kubectl delete job jingra-load jingra-eval -n $(NAMESPACE) --ignore-not-found
	@kubectl delete kibana es-cluster -n $(NAMESPACE) --ignore-not-found
	@kubectl delete elasticsearch es-cluster -n $(NAMESPACE) --ignore-not-found
	@kubectl delete pvc -l elasticsearch.k8s.elastic.co/cluster-name=es-cluster -n $(NAMESPACE) --ignore-not-found
	@kubectl delete -f $(STACK_DIR)k8s/ --ignore-not-found

logs-ui: connect-k8s
	@POD=$$(kubectl get pods -l common.k8s.elastic.co/type=kibana -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -z "$$POD" ]; then \
		echo >&2 "ERROR: No UI pods found"; \
		exit 1; \
	fi; \
	kubectl logs -f "$$POD"
