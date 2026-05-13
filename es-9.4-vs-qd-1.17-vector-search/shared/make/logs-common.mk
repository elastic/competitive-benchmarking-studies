.PHONY: jingra-load-logs jingra-eval-logs logs-eval copy-query-dumps metrics

# Shared function to follow logs from a Kubernetes job
# Args: $(1)=job name
define FOLLOW_JOB_LOGS
	@POD=$$(kubectl get pods -l job-name=$(1) --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -z "$$POD" ]; then \
		POD=$$(kubectl get pods -l job-name=$(1) --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null); \
	fi; \
	if [ -z "$$POD" ]; then \
		echo >&2 "ERROR: No pods found for job $(1)"; \
		exit 1; \
	fi; \
	if [ -n "$(LINES)" ]; then \
		kubectl logs -f --tail=$(LINES) $$POD; \
	else \
		kubectl logs -f $$POD; \
	fi
endef

metrics: connect-k8s
	@kubectl top pods --all-namespaces 2>/dev/null || echo >&2 "Note: metrics server not available or pods not ready"
	@kubectl top nodes 2>/dev/null || echo >&2 "Note: metrics server not available"

jingra-load-logs: connect-k8s
	$(call FOLLOW_JOB_LOGS,jingra-load)

jingra-eval-logs: connect-k8s
	$(call FOLLOW_JOB_LOGS,jingra-eval)

logs-eval: jingra-eval-logs

copy-query-dumps: connect-k8s
	@NS="$(or $(NAMESPACE),default)"; \
	REMOTE="$(or $(QUERY_DUMP_REMOTE),/app/query-dumps/)"; \
	LOCAL="$(or $(QUERY_DUMP_LOCAL),./query-dumps)"; \
	POD="$(strip $(EVAL_POD))"; \
	if [ -z "$$POD" ]; then \
		POD=$$(kubectl get pods -n "$$NS" -l job-name=jingra-eval --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	fi; \
	if [ -z "$$POD" ]; then \
		echo >&2 "ERROR: No Running jingra-eval pod in namespace $$NS."; \
		echo >&2 "  kubectl cp needs the eval container running; Completed pods cannot be copied."; \
		echo >&2 "  Run \`make copy-query-dumps\` in another terminal while eval is in progress,"; \
		echo >&2 "  or set EVAL_POD to a currently Running pod: kubectl get pods -l job-name=jingra-eval"; \
		exit 1; \
	fi; \
	mkdir -p "$$LOCAL"; \
	echo "Copying $$NS/$$POD:$$REMOTE -> $$LOCAL"; \
	kubectl cp "$$NS/$$POD:$$REMOTE" "$$LOCAL"
