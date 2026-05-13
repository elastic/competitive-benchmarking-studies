# Shared access/log targets for engine stacks.
# Requires: connect-k8s (from gke-connect.mk)
#
# Set by stack Makefile:
# - ENGINE_LOG_POD_SELECTOR (kubectl -l selector for engine pods)
#
# For connect-ui:
# - UI_SERVICE_NAME (service name providing a LoadBalancer IP)
# - UI_URL_TEMPLATE (printed with $EXTERNAL_IP available)
# - UI_CREDENTIAL_LINES (shell snippet that prints credential lines; must end lines with `; \`)
#
# Optional:
# - UI_NAMESPACE (defaults to $(or $(NAMESPACE),default))

.PHONY: logs-engine connect-ui

ENGINE_LOG_POD_SELECTOR ?=

UI_SERVICE_NAME ?=
UI_URL_TEMPLATE ?=
UI_CREDENTIAL_LINES ?=
UI_NAMESPACE ?= $(or $(NAMESPACE),default)

logs-engine: connect-k8s
	@POD=$$(kubectl get pods -l "$(ENGINE_LOG_POD_SELECTOR)" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -z "$$POD" ]; then \
		echo >&2 "ERROR: No engine pods found"; \
		exit 1; \
	fi; \
	kubectl logs -f "$$POD"

connect-ui: connect-k8s
	@if [ -z "$(strip $(UI_SERVICE_NAME))" ]; then \
		echo >&2 "ERROR: UI_SERVICE_NAME is not set"; \
		exit 1; \
	fi; \
	if [ -z "$(strip $(UI_URL_TEMPLATE))" ]; then \
		echo >&2 "ERROR: UI_URL_TEMPLATE is not set"; \
		exit 1; \
	fi; \
	NS="$(UI_NAMESPACE)"; \
	EXTERNAL_IP=$$(kubectl get svc "$(UI_SERVICE_NAME)" -n "$$NS" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null); \
	if [ -z "$$EXTERNAL_IP" ]; then \
		echo >&2 "LoadBalancer IP not ready."; \
		kubectl get svc "$(UI_SERVICE_NAME)" -n "$$NS" >&2; \
		exit 1; \
	fi; \
	eval "echo \"$(UI_URL_TEMPLATE)\""; \
	$(UI_CREDENTIAL_LINES)

