.PHONY: status

STACK_ENGINE_LABEL ?= Engine
ENGINE_POD_SELECTOR ?=
UI_POD_SELECTOR ?=
SERVICE_LABEL_SELECTOR ?=
SERVICE_NAMES ?=

status: connect-k8s
	@echo "$(STACK_ENGINE_LABEL) pods:"; \
	if [ -n "$(strip $(ENGINE_POD_SELECTOR))" ]; then \
		kubectl get pods -n "$(NAMESPACE)" -l "$(ENGINE_POD_SELECTOR)" 2>/dev/null || echo "  (none)"; \
	else \
		echo "  (not configured)"; \
	fi; \
	if [ -n "$(strip $(UI_POD_SELECTOR))" ]; then \
		echo ""; \
		echo "UI pods:"; \
		kubectl get pods -n "$(NAMESPACE)" -l "$(UI_POD_SELECTOR)" 2>/dev/null || echo "  (none)"; \
	fi; \
	echo ""; \
	echo "Jingra jobs:"; \
	kubectl get jobs -n "$(NAMESPACE)" jingra-load jingra-eval --ignore-not-found 2>/dev/null || echo "  (none)"; \
	echo ""; \
	echo "Jingra pods:"; \
	kubectl get pods -n "$(NAMESPACE)" -l 'job-name in (jingra-load,jingra-eval)' 2>/dev/null || echo "  (none)"; \
	echo ""; \
	echo "Services:"; \
	if [ -n "$(strip $(SERVICE_LABEL_SELECTOR))" ]; then \
		kubectl get svc -n "$(NAMESPACE)" -l "$(SERVICE_LABEL_SELECTOR)" 2>/dev/null || true; \
	elif [ -n "$(strip $(SERVICE_NAMES))" ]; then \
		kubectl get svc -n "$(NAMESPACE)" $(SERVICE_NAMES) 2>/dev/null || true; \
	else \
		echo "  (not configured)"; \
	fi

