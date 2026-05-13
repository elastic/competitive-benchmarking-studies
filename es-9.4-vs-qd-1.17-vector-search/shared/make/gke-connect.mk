.PHONY: connect-k8s

GKE_ZONE ?= us-central1-a

connect-k8s:
	@PROJECT_ID=$$(grep '^project_id' $(REPO_ROOT)/shared/secrets/terraform.tfvars | awk -F'=' '{print $$2}' | tr -d ' "'); \
	gcloud container clusters get-credentials $(GKE_CLUSTER_NAME) \
		--zone $(GKE_ZONE) --project $$PROJECT_ID; \
	if ! kubectl cluster-info --request-timeout=20s >/dev/null; then \
		echo >&2 "ERROR: Cannot reach the Kubernetes API (timeout or connection failed)."; \
		echo >&2 "  Typical causes: master authorized networks (add this machine's public IP),"; \
		echo >&2 "  private GKE control plane without VPN, or offline/restrictive network."; \
		echo >&2 "  Try: gcloud container clusters describe $(GKE_CLUSTER_NAME) --zone $(GKE_ZONE) --project $$PROJECT_ID"; \
		exit 1; \
	fi; \
	:
