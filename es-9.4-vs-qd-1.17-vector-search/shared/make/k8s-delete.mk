# Interactive guard for all stacks: engines/*/engine.mk should define k8s-delete-impl (actual kubectl deletes only).
.PHONY: k8s-delete confirm-k8s-delete

confirm-k8s-delete:
	@echo >&2 "WARNING: This will delete Kubernetes workloads and persistent data for this engine."
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ]

k8s-delete: connect-k8s confirm-k8s-delete k8s-delete-impl
	@if [[ -f '$(REPO_ROOT)/shared/infra/k8s/storage-class.yml' ]]; then \
		kubectl delete -f '$(REPO_ROOT)/shared/infra/k8s/storage-class.yml' --ignore-not-found; \
	fi
