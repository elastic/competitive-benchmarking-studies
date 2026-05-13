# Shared kubectl apply pipeline for engine stacks: cluster-scoped StorageClass, then
# envsubst-rendered manifests from engines/$(STACK)/k8s/ using K8S_VARS + stack version.
# Included from common.mk. Engines override K8S_VARS_EXTRA_SHELL / K8S_ENVSUBST_EXTRA_VARS as needed.

# Renders engines/$(STACK)/k8s/*.yaml with sizing from K8S_VARS plus
# shared/variables/versions/.<stack> (plain text → env engineVersion), then kubectl apply.
# Reads shared sizing YAML with mikefarah yq v4 (same as jingra run_id merge).
# Requires: envsubst (gettext), yq.
# Only substitutes known keys so other ${...} (e.g. ${HOSTNAME} in ConfigMaps) is untouched.

K8S_VARS ?= $(REPO_ROOT)/shared/variables/k8s.yml

# Explicit list so envsubst does not expand unrelated ${...} in manifests.
K8S_ENVSUBST_VARS := $${workerCount} $${storageClassName} $${storageSize} $${memoryRequest} $${cpuRequest} $${memoryLimit} $${cpuLimit} $${engineVersion}

define KUBECTL_APPLY_K8S_FROM_VARS
	@set -euo pipefail; \
	k8s_read_yaml_value() { \
		local key="$$1" file="$$2"; \
		[[ -f "$$file" ]] || { echo >&2 "ERROR: file not found: $$file"; exit 1; }; \
		command -v yq >/dev/null 2>&1 || { echo >&2 "ERROR: yq (https://github.com/mikefarah/yq) is required for k8s-apply"; exit 1; }; \
		yq ".$$key" "$$file"; \
	}; \
	command -v envsubst >/dev/null 2>&1 || { echo >&2 "ERROR: envsubst is required for k8s-apply (gettext package)"; exit 1; }; \
	K8S_VARS_FILE='$(K8S_VARS)'; \
	STACK_K8S_VARS='$(REPO_ROOT)/shared/variables/versions/.$(STACK)'; \
	if [[ ! -f "$$K8S_VARS_FILE" ]]; then \
		echo >&2 "ERROR: K8S vars file not found: $$K8S_VARS_FILE"; \
		exit 1; \
	fi; \
	if [[ ! -f "$$STACK_K8S_VARS" ]]; then \
		echo >&2 "ERROR: Stack version file not found: $$STACK_K8S_VARS"; \
		exit 1; \
	fi; \
	export storageClassName="$$( k8s_read_yaml_value storageClassName "$$K8S_VARS_FILE" )"; \
	export storageSize="$$( k8s_read_yaml_value storageSize "$$K8S_VARS_FILE" )"; \
	export memoryRequest="$$( k8s_read_yaml_value memoryRequest "$$K8S_VARS_FILE" )"; \
	export cpuRequest="$$( k8s_read_yaml_value cpuRequest "$$K8S_VARS_FILE" )"; \
	export memoryLimit="$$( k8s_read_yaml_value memoryLimit "$$K8S_VARS_FILE" )"; \
	export cpuLimit="$$( k8s_read_yaml_value cpuLimit "$$K8S_VARS_FILE" )"; \
	export workerCount="$$( k8s_read_yaml_value workerCount "$$K8S_VARS_FILE" )"; \
	engineVersion="$$(head -n1 "$$STACK_K8S_VARS" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$$//')"; \
	if [[ -z "$$engineVersion" ]] || [[ "$$engineVersion" == \#* ]]; then \
		echo >&2 "ERROR: missing or invalid version in $$STACK_K8S_VARS (expected first line: version string)"; \
		exit 1; \
	fi; \
	export engineVersion; \
	$(K8S_VARS_EXTRA_SHELL) \
	if [[ -f '$(REPO_ROOT)/shared/infra/k8s/storage-class.yml' ]]; then \
		echo "kubectl apply: shared cluster StorageClass"; \
		kubectl apply -f '$(REPO_ROOT)/shared/infra/k8s/storage-class.yml'; \
	fi; \
	RENDER_DIR="$$(mktemp -d)"; \
	trap 'rm -rf "$$RENDER_DIR"' EXIT; \
	STACK_K8S='$(STACK_DIR)k8s'; \
	shopt -s nullglob; \
	ALL_ENVSUBST_VARS='$(K8S_ENVSUBST_VARS) $(K8S_ENVSUBST_EXTRA_VARS)'; \
	for f in "$$STACK_K8S"/*.yaml "$$STACK_K8S"/*.yml; do \
		envsubst "$$ALL_ENVSUBST_VARS" < "$$f" > "$$RENDER_DIR/$$(basename "$$f")"; \
	done; \
	kubectl apply -f "$$RENDER_DIR/"
endef
