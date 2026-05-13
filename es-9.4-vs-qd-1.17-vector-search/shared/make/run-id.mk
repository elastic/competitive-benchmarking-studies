# Shared function for merging run_id from shared/variables/run_id.yml into YAML files
# Reads run_id from shared/variables/run_id.yml and merges it into a YAML file
# Args: $(1)=input yaml path, $(2)=output file path, $(3)=yaml path (e.g., ".evaluation.run_id")
# Note: This function is meant to be used with $(call ...) inside recipe blocks
define MERGE_RUN_ID_TO_FILE
command -v yq >/dev/null 2>&1 || { echo >&2 "ERROR: yq (https://github.com/mikefarah/yq) is required"; exit 1; }; \
if [ -n "$(RUN_ID)" ]; then rid="$(RUN_ID)"; else rid="$$(yq eval '.run_id' "$(REPO_ROOT)/shared/variables/run_id.yml")"; fi; \
yq eval '$(3) = "'"$$rid"'"' "$(1)" >"$(2)"
endef
