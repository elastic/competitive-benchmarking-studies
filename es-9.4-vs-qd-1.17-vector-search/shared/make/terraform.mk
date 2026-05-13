.PHONY: terraform-init terraform-plan terraform-apply terraform-destroy

terraform-init:
	cp "$(REPO_ROOT)/shared/infra/terraform/modules/gke-benchmark/gke-engine-root-variables.tf" "$(STACK_DIR)terraform/gke-engine-root-variables.tf"
	cd $(STACK_DIR)terraform && terraform init

terraform-plan: terraform-init
	cd $(STACK_DIR)terraform && terraform plan -var-file=$(REPO_ROOT)/shared/secrets/terraform.tfvars -var-file=$(REPO_ROOT)/shared/variables/terraform.tfvars

terraform-apply: terraform-init
	cd $(STACK_DIR)terraform && terraform apply -var-file=$(REPO_ROOT)/shared/secrets/terraform.tfvars -var-file=$(REPO_ROOT)/shared/variables/terraform.tfvars

terraform-destroy: terraform-init
	@echo >&2 "WARNING: This will delete ALL resources."
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ]
	cd $(STACK_DIR)terraform && terraform destroy -var-file=$(REPO_ROOT)/shared/secrets/terraform.tfvars -var-file=$(REPO_ROOT)/shared/variables/terraform.tfvars
