variable "github_token" {
  description = "GitHub token used to clone the repo"
  type        = string
  sensitive   = true
}

variable "repo" {
  description = "GitHub repo to clone (org/name)"
  type        = string
  default     = "elastic/competitive-benchmarking-studies"
}

variable "branch" {
  description = "Git branch to check out"
  type        = string
  default     = "main"
}

variable "machine" {
  description = "EC2 instance type — must have local NVMe storage"
  type        = string
  default     = "c8gd.4xlarge"
}

variable "nvme_mount" {
  description = "Mount point for the local NVMe device"
  type        = string
  default     = "/data"
}

variable "shutdown" {
  description = "Shut down (and terminate) the instance when the benchmark finishes"
  type        = bool
  default     = false
}

variable "key_name" {
  description = "EC2 key pair name for SSH access (optional)"
  type        = string
  default     = null
}

variable "subnet_id" {
  description = "Subnet to launch the instance in (optional)"
  type        = string
  default     = null
}

variable "security_group_ids" {
  description = "Security group IDs to attach (optional)"
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Additional tags to apply to the EC2 instance (optional)"
  type        = map(string)
  default     = {}
}
