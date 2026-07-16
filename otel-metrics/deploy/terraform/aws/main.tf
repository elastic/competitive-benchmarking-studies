terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {}

# Look up the architecture of the chosen instance type so the AMI filter matches
data "aws_ec2_instance_type" "selected" {
  instance_type = var.machine
}

# Latest Ubuntu 24.04 AMI for the instance's architecture
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04*"]
  }

  filter {
    name   = "architecture"
    values = [data.aws_ec2_instance_type.selected.supported_architectures[0]]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

resource "aws_instance" "benchmark" {
  ami                                  = data.aws_ami.ubuntu.id
  instance_type                        = var.machine
  key_name                             = var.key_name
  subnet_id                            = var.subnet_id
  vpc_security_group_ids               = length(var.security_group_ids) > 0 ? var.security_group_ids : null
  associate_public_ip_address          = var.subnet_id != null || var.key_name != null
  instance_initiated_shutdown_behavior = var.terminate_on_shutdown ? "terminate" : "stop"

  user_data = templatefile("${path.module}/cloud-init.sh.tftpl", {
    repo         = var.repo
    branch       = var.branch
    github_token = var.github_token
    nvme_mount   = var.nvme_mount
    run_command  = var.run_command
    shutdown     = var.shutdown ? "true" : "false"
  })

  root_block_device {
    volume_size           = 30
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = merge(
    {
      Name = "metrics-benchmark"
      repo = var.repo
    },
    var.tags
  )
}
