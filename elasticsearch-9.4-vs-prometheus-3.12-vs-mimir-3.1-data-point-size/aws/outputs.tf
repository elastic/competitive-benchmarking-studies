output "instance_id" {
  value = aws_instance.benchmark.id
}

output "public_ip" {
  value = aws_instance.benchmark.public_ip
}

output "console_log_command" {
  value = "aws ec2 get-console-output --instance-id ${aws_instance.benchmark.id} --latest --output text"
}
