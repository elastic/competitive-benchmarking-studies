resource "aws_vpc" "benchmarking_elasticsearch_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
}

resource "aws_internet_gateway" "benchmarking_elasticsearch_igw" {
  vpc_id = aws_vpc.benchmarking_elasticsearch_vpc.id
}

resource "aws_subnet" "benchmarking_elasticsearch_subnet_public1" {
  vpc_id                  = aws_vpc.benchmarking_elasticsearch_vpc.id
  cidr_block              = "10.0.21.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true
}

resource "aws_subnet" "benchmarking_elasticsearch_subnet_public2" {
  vpc_id                  = aws_vpc.benchmarking_elasticsearch_vpc.id
  cidr_block              = "10.0.22.0/24"
  availability_zone       = "us-east-1b"
  map_public_ip_on_launch = true
}

resource "aws_route_table" "benchmarking_elasticsearch_route_table" {
  vpc_id = aws_vpc.benchmarking_elasticsearch_vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.benchmarking_elasticsearch_igw.id
  }
}

resource "aws_route_table_association" "benchmarking_elasticsearch_route_table_association_public1" {
  subnet_id      = aws_subnet.benchmarking_elasticsearch_subnet_public1.id
  route_table_id = aws_route_table.benchmarking_elasticsearch_route_table.id
}

resource "aws_route_table_association" "benchmarking_elasticsearch_route_table_association_public2" {
  subnet_id      = aws_subnet.benchmarking_elasticsearch_subnet_public2.id
  route_table_id = aws_route_table.benchmarking_elasticsearch_route_table.id
}
