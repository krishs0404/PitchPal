variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
  default     = "staging"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "cold-email-io-cluster"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "cold-email-io"
}

variable "github_repo" {
  description = "GitHub repository in format 'owner/repo'"
  type        = string
  default     = "krishs0404/ColdEmailIO"
}
