terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.12"
    }
  }
  required_version = ">= 1.2.0"

  backend "s3" {
    bucket = "email-summarizer-tf-state"
    key    = "terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "18.30.2"
  cluster_name    = var.cluster_name
  cluster_version = "1.24"
  subnet_ids      = module.vpc.private_subnets

  vpc_id = module.vpc.vpc_id

  eks_managed_node_groups = {
    main = {
      desired_capacity = 2
      max_capacity     = 5
      min_capacity     = 1

      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
      
      labels = {
        Environment = var.environment
      }
    }
  }

  tags = {
    Environment = var.environment
    Project     = "EmailSummarizer"
  }
}

# VPC for EKS
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "3.14.2"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway     = true
  single_nat_gateway     = true
  one_nat_gateway_per_az = false

  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Environment = var.environment
    Project     = "EmailSummarizer"
  }
}

# ECR Repositories
resource "aws_ecr_repository" "api_service" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_ecr_repository" "vector_service" {
  name                 = "${var.project_name}-vector"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_ecr_repository" "ml_service" {
  name                 = "${var.project_name}-ml"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_ecr_repository" "frontend" {
  name                 = "${var.project_name}-frontend"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# IAM role for GitHub Actions
resource "aws_iam_role" "github_actions" {
  name = "${var.project_name}-github-actions-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.github_actions.arn
        }
        Condition = {
          StringLike = {
            "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:*"
          }
        }
      }
    ]
  })

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# IAM policy for GitHub Actions
resource "aws_iam_policy" "github_actions" {
  name        = "${var.project_name}-github-actions-policy"
  description = "Policy for GitHub Actions to push to ECR and deploy to EKS"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage"
        ]
        Effect   = "Allow"
        Resource = "*"
      },
      {
        Action = [
          "eks:DescribeCluster",
          "eks:ListClusters"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "github_actions" {
  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.github_actions.arn
}

# GitHub Actions OIDC Provider
resource "aws_iam_openid_connect_provider" "github_actions" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = [
    "sts.amazonaws.com",
  ]

  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1"
  ]

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# Output the ECR repository URLs and cluster endpoint
output "api_service_repository_url" {
  value = aws_ecr_repository.api_service.repository_url
}

output "vector_service_repository_url" {
  value = aws_ecr_repository.vector_service.repository_url
}

output "ml_service_repository_url" {
  value = aws_ecr_repository.ml_service.repository_url
}

output "frontend_repository_url" {
  value = aws_ecr_repository.frontend.repository_url
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions.arn
}
