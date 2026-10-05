# Infrastructure for the deployed demo.
#
# Shape: one Lambda function running the container image, reachable through a
# Function URL. No API Gateway, no load balancer, no vector database — the index
# is baked into the image (docs/decisions/006), so there is nothing running, and
# nothing billed, between requests.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

locals {
  name = "rag-eval-pipeline"
}

# ----------------------------------------------------------------- ECR ------

resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# The image is 2.4 GB. Without this, every deploy leaves the previous one behind
# at roughly €0.24/month each, forever.
resource "aws_ecr_lifecycle_policy" "keep_recent" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 3 most recent images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 3 }
      action       = { type = "expire" }
    }]
  })
}

# ----------------------------------------------------------------- IAM ------

resource "aws_iam_role" "lambda" {
  name = "${local.name}-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Least privilege: invoke one model, in one region. Not bedrock:* on "*".
resource "aws_iam_role_policy" "bedrock" {
  name = "${local.name}-bedrock-invoke"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["bedrock:InvokeModel"]
      Resource = [
        "arn:aws:bedrock:${var.region}::foundation-model/${var.bedrock_foundation_model}",
        "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:inference-profile/${var.bedrock_model_id}",
      ]
    }]
  })
}

# -------------------------------------------------------------- Lambda ------

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.name}"
  retention_in_days = 14 # logs are not the product; do not accumulate them forever
}

resource "aws_lambda_function" "app" {
  function_name = local.name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"

  # The embedding model loads on cold start and the index is copied to /tmp.
  # 2 GB buys proportionally more CPU, which is what actually shortens that.
  memory_size = 2048
  timeout     = 60

  # The ceiling on a surprise bill. A public endpoint with unbounded concurrency
  # is an invitation; two at a time is plenty for a demo and caps the blast radius.
  reserved_concurrent_executions = var.max_concurrency

  environment {
    variables = {
      QDRANT_BAKED_INDEX = "/opt/index"
      HF_HOME            = "/opt/hf"
      BEDROCK_MODEL_ID   = var.bedrock_model_id
      RATE_LIMIT_PER_MIN = tostring(var.rate_limit_per_min)
    }
  }

  depends_on = [aws_cloudwatch_log_group.lambda]
}

resource "aws_lambda_function_url" "app" {
  function_name      = aws_lambda_function.app.function_name
  authorization_type = "NONE" # public on purpose: a link nobody can open proves nothing

  cors {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST"]
  }
}

# ---------------------------------------------------------------- cost ------

# Lambda is free at idle, but "free" assumes nobody hammers a public endpoint.
# This is the tripwire, not the defence.
resource "aws_budgets_budget" "monthly" {
  name         = "${local.name}-monthly"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_eur
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }
}
