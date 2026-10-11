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

# Lambda pulls the image as a service, not as the caller, so the execution role
# cannot grant this — it needs a resource policy on the repository itself.
# Without it CreateFunction fails with "Lambda does not have permission to access
# the ECR image", which names ECR without saying that a repository policy is what
# is missing. Scoped by sourceArn so only this function may pull.
resource "aws_ecr_repository_policy" "lambda_pull" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "LambdaECRImageRetrievalPolicy"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action = [
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
      ]
      Condition = {
        StringLike = {
          "aws:sourceArn" = "arn:aws:lambda:${var.region}:${data.aws_caller_identity.current.account_id}:function:${local.name}"
        }
      }
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
  # The image's own CMD runs uvicorn, which is right for `docker run` and wrong for
  # Lambda: the runtime polls a client for events rather than speaking HTTP to the
  # container. Overriding the entry point here keeps one image serving both, instead
  # of maintaining a second Dockerfile that drifts from the first.
  image_config {
    entry_point = ["/usr/local/bin/python", "-m", "awslambdaric"]
    command     = ["api.lambda_handler.handler"]
  }

  memory_size = 2048

  # Cold start loads sentence-transformers and copies the baked index to /tmp. 60s
  # was a guess made before anything had ever run; raise it until a real cold start
  # is measured, then lower it to something justified by the number.
  timeout = 120

  # The ceiling on a surprise bill. A public endpoint with unbounded concurrency is
  # an invitation. On an account whose total limit is 10 this must be -1 (see
  # variables.tf) and the account limit is the ceiling; the per-IP rate limit in the
  # application and the budget alarm are the other two layers.
  reserved_concurrent_executions = var.max_concurrency

  environment {
    variables = {
      QDRANT_BAKED_INDEX = "/opt/index"
      HF_HOME            = "/opt/hf"
      BEDROCK_MODEL_ID   = var.bedrock_model_id
      RATE_LIMIT_PER_MIN = tostring(var.rate_limit_per_min)
    }
  }

  # The repository policy must exist before CreateFunction, which verifies the
  # image is pullable at create time rather than on first invoke.
  depends_on = [
    aws_cloudwatch_log_group.lambda,
    aws_ecr_repository_policy.lambda_pull,
  ]
}

# authorization_type = "NONE" only says the URL does not check IAM; it does not grant
# anyone the right to call it. Without this the URL answers every request with 403.
# The FunctionUrlAuthType condition keeps the grant to the URL: it does not let the
# world call the function through the Lambda API.
resource "aws_lambda_permission" "public_url" {
  statement_id           = "AllowPublicFunctionUrlInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.app.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
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
