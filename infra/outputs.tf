output "function_url" {
  description = "The public demo endpoint"
  value       = aws_lambda_function_url.app.function_url
}

output "ecr_repository_url" {
  description = "Push the image here before applying with a new tag"
  value       = aws_ecr_repository.app.repository_url
}

output "log_group" {
  value = aws_cloudwatch_log_group.lambda.name
}
