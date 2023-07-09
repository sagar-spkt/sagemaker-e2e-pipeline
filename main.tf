terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.7"
    }
    awscc = {
      source  = "hashicorp/awscc"
      version = "~> 0.55.0"
    }
  }

  required_version = ">= 1.5.0"
}

variable "pipeline-name" {
  type    = string
  default = "sklearn-multimodel"
}

variable "aws-region" {
  type    = string
  default = "us-east-1"
}

variable "preprocessing-instance" {
  type    = string
  default = "ml.t3.xlarge"
}

variable "training-instance" {
  type    = string
  default = "ml.m5.large"
}

variable "evaluation-instance" {
  type    = string
  default = "ml.t3.xlarge"
}

variable "inference-instance" {
  type    = string
  default = "ml.m5.large"
}

variable "max-endpoint-instances" {
  type    = number
  default = 4
}

provider "aws" {
  region = var.aws-region
}

provider "awscc" {
  region = var.aws-region
}

resource "aws_iam_role" "pipeline_iam_role" {
  name               = "${var.pipeline-name}-pipeline-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = {
          Service = [
            "sagemaker.amazonaws.com",
            "lambda.amazonaws.com"
          ]
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "pipeline_iam_policies" {
  for_each = toset([
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    "arn:aws:iam::aws:policy/AWSLambda_FullAccess",
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    "arn:aws:iam::aws:policy/AmazonSageMakerCanvasFullAccess",
    "arn:aws:iam::aws:policy/AmazonSageMakerCanvasAIServicesAccess",
  ])
  role       = aws_iam_role.pipeline_iam_role.name
  policy_arn = each.value
}

resource "aws_ecr_repository" "ecr_repo" {
  name                 = "${var.pipeline-name}-image"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true
  encryption_configuration {
    encryption_type = "AES256"
  }
}

data "aws_caller_identity" "current_caller" {

}

data "aws_region" "current_region" {

}

resource "null_resource" "image_build_push" {
  triggers = {
    # changes in `image` will trigger docker image build and push step
    file_hashes = sha256(join("", [
      for file_path in fileset("${path.module}/image", "**") : filesha256("${path.module}/image/${file_path}")
    ]))
  }

  provisioner "local-exec" {
    command = <<EOT
      sh ${path.module}/image/build_and_push.sh ${data.aws_caller_identity.current_caller.account_id} ${data.aws_region.current_region.name} ${aws_ecr_repository.ecr_repo.name}
    EOT
  }
}

data "aws_ecr_image" "latest_image" {
  repository_name = aws_ecr_repository.ecr_repo.name
  image_tag       = "latest"
  depends_on      = [null_resource.image_build_push]
}

resource "aws_s3_bucket" "pipeline_bucket" {
  bucket        = "sagemaker-${var.pipeline-name}"
  force_destroy = true
}

resource "null_resource" "source_tar" {
  triggers = {
    file_hashes = sha256(join("", [
      for file_path in fileset("${path.module}/scripts", "**") : filesha256("${path.module}/scripts/${file_path}")
    ]))
  }
  provisioner "local-exec" {
    command = <<EOT
       tar -czvf ${path.module}/.terraform_artifacts/source.tar.gz -C ${path.module}/scripts  $(ls -A ${path.module}/scripts)
    EOT
  }
}

resource "aws_s3_object" "source_tar" {
  bucket      = aws_s3_bucket.pipeline_bucket.bucket
  key         = "codes/source.tar.gz"
  source      = "${path.module}/.terraform_artifacts/source.tar.gz"
  source_hash = sha256(join("", [
    for file_path in fileset("${path.module}/scripts", "**") : filesha256("${path.module}/scripts/${file_path}")
  ]))
  depends_on = [null_resource.source_tar]
}

resource "aws_s3_object" "preprocessing_script" {
  bucket = aws_s3_bucket.pipeline_bucket.bucket
  key    = "codes/preprocessing.py"
  source = "${path.module}/scripts/preprocessing.py"
  etag   = filemd5("${path.module}/scripts/preprocessing.py")
}

resource "aws_s3_object" "evaluation_script" {
  bucket = aws_s3_bucket.pipeline_bucket.bucket
  key    = "codes/evaluate.py"
  source = "${path.module}/scripts/evaluate.py"
  etag   = filemd5("${path.module}/scripts/evaluate.py")
}

locals {
  pipeline_definition_command = <<EOT
    python ${path.module}/pipeline.py \
    --bucket ${aws_s3_bucket.pipeline_bucket.bucket} \
    --role ${aws_iam_role.pipeline_iam_role.arn} \
    --image-uri ${data.aws_caller_identity.current_caller.account_id}.dkr.ecr.${data.aws_region.current_region.name}.amazonaws.com/${aws_ecr_repository.ecr_repo.name}@${data.aws_ecr_image.latest_image.image_digest} \
    --pipeline-name ${var.pipeline-name} \
    --aws-region ${data.aws_region.current_region.name} \
    --source-s3-uri s3://${aws_s3_object.source_tar.bucket}/${aws_s3_object.source_tar.key} \
    --preprocessing-script-s3 s3://${aws_s3_object.preprocessing_script.bucket}/${aws_s3_object.preprocessing_script.key} \
    --evaluation-script-s3 s3://${aws_s3_object.evaluation_script.bucket}/${aws_s3_object.evaluation_script.key} \
    --definition-output ${path.module}/.terraform_artifacts/pipeline_definition.json \
    --preprocessing-instance ${var.preprocessing-instance} \
    --training-instance ${var.training-instance} \
    --evaluation-instance ${var.evaluation-instance} \
    --inference-instance ${var.inference-instance} \
  EOT
}

resource "null_resource" "pipeline_definition" {
  triggers = {
    file_hashes = sha256(join("", [
      filesha256("${path.module}/pipeline.py"), sha256(local.pipeline_definition_command)
    ]))
  }
  provisioner "local-exec" {
    command = local.pipeline_definition_command
  }
}

resource "aws_s3_object" "pipeline_definition" {
  bucket      = aws_s3_bucket.pipeline_bucket.bucket
  key         = "codes/pipeline_definition.json"
  content_type = "application/json"
  source      = "${path.module}/.terraform_artifacts/pipeline_definition.json"
  source_hash = sha256(join("", [
    filesha256("${path.module}/pipeline.py"), sha256(local.pipeline_definition_command)
  ]))
  depends_on = [null_resource.pipeline_definition]
}

resource "awscc_sagemaker_pipeline" "pipeline" {
  pipeline_name        = var.pipeline-name
  role_arn             = aws_iam_role.pipeline_iam_role.arn
  pipeline_description = "E2E Hyperparameter Optimization Multi-Model Pipeline"
  pipeline_definition  = {
    pipeline_definition_body = aws_s3_object.pipeline_definition.content
  }
}

locals {
  multimodel_artifacts_key = "multimodel-artifacts"
}

resource "aws_sagemaker_model" "endpoint_model" {
  name               = "${var.pipeline-name}-multimodel"
  execution_role_arn = aws_iam_role.pipeline_iam_role.arn
  primary_container {
    image          = "${data.aws_caller_identity.current_caller.account_id}.dkr.ecr.${data.aws_region.current_region.name}.amazonaws.com/${aws_ecr_repository.ecr_repo.name}@${data.aws_ecr_image.latest_image.image_digest}"
    model_data_url = "s3://${aws_s3_bucket.pipeline_bucket.bucket}/${local.multimodel_artifacts_key}/"
    mode           = "MultiModel"
    environment    = {
      SAGEMAKER_CONTAINER_LOG_LEVEL = 20
      SAGEMAKER_PROGRAM             = "inference.py"
      SAGEMAKER_REGION              = data.aws_region.current_region.name
      SAGEMAKER_SUBMIT_DIRECTORY    = "s3://${aws_s3_object.source_tar.bucket}/${aws_s3_object.source_tar.key}"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "endpoint_configuration" {
  name = "${var.pipeline-name}-endpoint-config"
  production_variants {
    model_name             = aws_sagemaker_model.endpoint_model.name
    variant_name           = "AllTraffic"
    instance_type          = var.inference-instance
    initial_instance_count = 1
    initial_variant_weight = 1
  }
}

resource "aws_sagemaker_endpoint" "endpoint" {
  name                 = var.pipeline-name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.endpoint_configuration.name
}

resource "aws_appautoscaling_target" "pipeline_endpoint" {
  max_capacity       = var.max-endpoint-instances
  min_capacity       = 1
  resource_id        = "endpoint/${aws_sagemaker_endpoint.endpoint.name}/variant/${aws_sagemaker_endpoint_configuration.endpoint_configuration.production_variants[0].variant_name}"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

resource "aws_appautoscaling_policy" "pipeline_endpoint" {
  name               = "${var.pipeline-name}:${aws_appautoscaling_target.pipeline_endpoint.resource_id}"
  resource_id        = aws_appautoscaling_target.pipeline_endpoint.resource_id
  scalable_dimension = aws_appautoscaling_target.pipeline_endpoint.scalable_dimension
  service_namespace  = aws_appautoscaling_target.pipeline_endpoint.service_namespace
  policy_type        = "TargetTrackingScaling"
  target_tracking_scaling_policy_configuration {
    target_value = 85
    customized_metric_specification {
      metric_name = "CPUUtilization"
      namespace   = "/aws/sagemaker/Endpoints"
      statistic   = "Average"
      unit        = "Percent"
      dimensions {
        name  = "EndpointName"
        value = aws_sagemaker_endpoint.endpoint.name
      }
      dimensions {
        name  = "VariantName"
        value = aws_sagemaker_endpoint_configuration.endpoint_configuration.production_variants[0].variant_name
      }
    }
  }
}

data "archive_file" "deploy_lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/lambda/endpoint_deploy.py"
  output_path = "${path.module}/.terraform_artifacts/deploy_lambda.zip"
}

resource "aws_lambda_function" "lambda_endpoint_deployer" {
  function_name                  = "sagemaker-${var.pipeline-name}-endpoint-deploy"
  role                           = aws_iam_role.pipeline_iam_role.arn
  handler                        = "endpoint_deploy.lambda_handler"
  runtime                        = "python3.10"
  filename                       = data.archive_file.deploy_lambda_zip.output_path
#  reserved_concurrent_executions = 1
  source_code_hash               = data.archive_file.deploy_lambda_zip.output_base64sha256
  environment {
    variables = {
      PIPELINE_BUCKET     = aws_s3_bucket.pipeline_bucket.bucket
      MODEL_ARTIFACTS_KEY = local.multimodel_artifacts_key
      GROUPNAME_MODEL_MAP = "groupname2model_map.json"
    }
  }
}

resource "aws_cloudwatch_event_rule" "model_package_update_rule" {
  name          = "sagemaker-${var.pipeline-name}-model-approve-or-reject"
  description   = "Listens to every sagemaker model package state change to either reject or update and invokes lambda function to deploy the latest approved model in the model package group to the endpoint."
  is_enabled    = true
  event_pattern = jsonencode({
    source      = ["aws.sagemaker"],
    detail-type = ["SageMaker Model Package State Change"],
    detail      = {
      ModelPackageGroupName = [
        {
          prefix = "${var.pipeline-name}-"
        }
      ]
      ModelApprovalStatus = [
        {
          anything-but = ["PendingManualApproval"]
        }
      ]
    }
  })
}

resource "aws_cloudwatch_event_target" "model_package_update_rule_target" {
  arn  = aws_lambda_function.lambda_endpoint_deployer.arn
  rule = aws_cloudwatch_event_rule.model_package_update_rule.id
  retry_policy {
    maximum_event_age_in_seconds = 3600
    maximum_retry_attempts       = 3
  }
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lambda_endpoint_deployer.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.model_package_update_rule.arn
}

data "archive_file" "invoke_lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/lambda/endpoint_invoke.py"
  output_path = "${path.module}/.terraform_artifacts/invoke_lambda.zip"
}

resource "aws_lambda_function" "lambda_endpoint_invoker" {
  function_name    = "sagemaker-${var.pipeline-name}-endpoint-invoker"
  role             = aws_iam_role.pipeline_iam_role.arn
  handler          = "endpoint_invoke.lambda_handler"
  runtime          = "python3.10"
  filename         = data.archive_file.invoke_lambda_zip.output_path
  source_code_hash = data.archive_file.invoke_lambda_zip.output_base64sha256
  environment {
    variables = {
      PIPELINE_BUCKET     = aws_lambda_function.lambda_endpoint_deployer.environment[0].variables.PIPELINE_BUCKET
      MODEL_ARTIFACTS_KEY = aws_lambda_function.lambda_endpoint_deployer.environment[0].variables.MODEL_ARTIFACTS_KEY
      GROUPNAME_MODEL_MAP = aws_lambda_function.lambda_endpoint_deployer.environment[0].variables.GROUPNAME_MODEL_MAP
      ENDPOINT_NAME       = aws_sagemaker_endpoint.endpoint.name
    }
  }
}

resource "aws_lambda_function_url" "function_url_endpoint" {
  authorization_type = "NONE"
  function_name      = aws_lambda_function.lambda_endpoint_invoker.function_name
}

output "function_url_endpoint" {
  value = aws_lambda_function_url.function_url_endpoint.function_url
}