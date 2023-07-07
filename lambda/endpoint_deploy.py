import os
import json
import boto3
import logging
from time import gmtime, strftime

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

sm_client = boto3.client("sagemaker")
lambda_client = boto3.client('lambda')
s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")


def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group."""
    # Get the latest approved model package
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        MaxResults=100,
    )
    approved_packages = response["ModelPackageSummaryList"]

    # Fetch more packages if none returned with continuation token
    while len(approved_packages) == 0 and "NextToken" in response:
        logger.info("Getting more packages for token: {}".format(
            response["NextToken"]))
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
            NextToken=response["NextToken"],
        )
        approved_packages.extend(response["ModelPackageSummaryList"])

    # Return error if no packages found
    if len(approved_packages) == 0:
        error_message = (
            f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
        )
        logger.error(error_message)
        raise Exception(error_message)

    # Return the pmodel package arn
    model_package_arn = approved_packages[0]["ModelPackageArn"]
    logger.info(
        f"Identified the latest approved model package: {model_package_arn}")
    return model_package_arn


def get_groupname2model_map():
    try:
        groupname2model_map = s3_client.get_object(
            Bucket=os.environ["PIPELINE_BUCKET"],
            Key=f"{os.environ['MODEL_ARTIFACTS_KEY']}/{os.environ['GROUPNAME_MODEL_MAP']}",
        )["Body"].read().decode('utf-8')
        groupname2model_map = json.loads(groupname2model_map)
        return groupname2model_map
    except s3_client.exceptions.NoSuchKey:
        return {}


def set_groupname2model_map(groupname2model_map):
    groupname2model_map = json.dumps(groupname2model_map)
    s3_object = s3_resource.Object(
        bucket_name=os.environ["PIPELINE_BUCKET"],
        key=f"{os.environ['MODEL_ARTIFACTS_KEY']}/{os.environ['GROUPNAME_MODEL_MAP']}"
    )
    s3_object.put(Body=groupname2model_map, ContentType="application/json")


# def s3tos3_copy(src_bucket, src_key, dest_bucket, dest_key):
def s3tos3_copy(src_s3_uri, dest_s3_uri):
    src_s3_uri = src_s3_uri.split("/")
    dest_s3_uri = dest_s3_uri.split("/")
    src_bucket, src_key = src_s3_uri[2], "/".join(src_s3_uri[3:])
    dest_bucket, dest_key = dest_s3_uri[2], "/".join(dest_s3_uri[3:])

    s3_resource.Bucket(
        dest_bucket
    ).copy(
        {"Bucket": src_bucket, "Key": src_key},
        dest_key
    )


def lambda_handler(event, context):
    logger.info(f"Got Event: {json.dumps(event)}")

    model_package_group_name = event["detail"]["ModelPackageGroupName"]
    model_package_arn = get_approved_package(model_package_group_name)

    new_model_path = sm_client.describe_model_package(
        ModelPackageName=model_package_arn
    )["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    model_name = model_package_group_name + "-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + ".tar.gz"
    dest_model_path = f's3://{os.environ["PIPELINE_BUCKET"]}/{os.environ["MODEL_ARTIFACTS_KEY"]}/{model_name}'

    s3tos3_copy(new_model_path, dest_model_path)

    groupname2model_map = get_groupname2model_map()
    if old_model_name := groupname2model_map.get(model_package_group_name):
        groupname2model_map[model_package_group_name] = model_name
        set_groupname2model_map(groupname2model_map)

        s3_client.delete_object(
            Bucket=os.environ["PIPELINE_BUCKET"],
            Key=f"{os.environ['MODEL_ARTIFACTS_KEY']}/{old_model_name}"
        )
    else:
        groupname2model_map[model_package_group_name] = model_name
        set_groupname2model_map(groupname2model_map)

    return {
        "statusCode": 200,
        "body": json.dumps("Endpoint Deployed!"),
    }
