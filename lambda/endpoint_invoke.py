import os
import json
import boto3

runtime = boto3.client("runtime.sagemaker")
s3_client = boto3.client("s3")


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


def lambda_handler(event, context):
    print(f"Got Event: {json.dumps(event)}")

    groupname2model_map = get_groupname2model_map()
    if model_name := groupname2model_map.get(event['model']):
        response = runtime.invoke_endpoint(
            EndpointName=os.environ['ENDPOINT_NAME'],
            ContentType='text/csv',
            TargetModel=model_name,
            Body=event['data']
        )["Body"].read().decode("utf-8")
        return {
            "statusCode": 200,
            "body": response,
        }
    else:
        return {
            "statusCode": 404,
            "body": f"No model found for {event['model']}",
        }
