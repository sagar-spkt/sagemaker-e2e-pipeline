# Table of Contents
1. [Preparing Environment](#preparing-environment)
2. [Running Pipeline With Sagemaker Studio](#running-pipeline-with-sagemaker-studio)
3. [Running Pipeline with Boto3](#running-pipeline-with-boto3)
4. [Invoking the Endpoint](#invoking-the-endpoint)
5. [Clearing AWS Resources and Artifacts](#clearing-aws-resources-and-artifacts-created-by-pipeline)

## Preparing Environment

### AWS Setup

Install the AWS Command Line Interface (CLI), which is used to interact with various AWS services from the command line.
```shell
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

Setup default AWS credentials and region. This step configures the default AWS credentials and region for the CLI. It prompts you to enter the Access Key ID, Secret Access Key, default region name, and default output format.

```shell
aws configure
```

Enter your credentials and default AWS region. For example:
```mathematica
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-east-1
Default output format [None]:
```

Make sure your AWS credentials have full access to following AWS resources that are used in the pipeline:
- AWS App Autoscaling: This resource is used for automatically scaling AWS resources based on defined policies.
- AWS EventBridge(Previously Cloudwatch Event): This resource is used for monitoring and responding to Sagemaker Model Registry events.
- AWS Elastic Container Registry(ECR): This resource is used for storing Docker container images used in the pipeline.
- AWS IAM Role/Policies: This resource is used for defining and managing access permissions for AWS services used by various components in the pipeline.
- AWS Lambda Function: This resource is used for handling various events in pipeline, e.g., deployment and  of endpoint.
- AWS S3: This resource is used for storing input and output data for the pipeline.
- AWS Sagemaker: This resource is the main service used for training and deploying machine learning models.

### Install Terraform

Install Terraform, an infrastructure as code tool, which will be used to create and manage the AWS resources required for the pipeline. For linux, you can use following instructions. For other platforms refer official terraform installation instructions.
```shell
sudo apt-get update && sudo apt-get install -y gnupg software-properties-common
wget -O- https://apt.releases.hashicorp.com/gpg | \
gpg --dearmor | \
sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
gpg --no-default-keyring \
--keyring /usr/share/keyrings/hashicorp-archive-keyring.gpg \
--fingerprint
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update
sudo apt-get install terraform
```

### Install Docker

Installs Docker, a containerization platform, which will be used to build and run Docker containers for preprocessing, training, evaluation, and inference in the pipeline. For linux, you can use the following set of commands. For other platforms, refer official docker installation instructions.
```shell
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ${USER}
su - ${USER}
```

### Clone the Repo

Clone the Git repository that contains the source code and configuration files for the SageMaker end-to-end pipeline.
```shell
git clone https://github.com/sagar-spkt/sagemaker-e2e-pipeline.git
cd sagemaker-e2e-pipeline
```

### Python Environment

Create virtual environment and install required dependencies from `requirements.txt`
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Create AWS Resources

Initialize Terraform and apply the Terraform configuration to create the AWS resources needed for the pipeline. The provided variables specify the pipeline name, AWS region, instance types for preprocessing, training, evaluation, and inference, and the maximum number of endpoint instances. Adjust these variables as needed for your specific requirements.
```shell
terraform init
terraform apply \
-var pipeline-name="sklearn-multimodel" \
-var aws-region="us-east-1" \
-var preprocessing-instance="ml.t3.xlarge" \
-var training-instance="ml.m5.large" \
-var evaluation-instance="ml.t3.xlarge" \
-var inference-instance="ml.m5.large" \
-var max-endpoint-instances=4
```

**Note:** If you have use Sagemaker before in your account, it is possible that the default bucket for Sagemaker already exists and terraform throws error. In that case, you can import existing bucket configuration to terraform with the following command:
```shell
terraform import aws_s3_bucket.pipeline_bucket sagemaker-<YOUR_AWS_REGION>-<YOUR AWS ACCOUNT ID>
```

## Running Pipeline With Sagemaker Studio

After terraform creates the required infrastructures, you can see the newly created pipeline under **Home**>**Pipelines**.
![](images/image1.png)

Select the pipeline and click the `Create execution` button. You'll see pop like below where you can input the parameters for pipeline execution. Give the execution a unique name and input other parameters and hit `Start` button. Following are the descriptions of the input parameters to the pipeline:
- **DatasetName**: Name of the dataset against which the pipeline performs modeling
- **StratifySplit**: Whether to perform stratified train/test split of the dataset
- **TestSetSize**: Test set size to create.
- **RandomState**: Seed used for random operation in the pipeline
- **CrossValidationScorer**: Metric used to select best model during cross-validation
- **MaxTuningJobs**: Maximum number of tuning jobs to run. Hyperparameter search strategy used in the pipeline is `Random Search`.
- **MaxTuningParallelJobs**: Maximum parallel runs during hyperparameter tuning
- **MetricForRegistrationThreshold**: Metric whose value is used to decide whether to register the model to registry or not.
- **MinThesholdForRegisterMetric**: Minimum value for `MetricForRegistrationThreshold` the model should pass to be registered in the model registry.
- **RegisterModelApprovalStatus**: Whether to automatically deploy the model to endpoint or not if it passes the `MetricForRegistrationThreshold` constraint.
![](images/image2.png)

You can see the list of pipeline executions in the dashboard. You'll find information like the execution status, time taken, and many more.
![](images/image3.png)

While one execution is running, you can run as many other executions you like provided that you don't hit the services quota limit of the instances and other resources used by the pipeline.
![](images/image4.png)

When you click the executions, you'll find a nice graph of pipeline steps with their dependency links, execution progress, and inputs, outputs, logs and other information for each steps. 
![](images/image5.png)

After pipeline execution completes, you'll find model groups for the respective dataset under **Home**>**Models**>**Model registry**. For each dataset, the pipeline creates model group in the registry with the name formatted as `<PIPELINE_NAME>-<DATASET_NAME>`.
![](images/image6.png)

Inside the model groups, you'll find all the model version that has been registered in model registry for the group. The new version of the model will have wither status of `Pending` and `Approved` based on the value of the parameter `RegisterModelApprovalStatus` passed during pipeline execution.
![](images/image7.png)

For evaluating the model version in order to decide whether to approve or reject the model, you can go to **Version#**>**Model Quality** and have a look at the model performance metrices. 
![](images/image8.png)

After looking at the performance metrices, you can approve or reject the model version for deployment from **Version#**>**Actions**>**Update Status**.
![](images/image9.png)

Approving or rejecting the model will trigger an EventBridge rule whose target is a Lambda function responsible for deploying the latest approved model from the model registry to the endpoint.

## Running Pipeline with Boto3

Import the required libraries.


```python
import json
import boto3
```

Initialize sagemaker and s3 boto3 client. `s3_client` is required as it will be used to read best model evaluation results saved by the pipeline in s3 bucket.


```python
pipeline_name = "sklearn-multimodel"
sm_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")
```

Retrieve the pipeline parameters. Following code will return input parameters to the pipeline, their type, and if available the default value. Following are the descriptions of the input parameters to the pipeline:
- **DatasetName**: Name of the dataset against which the pipeline performs modeling
- **StratifySplit**: Whether to perform stratified train/test split of the dataset
- **TestSetSize**: Test set size to create.
- **RandomState**: Seed used for random operation in the pipeline
- **CrossValidationScorer**: Metric used to select best model during cross-validation
- **MaxTuningJobs**: Maximum number of tuning jobs to run. Hyperparameter search strategy used in the pipeline is `Random Search`.
- **MaxTuningParallelJobs**: Maximum parallel runs during hyperparameter tuning
- **MetricForRegistrationThreshold**: Metric whose value is used to decide whether to register the model to registry or not.
- **MinThesholdForRegisterMetric**: Minimum value for `MetricForRegistrationThreshold` the model should pass to be registered in the model registry.
- **RegisterModelApprovalStatus**: Whether to automatically deploy the model to endpoint or not if it passes the `MetricForRegistrationThreshold` constraint.


```python
json.loads(sm_client.describe_pipeline(
    PipelineName=pipeline_name
)["PipelineDefinition"])["Parameters"]
```

Start the pipeline execution with the desired parameters. The pipeline creates individual model for each dataset. If no previous training for the dataset is done, it creates new model group in Sagemaker registry and if the model group already exists in the registry, model will be registered with new version.


```python
pipeline_start_resp = sm_client.start_pipeline_execution(
    PipelineName=pipeline_name,
    PipelineParameters=[
        # specify dataset name on which to train the model
        {'Name': 'DatasetName', 'Value': 'breast-cancer'},
        {'Name': 'MaxTuningJobs', 'Value': '4'},
        {'Name': 'MaxTuningParallelJobs', 'Value': '2'},
        # pass `Approved` if you want to auto approve the model in registry for deployment
        # if it meets the requirement for model registry
        {'Name': 'RegisterModelApprovalStatus', 'Value': 'PendingManualApproval'},
        # mention other parameters if you do not want to use default value for them
    ],
)
pipeline_start_resp["PipelineExecutionArn"]
```

Check pipeline execution status. Possible state of pipeline execution are: `Executing`|`Stopping`|`Stopped`|`Failed`|`Succeeded`. Wait until the value for `PipelineExecutionStatus` in the response is `Succeeded` otherwise subsequent steps will raise error. If the pipeline execution fails in anyway, you can find find the region for failure in the response of the following code.


```python
sm_client.describe_pipeline_execution(
    PipelineExecutionArn=pipeline_start_resp["PipelineExecutionArn"],
)
```

After pipeline execution status is `Succeeded`, you can extract the latest model package from the registry. Following code will raise error if `PipelineExecutionStatus` is not `Succeeded`.


```python
response = sm_client.list_pipeline_execution_steps(
    # replace the arn with the arn of a package whose performance you want to look and do approve/reject operation
    PipelineExecutionArn=pipeline_start_resp["PipelineExecutionArn"],
    MaxResults=100,
    SortOrder='Descending'
)
latest_model_package_arn = response["PipelineExecutionSteps"][0]["Metadata"]["RegisterModel"]["Arn"]
latest_model_package_arn
```

From this arn, we can extract the model's evaluation metric that is stored in s3. Following code will return stringified json eval results. This json is prepared by `scripts/evaluate.py`. Refer to the evaluation code there if you want to know how these results were prepared.



```python
model_package_resp = sm_client.describe_model_package(ModelPackageName=latest_model_package_arn)
model_eval_s3_path = model_package_resp["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]
eval_results = s3_client.get_object(Bucket=model_eval_s3_path.split("/")[2], Key="/".join(model_eval_s3_path.split("/")[3:]))["Body"].read().decode('utf-8')
eval_results
```

Looking at the performance, you can approve or reject the model for deployment. If the pipeline is executed with auto approval status enabled, it will already be deployed in the endpoint and rejecting the model will revert the model used in endpoint to the last approved model version.


```python
# Approve the model
sm_client.update_model_package(
    ModelPackageArn=latest_model_package_arn,
    ModelApprovalStatus='Approved',
    ApprovalDescription='<Comment for approving the model>',
)

# Reject the model
sm_client.update_model_package(
    ModelPackageArn=latest_model_package_arn,
    ModelApprovalStatus='Rejected',
    ApprovalDescription='<Comment for rejecting the model>',
)
```

Approving or rejecting the model will trigger an EventBridge rule whose target is a Lambda function responsible for deploying the latest approved model from the model registry to the endpoint.

## Invoking the Endpoint

Since all models are hosted in a single endpoint, we've to mention with which model to do the inference. All of these models are stored in a s3 bucket folder and Sagemaker downloads the requested model from that s3 location upon invocation if it is not already instantiated in the endpoint instances. The catch here is that with every new version deployed, the model name will be timestamped as shown in the screenshot below. For the enduser, it is not convenient to mention timestamped model name.
![](images/image10.png)
So, the map between the model group name(dataset name) and the latest model for that group is maintained in a JSON file and a lambda function is created that extracts latest model for the user requested model group and do the inference using sagemaker endpoint. We'll use its function url. The function url can be found using the following terraform command.
```shell
terraform output
```
Note the function url from the output of this command. The output will be in the following format:
```mathematica
function_url_endpoint = "https://yugkwzcne3rop4meh2x3g32xbq0fbhrp.lambda-url.us-east-1.on.aws/"
```

#### Payload Format

The payload to this function url is a json with the following format:
```json
{
    "model": "<PIPELINE_NAME>-<DATASET-NAME>"
    "data": "<CSV DUMP WITHOUT HEADERS WHICH SATISFIES INPUT TO SKLEARN'S MODEL.PREDICT FUNCTION>"
}
```
Following function uses pandas and json libraries to prepare payload in python and use requests to make the `POST` request to the function url.


```python
import requests
import json
import pandas as pd

def invoke_endpoint(function_url, pipeline_name, dataset, data):
    model = f"{pipeline_name}-{dataset}"
    data = pd.DataFrame(data).to_csv(index=None, header=None)
    payload = json.dumps({
        "model": model,
        "data": data,
    })
    headers = {
        'Content-Type': 'application/json',
    }
    return requests.request("POST", function_url, headers=headers, data=payload).text
```

Create variables to store function url and pipeline name.


```python
# assign `function_url_endpoint` your function url endpoint
function_url_endpoint = "https://yugkwzcne3rop4meh2x3g32xbq0fbhrp.lambda-url.us-east-1.on.aws/"
pipeline_name = "sklearn-multimodel"
```

Create a test banknote data and invoke the `banknote-authentication` model.


```python
banknote_testdata = [
    [3.6216,8.6661,-2.8073,-0.44699],
    [4.5459,8.1674,-2.4586,-1.4621],
    [3.866,-2.6383,1.9242,0.10645],
    [3.4566,9.5228,-4.0112,-3.5944],
]
dataset = "banknote-authentication"
invoke_endpoint(function_url_endpoint, pipeline_name, dataset, banknote_testdata)
```

Create a test breast cancer data and invoke the `breast-cancer` model.


```python
# dummay data
breastcancer_testdata = [
    [1]*30,
    [0]*30,
]
dataset = "breast-cancer"
invoke_endpoint(function_url_endpoint, pipeline_name, dataset, breastcancer_testdata)
```

## Clearing AWS Resources and Artifacts Created By Pipeline

You can use terraform to destroy most of the resources created in this guideline with the following command.
```shell
terraform destroy
```
However, this command won't delete following resources. You've to delete them manually.
- AWS Sagemaker Model Packages in Model Registry since it is not created by the Terraform. They're created dynamically by the pipeline based on input hance not destroyed by Terraform.
