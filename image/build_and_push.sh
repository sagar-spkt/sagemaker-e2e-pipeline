#!/bin/bash

set -e

cd "$(dirname "$0")"

export AWS_PAGER=""

# Login for Sagemaker SKLearn Image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 683313688378.dkr.ecr.us-east-1.amazonaws.com
# Login for building and pushing custom image
aws ecr get-login-password --region $2 | docker login --username AWS --password-stdin $1.dkr.ecr.$2.amazonaws.com

docker build -t $3 .
docker tag $3:latest $1.dkr.ecr.$2.amazonaws.com/$3:latest
docker push $1.dkr.ecr.$2.amazonaws.com/$3:latest

