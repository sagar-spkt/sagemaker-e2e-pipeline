#!/bin/bash

set -e

cd "$(dirname "$0")"

export AWS_PAGER=""

# Login
aws ecr get-login-password --region $2 | docker login --username AWS --password-stdin $1.dkr.ecr.$2.amazonaws.com

docker build -t $3 .
docker tag $3:latest $1.dkr.ecr.$2.amazonaws.com/$3:latest
docker push $1.dkr.ecr.$2.amazonaws.com/$3:latest

