import os
import sys
import argparse
import subprocess

import boto3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--best-training-job")
    parser.add_argument("--best-algorithm")
    parser.add_argument("--aws-region")
    
    args, _ = parser.parse_known_args()
    
    sm_client = boto3.client("sagemaker", region_name=args.aws_region)
    best_hyperparameters = sm_client.describe_training_job(TrainingJobName=args.best_training_job)["HyperParameters"]
    best_hyperparameters["cross-validation"] = 'false'
    
    command = [sys.executable, os.path.join(os.path.dirname(__file__), "estimators", f"{args.best_algorithm}.py")]
    for k, v in best_hyperparameters.items():
        command += [f"--{k}", v.strip('"\'') if isinstance(v, str) else v]
    
    subprocess.check_output(command)
