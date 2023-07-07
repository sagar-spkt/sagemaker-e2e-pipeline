import os
import json
import joblib
import tarfile
import argparse
import pathlib

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, default="dataset.csv")
    parser.add_argument("--register-metric", type=str, default="auc")
    parser.add_argument("--features", default=None)
    parser.add_argument("--target")
    
    args, _ = parser.parse_known_args()
    
    print("Extracting Model.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    print("Loading Model")
    model = joblib.load("./model.joblib")

    print("Loading test input data.")
    test_path = f"/opt/ml/processing/test/{args.test_file}"
    test_df = pd.read_csv(test_path)

    print("building testing datasets")
    feature_columns = args.features.split() if args.features else [
        col for col in test_df.columns
        if col != args.target
    ]
    X_test = test_df[feature_columns]
    y_test = test_df[args.target]

    print("Performing predictions against test data.")
    predictions, pred_proba = model.predict(X_test), model.predict_proba(X_test)
    prediction_probabilities = pred_proba[:, 1]
    
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, prediction_probabilities)
    conf_matrix = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, prediction_probabilities)
    precs, recs, _ = precision_recall_curve(y_test, prediction_probabilities)

    if len(fpr) > 10000:
        roc_curve_index = np.sort(np.random.choice(np.arange(len(fpr)), size=10000, replace=False))
        fpr = fpr[roc_curve_index]
        tpr = tpr[roc_curve_index]

    if len(precs) > 10000:
        pr_curve_index = np.sort(np.random.choice(np.arange(len(precs)), size=10000, replace=False))
        precs = precs[pr_curve_index]
        recs = recs[pr_curve_index]

    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Confusion matrix: {}".format(conf_matrix))

    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "f1": {"value": f1, "standard_deviation": "NaN"},
            "auc": {"value": roc_auc, "standard_deviation": "NaN"},
            "confusion_matrix": {
                "0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1])},
                "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1])},
            },
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": list(fpr),
                "true_positive_rates": list(tpr),
            },
            "precision_recall_curve": {
                "precisions": list(precs),
                "recalls": list(recs),
            },
        },
    }
    report_dict["register_metric"] = report_dict["binary_classification_metrics"][args.register_metric]

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
