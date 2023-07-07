import joblib
import os

import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    # Can't use argparse action
    argparse_bool_type = lambda arg: False if not arg or arg.lower() == "false" else True
    none_or_str = lambda arg: None if arg.lower() == "none" or arg.lower() == "null" else str(arg)

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--penalty", type=none_or_str, choices=["l1", "l2", "elasticnet", None], default="l2")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", choices=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], default="lbfgs")
    parser.add_argument("--class-weight", type=none_or_str, choices=["balanced", None], default=None)
    parser.add_argument("--max-iter", type=int, default=100)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--train-file", type=str, default="dataset.csv")
    parser.add_argument("--features", default=None)
    parser.add_argument("--target")
    # cross-validation arguments
    parser.add_argument("--cross-validation", type=argparse_bool_type, default=False)
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    parser.add_argument("--cv-scorer", default="accuracy")
    parser.add_argument("--cv-folds", type=int, default=5)
    
    args, _ = parser.parse_known_args()
    
    print("reading data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    
    print("building training datasets")
    feature_columns = args.features.split() if args.features else [
        col for col in train_df.columns
        if col != args.target
    ]
    X_train = train_df[feature_columns]
    y_train = train_df[args.target]

    # train
    print("Creating model instance")
    model = LogisticRegression(
        penalty=args.penalty,
        C=args.C,
        solver=args.solver,
        class_weight=args.class_weight,
        max_iter=args.max_iter,
    )

    if args.cross_validation:
        print("Cross-Validating Model")
        cv_score = cross_val_score(
            model, X_train, y_train,
            cv=args.cv_folds, scoring=args.cv_scorer,
        )
        print(f"Cross-Validation Score: {cv_score.mean()}")
    else:
        print("Training Model")
        model.fit(X_train, y_train)

        path = os.path.join(args.model_dir, "model.joblib")
        joblib.dump(model, path)
        print("model saved at " + path)
