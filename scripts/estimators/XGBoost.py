import joblib
import os

import argparse
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    # Can't use argparse action
    argparse_bool_type = lambda arg: False if not arg or arg.lower() == "false" else True
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--booster", type=str, default="gbtree")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=1)
    parser.add_argument("--subsample", type=float, default=1)
    parser.add_argument("--colsample-bytree", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--reg-alpha", type=float, default=0)
    parser.add_argument("--reg-lambda", type=float, default=0)
    
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
    model = XGBClassifier(
        booster=args.booster,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        gamma=args.gamma,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
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
