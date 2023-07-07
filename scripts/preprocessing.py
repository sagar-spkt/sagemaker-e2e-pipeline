import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def get_breast_cancer_dataset(test_size, target, random_state, stratify):
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target,
        test_size=test_size,
        stratify=data.target if stratify else None,
        random_state=random_state,
    )

    train_df = pd.DataFrame(X_train, columns=data.feature_names)
    train_df[target] = y_train

    test_df = pd.DataFrame(X_test, columns=data.feature_names)
    test_df[target] = y_test

    return train_df, test_df


def get_banknote_authentication_dataset(test_size, target, random_state, stratify):
    # http://archive.ics.uci.edu/dataset/267/banknote+authentication
    df = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
        header=None,
        names=["var", "skew", "kurt", "entropy", "class"],
    )
    df = df.rename(columns={"class": target})

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target] if stratify else None,
        random_state=random_state,
    )

    return train_df, test_df


if __name__ == "__main__":
    # Can't use argparse action
    argparse_bool_type = lambda arg: False if not arg or arg.lower() == "false" else True

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--stratify", type=argparse_bool_type, default=True)
    parser.add_argument("--target")
    parser.add_argument("--file-name", default="dataset.csv")

    args, _ = parser.parse_known_args()

    if args.dataset == "breast-cancer":
        train_df, test_df = get_breast_cancer_dataset(
            args.test_size, args.target, args.random_state, args.stratify,
        )
    elif args.dataset == "banknote-authentication":
        train_df, test_df = get_banknote_authentication_dataset(
            args.test_size, args.target, args.random_state, args.stratify,
        )
    else:
        raise ValueError("Invalid dataset chosen.")

    os.makedirs("/opt/ml/processing/train/", exist_ok=True)
    os.makedirs("/opt/ml/processing/test/", exist_ok=True)
    train_df.to_csv(f"/opt/ml/processing/train/{args.file_name}", index=False)
    test_df.to_csv(f"/opt/ml/processing/test/{args.file_name}", index=False)
