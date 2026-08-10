"""
train_models.py

Trains 5 classification models on the Telco Customer Churn dataset and
evaluates each on the held-out test split:

    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbor Classifier
    4. Naive Bayes Classifier (Gaussian)
    5. Random Forest Classifier (Ensemble)

Each model is wrapped in a single sklearn Pipeline that includes all
preprocessing (imputation, scaling, one-hot encoding), so the saved .pkl
files can be applied directly to raw test_data.csv rows with no manual
feature engineering step required in the Streamlit app.

Run:
    python train_models.py

Outputs (written next to this script):
    logistic_regression.pkl
    decision_tree.pkl
    knn.pkl
    naive_bayes.pkl
    random_forest.pkl
    comparison_metrics.csv
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(HERE, "train_data.csv")
TEST_PATH = os.path.join(HERE, "..", "test_data.csv")
METRICS_OUT = os.path.join(HERE, "comparison_metrics.csv")

TARGET = "Churn"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical_features = [
        c for c in df.columns if c not in NUMERIC_FEATURES + [TARGET]
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, categorical_features),
    ])


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
    ),
}

FILE_NAMES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    preprocessor = build_preprocessor(train_df)

    results = []
    for name, clf in MODELS.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["ML Model Name"] = name
        results.append(metrics)

        out_path = os.path.join(HERE, FILE_NAMES[name])
        joblib.dump(pipe, out_path)
        print(f"[{name}] {metrics}  -> saved to {out_path}")

    comparison_df = pd.DataFrame(results)[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    comparison_df.to_csv(METRICS_OUT, index=False)
    print("\nComparison table:\n", comparison_df.to_string(index=False))
    print(f"\nSaved comparison table -> {METRICS_OUT}")


if __name__ == "__main__":
    main()
