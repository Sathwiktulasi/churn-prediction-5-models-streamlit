"""
Telco Customer Churn - Multi-Model Classification Demo
--------------------------------------------------------
Streamlit app for BITS Pilani M.Tech ML Assignment 2.

Features:
    a. CSV upload (upload test_data.csv or any file with the same schema)
    b. Model selection dropdown (5 trained classifiers)
    c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
    d. Confusion matrix + classification report

Run locally:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
TARGET = "Churn"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

st.set_page_config(
    page_title="Telco Churn Classifier Demo",
    layout="wide",
)


@st.cache_resource
def load_model(model_name: str):
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    return joblib.load(path)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("📊 Telco Customer Churn — Multi-Model Classifier Demo")
    st.caption(
        "BITS Pilani M.Tech (AIML) — Machine Learning Assignment 2. "
        "Upload the test dataset, pick a model, and view live evaluation results."
    )

    with st.sidebar:
        st.header("⚙️ Controls")

        model_name = st.selectbox(
            "Choose a classification model",
            options=list(MODEL_FILES.keys()),
        )

        uploaded_file = st.file_uploader(
            "Upload test CSV (must include the 'Churn' column)",
            type=["csv"],
        )

        st.markdown("---")
        st.markdown(
            "**Dataset:** Telco Customer Churn (IBM sample data, 7,043 "
            "customers, 19 features, binary target `Churn`)."
        )
        st.markdown(
            "Only the held-out **test_data.csv** should be uploaded here "
            "(Streamlit Community Cloud free tier has limited memory)."
        )

    if uploaded_file is None:
        st.info("👈 Upload `test_data.csv` from the repository root to see results.")
        return

    df = pd.read_csv(uploaded_file)

    if TARGET not in df.columns:
        st.error(f"Uploaded CSV must contain a '{TARGET}' column with true labels.")
        return

    X = df.drop(columns=[TARGET])
    y_true = df[TARGET]
    # Allow either 'Yes'/'No' strings or already-encoded 0/1 ints.
    if y_true.dtype == object:
        y_true = y_true.map({"Yes": 1, "No": 0})

    pipe = load_model(model_name)

    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]

    metrics = compute_metrics(y_true, y_pred, y_proba)

    st.subheader(f"Results — {model_name}")

    cols = st.columns(6)
    labels = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
    for c, label in zip(cols, labels):
        c.metric(label, f"{metrics[label]:.4f}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_right:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3))

    with st.expander("Show uploaded data / raw predictions"):
        preview = df.copy()
        preview["Predicted"] = np.where(y_pred == 1, "Yes", "No")
        preview["Churn Probability"] = y_proba.round(4)
        st.dataframe(preview.head(200))


if __name__ == "__main__":
    main()
