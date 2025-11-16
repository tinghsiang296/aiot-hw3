import os
import json
from datetime import datetime
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def train_and_save(
    csv_path="data/sms_spam_no_header.csv",
    text_col=1,
    label_col=0,
    model_dir="models",
    test_size=0.2,
    random_state=42,
):
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(csv_path, header=None)
    texts = df.iloc[:, text_col].astype(str)
    labels = df.iloc[:, label_col].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(stop_words="english", max_features=5000),
            ),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Save model pipeline and label encoder
    model_path = os.path.join(model_dir, "model.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    joblib.dump(pipeline, model_path)
    joblib.dump(le, le_path)

    # Evaluate on test set
    probs = pipeline.predict_proba(X_test)
    if probs.shape[1] == 2:
        positive_probs = probs[:, 1]
    else:
        # fallback: take max col
        positive_probs = probs.max(axis=1)

    preds = (positive_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }

    cm = confusion_matrix(y_test, preds).tolist()

    # Save test split for the app to generate plots
    test_save = os.path.join(model_dir, "test_split.joblib")
    joblib.dump({"X_test": X_test.tolist(), "y_test": y_test.tolist(), "probs": positive_probs.tolist()}, test_save)

    # Save metrics and confusion matrix
    timestamp = datetime.utcnow().isoformat()
    metrics_payload = {
        "timestamp": timestamp,
        "metrics": metrics,
        "confusion_matrix": cm,
        "model_path": model_path,
        "label_encoder_path": le_path,
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh, indent=2)

    return {
        "model_path": model_path,
        "label_encoder_path": le_path,
        "metrics": metrics_payload,
        "test_split_path": test_save,
    }


if __name__ == "__main__":
    print("Training model...")
    out = train_and_save()
    print("Saved:", out)
