from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC

from coverage_pipeline.config import settings
from coverage_pipeline.db import get_conn
from coverage_pipeline.ml.artifact_store import save_artifact
from coverage_pipeline.ml.registry import get_snapshot, register_model
from coverage_pipeline.normalization import VALID_CANONICAL_CLASSES

TFIDF_MAX_FEAT = 500
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42


def load_snapshot_df(snapshot_table: str) -> pd.DataFrame:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT payer_name, state_name, acronym, expansion, explanation, coverage_status
                FROM {snapshot_table}
                """
            )
            rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"Snapshot table has no rows: {snapshot_table}")
    return pd.DataFrame(rows)


def build_features(df: pd.DataFrame):
    df = df.copy()
    df = df[df["coverage_status"].isin(VALID_CANONICAL_CLASSES)]

    df["expansion"] = df["expansion"].fillna("")
    df["explanation"] = df["explanation"].fillna("")
    df["combined_text"] = (df["expansion"].astype(str) + " " + df["explanation"].astype(str)).str.lower().str.strip()

    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEAT, ngram_range=(1, 2), stop_words="english")
    x_text = tfidf.fit_transform(df["combined_text"]).toarray()

    df["acronym_clean"] = df["acronym"].fillna("").str.strip().str.upper()
    cat = df[["payer_name", "state_name", "acronym_clean"]].fillna("UNKNOWN")
    onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_cat = onehot.fit_transform(cat)

    x = np.hstack([x_text, x_cat])
    le = LabelEncoder()
    y = le.fit_transform(df["coverage_status"])

    return x, y, tfidf, onehot, le


def train_best_model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(C=0.5, kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict] = {}

    for name, clf in models.items():
        started = time.time()
        pipe = ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", clf)])
        cv_scores = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="accuracy")

        smote = SMOTE(random_state=RANDOM_STATE)
        x_res, y_res = smote.fit_resample(x_train, y_train)
        clf.fit(x_res, y_res)

        results[name] = {
            "model": clf,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "train_acc": float(accuracy_score(y_train, clf.predict(x_train))),
            "test_acc": float(accuracy_score(y_test, clf.predict(x_test))),
            "training_time_sec": round(time.time() - started, 2),
        }

    best_name = max(results.keys(), key=lambda k: results[k]["test_acc"])
    return best_name, results[best_name], results


def run_training(snapshot_id: str | None = None) -> dict:
    snapshot = get_snapshot(snapshot_id)
    df = load_snapshot_df(snapshot["snapshot_table"])
    x, y, tfidf, onehot, le = build_features(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    best_name, best_result, all_results = train_best_model(x_train, y_train, x_test, y_test)
    model_version = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]

    artifact = {
        "model": best_result["model"],
        "tfidf": tfidf,
        "onehot": onehot,
        "label_encoder": le,
        "metrics": {
            "model_name": best_name,
            "test_accuracy": best_result["test_acc"],
            "train_accuracy": best_result["train_acc"],
            "cv_accuracy_mean": best_result["cv_mean"],
            "cv_accuracy_std": best_result["cv_std"],
            "training_time_sec": best_result["training_time_sec"],
            "num_classes": len(le.classes_),
            "classes": list(le.classes_),
            "all_model_results": {
                k: {
                    "cv_mean": v["cv_mean"],
                    "cv_std": v["cv_std"],
                    "train_acc": v["train_acc"],
                    "test_acc": v["test_acc"],
                    "training_time_sec": v["training_time_sec"],
                }
                for k, v in all_results.items()
            },
        },
    }

    os.makedirs("artifacts/tmp", exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir="artifacts/tmp") as tmp:
        local_tmp_path = tmp.name
    joblib.dump(artifact, local_tmp_path)

    artifact_uri = save_artifact(local_tmp_path, model_version, settings.artifact_root_uri)
    os.remove(local_tmp_path)

    metrics_json = json.dumps(artifact["metrics"])
    register_model(
        model_version=model_version,
        artifact_uri=artifact_uri,
        snapshot_id=snapshot["snapshot_id"],
        feature_version=snapshot["feature_version"],
        metrics_json=metrics_json,
        status="staging",
    )

    return {
        "model_version": model_version,
        "artifact_uri": artifact_uri,
        "snapshot_id": snapshot["snapshot_id"],
        "feature_version": snapshot["feature_version"],
        "best_model": best_name,
        "test_accuracy": best_result["test_acc"],
        "status": "staging",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model from Postgres snapshot and register staging model")
    parser.add_argument("--snapshot-id", default=None)
    args = parser.parse_args()

    result = run_training(snapshot_id=args.snapshot_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
