"""
Coverage Classification Model – Training Script
=================================================
Trains on the augmented 5-class dataset and saves the best model.

Target classes:
    1. Covered
    2. Not Covered
    3. Coverage with Conditions
    4. Coverage with Conditions(PA Required)
    5. Coverage with Conditions(ST Required)
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
import json
import logging
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI / headless
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import mlflow
import mlflow.sklearn

# ─── LOGGING CONFIGURATION ──────────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                                      # console output
        logging.FileHandler(os.path.join(LOG_DIR, "training.log"), mode="a"),  # file output
    ],
)
logger = logging.getLogger(__name__)

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATA_PATH       = r"data\Cleaned Output\Cleaned_January_Acronym_2026_augmented.csv"
MODEL_DIR       = "models"
REPORT_DIR      = "reports"
TFIDF_MAX_FEAT  = 500
TEST_SIZE       = 0.2
CV_FOLDS        = 5
RANDOM_STATE    = 42

VALID_CLASSES = [
    "Covered",
    "Not Covered",
    "Coverage with Conditions",
    "Coverage with Conditions(PA Required)",
    "Coverage with Conditions(ST Required)",
]

# ─── MLFLOW CONFIGURATION ───────────────────────────────────────────────────
MLFLOW_EXPERIMENT  = "Coverage_Classification"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # SQLite backend (recommended over filesystem)


# ─── DATA LOADING ───────────────────────────────────────────────────────────
def load_data(path):
    """Load augmented CSV and return feature-ready DataFrame."""
    logger.info("=" * 70)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 70)
    logger.debug(f"Attempting to load data from: {path}")

    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.debug(f"Successfully loaded with encoding: {enc}")
            break
        except UnicodeDecodeError:
            logger.debug(f"Failed to decode with encoding: {enc}")
            continue
    else:
        logger.error(f"Cannot decode file: {path}")
        raise RuntimeError(f"Cannot decode {path}")

    df.columns = df.columns.str.strip()

    # Standardise column names
    col_map = {
        "PAYER NAME": "PAYER NAME",
        "STATE NAME": "STATE NAME",
        "ACRONYM": "Acronym",
        "Acronym": "Acronym",
        "EXPANSION": "EXPANSION",
        "Explanation": "Explanation",
        "Coverage Status": "Coverage Status",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Drop rows with ANY NaN / null value in any column
    null_before = len(df)
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning("Null values detected in dataset:")
        for col, cnt in null_counts[null_counts > 0].items():
            logger.warning(f"  {col}: {cnt} null(s)")
        df = df.dropna().copy()
        logger.info(f"Dropped {null_before - len(df)} rows with NaN/null values")
    else:
        logger.info("No null values found in dataset")

    # Keep only rows with valid target classes
    df = df[df["Coverage Status"].isin(VALID_CLASSES)].copy()

    logger.info(f"Total records loaded: {len(df)}")
    logger.info("Class distribution:")
    for cls, cnt in df["Coverage Status"].value_counts().sort_index().items():
        logger.info(f"  {cls:50s} {cnt:>5}  ({cnt/len(df)*100:.1f}%)")

    return df


# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(df):
    """Create TF-IDF text + OneHot categorical features and LabelEncoder."""
    logger.info("=" * 70)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 70)

    # --- Text features ---
    logger.debug("Building TF-IDF text features...")
    df["EXPANSION"]   = df["EXPANSION"].fillna("")
    df["Explanation"]  = df["Explanation"].fillna("")
    df["combined_text"] = (
        df["EXPANSION"].astype(str) + " " + df["Explanation"].astype(str)
    ).str.lower().str.strip()

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEAT,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_text = tfidf.fit_transform(df["combined_text"]).toarray()
    logger.info(f"TF-IDF features: {X_text.shape[1]}")

    # --- Categorical features ---
    logger.debug("Building OneHot categorical features...")
    df["ACRONYM_clean"] = df["Acronym"].fillna("").str.strip().str.upper()
    cat_cols = ["PAYER NAME", "STATE NAME", "ACRONYM_clean"]
    for c in cat_cols:
        df[c] = df[c].fillna("UNKNOWN")

    onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = onehot.fit_transform(df[cat_cols])
    logger.info(f"OneHot features: {X_cat.shape[1]}")

    X = np.hstack([X_text, X_cat])
    logger.info(f"Combined feature matrix shape: {X.shape}")

    # --- Target ---
    le = LabelEncoder()
    y = le.fit_transform(df["Coverage Status"])
    logger.info(f"Target classes ({len(le.classes_)}): {list(le.classes_)}")

    return X, y, tfidf, onehot, le


# ─── TRAINING ───────────────────────────────────────────────────────────────
# ─── VISUALISATION HELPERS ───────────────────────────────────────────────────
def _plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Return a matplotlib Figure of the confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def _plot_model_comparison(results):
    """Return a bar-chart Figure comparing all models on CV / Test accuracy."""
    names = list(results.keys())
    cv_accs = [results[n]["cv_mean"] for n in names]
    test_accs = [results[n]["test_acc"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, cv_accs,  width, label="CV Accuracy", color="steelblue")
    ax.bar(x + width / 2, test_accs, width, label="Test Accuracy", color="coral")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison – CV vs Test Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    for i, (cv, te) in enumerate(zip(cv_accs, test_accs)):
        ax.text(i - width / 2, cv + 0.01, f"{cv:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, te + 0.01, f"{te:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    return fig


def _plot_feature_importances(importances, feat_names, top_n=20):
    """Return a horizontal bar-chart Figure of the top-N feature importances."""
    idx = np.argsort(importances)[::-1][:top_n]
    top_imp = importances[idx]
    top_names = [feat_names[i] if i < len(feat_names) else f"feature_{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_imp)), top_imp[::-1], color="teal")
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    return fig


def _plot_class_distribution(y, le, title="Class Distribution"):
    """Return a bar-chart Figure of the class distribution."""
    classes = le.classes_
    counts = np.bincount(y, minlength=len(classes))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(classes)), counts, color="mediumpurple")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(title)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(cnt), ha="center", fontsize=9)
    plt.tight_layout()
    return fig


# ─── TRAINING ───────────────────────────────────────────────────────────────
def train_models(X_train, y_train, X_test, y_test, le):
    """Train multiple models with SMOTE + CV; return results dict."""
    logger.info("=" * 70)
    logger.info("STEP 3: MODEL TRAINING (SMOTE + 5-fold Stratified CV)")
    logger.info("=" * 70)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=2, random_state=RANDOM_STATE,
        ),
        "SVM": SVC(C=0.5, kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            C=0.5, max_iter=1000, random_state=RANDOM_STATE
        ),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, clf in models.items():
        logger.info(f"Training model: {name}")
        t0 = time.time()

        # SMOTE pipeline (applied inside each CV fold → no data leakage)
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", clf),
        ])

        # Cross-validation on training set
        logger.debug(f"Running {CV_FOLDS}-fold cross-validation for {name}...")
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        logger.info(f"  [{name}] CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

        # Fit on full training set (with SMOTE)
        logger.debug(f"Applying SMOTE and fitting {name} on full training set...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        clf.fit(X_res, y_res)

        # Evaluate on test set
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc  = accuracy_score(y_test, clf.predict(X_test))
        elapsed   = time.time() - t0

        logger.info(f"  [{name}] Train Accuracy: {train_acc:.4f}")
        logger.info(f"  [{name}] Test Accuracy:  {test_acc:.4f}")
        logger.info(f"  [{name}] Training time:  {elapsed:.1f}s")

        results[name] = {
            "model": clf,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "time": elapsed,
        }

    logger.info(f"Completed training {len(models)} models")
    return results


# ─── SELECTION & REPORTING ──────────────────────────────────────────────────
def select_and_save(results, X_train, y_train, X_test, y_test, le, tfidf, onehot):
    """Pick the best model, print detailed report, log everything to MLflow, save artefacts."""
    logger.info("=" * 70)
    logger.info("STEP 4: MODEL COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Model':<25s} {'CV Acc':>15s} {'Test Acc':>12s} {'Time':>10s}")
    logger.info("-" * 65)
    for name, r in results.items():
        logger.info(f"{name:<25s} {r['cv_mean']:.4f}+/-{r['cv_std']:.4f}   {r['test_acc']:.4f}   {r['time']:>8.1f}s")

    best_name = max(results, key=lambda k: results[k]["test_acc"])
    best = results[best_name]
    model = best["model"]
    logger.info(f"BEST MODEL: {best_name} (Test Acc = {best['test_acc']:.4f})")

    # ── Detailed report on test set ──
    y_pred = model.predict(X_test)
    class_names = list(le.classes_)

    logger.info("=" * 70)
    logger.info("STEP 5: CLASSIFICATION REPORT")
    logger.info("=" * 70)
    report_text = classification_report(y_test, y_pred, target_names=class_names)
    for line in report_text.strip().split("\n"):
        logger.info(line)

    logger.info("=" * 70)
    logger.info("STEP 6: CONFUSION MATRIX")
    logger.info("=" * 70)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    for line in cm_df.to_string().split("\n"):
        logger.info(line)

    # ── Fit-status analysis ──
    train_test_gap = best["train_acc"] - best["test_acc"]
    train_cv_gap   = best["train_acc"] - best["cv_mean"]
    if best["train_acc"] < 0.85:
        fit_status = "UNDERFITTED"
        logger.warning(f"Model is UNDERFITTED (Train Acc < 0.85)")
    elif train_test_gap > 0.05:
        fit_status = "OVERFITTED"
        logger.warning(f"Model is OVERFITTED (Train-Test gap > 0.05)")
    else:
        fit_status = "WELL-FITTED"
        logger.info(f"Model is WELL-FITTED")

    logger.info(f"Fit status: {fit_status}")
    logger.info(f"  Train-Test gap: {train_test_gap:.4f}")
    logger.info(f"  Train-CV gap:   {train_cv_gap:.4f}")

    # ── Feature importance (if available) ──
    top_features = []
    if hasattr(model, "feature_importances_"):
        feat_names = list(tfidf.get_feature_names_out()) + list(onehot.get_feature_names_out())
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:20]
        logger.info("=" * 70)
        logger.info("STEP 7: TOP 20 FEATURES")
        logger.info("=" * 70)
        for i in idx:
            fname = feat_names[i] if i < len(feat_names) else f"feature_{i}"
            logger.info(f"  {fname:<55s} {importances[i]:.6f}")
            top_features.append((fname, float(importances[i])))

    # ── Save model (local) ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "coverage_model.pkl")
    artefact = {
        "model": model,
        "tfidf": tfidf,
        "onehot": onehot,
        "label_encoder": le,
        "metrics": {
            "model_name": best_name,
            "test_accuracy": best["test_acc"],
            "train_accuracy": best["train_acc"],
            "cv_accuracy_mean": best["cv_mean"],
            "cv_accuracy_std": best["cv_std"],
            "training_time": best["time"],
            "num_classes": len(class_names),
            "classes": class_names,
            "fit_status": fit_status,
        },
    }
    joblib.dump(artefact, model_path)
    logger.info(f"Model saved to {model_path}")

    # ── Save text report ──
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"training_report_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("COVERAGE CLASSIFICATION MODEL - TRAINING REPORT (5 CLASSES)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 70 + "\n1. DATA SUMMARY\n" + "-" * 70 + "\n")
        f.write(f"Total Records: {len(y_test) + len(y_test) * 4}\n")
        f.write(f"Features Used: {TFIDF_MAX_FEAT} TF-IDF + {onehot.get_feature_names_out().shape[0]} Categorical\n\n")
        f.write("Class Distribution (target):\n")
        for cls in class_names:
            f.write(f"  {cls}\n")

        f.write("\n" + "-" * 70 + "\n2. MODEL COMPARISON\n" + "-" * 70 + "\n")
        f.write(f"{'Model':<25s} {'CV Accuracy':>20s} {'Test Accuracy':>15s} {'Time':>10s}\n")
        f.write("-" * 70 + "\n")
        for name, r in results.items():
            f.write(f"{name:<25s} {r['cv_mean']:.4f} (+/-{r['cv_std']:.4f})   {r['test_acc']:.4f}      {r['time']:>8.1f}s\n")

        f.write("\n" + "-" * 70 + "\n3. BEST MODEL DETAILS\n" + "-" * 70 + "\n")
        f.write(f"Selected Model: {best_name}\n")
        f.write(f"Training Accuracy: {best['train_acc']:.4f}\n")
        f.write(f"CV Accuracy: {best['cv_mean']:.4f} (+/- {best['cv_std']:.4f})\n")
        f.write(f"Test Accuracy: {best['test_acc']:.4f}\n")

        f.write("\n" + "-" * 70 + "\n4. FIT STATUS\n" + "-" * 70 + "\n")
        f.write(f"Status: {fit_status}\n")
        f.write(f"Train-Test gap: {train_test_gap:.4f}\n")
        f.write(f"Train-CV gap:   {train_cv_gap:.4f}\n")

        if top_features:
            f.write("\n" + "-" * 70 + "\n5. TOP 20 FEATURES\n" + "-" * 70 + "\n")
            for fname, imp in top_features:
                f.write(f"  {fname:<55s} {imp:.6f}\n")

        f.write("\n" + "-" * 70 + "\n6. CLASSIFICATION REPORT\n" + "-" * 70 + "\n")
        f.write(report_text + "\n")

        f.write("-" * 70 + "\n7. CONFUSION MATRIX\n" + "-" * 70 + "\n")
        f.write(cm_df.to_string() + "\n")

        f.write("\n" + "=" * 70 + "\nEND OF REPORT\n" + "=" * 70 + "\n")

    logger.info(f"Training report saved to {report_path}")

    # ==================================================================
    #  MLFLOW TRACKING & MODEL REGISTRY
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STEP 8: MLFLOW EXPERIMENT TRACKING")
    logger.info("=" * 70)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # --- Child runs: one per candidate model ---
    parent_run_name = f"training_run_{ts}"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        # Log shared parameters on the parent run
        mlflow.log_params({
            "tfidf_max_features": TFIDF_MAX_FEAT,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "num_classes": len(class_names),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
        })
        mlflow.set_tag("best_model", best_name)
        mlflow.set_tag("fit_status", fit_status)

        # -- Log each candidate model as a nested (child) run --
        for name, r in results.items():
            with mlflow.start_run(run_name=name, nested=True) as child_run:
                mlflow.set_tag("model_type", type(r["model"]).__name__)
                mlflow.log_metrics({
                    "cv_accuracy_mean": r["cv_mean"],
                    "cv_accuracy_std": r["cv_std"],
                    "train_accuracy": r["train_acc"],
                    "test_accuracy": r["test_acc"],
                    "training_time_s": r["time"],
                })
                # Log the sklearn model
                mlflow.sklearn.log_model(r["model"], name=f"model_{name.replace(' ', '_')}")
                logger.debug(f"Child run logged: {name} (run_id={child_run.info.run_id[:8]}...)")

        # -- Log best-model metrics on parent --
        mlflow.log_metrics({
            "best_test_accuracy": best["test_acc"],
            "best_train_accuracy": best["train_acc"],
            "best_cv_accuracy_mean": best["cv_mean"],
            "train_test_gap": train_test_gap,
            "train_cv_gap": train_cv_gap,
        })

        # -- Log per-class precision / recall / f1 --
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        for cls_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                safe = cls_name.replace(" ", "_").replace("(", "").replace(")", "")
                for metric_key, metric_val in metrics.items():
                    mlflow.log_metric(f"{safe}_{metric_key}", metric_val)

        # -- Log visualisation artifacts --
        artifact_dir = os.path.join(REPORT_DIR, "mlflow_plots")
        os.makedirs(artifact_dir, exist_ok=True)

        # 1. Confusion matrix
        fig_cm = _plot_confusion_matrix(cm, class_names, title=f"Confusion Matrix – {best_name}")
        cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
        fig_cm.savefig(cm_path, dpi=150)
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # 2. Model comparison chart
        fig_cmp = _plot_model_comparison(results)
        cmp_path = os.path.join(artifact_dir, "model_comparison.png")
        fig_cmp.savefig(cmp_path, dpi=150)
        plt.close(fig_cmp)
        mlflow.log_artifact(cmp_path, artifact_path="plots")

        # 3. Class distribution
        fig_dist = _plot_class_distribution(y_test, le, title="Test-Set Class Distribution")
        dist_path = os.path.join(artifact_dir, "class_distribution.png")
        fig_dist.savefig(dist_path, dpi=150)
        plt.close(fig_dist)
        mlflow.log_artifact(dist_path, artifact_path="plots")

        # 4. Feature importances (if available)
        if hasattr(model, "feature_importances_"):
            feat_names_all = list(tfidf.get_feature_names_out()) + list(onehot.get_feature_names_out())
            fig_fi = _plot_feature_importances(model.feature_importances_, feat_names_all)
            fi_path = os.path.join(artifact_dir, "feature_importances.png")
            fig_fi.savefig(fi_path, dpi=150)
            plt.close(fig_fi)
            mlflow.log_artifact(fi_path, artifact_path="plots")

        # 5. Log text report & model pickle as artifacts
        mlflow.log_artifact(report_path, artifact_path="reports")
        mlflow.log_artifact(model_path, artifact_path="model")

        # -- Register the best model in the MLflow Model Registry --
        reg_model_name = "CoverageClassifier"
        model_uri = f"runs:/{parent_run.info.run_id}/best_model"
        mlflow.sklearn.log_model(
            model, name="best_model",
            registered_model_name=reg_model_name,
        )
        logger.info(f"Best model registered as '{reg_model_name}' in MLflow Model Registry")
        logger.info(f"Parent MLflow run ID: {parent_run.info.run_id}")
        logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Start UI with: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

    return artefact


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 70)
    logger.info("COVERAGE CLASSIFICATION MODEL - 5-CLASS TRAINING")
    logger.info("=" * 70)

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Build features
    X, y, tfidf, onehot, le = build_features(df)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train/Test split: {len(X_train)} train, {len(X_test)} test")

    # 4. Train models
    results = train_models(X_train, y_train, X_test, y_test, le)

    # 5. Select best, save model & report  (+ MLflow logging)
    artefact = select_and_save(results, X_train, y_train, X_test, y_test, le, tfidf, onehot)

    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    return artefact


if __name__ == "__main__":
    main()
