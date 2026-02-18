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
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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


# ─── DATA LOADING ───────────────────────────────────────────────────────────
def load_data(path):
    """Load augmented CSV and return feature-ready DataFrame."""
    print("=" * 70)
    print("1. LOADING DATA")
    print("=" * 70)

    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
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
        print(f"\n  Null values detected:")
        for col, cnt in null_counts[null_counts > 0].items():
            print(f"    {col}: {cnt}")
        df = df.dropna().copy()
        print(f"  Dropped {null_before - len(df)} rows with NaN/null values")
    else:
        print("  No null values found")

    # Keep only rows with valid target classes
    df = df[df["Coverage Status"].isin(VALID_CLASSES)].copy()

    print(f"  Records: {len(df)}")
    print(f"\n  Class distribution:")
    for cls, cnt in df["Coverage Status"].value_counts().sort_index().items():
        print(f"    {cls:50s} {cnt:>5}  ({cnt/len(df)*100:.1f}%)")

    return df


# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(df):
    """Create TF-IDF text + OneHot categorical features and LabelEncoder."""
    print("\n" + "=" * 70)
    print("2. FEATURE ENGINEERING")
    print("=" * 70)

    # --- Text features ---
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
    print(f"  TF-IDF features : {X_text.shape[1]}")

    # --- Categorical features ---
    df["ACRONYM_clean"] = df["Acronym"].fillna("").str.strip().str.upper()
    cat_cols = ["PAYER NAME", "STATE NAME", "ACRONYM_clean"]
    for c in cat_cols:
        df[c] = df[c].fillna("UNKNOWN")

    onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = onehot.fit_transform(df[cat_cols])
    print(f"  OneHot features : {X_cat.shape[1]}")

    X = np.hstack([X_text, X_cat])
    print(f"  Combined features: {X.shape[1]}")

    # --- Target ---
    le = LabelEncoder()
    y = le.fit_transform(df["Coverage Status"])
    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")

    return X, y, tfidf, onehot, le


# ─── TRAINING ───────────────────────────────────────────────────────────────
def train_models(X_train, y_train, X_test, y_test, le):
    """Train multiple models with SMOTE + CV; return results dict."""
    print("\n" + "=" * 70)
    print("3. MODEL TRAINING  (SMOTE + 5-fold Stratified CV)")
    print("=" * 70)

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
            C=0.5, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, clf in models.items():
        print(f"\n  ── {name} ──")
        t0 = time.time()

        # SMOTE pipeline (applied inside each CV fold → no data leakage)
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", clf),
        ])

        # Cross-validation on training set
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        print(f"    CV Accuracy : {cv_mean:.4f} (+/- {cv_std:.4f})")

        # Fit on full training set (with SMOTE)
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        clf.fit(X_res, y_res)

        # Evaluate on test set
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc  = accuracy_score(y_test, clf.predict(X_test))
        elapsed   = time.time() - t0

        print(f"    Train Acc   : {train_acc:.4f}")
        print(f"    Test  Acc   : {test_acc:.4f}")
        print(f"    Time        : {elapsed:.1f}s")

        results[name] = {
            "model": clf,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "time": elapsed,
        }

    return results


# ─── SELECTION & REPORTING ──────────────────────────────────────────────────
def select_and_save(results, X_test, y_test, le, tfidf, onehot):
    """Pick the best model, print detailed report, save artefacts."""
    print("\n" + "=" * 70)
    print("4. MODEL COMPARISON")
    print("=" * 70)
    header = f"  {'Model':<25s} {'CV Acc':>10s} {'Test Acc':>10s} {'Time':>8s}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for name, r in results.items():
        print(f"  {name:<25s} {r['cv_mean']:.4f}+/-{r['cv_std']:.4f}  {r['test_acc']:.4f}    {r['time']:>6.1f}s")

    best_name = max(results, key=lambda k: results[k]["test_acc"])
    best = results[best_name]
    model = best["model"]
    print(f"\n  ★ Best model: {best_name}  (Test Acc = {best['test_acc']:.4f})")

    # ── Detailed report on test set ──
    y_pred = model.predict(X_test)
    class_names = list(le.classes_)

    print("\n" + "=" * 70)
    print("5. CLASSIFICATION REPORT")
    print("=" * 70)
    report_text = classification_report(y_test, y_pred, target_names=class_names)
    print(report_text)

    print("=" * 70)
    print("6. CONFUSION MATRIX")
    print("=" * 70)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df.to_string())

    # ── Fit-status analysis ──
    train_test_gap = best["train_acc"] - best["test_acc"]
    train_cv_gap   = best["train_acc"] - best["cv_mean"]
    if best["train_acc"] < 0.85:
        fit_status = "UNDERFITTED"
    elif train_test_gap > 0.05:
        fit_status = "OVERFITTED"
    else:
        fit_status = "WELL-FITTED"

    print(f"\n  Fit status: {fit_status}")
    print(f"    Train-Test gap : {train_test_gap:.4f}")
    print(f"    Train-CV gap   : {train_cv_gap:.4f}")

    # ── Feature importance (if available) ──
    top_features = []
    if hasattr(model, "feature_importances_"):
        feat_names = list(tfidf.get_feature_names_out()) + list(onehot.get_feature_names_out())
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:20]
        print("\n" + "=" * 70)
        print("7. TOP 20 FEATURES")
        print("=" * 70)
        for i in idx:
            fname = feat_names[i] if i < len(feat_names) else f"feature_{i}"
            print(f"    {fname:<55s} {importances[i]:.6f}")
            top_features.append((fname, float(importances[i])))

    # ── Save model ──
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
    print(f"\n  ✓ Model saved to {model_path}")

    # ── Save report ──
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

    print(f"  ✓ Report saved to {report_path}")
    return artefact


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    print("\n" + "★" * 70)
    print("  COVERAGE CLASSIFICATION MODEL – 5-CLASS TRAINING")
    print("★" * 70 + "\n")

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Build features
    X, y, tfidf, onehot, le = build_features(df)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 4. Train models
    results = train_models(X_train, y_train, X_test, y_test, le)

    # 5. Select best, save model & report
    artefact = select_and_save(results, X_test, y_test, le, tfidf, onehot)

    print("\n" + "★" * 70)
    print("  TRAINING COMPLETE")
    print("★" * 70 + "\n")
    return artefact


if __name__ == "__main__":
    main()
