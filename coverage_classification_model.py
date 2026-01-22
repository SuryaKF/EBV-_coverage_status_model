"""
Coverage Status Classification Model
=====================================
Predicts Coverage Status based on Payer Name, State, Acronym, Expansion, and Explanation.
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND EXPLORE DATA
# =============================================================================

def load_and_explore_data(filepath):
    """Load data and display basic statistics."""
    # Try multiple encodings for compatibility
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='cp1252')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin-1')
    
    # Standardize column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Clean Coverage Status (strip whitespace to consolidate duplicates)
    df['Coverage Status'] = df['Coverage Status'].str.strip()
    
    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nCoverage Status Distribution:")
    print(df['Coverage Status'].value_counts())
    print(f"\nUnique Payers: {df['PAYER NAME'].nunique()}")
    print(f"Unique States: {df['STATE NAME'].nunique()}")
    print(f"Unique Acronyms: {df['ACRONYM'].nunique()}")
    
    return df

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def prepare_features(df):
    """Prepare features for modeling."""
    
    # Create a copy
    data = df.copy()
    
    # Handle missing values (use uppercase column names as in CSV)
    data['EXPANSION'] = data['EXPANSION'].fillna('')
    data['Explanation'] = data['Explanation'].fillna('')
    
    # Combine text features for richer representation
    data['combined_text'] = (
        data['EXPANSION'].astype(str) + ' ' + 
        data['Explanation'].astype(str)
    )
    
    # Clean text
    data['combined_text'] = data['combined_text'].str.lower().str.strip()
    
    # Standardize ACRONYM (remove spaces, special chars)
    data['ACRONYM_clean'] = data['ACRONYM'].str.strip().str.upper()
    
    return data

# =============================================================================
# 3. BUILD CLASSIFICATION PIPELINE
# =============================================================================

def build_model_pipeline(classifier_name='random_forest'):
    """
    Build a complete preprocessing + classification pipeline with regularization.
    
    Parameters:
    -----------
    classifier_name : str
        One of: 'random_forest', 'gradient_boosting', 'svm', 'logistic_regression'
    """
    
    # Define classifiers with regularization to prevent overfitting
    classifiers = {
        'random_forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=8,  # Reduced from 10
            min_samples_split=5,  # Added regularization
            min_samples_leaf=2,   # Added regularization
            random_state=42,
            class_weight='balanced'
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=4,  # Reduced from 5
            min_samples_split=5,  # Added regularization
            min_samples_leaf=2,   # Added regularization
            learning_rate=0.1,    # Moderate learning rate
            subsample=0.8,        # Use 80% of samples per tree
            random_state=42
        ),
        'svm': SVC(
            kernel='rbf', 
            C=0.5,  # Reduced from 1.0 for more regularization
            gamma='scale',
            random_state=42,
            class_weight='balanced',
            probability=True  # Enable probability estimates
        ),
        'logistic_regression': LogisticRegression(
            max_iter=1000, 
            C=0.5,  # Reduced from default 1.0 for more regularization
            random_state=42,
            class_weight='balanced'
        )
    }
    
    return classifiers.get(classifier_name, classifiers['random_forest'])

# =============================================================================
# 4. TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate(df, test_size=0.2):
    """
    Train multiple models and compare performance.
    """
    
    # Prepare data
    data = prepare_features(df)
    
    # Features and target
    X_text = data['combined_text']
    X_categorical = data[['PAYER NAME', 'STATE NAME', 'ACRONYM_clean']]
    y = data['Coverage Status']
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("\n" + "=" * 60)
    print("TARGET CLASSES")
    print("=" * 60)
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {i}: {cls}")
    
    # TF-IDF for text
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words='english',
        min_df=2
    )
    X_text_tfidf = tfidf.fit_transform(X_text).toarray()
    
    # One-hot encode categorical features
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = onehot.fit_transform(X_categorical)
    
    # Combine features
    X_combined = np.hstack([X_text_tfidf, X_cat_encoded])
    
    print(f"\nFeature dimensions:")
    print(f"  TF-IDF features: {X_text_tfidf.shape[1]}")
    print(f"  Categorical features: {X_cat_encoded.shape[1]}")
    print(f"  Total features: {X_combined.shape[1]}")
    
    # Split data - handle case where some classes have very few samples
    # Check if stratification is possible
    min_class_count = pd.Series(y_encoded).value_counts().min()
    
    if min_class_count >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_encoded, 
            test_size=test_size, 
            random_state=42, 
            stratify=y_encoded
        )
    else:
        # Cannot stratify - use regular split
        print(f"\nWarning: Some classes have <2 samples. Using non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_encoded, 
            test_size=test_size, 
            random_state=42
        )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Show class distribution in training data
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION (Training Data)")
    print("=" * 60)
    
    print("\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u} ({label_encoder.classes_[u]}): {c}")
    
    print("\nNote: SMOTE will be applied INSIDE each CV fold to prevent data leakage.")
    print("This provides more realistic CV accuracy estimates.")
    
    # Build pipelines with SMOTE inside (applied per CV fold)
    # This prevents data leakage and gives more realistic accuracy estimates
    base_models = {
        'Random Forest': build_model_pipeline('random_forest'),
        'Gradient Boosting': build_model_pipeline('gradient_boosting'),
        'SVM': build_model_pipeline('svm'),
        'Logistic Regression': build_model_pipeline('logistic_regression')
    }
    
    # Create pipelines with SMOTE inside for proper CV
    models = {}
    for name, clf in base_models.items():
        models[name] = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)
        ])
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    total_training_start = time.time()
    
    for name, model in models.items():
        print(f"\n{name}:")
        print(f"  Training started...")
        
        model_start_time = time.time()
        
        # Cross-validation
        cv_start = time.time()
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        cv_time = time.time() - cv_start
        
        # Train on full training set
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        
        # Predict
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        model_total_time = time.time() - model_start_time
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'model': model,
            'cv_time': cv_time,
            'train_time': train_time,
            'total_time': model_total_time
        }
        
        print(f"  Cross-validation time: {cv_time:.2f}s")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Total time: {model_total_time:.2f}s")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = (name, model)
    
    total_training_time = time.time() - total_training_start
    print(f"\n" + "-" * 60)
    print(f"Total training time for all models: {total_training_time:.2f}s")
    
    # Detailed results for best model
    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_model[0]}")
    print("=" * 60)
    
    y_pred_best = best_model[1].predict(X_test)
    
    # Get unique labels in test set
    unique_test_labels = np.unique(np.concatenate([y_test, y_pred_best]))
    test_target_names = [label_encoder.classes_[i] for i in unique_test_labels]
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred_best, 
        labels=unique_test_labels,
        target_names=test_target_names
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best, labels=unique_test_labels)
    print(pd.DataFrame(
        cm, 
        index=test_target_names, 
        columns=test_target_names
    ))
    
    return {
        'results': results,
        'best_model': best_model,
        'label_encoder': label_encoder,
        'tfidf': tfidf,
        'onehot': onehot,
        'X_test': X_test,
        'y_test': y_test
    }

# =============================================================================
# 5. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance(model_output, df):
    """Analyze which features are most important for prediction."""
    
    best_model_name, best_model_pipeline = model_output['best_model']
    
    # Extract the classifier from the pipeline
    if hasattr(best_model_pipeline, 'named_steps'):
        best_model = best_model_pipeline.named_steps['classifier']
    else:
        best_model = best_model_pipeline
    
    if hasattr(best_model, 'feature_importances_'):
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        # Get feature names
        tfidf_features = model_output['tfidf'].get_feature_names_out()
        cat_features = model_output['onehot'].get_feature_names_out()
        all_features = list(tfidf_features) + list(cat_features)
        
        # Get importances
        importances = best_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': all_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Most Important Features:")
        print(importance_df.head(20).to_string(index=False))
        
        return importance_df
    else:
        print(f"\n{best_model_name} doesn't support feature importance directly.")
        return None

# =============================================================================
# 6. PREDICTION FUNCTION
# =============================================================================

def predict_coverage_status(model_output, payer_name, state_name, acronym, expansion, explanation):
    """
    Predict coverage status for new data.
    
    Parameters:
    -----------
    model_output : dict
        Output from train_and_evaluate()
    payer_name : str
        Name of the payer (e.g., 'Aetna', 'Cigna')
    state_name : str
        State name (e.g., 'Florida', 'Ohio')
    acronym : str
        Drug coverage acronym (e.g., 'PA', 'QL', 'ST')
    expansion : str
        Short description/expansion of acronym
    explanation : str
        Detailed explanation
    
    Returns:
    --------
    str : Predicted coverage status
    """
    
    # Prepare input
    combined_text = f"{expansion} {explanation}".lower().strip()
    acronym_clean = acronym.strip().upper()
    
    # Transform text
    text_features = model_output['tfidf'].transform([combined_text]).toarray()
    
    # Transform categorical
    cat_df = pd.DataFrame({
        'PAYER NAME': [payer_name],
        'STATE NAME': [state_name],
        'ACRONYM_clean': [acronym_clean]
    })
    cat_features = model_output['onehot'].transform(cat_df)
    
    # Combine
    X = np.hstack([text_features, cat_features])
    
    # Get the classifier from pipeline if needed
    model_or_pipeline = model_output['best_model'][1]
    if hasattr(model_or_pipeline, 'named_steps'):
        classifier = model_or_pipeline.named_steps['classifier']
    else:
        classifier = model_or_pipeline
    
    # Predict
    prediction_encoded = classifier.predict(X)
    prediction = model_output['label_encoder'].inverse_transform(prediction_encoded)
    
    return prediction[0]

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Load cleaned data (rows with null Coverage Status removed)
    DATA_PATH = r"c:\Users\VH0000812\Desktop\Coverage\data\Acronym_cleaned.csv"
    
    df = load_and_explore_data(DATA_PATH)
    
    # Train and evaluate models
    model_output = train_and_evaluate(df)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(model_output, df)
    
    # Example prediction
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    
    test_prediction = predict_coverage_status(
        model_output,
        payer_name="Aetna",
        state_name="Florida",
        acronym="PA",
        expansion="Prior authorization",
        explanation="You need to get approval from the plan before filling prescription."
    )
    print(f"\nPredicted Coverage Status: {test_prediction}")
    
    # Save the model to models directory
    import joblib
    
    # Create models directory if it doesn't exist
    models_dir = r"c:\Users\VH0000812\Desktop\Coverage\models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "coverage_model.pkl")
    
    # Get best model metrics
    best_model_name = model_output['best_model'][0]
    best_model_pipeline = model_output['best_model'][1]
    best_model_results = model_output['results'][best_model_name]
    
    # Extract the classifier from the pipeline for saving
    if hasattr(best_model_pipeline, 'named_steps'):
        classifier = best_model_pipeline.named_steps['classifier']
    else:
        classifier = best_model_pipeline
    
    joblib.dump({
        'model': classifier,
        'label_encoder': model_output['label_encoder'],
        'tfidf': model_output['tfidf'],
        'onehot': model_output['onehot'],
        'metrics': {
            'model_name': best_model_name,
            'test_accuracy': best_model_results['test_accuracy'],
            'cv_accuracy_mean': best_model_results['cv_mean'],
            'cv_accuracy_std': best_model_results['cv_std'],
            'training_time': best_model_results['train_time']
        }
    }, model_path)
    print(f"\nModel saved to '{model_path}'")
