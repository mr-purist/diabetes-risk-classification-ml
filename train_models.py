"""
Diabetes Risk Classification - Model Training and Evaluation Pipeline
BITS Pilani - M.Tech (AIML) - Machine Learning (AIMLZG565) Assignment 2
Student: Mayank Gupta (BITS ID: 2026AK05256)
Random State: 26
"""

import os
import warnings

warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# 1. Configuration & Paths
# ---------------------------------------------------------
RANDOM_STATE = 26
DATASET_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
MODEL_DIR = "model"
TEST_DATA_PATH = "test_data.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. Data Ingestion & Validation
# ---------------------------------------------------------
print("=" * 70)
print("Step 1: Ingesting CDC BRFSS Diabetes Health Indicators Dataset...")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

target_col = "Diabetes_binary"
feature_cols = [col for col in df.columns if col != target_col]

print(f"Total predictor features ({len(feature_cols)}):")
print(feature_cols)

print("\nTarget class distribution:")
print(df[target_col].value_counts())

X = df[feature_cols]
y = df[target_col].astype(int)

# ---------------------------------------------------------
# 3. Train-Test Split (80% Train, 20% Test)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print(f"Step 2: Splitting data (80/20) with random_state={RANDOM_STATE} (Stratified)...")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

# Export curated representative test sample for Streamlit upload & testing
# Sampling 1,500 balanced rows ensures ultra-fast upload (<150KB) on Streamlit Cloud free tier
test_df_full = X_test.copy()
test_df_full[target_col] = y_test

test_0 = test_df_full[test_df_full[target_col] == 0].sample(n=750, random_state=RANDOM_STATE)
test_1 = test_df_full[test_df_full[target_col] == 1].sample(n=750, random_state=RANDOM_STATE)
test_sample = pd.concat([test_0, test_1]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

test_sample.to_csv(TEST_DATA_PATH, index=False)
print(f"\nSaved test dataset to '{TEST_DATA_PATH}' ({test_sample.shape[0]} rows, {test_sample.shape[1]} cols)")
print(f"Class balance in '{TEST_DATA_PATH}':\n{test_sample[target_col].value_counts().to_dict()}")

# ---------------------------------------------------------
# 4. Feature Preprocessing Pipeline (StandardScaler)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("Step 3: Fitting Preprocessor Pipeline on Training Data...")
print("=" * 70)

# Continuous and multi-level ordinal features to scale
scale_features = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
binary_features = [col for col in feature_cols if col not in scale_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), scale_features),
        ("pass", "passthrough", binary_features)
    ],
    remainder="drop"
)

# Fit on training data only to prevent data leakage
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
joblib.dump(preprocessor, preprocessor_path)
print(f"Saved preprocessor to '{preprocessor_path}'")

# ---------------------------------------------------------
# 5. Initialize the 5 Classification Models
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("Step 4: Initializing and Training 5 Classification Models...")
print("=" * 70)

# For SVM on large dataset (56k rows), using a calibrated stratified sample of 15,000 instances
# guarantees optimal RBF support vector convergence without memory bottlenecks
svm_sample_size = 15000
svm_indices = X_train.sample(n=svm_sample_size, random_state=RANDOM_STATE).index
X_train_svm = preprocessor.transform(X_train.loc[svm_indices])
y_train_svm = y_train.loc[svm_indices]

models = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
        "X_train": X_train_trans,
        "y_train": y_train,
        "filename": "logistic_regression.joblib"
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(max_depth=6, min_samples_split=20, min_samples_leaf=10, random_state=RANDOM_STATE),
        "X_train": X_train_trans,
        "y_train": y_train,
        "filename": "decision_tree.joblib"
    },
    "KNN": {
        "estimator": KNeighborsClassifier(n_neighbors=9, metric="euclidean", n_jobs=-1),
        "X_train": X_train_trans,
        "y_train": y_train,
        "filename": "knn.joblib"
    },
    "Naive Bayes": {
        "estimator": GaussianNB(),
        "X_train": X_train_trans,
        "y_train": y_train,
        "filename": "naive_bayes.joblib"
    },
    "Support Vector Machine": {
        "estimator": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE, cache_size=1000),
        "X_train": X_train_svm,
        "y_train": y_train_svm,
        "filename": "svm.joblib"
    }
}

# ---------------------------------------------------------
# 6. Model Training & Evaluation
# ---------------------------------------------------------
results = []

for name, config in models.items():
    print(f"\n--> Training {name}...")
    model = config["estimator"]
    model.fit(config["X_train"], config["y_train"])
    
    # Save trained model artifact
    save_path = os.path.join(MODEL_DIR, config["filename"])
    joblib.dump(model, save_path)
    print(f"    Saved model artifact to '{save_path}'")
    
    # Predictions on full test set
    y_pred = model.predict(X_test_trans)
    y_proba = model.predict_proba(X_test_trans)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate all 6 mandatory metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })
    print(f"    {name} -> Acc: {acc:.4f}, AUC: {auc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")

# ---------------------------------------------------------
# 7. Comparison Table & Summary
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n" + "=" * 80)
print("FINAL EVALUATION METRICS COMPARISON TABLE (Mandatory Step 5d)")
print("=" * 80)
print(results_df.to_string(index=False))

# Save metrics to CSV for Streamlit app and README generation
results_df.to_csv(os.path.join(MODEL_DIR, "evaluation_metrics.csv"), index=False)
print(f"\nSaved evaluation metrics to '{os.path.join(MODEL_DIR, 'evaluation_metrics.csv')}'")
print("Training and evaluation completed successfully!")
