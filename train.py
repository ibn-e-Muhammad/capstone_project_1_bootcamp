# =============================================================================
# train.py — Machine Failure Prediction System
# Phase 1: Data Preprocessing, Model Training, and Asset Export
# =============================================================================
# Dataset: AI4I 2020 Predictive Maintenance Dataset
# Target: Machine failure (binary: 0 = No Failure, 1 = Failure)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# =============================================================================
# STEP 1: DATA INGESTION
# =============================================================================
print("=" * 60)
print("STEP 1: Loading dataset...")
print("=" * 60)

df = pd.read_csv("ai4i2020.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head(3))

# =============================================================================
# STEP 2: DATA CLEANING & FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Cleaning & Feature Engineering...")
print("=" * 60)

# --- Drop identifier columns ---
# 'UDI' is a row index and 'Product ID' is a product serial number.
# Neither contains any predictive signal; keeping them would cause the model
# to memorize IDs rather than learn meaningful patterns.
df.drop(columns=["UDI", "Product ID"], inplace=True)
print("Dropped: UDI, Product ID")

# --- Drop specific failure mode columns (ANTI-LEAKAGE RULE) ---
# TWF (Tool Wear Failure), HDF (Heat Dissipation Failure),
# PWF (Power Failure), OSF (Overstrain Failure), RNF (Random Failure)
# are direct sub-causes of 'Machine failure'. Including them as features
# would cause the model to cheat — it would perfectly predict failure
# because these columns ARE the failure. In production, these sub-causes
# are unknown at prediction time; only sensor readings are available.
failure_mode_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
df.drop(columns=failure_mode_cols, inplace=True)
print(f"Dropped failure mode columns (anti-leakage): {failure_mode_cols}")

# --- One-Hot Encode the 'Type' column ---
# 'Type' encodes machine quality variant: L (Low), M (Medium), H (High).
# Machine learning models require numeric input, so we convert this
# categorical variable into binary indicator columns.
# drop_first=True drops the 'L' (Low) category to avoid multicollinearity
# (the "dummy variable trap"). The dropped category becomes the baseline.
df = pd.get_dummies(df, columns=["Type"], drop_first=True)
print(f"One-hot encoded 'Type'. New columns: {[c for c in df.columns if 'Type' in c]}")

print(f"\nFinal columns after cleaning: {df.columns.tolist()}")
print(f"Target distribution:\n{df['Machine failure'].value_counts()}")

# =============================================================================
# STEP 3: TRAIN/TEST SPLIT & SELECTIVE SCALING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Train/Test Split & Scaling...")
print("=" * 60)

# Separate features (X) from the target label (y)
X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape:  {y.shape}")

# 80/20 split with a fixed random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# --- Selective Scaling ---
# StandardScaler normalizes features to zero mean and unit variance.
# This is critical for distance-based algorithms (e.g., KNN) and helps
# gradient-based solvers (Logistic Regression) converge faster.
#
# WHY selective? One-hot encoded columns (0/1) are already on a bounded
# scale. Scaling them would distort their meaning. We ONLY scale the
# continuous numerical sensor readings.
numerical_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

scaler = StandardScaler()

# --- CRITICAL: Fit ONLY on training data ---
# We fit the scaler on X_train so it learns the mean/std of the training set.
# Then we ONLY transform X_test using those same parameters.
# If we fit on the full dataset (including test), we introduce "data leakage"
# — the model indirectly sees test set statistics during training.
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"Scaled numerical columns: {numerical_cols}")
print("Scaler fitted on X_train only (no leakage into X_test).")

# Store column order for consistent inference in app.py
feature_columns = X_train.columns.tolist()
print(f"\nFinal feature columns (order matters for inference): {feature_columns}")

# =============================================================================
# STEP 4: MODEL TRAINING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Training Models...")
print("=" * 60)

# --- Logistic Regression ---
# A linear classifier that estimates failure probability via the logistic
# (sigmoid) function. Fast, interpretable, and a strong baseline.
# max_iter=1000 ensures convergence on this dataset without warnings.
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("LogisticRegression trained.")

# --- K-Nearest Neighbors ---
# A non-parametric, instance-based learner. It classifies a new point
# by majority vote of its k=5 nearest neighbours in feature space.
# Scaling is essential here because KNN relies on Euclidean distance.
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
print("KNeighborsClassifier (k=5) trained.")

# =============================================================================
# STEP 5: EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Model Evaluation on Test Set")
print("=" * 60)

lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
knn_acc = accuracy_score(y_test, knn_model.predict(X_test))

print(f"  Logistic Regression Accuracy : {lr_acc:.4f} ({lr_acc*100:.2f}%)")
print(f"  KNN (k=5) Accuracy           : {knn_acc:.4f} ({knn_acc*100:.2f}%)")

# =============================================================================
# STEP 6: ASSET EXPORT
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Exporting Best Model & Scaler...")
print("=" * 60)

# Select the winner based on test accuracy
if lr_acc >= knn_acc:
    best_model = lr_model
    winner_name = "LogisticRegression"
else:
    best_model = knn_model
    winner_name = "KNeighborsClassifier"

print(f"Winner: {winner_name} with accuracy {max(lr_acc, knn_acc)*100:.2f}%")

# Export the winning model
joblib.dump(best_model, "best_model.pkl")
print("Saved: best_model.pkl")

# Export the fitted scaler — must be the same scaler used during training
# so app.py can apply identical transformations at inference time
joblib.dump(scaler, "scaler.pkl")
print("Saved: scaler.pkl")

# Export the feature column list — ensures app.py builds the DataFrame
# in the exact same column order the model was trained on
joblib.dump(feature_columns, "feature_columns.pkl")
print("Saved: feature_columns.pkl")

print("\n" + "=" * 60)
print("Phase 1 Complete! Assets ready for Streamlit deployment.")
print("=" * 60)
