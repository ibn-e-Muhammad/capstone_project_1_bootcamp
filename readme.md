# KINETIC ENGINE — Development Report

## 1. Project Summary
**KINETIC ENGINE** is a machine failure prediction application with a cyber‑industrial Streamlit dashboard. It combines classical ML classification with a custom “Kinetic Engine” UI design system. The project is split into two phases:

- **Phase 1 (Training & Export):** Data preprocessing, model training, evaluation, and asset export.
- **Phase 2 (Deployment & Inference):** Streamlit app loads the trained model, applies identical preprocessing, and visualizes predictions with a high‑fidelity UI.

---

## 2. Dataset
**Source:** "[kaggle dataset link](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)"

**File:** `ai4i2020.csv`

**Target:** `Machine failure` (binary classification; 0 = No Failure, 1 = Failure).

**Key raw fields used as sensor inputs:**
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Machine quality type (`Type` → L/M/H)

**Fields removed for model integrity:**
- Identifier columns: `UDI`, `Product ID`
- Failure sub‑cause columns: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
  - These are direct causes of `Machine failure` and would leak the label if used as features.

---

## 3. Data Cleaning & Feature Engineering
The pipeline applies the following steps during training:

1. **Drop IDs** (`UDI`, `Product ID`) to avoid memorization and leakage.
2. **Drop failure sub‑causes** (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
3. **One‑hot encode `Type`** (machine quality variant).
   - `Type_L` is dropped (baseline) using `drop_first=True`.
4. **Selective scaling** with `StandardScaler`:
   - Only continuous sensor columns are scaled.
   - One‑hot features remain unscaled.

This produces the final, ordered feature list used consistently in both training and inference.

---

## 4. Train/Test Split
- **Split:** 80/20 train/test
- **Random seed:** 42
- **Leakage prevention:** scaler is fit **only** on training data, then applied to test data.

---

## 5. Models Trained
Two classical classifiers are trained and evaluated:

1. **Logistic Regression**
   - `max_iter=1000`
   - Fast, interpretable baseline classifier

2. **K‑Nearest Neighbors (KNN)**
   - `n_neighbors=5`
   - Distance‑based, non‑parametric
   - Requires feature scaling for stable Euclidean distances

---

## 6. Evaluation & Model Selection
**Metric used:** Accuracy on the held‑out test set.

- Both models are evaluated on the same test split.
- The model with higher accuracy is selected as the **production model**.

**Exported model:** `best_model.pkl` (Logistic Regression or KNN, depending on test accuracy).

---

## 7. Training Artifacts
The training phase exports three critical artifacts used in the app:

- `best_model.pkl` — the final trained classifier
- `scaler.pkl` — fitted `StandardScaler`
- `feature_columns.pkl` — exact training column order

These ensure inference uses **identical preprocessing** to training.

---

## 8. Inference Pipeline (Streamlit App)
The Streamlit app loads assets and performs inference as follows:

1. **Load assets** with joblib:
   - `best_model.pkl`
   - `scaler.pkl`
   - `feature_columns.pkl`

2. **Collect user inputs** from the sidebar:
   - 5 sensor values
   - Machine type (Low / Medium / High)

3. **Feature encoding:**
   - Converts `Type` into one‑hot fields
   - Rebuilds dataframe in the **exact** training column order

4. **Scaling:**
   - Applies the saved scaler to numerical columns

5. **Prediction:**
   - `predict_proba` used to compute failure probability
   - Health probability = 1 − failure probability

---

## 9. Results & Output
The dashboard surfaces two core outputs:

- **Failure Probability (%)**
- **Machine Health (%)**

The UI provides clear status signals:
- **Nominal** (safe zone)
- **Critical** if probability ≥ 85%

The app also shows:
- Live styled metrics
- Dynamic progress bars
- Status badges and alert cards
- Simulated sensor time‑series using Plotly

---

## 10. UI/UX & Design System
The interface adheres to the **Kinetic Engine Design System**, emphasizing:

- OLED‑dark industrial theme
- Neon cyan for normal, crimson for critical
- Space Grotesk + Inter typography pairing
- Glassmorphism + layered depth
- Ghost borders and minimal dividers

This system is specified in the design documentation and implemented via custom CSS and HTML injected into Streamlit.

---

## 11. Security & Access
The app uses a **session‑state login gate** for access to the dashboard:

- Operator ID: `STATION_NODE_492`
- Security Key: `kinetic2024`

This prevents dashboard rendering until valid credentials are entered.

---

## 12. Dependencies
Core runtime and ML stack:

- streamlit
- pandas
- numpy
- scikit‑learn
- joblib
- plotly

---

## 13. Project Assets
Key artifacts in the project root:

- `ai4i2020.csv` — training dataset
- `train.py` — training pipeline
- `app.py` — Streamlit app
- `best_model.pkl`, `scaler.pkl`, `feature_columns.pkl` — trained assets
- `DESIGN.md` — design system specification
- UI images: `screen.png`, `login_screen.png`

---

## 14. Limitations & Future Improvements
- Only accuracy is used for model selection; consider ROC‑AUC, F1, or recall for failure detection.
- No hyperparameter tuning is included; could improve KNN or add tree‑based models.
- Synthetic telemetry history is simulated; integration with real sensor streams would improve realism.
- Authentication is static (demo credentials); production should use secure auth.

---

## 15. Reproducibility Summary
To reproduce training and inference:

1. Ensure dataset is present as `ai4i2020.csv`.
2. Run `train.py` to produce the artifacts.
3. Run the app with Streamlit and authenticate.

This ensures the end‑to‑end pipeline—from dataset to dashboard—is fully repeatable.
