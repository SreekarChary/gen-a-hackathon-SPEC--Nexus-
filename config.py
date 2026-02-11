"""
ClaimWatch AI — Global Configuration
"""
import os

# ─── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─── Multi-Insurance Config ──────────────────────────────
CATEGORIES = {
    "vehicle": {
        "target": "fraud_reported",
        "raw_file": "insurance_claims.csv",
        "junk_cols": ['_c39', 'policy_number', 'insured_zip', 'incident_location'],
        "threshold": 0.2
    },
    "health": {
        "target": "Is_Fraudulent",
        "raw_file": "synthetic_health_claims.csv",
        "junk_cols": ['Patient_ID', 'Policy_Number', 'Claim_ID', 'Hospital_ID', 'Diagnosis_Code', 'Procedure_Code'],
        "threshold": 0.2
    }
}

DEFAULT_CATEGORY = "vehicle"

def get_raw_path(category):
    return os.path.join(DATA_RAW_DIR, category, CATEGORIES[category]["raw_file"])

def get_processed_dir(category):
    return os.path.join(DATA_PROCESSED_DIR, category)

def get_model_dir(category):
    return os.path.join(MODELS_DIR, category)

# ─── Model Hyperparameters ──────────────────────────────
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "class_weight": "balanced",
}

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "scale_pos_weight": 5.0,
}

# ─── Train / Test Split ────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─── SHAP / Explainability ─────────────────────────────
TOP_N_FEATURES = 5
