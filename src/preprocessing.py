import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Add root dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(category: str, include_augmented=False) -> pd.DataFrame:
    """Load raw dataset and optionally the augmented synthetic data."""
    path = config.get_raw_path(category)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found for {category}: {path}")
    
    df = pd.read_csv(path)
    
    if include_augmented:
        aug_path = os.path.join(config.get_processed_dir(category), "augmented_data.csv")
        if os.path.exists(aug_path):
            df_aug = pd.read_csv(aug_path)
            # Align augmented data with raw data columns
            df_aug = df_aug[df.columns]
            df = pd.concat([df, df_aug], ignore_index=True)
            print(f"[+] Combined with augmented data: {len(df_aug)} rows added")
    
    print(f"[✓] Data Loaded: {len(df)} total rows")
    return df


def clean_data(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Drop junk, handle missing values, and fix data types per category."""
    initial_rows = len(df)
    cat_cfg = config.CATEGORIES[category]

    # Drop category-specific junk columns
    junk_cols = cat_cfg["junk_cols"]
    df = df.drop(columns=[c for c in junk_cols if c in df.columns])

    # Replace '?' or 'None' with NaN
    df = df.replace(['?', 'None'], np.nan)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns.tolist()
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Drop duplicates
    df = df.drop_duplicates()

    # Fill NaNs with mode (simple)
    if not df.empty:
        df = df.fillna(df.mode().iloc[0])

    removed = initial_rows - len(df)
    print(f"[✓] Cleaned {category} data: removed {removed} rows")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Create derived features specific to the insurance category."""
    if category == "vehicle":
        # Existing vehicle logic
        df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
        df['incident_date'] = pd.to_datetime(df['incident_date'])
        df['days_to_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days
        df['incident_hour_of_the_day'] = df['incident_hour_of_the_day'].astype(int)
        df['injury_claim_ratio'] = df['injury_claim'] / (df['total_claim_amount'] + 1)
        df['property_claim_ratio'] = df['property_claim'] / (df['total_claim_amount'] + 1)
        df = df.drop(columns=['policy_bind_date', 'incident_date'])
        
    elif category == "health":
        # Health insurance logic
        df['Claim_Date'] = pd.to_datetime(df['Claim_Date'])
        df['Service_Date'] = pd.to_datetime(df['Service_Date'])
        df['Policy_Expiration_Date'] = pd.to_datetime(df['Policy_Expiration_Date'])
        
        # Tenure: days between start of service and policy expiry
        df['days_to_expiry'] = (df['Policy_Expiration_Date'] - df['Service_Date']).dt.days
        # Claim lag: days between service and claim filing
        df['claim_lag'] = (df['Claim_Date'] - df['Service_Date']).dt.days
        
        # Casting
        df['Claim_Submitted_Late'] = df['Claim_Submitted_Late'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)
        
        # Drop raw dates
        df = df.drop(columns=['Claim_Date', 'Service_Date', 'Policy_Expiration_Date'])

    print(f"[✓] Feature engineering complete for {category}")
    return df


def encode_and_scale(df: pd.DataFrame, category: str):
    """Encodes categorical data and scales numbers."""
    cat_cfg = config.CATEGORIES[category]
    target_col = cat_cfg["target"]

    # Separate target
    y = df[target_col].apply(lambda x: 1 if str(x).upper() in ['Y', 'TRUE', '1'] else 0).values
    X = df.drop(columns=[target_col])

    # Identify types
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Encoders
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Scaler
    scaler = StandardScaler()
    if num_cols:
        X[num_cols] = scaler.fit_transform(X[num_cols])

    print(f"[✓] Encoding & scaling done for {category}  —  {len(cat_cols)} categorical, {len(num_cols)} numeric features")
    return X, y, label_encoders, scaler, X.columns.tolist(), num_cols


def split_data(X, y):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"[✓] Split: {len(X_train)} train / {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test


# ─── CLI entry point ───────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess insurance data.")
    parser.add_argument("--category", type=str, default=config.DEFAULT_CATEGORY, help="vehicle or health")
    args = parser.parse_args()

    cat = args.category
    
    # In CLI mode, we usually want to include augmented data if it exists for full retraining
    df = load_data(cat, include_augmented=True)
    df = clean_data(df, cat)
    df = engineer_features(df, cat)
    X, y, encoders, scaler, features, num_cols = encode_and_scale(df, cat)
    
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save artifacts in category-specific folder
    processed_dir = config.get_processed_dir(cat)
    model_dir = config.get_model_dir(cat)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    data_bundle = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": features,
        "num_cols": num_cols
    }
    
    joblib.dump(data_bundle, os.path.join(processed_dir, "processed_data.pkl"))
    joblib.dump(encoders, os.path.join(model_dir, "label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    print(f"\n✅ Preprocessing complete for {cat}!")
    print(f"   Artifacts saved to {processed_dir} and {model_dir}")
