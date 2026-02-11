import os
import sys
import joblib
import pandas as pd
import numpy as np
import shap

# Add root dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.explainer import generate_reasoning

INR_TO_USD = 1 / 83.0 # Conversion rate


def load_artifacts(category: str, model_name="best_model"):
    """Load model, encoders, and scaler for a specific category."""
    model_dir = config.get_model_dir(category)
    model = joblib.load(os.path.join(model_dir, f"{model_name}.pkl"))
    label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    
    try:
        data_bundle = joblib.load(os.path.join(config.get_processed_dir(category), "processed_data.pkl"))
        feature_names = data_bundle["feature_names"]
        num_cols = data_bundle.get("num_cols", [])
    except:
        feature_names = [] 
        num_cols = []

    return model, label_encoders, scaler, feature_names, num_cols


def preprocess_input(claim_dict: dict, category: str, label_encoders: dict, scaler, feature_names: list, num_cols: list):
    """Transform raw dictionary into model-ready dataframe for specific category."""
    df = pd.DataFrame([claim_dict])

    # Convert Monetary fields from INR to USD (Health only, Vehicle trained on INR)
    monetary_cols = {
        "health": ['Claim_Amount', 'Deductible_Amount', 'CoPay_Amount']
    }
    for col in monetary_cols.get(category, []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) * INR_TO_USD

    if category == "vehicle":
        # Vehicle Feature Engineering
        df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
        df['incident_date'] = pd.to_datetime(df['incident_date'])
        df['days_to_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days
        
        numeric_cols = [
            'total_claim_amount', 'injury_claim', 'property_claim', 
            'vehicle_claim', 'policy_annual_premium', 'age', 
            'incident_hour_of_the_day', 'number_of_vehicles_involved',
            'policy_deductable', 'umbrella_limit', 'bodily_injuries', 'witnesses'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['injury_claim_ratio'] = df['injury_claim'] / (df['total_claim_amount'] + 1)
        df['property_claim_ratio'] = df['property_claim'] / (df['total_claim_amount'] + 1)
        
    elif category == "health":
        # Health Feature Engineering
        df['Claim_Date'] = pd.to_datetime(df['Claim_Date'])
        df['Service_Date'] = pd.to_datetime(df['Service_Date'])
        df['Policy_Expiration_Date'] = pd.to_datetime(df['Policy_Expiration_Date'])
        
        df['days_to_expiry'] = (df['Policy_Expiration_Date'] - df['Service_Date']).dt.days
        df['claim_lag'] = (df['Claim_Date'] - df['Service_Date']).dt.days
        
        # Numeric casting
        health_num_cols = ['Claim_Amount', 'Patient_Age', 'Number_of_Procedures', 'Length_of_Stay_Days', 'Deductible_Amount', 'CoPay_Amount']
        for col in health_num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Boolean to int
        if 'Claim_Submitted_Late' in df.columns:
            df['Claim_Submitted_Late'] = df['Claim_Submitted_Late'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)

    # Encoding
    for col, le in label_encoders.items():
        if col in df.columns:
            known = set(str(c) for c in le.classes_)
            df[col] = df[col].apply(lambda x: str(x) if str(x) in known else str(le.classes_[0]))
            df[col] = le.transform(df[col])

    # Ensure correct column order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale numeric columns (strictly matching training columns)
    if num_cols:
        X_scaled = scaler.transform(df[num_cols])
        df[num_cols] = X_scaled

    return df


def get_shap_values(model, X):
    """Generate SHAP values for the prediction."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def top_features(shap_row, feature_names, n=config.TOP_N_FEATURES):
    """Extract top N contributing features from SHAP values."""
    feat_impact = []
    for i in range(len(feature_names)):
        feat_impact.append({
            "feature": feature_names[i],
            "shap_value": float(shap_row[i]),
            "direction": "increases fraud risk" if shap_row[i] > 0 else "decreases fraud risk"
        })
    
    # Sort by absolute impact
    feat_impact.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return feat_impact[:n]



def predict_claim(claim_dict: dict, category=config.DEFAULT_CATEGORY, model_name="best_model"):
    """Full prediction pipeline for a single claim."""
    model, label_encoders, scaler, feature_names, num_cols = load_artifacts(category, model_name)
    X = preprocess_input(claim_dict, category, label_encoders, scaler, feature_names, num_cols)

    # Probability of fraud (class 1)
    risk_score = float(model.predict_proba(X)[0][1])
    
    threshold = config.CATEGORIES[category]["threshold"]
    verdict = "Fraudulent" if risk_score >= threshold else "Legitimate"

    # Explainability
    _, shap_vals = get_shap_values(model, X)
    
    # Handle different SHAP output formats
    if isinstance(shap_vals, list): 
        shap_row = shap_vals[1][0]
    elif shap_vals.ndim == 3:
        shap_row = shap_vals[0, :, 1]
    else:
        shap_row = shap_vals[0]

    top_feats = top_features(shap_row, feature_names)
    reasoning = generate_reasoning(top_feats, risk_score, verdict)

    return {
        "verdict": verdict,
        "risk_score": round(risk_score, 4),
        "reasoning": reasoning,
        "top_features": top_feats,
    }
