import os
import pandas as pd
import numpy as np
import sys
import joblib

# Add root dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_augmented_data(category: str, num_samples=10000):
    """Generate synthetic samples that preserve logical correlations for fraud detection."""
    raw_path = config.get_raw_path(category)
    df_raw = pd.read_csv(raw_path)
    
    cat_cfg = config.CATEGORIES[category]
    target = cat_cfg["target"]
    
    # Separate existing fraud and legitimate cases
    if category == 'vehicle':
        df_fraud = df_raw[df_raw[target] == 'Y']
        df_legit = df_raw[df_raw[target] == 'N']
    else:
        df_fraud = df_raw[df_raw[target] == 1]
        df_legit = df_raw[df_raw[target] == 0]

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]
    
    augmented_samples = []
    
    print(f"[*] Generating {num_samples} logically-augmented samples for {category}...")
    
    for _ in range(num_samples):
        # 50/50 split of generated data
        is_fraud = np.random.random() < 0.5
        
        # Pick a base sample from the appropriate class to preserve correlations
        if is_fraud and not df_fraud.empty:
            base_sample = df_fraud.sample(1).iloc[0].to_dict()
        else:
            base_sample = df_legit.sample(1).iloc[0].to_dict()
            
        # 1. Noise Injection removed per user request for cleaner logic.
                
        # 2. Inject Suspicious Signal for Fraud Samples
        if is_fraud:
            if category == "vehicle":
                # High severity is a strong indicator
                if np.random.random() < 0.3:
                    base_sample['incident_severity'] = 'Major Damage'
                # Recent policy binding
                if np.random.random() < 0.2:
                    base_sample['policy_bind_date'] = '2014-12-01' # Incident is usually 2015 in this set
                # High witness/injury for some
                if np.random.random() < 0.2:
                    base_sample['witnesses'] = np.random.randint(2, 6)
            elif category == "health":
                # High stay days or procedures
                if np.random.random() < 0.3:
                    base_sample['Length_of_Stay_Days'] = np.random.randint(15, 60)
                if np.random.random() < 0.3:
                    base_sample['Number_of_Procedures'] = np.random.randint(10, 20)

        # Ensure label matches the logic
        if category == 'vehicle':
            base_sample[target] = 'Y' if is_fraud else 'N'
        else:
            base_sample[target] = 1 if is_fraud else 0
            
        augmented_samples.append(base_sample)
        
    df_augmented = pd.DataFrame(augmented_samples)
    
    save_dir = config.get_processed_dir(category)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "augmented_data.csv")
    df_augmented.to_csv(save_path, index=False)
    print(f"[✓] Logical augmented data saved to {save_path}")

if __name__ == "__main__":
    generate_augmented_data("vehicle")
    generate_augmented_data("health")
