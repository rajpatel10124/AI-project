import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def run_unsw_test():
    print("Starting UNSW-NB15 Cross-Verification Pipeline...")

    # 1. LOAD THE TRAINED BRAIN
    try:
        model = joblib.load('honest_security_model.pkl')
        scaler = joblib.load('honest_scaler.pkl')
        trained_features = joblib.load('honest_features.pkl')
        print("CIC-IDS-2017 Model Loaded.")
    except Exception as e:
        print(f"Error loading model files: {e}")
        return

    # 2. LOAD UNSW-NB15 DATA
    # Update this path to your UNSW CSV file
    DATA_PATH = '/home/snr/D-Drive/sem-6/ai/project/unsb-nb15/UNSW_NB15_testing-set.csv' 
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    df_new = pd.read_csv(DATA_PATH)
    df_new.columns = df_new.columns.str.strip()

    # 3. FEATURE MAPPING (Translating UNSW to CIC)
    # Mapping the most critical behavioral features
    mapping = {
        'dur': 'Flow Duration',
        'sbytes': 'Total Fwd Packets',
        'dbytes': 'Total Backward Packets',
        'sloss': 'Fwd Packets/s',
        'dloss': 'Bwd Packets/s',
        'sttl': 'Fwd Header Length',
        'dttl': 'Bwd Header Length'
    }
    df_new = df_new.rename(columns=mapping)

 # ... (Part 1: Loading model remains same) ...

    # 4. FIND AND CLEAN THE LABEL COLUMN (AGGRESSIVE FIX)
    label_options = ['label', 'attack_cat', 'Label']
    actual_label_col = next((c for c in df_new.columns if c in label_options), None)
    
    if actual_label_col is None:
        print(f"Could not find label column. Columns: {df_new.columns.tolist()}")
        return
    
    # NEW: Force conversion to string first, then map to 0/1 to avoid "Mix of Types"
    def clean_label(val):
        str_val = str(val).strip().lower()
        # If it's already '0' or 'benign' or 'normal', it's 0
        if str_val in ['0', '0.0', 'benign', 'normal']:
            return 0
        # Everything else is an attack (1)
        return 1

    y_true = df_new[actual_label_col].apply(clean_label).astype(int)
    print(f"Cleaned Labels. Unique values in y_true: {y_true.unique()}")

    # 5. ALIGN FEATURES (Performance Warning Fix)
    # Instead of inserting columns one by one, we create a dictionary and join
    missing_cols = {}
    for col in trained_features:
        if col not in df_new.columns:
            missing_cols[col] = 0
    
    if missing_cols:
        df_new = pd.concat([df_new, pd.DataFrame(missing_cols, index=df_new.index)], axis=1)
            
    # Ensure exact order as training
    X_new = df_new[trained_features].copy()
    
    # 6. SCALE & PREDICT
    X_new_scaled = scaler.transform(X_new)
    print("AI is analyzing the UNSW-NB15 traffic patterns...")
    y_pred = model.predict(X_new_scaled)

    # 7. RESULTS
    print("\n" + "="*40)
    print(f"CROSS-DATASET RESULTS (UNSW-NB15)")
    print("="*40)
    print(f"Generalization Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # 8. VISUALIZE
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Performance on UNSW-NB15 Dataset')
    plt.savefig('unsw_results.png')
    print("\n Confusion Matrix saved as 'unsw_results.png'")

if __name__ == "__main__":
    run_unsw_test()