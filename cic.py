import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
FOLDER_PATH = '/home/snr/D-Drive/sem-6/ai/project/MachineLearningCSV/MachineLearningCVE' 
MODEL_NAME = 'honest_security_model.pkl'
SCALER_NAME = 'honest_scaler.pkl'
FEATURES_NAME = 'honest_features.pkl'

# --- 1. DATA AGGREGATION & CLEANING ---
def build_honest_dataset(path):
    if not os.path.exists(path):
        print(f" ERROR: Folder '{path}' not found!")
        exit()

    all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    master_chunks = []
    
    # DROP CHEAT COLUMNS: Forcing AI to learn behavior, not just port numbers
    cols_to_drop = ['Destination Port', 'Protocol', 'Timestamp', 'External IP']

    print(f" Found {len(all_files)} files. Starting Honest Aggregation...")
    
    for file in all_files:
        file_path = os.path.join(path, file)
        for chunk in pd.read_csv(file_path, chunksize=100000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.dropna(inplace=True)
            
            # Remove "Cheat" columns
            existing_drops = [c for c in cols_to_drop if c in chunk.columns]
            chunk.drop(columns=existing_drops, inplace=True)
            
            # BALANCED SAMPLING: 100% Attacks, 1% Benign
            attacks = chunk[chunk['Label'] != 'BENIGN']
            benign = chunk[chunk['Label'] == 'BENIGN'].sample(frac=0.01, random_state=42)
            
            master_chunks.append(pd.concat([attacks, benign]))
        print(f" Processed: {file}")

    return pd.concat(master_chunks)

# --- 2. TRAINING & EVALUATION ---
def run_full_pipeline():
    # Load and Prepare
    df = build_honest_dataset(FOLDER_PATH)
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Honest Random Forest
    print(f" Training on {len(X_train)} rows with 8GB RAM Optimization...")
    model = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\n✨ Honest Accuracy: {model.score(X_test, y_test)*100:.2f}%")
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))
    
    # --- 3. GENERATE VISUAL ASSETS ---
    print("📊 Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Security Model Performance (Unbiased)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    print(" Saved 'confusion_matrix.png'")

    # --- 4. SAVE MODEL FOR REAL-TIME USE ---
    joblib.dump(model, MODEL_NAME)
    joblib.dump(scaler, SCALER_NAME)
    joblib.dump(list(X.columns), FEATURES_NAME)
    print(f" All model files saved successfully!")

if __name__ == "__main__":
    run_full_pipeline()