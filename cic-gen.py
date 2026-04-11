import pandas as pd
import numpy as np
import os, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

FOLDER_PATH = '/home/snr/D-Drive/sem-6/ai/project/MachineLearningCSV/MachineLearningCVE'

# These 5 features are GUARANTEED to exist in both datasets
GOLDEN_FEATURES = [
    'Flow Duration', 
    'Total Fwd Packets', 
    'Total Backward Packets',
    'Flow Packets/s', 
    'Flow Bytes/s'
]

def train_generalist():
    print("🏗️ Step 1: Training on Golden Features...")
    all_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.csv')]
    master_chunks = []
    
    for file in all_files:
        df_temp = pd.read_csv(os.path.join(FOLDER_PATH, file))
        df_temp.columns = df_temp.columns.str.strip()
        
        # Select only the 5 features + Label
        df_temp = df_temp[GOLDEN_FEATURES + ['Label']]
        df_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_temp.dropna(inplace=True)
        
        attacks = df_temp[df_temp['Label'] != 'BENIGN']
        benign = df_temp[df_temp['Label'] == 'BENIGN'].sample(frac=0.01)
        master_chunks.append(pd.concat([attacks, benign]))
            
    final_df = pd.concat(master_chunks)
    X = final_df[GOLDEN_FEATURES]
    y = final_df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=GOLDEN_FEATURES)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
    model.fit(X_scaled_df, y)

    joblib.dump(model, 'general_model.pkl')
    joblib.dump(scaler, 'general_scaler.pkl')
    print("✅ Training Complete.")

if __name__ == "__main__":
    train_generalist()