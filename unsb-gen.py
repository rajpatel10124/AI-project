import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

def cross_verify():
    print("🧪 Step 2: Testing on UNSW-NB15...")
    
    model = joblib.load('general_model.pkl')
    scaler = joblib.load('general_scaler.pkl')

    df_new = pd.read_csv('unsb-nb15/UNSW_NB15_testing-set.csv')
    
    # Correct mapping for UNSW-NB15 testing-set.csv
    mapping = {
        'dur': 'Flow Duration',
        'spkts': 'Total Fwd Packets',
        'dpkts': 'Total Backward Packets',
        'sload': 'Flow Packets/s',
        'dload': 'Flow Bytes/s'
    }
    
    GOLDEN_FEATURES = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Flow Packets/s', 'Flow Bytes/s']
    
    # Rename what we can
    df_mapped = df_new.rename(columns=mapping)
    
    # Ensure all 5 columns exist, if not, fill with 0
    for col in GOLDEN_FEATURES:
        if col not in df_mapped.columns:
            df_mapped[col] = 0
            
    X_test = df_mapped[GOLDEN_FEATURES]
    X_test.replace([np.inf, -np.inf], 0, inplace=True)
    X_test.fillna(0, inplace=True)
    
    # Predict
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    y_true = df_new['label'].astype(int)

    print("\n" + "="*40)
    print(f"🚀 FINAL ACCURACY: {accuracy_score(y_true, y_pred)*100:.2f}%")
    print("="*40)
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    cross_verify()