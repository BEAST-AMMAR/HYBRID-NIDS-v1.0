import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.backend.detector import HybridDetector
from src.backend.sniffer import RealTimeFeatureExtractor

def main():
    print("==========================================================================")
    print("      HYBRID NIDS MODEL TRAINING & EVALUATION METRICS REPORT              ")
    print("==========================================================================\n")

    model_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    kdd_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data/KDDTest+.txt")
    mock_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data/mock_dataset.csv")
    data_path = kdd_path if os.path.exists(kdd_path) else mock_path

    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return

    print("[1] Loading Trained Models & Scalers...")
    detector = HybridDetector(model_dir)
    extractor = RealTimeFeatureExtractor(
        feature_columns_path=os.path.join(model_dir, "feature_columns.json"),
        scaler_path=os.path.join(model_dir, "standard_scaler.joblib"),
        models_dir=model_dir
    )
    
    print("\n--- MODEL ARCHITECTURE & TRAINING HYPERPARAMETERS ---")
    print(f"  * Deep Autoencoder Architecture: 116 -> 64 -> 32 -> 16 -> 4 -> 16 -> 32 -> 64 -> 116 Dense Layers")
    print(f"  * Autoencoder Training Epochs  : 60 Epochs (Batch Size: 64, Adam Optimizer, MSE Loss)")
    print(f"  * Isolation Forest Ensemble    : 250 Trees (Contamination: 0.10, Max Samples: 'auto')")
    print(f"  * Score Fusion Weight (Alpha)  : 0.75 * AE_norm + 0.25 * IF_norm")
    print(f"  * Decision Threshold (Theta)   : {detector.base_threshold:.4f}")

    print(f"\n[2] Evaluating Model Performance on Benchmark Dataset ({os.path.basename(data_path)})...")
    columns = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
        'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
        'num_shells','num_access_files','num_outbound_cmds','is_host_login',
        'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
        'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
        'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate',
        'dst_host_rerror_rate','dst_host_srv_rerror_rate',
        'label','difficulty'
    ]

    df_raw = pd.read_csv(data_path, header=None)
    if len(df_raw.columns) >= 42:
        df_raw.columns = columns[:len(df_raw.columns)]

    labels = df_raw['label'].astype(str).str.strip().str.lower()
    df_features = df_raw.drop(['label', 'difficulty'], axis=1, errors='ignore')

    df_encoded = pd.get_dummies(df_features, columns=['protocol_type', 'service', 'flag'])
    df_aligned = df_encoded.reindex(columns=extractor.feature_columns, fill_value=0)

    scaled_matrix = extractor.scaler.transform(df_aligned.values)

    # Batch prediction
    reconstruction = detector.autoencoder.predict(scaled_matrix, batch_size=2048, verbose=0)
    mses = np.mean(np.power(scaled_matrix - reconstruction, 2), axis=1)
    if_scores = -detector.iso_forest.decision_function(scaled_matrix)

    if detector.ae_scaler is not None:
        ae_norm = np.clip(detector.ae_scaler.transform(mses.reshape(-1, 1)).flatten(), 0.0, 1.0)
    else:
        ae_norm = np.clip(mses / max(float(mses.max()), 0.1), 0.0, 1.0)

    if detector.if_scaler is not None:
        if_norm = np.clip(detector.if_scaler.transform(if_scores.reshape(-1, 1)).flatten(), 0.0, 1.0)
    else:
        if_norm = np.clip((if_scores - if_scores.min()) / max(if_scores.max() - if_scores.min(), 1e-6), 0.0, 1.0)

    hybrid_scores = detector.alpha * ae_norm + (1 - detector.alpha) * if_norm
    y_pred = (hybrid_scores > detector.base_threshold).astype(int)

    y_true = np.where(labels.isin(['normal', 'normal.']), 0, 1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, hybrid_scores)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print("\n==========================================================================")
    print("                    MODEL EVALUATION METRICS REPORT                       ")
    print("==========================================================================")
    print(f"Total Dataset Evaluation Samples : {len(y_true)}")
    print(f"Normal Samples (Class 0)         : {sum(y_true == 0)}")
    print(f"Anomaly Samples (Class 1)        : {sum(y_true == 1)}")
    print("--------------------------------------------------------------------------")
    print(f"  ACCURACY    : {acc * 100:.2f}%")
    print(f"  AUC-ROC     : {auc * 100:.2f}%")
    print(f"  PRECISION   : {prec * 100:.2f}%")
    print(f"  RECALL      : {rec * 100:.2f}%")
    print(f"  F1-SCORE    : {f1 * 100:.2f}%")
    print("--------------------------------------------------------------------------")
    print("\n--- CONFUSION MATRIX ---")
    print(f"                Predicted Normal (0)   Predicted Anomaly (1)")
    print(f"Actual Normal  (0)     TN = {tn:<8}     FP = {fp:<8}")
    print(f"Actual Anomaly (1)     FN = {fn:<8}     TP = {tp:<8}")
    print("--------------------------------------------------------------------------")
    print(f"  * True Negatives (TN)  : {tn} (Normal packets correctly classified)")
    print(f"  * False Positives (FP) : {fp} (Normal packets falsely flagged as anomaly)")
    print(f"  * False Negatives (FN) : {fn} (Attacks missed by model)")
    print(f"  * True Positives (TP)  : {tp} (Attacks correctly detected)")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()
