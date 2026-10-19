import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import json
import joblib
import tensorflow as tf

def main():
    print("==========================================================================", flush=True)
    print("      HYBRID NIDS MODEL TRAINING & EVALUATION METRICS REPORT              ", flush=True)
    print("==========================================================================\n", flush=True)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    kdd_path = os.path.join(base_dir, "data", "KDDTest+.txt")
    mock_path = os.path.join(base_dir, "data", "mock_dataset.csv")
    data_path = kdd_path if os.path.exists(kdd_path) else mock_path

    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}", flush=True)
        return

    print("[1] Loading Trained Models & Scalers...", flush=True)
    with open(os.path.join(model_dir, "feature_columns.json"), "r") as f:
        feature_columns = json.load(f)
    scaler = joblib.load(os.path.join(model_dir, "standard_scaler.joblib"))
    ae_scaler = joblib.load(os.path.join(model_dir, "ae_scaler.joblib")) if os.path.exists(os.path.join(model_dir, "ae_scaler.joblib")) else None
    if_scaler = joblib.load(os.path.join(model_dir, "if_scaler.joblib")) if os.path.exists(os.path.join(model_dir, "if_scaler.joblib")) else None
    autoencoder = tf.keras.models.load_model(os.path.join(model_dir, "autoencoder.keras"))
    iso_forest = joblib.load(os.path.join(model_dir, "iso_forest.joblib"))
    with open(os.path.join(model_dir, "threshold.txt"), "r") as f:
        base_threshold = float(f.read().strip())
    
    print("\n--- MODEL ARCHITECTURE & TRAINING HYPERPARAMETERS ---")
    alpha = 0.75
    print("\n--- MODEL ARCHITECTURE & TRAINING HYPERPARAMETERS ---", flush=True)
    print(f"  * Deep Autoencoder Architecture: {len(feature_columns)} -> 64 -> 32 -> 16 -> 4 -> 16 -> 32 -> 64 -> {len(feature_columns)} Dense Layers", flush=True)
    print(f"  * Autoencoder Training Epochs  : 60 Epochs (Batch Size: 64, Adam Optimizer, MSE Loss)", flush=True)
    print(f"  * Isolation Forest Ensemble    : 250 Trees (Contamination: 0.10, Max Samples: 'auto')", flush=True)
    print(f"  * Score Fusion Weight (Alpha)  : {alpha:.2f} * AE_norm + {1 - alpha:.2f} * IF_norm", flush=True)
    print(f"  * Decision Threshold (Theta)   : {base_threshold:.6f}", flush=True)

    print(f"\n[2] Evaluating Model Performance on Benchmark Dataset ({os.path.basename(data_path)})...", flush=True)
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
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    scaled_matrix = scaler.transform(df_aligned.values)

    # Batch prediction
    reconstruction = autoencoder.predict(scaled_matrix, batch_size=2048, verbose=0)
    mses = np.mean(np.power(scaled_matrix - reconstruction, 2), axis=1)
    if_scores = -iso_forest.decision_function(scaled_matrix)

    if ae_scaler is not None:
        ae_norm = np.clip(ae_scaler.transform(mses.reshape(-1, 1)).flatten(), 0.0, 1.0)
    else:
        ae_norm = np.clip(mses / max(float(mses.max()), 0.1), 0.0, 1.0)

    if if_scaler is not None:
        if_norm = np.clip(if_scaler.transform(if_scores.reshape(-1, 1)).flatten(), 0.0, 1.0)
    else:
        if_norm = np.clip((if_scores - if_scores.min()) / max(if_scores.max() - if_scores.min(), 1e-6), 0.0, 1.0)

    hybrid_scores = alpha * ae_norm + (1 - alpha) * if_norm
    y_pred = (hybrid_scores > base_threshold).astype(int)

    y_true = np.where(labels.isin(['normal', 'normal.']), 0, 1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, hybrid_scores)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Autoencoder alone
    ae_pred = (ae_norm > base_threshold).astype(int)
    ae_acc = accuracy_score(y_true, ae_pred)
    ae_prec = precision_score(y_true, ae_pred, zero_division=0)
    ae_rec = recall_score(y_true, ae_pred, zero_division=0)
    ae_f1 = f1_score(y_true, ae_pred, zero_division=0)
    ae_auc = roc_auc_score(y_true, ae_norm)

    # Isolation Forest alone
    if_pred = (if_norm > base_threshold).astype(int)
    if_acc = accuracy_score(y_true, if_pred)
    if_prec = precision_score(y_true, if_pred, zero_division=0)
    if_rec = recall_score(y_true, if_pred, zero_division=0)
    if_f1 = f1_score(y_true, if_pred, zero_division=0)
    if_auc = roc_auc_score(y_true, if_norm)

    print("\n==========================================================================", flush=True)
    print("                    MODEL EVALUATION METRICS REPORT                       ", flush=True)
    print("==========================================================================", flush=True)
    print(f"Total Dataset Evaluation Samples : {len(y_true)}", flush=True)
    print(f"Normal Samples (Class 0)         : {int(np.sum(y_true == 0))}", flush=True)
    print(f"Anomaly Samples (Class 1)        : {int(np.sum(y_true == 1))}", flush=True)
    print(f"Decision Threshold (Theta)       : {base_threshold:.6f}", flush=True)
    print("--------------------------------------------------------------------------", flush=True)
    print(f"  ACCURACY    : {acc * 100:.2f}% ({acc:.6f})", flush=True)
    print(f"  AUC-ROC     : {auc * 100:.2f}% ({auc:.6f})", flush=True)
    print(f"  PRECISION   : {prec * 100:.2f}% ({prec:.6f})", flush=True)
    print(f"  RECALL      : {rec * 100:.2f}% ({rec:.6f})", flush=True)
    print(f"  F1-SCORE    : {f1 * 100:.2f}% ({f1:.6f})", flush=True)
    print(f"  FPR (False Positive Rate): {fpr * 100:.2f}% ({fp} / {tn + fp})", flush=True)
    print(f"  FNR (False Negative Rate): {fnr * 100:.2f}% ({fn} / {fn + tp})", flush=True)
    print("--------------------------------------------------------------------------", flush=True)
    print("\n--- CONFUSION MATRIX ---", flush=True)
    print(f"                Predicted Normal (0)   Predicted Anomaly (1)", flush=True)
    print(f"Actual Normal  (0)     TN = {tn:<8}     FP = {fp:<8}", flush=True)
    print(f"Actual Anomaly (1)     FN = {fn:<8}     TP = {tp:<8}", flush=True)
    print("--------------------------------------------------------------------------", flush=True)
    print(f"  * True Negatives (TN)  : {tn:<6} ({tn / len(y_true) * 100:.2f}%) - Normal packets correctly classified", flush=True)
    print(f"  * False Positives (FP) : {fp:<6} ({fp / len(y_true) * 100:.2f}%) - Normal packets falsely flagged as anomaly", flush=True)
    print(f"  * False Negatives (FN) : {fn:<6} ({fn / len(y_true) * 100:.2f}%) - Attacks missed by model", flush=True)
    print(f"  * True Positives (TP)  : {tp:<6} ({tp / len(y_true) * 100:.2f}%) - Attacks correctly detected", flush=True)
    print("==========================================================================", flush=True)
    print("\n--- MODEL CONSTITUENT ABLATION COMPARISON ---", flush=True)
    print(f"  Autoencoder Alone      : Accuracy = {ae_acc*100:.2f}%, Precision = {ae_prec*100:.2f}%, Recall = {ae_rec*100:.2f}%, F1 = {ae_f1*100:.2f}%, AUC = {ae_auc*100:.2f}%", flush=True)
    print(f"  Isolation Forest Alone : Accuracy = {if_acc*100:.2f}%, Precision = {if_prec*100:.2f}%, Recall = {if_rec*100:.2f}%, F1 = {if_f1*100:.2f}%, AUC = {if_auc*100:.2f}%", flush=True)
    print(f"  Hybrid Ensemble (Ours) : Accuracy = {acc*100:.2f}%, Precision = {prec*100:.2f}%, Recall = {rec*100:.2f}%, F1 = {f1*100:.2f}%, AUC = {auc*100:.2f}%", flush=True)
    print("==========================================================================\n", flush=True)

    # Save to eval_results.txt
    eval_txt_path = os.path.join(base_dir, "eval_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write("="*75 + "\n")
        f.write("      HYBRID NIDS EXACT BENCHMARK EVALUATION RESULTS (KDDTest+.txt)\n")
        f.write("="*75 + "\n")
        f.write(f"Total Dataset Samples    : {len(y_true)}\n")
        f.write(f"Normal Samples (0)       : {int(np.sum(y_true == 0))}\n")
        f.write(f"Anomaly Samples (1)      : {int(np.sum(y_true == 1))}\n")
        f.write(f"Operating Threshold      : {base_threshold:.6f}\n")
        f.write(f"Fusion Alpha             : {alpha:.2f} * AE + {1-alpha:.2f} * IF\n")
        f.write("-"*75 + "\n")
        f.write(f"ACCURACY                 : {acc * 100:.2f}% ({acc:.6f})\n")
        f.write(f"AUC-ROC                  : {auc * 100:.2f}% ({auc:.6f})\n")
        f.write(f"PRECISION                : {prec * 100:.2f}% ({prec:.6f})\n")
        f.write(f"RECALL                   : {rec * 100:.2f}% ({rec:.6f})\n")
        f.write(f"F1-SCORE                 : {f1 * 100:.2f}% ({f1:.6f})\n")
        f.write(f"FALSE POSITIVE RATE (FPR): {fpr * 100:.2f}% ({fp} false alarms / {tn + fp} normal)\n")
        f.write(f"FALSE NEGATIVE RATE (FNR): {fnr * 100:.2f}% ({fn} missed attacks / {tp + fn} attacks)\n")
        f.write("-"*75 + "\n")
        f.write("CONFUSION MATRIX:\n")
        f.write(f"                Predicted Normal (0)   Predicted Anomaly (1)\n")
        f.write(f"Actual Normal  (0)     TN = {tn:<8}     FP = {fp:<8}\n")
        f.write(f"Actual Anomaly (1)     FN = {fn:<8}     TP = {tp:<8}\n")
        f.write("-"*75 + "\n")
        f.write(f"True Negatives (TN)      : {tn}\n")
        f.write(f"False Positives (FP)     : {fp}\n")
        f.write(f"False Negatives (FN)     : {fn}\n")
        f.write(f"True Positives (TP)      : {tp}\n")
        f.write("="*75 + "\n\n")
        f.write("--- COMPARISON WITH INDIVIDUAL CONSTITUENTS ---\n")
        f.write(f"Autoencoder Alone        : Accuracy = {ae_acc*100:.2f}%, Precision = {ae_prec*100:.2f}%, Recall = {ae_rec*100:.2f}%, F1 = {ae_f1*100:.2f}%, AUC = {ae_auc*100:.2f}%\n")
        f.write(f"Isolation Forest Alone   : Accuracy = {if_acc*100:.2f}%, Precision = {if_prec*100:.2f}%, Recall = {if_rec*100:.2f}%, F1 = {if_f1*100:.2f}%, AUC = {if_auc*100:.2f}%\n")
        f.write(f"Hybrid Ensemble (Ours)   : Accuracy = {acc*100:.2f}%, Precision = {prec*100:.2f}%, Recall = {rec*100:.2f}%, F1 = {f1*100:.2f}%, AUC = {auc*100:.2f}%\n")

    print(f"Results successfully saved to {eval_txt_path}", flush=True)

if __name__ == "__main__":
    main()
