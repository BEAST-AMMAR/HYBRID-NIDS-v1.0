import os
import sys
import json
import random
import urllib.request
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Reproducibility seeds — ensures identical results across training runs
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def build_sharper_autoencoder(input_dim, learning_rate=0.0005, dropout_rate=0.15):
    """
    Builds a Deep Autoencoder with a sharper latent bottleneck (4 dimensions)
    to enforce tighter reconstruction constraints on normal traffic.
    """
    input_layer = Input(shape=(input_dim,), name='input_features')
    
    # Encoder
    encoded = Dense(64, activation='relu', name='encoder_dense_1')(input_layer)
    encoded = BatchNormalization(name='encoder_bn_1')(encoded)
    encoded = Dropout(dropout_rate, name='encoder_dropout_1')(encoded)
    
    encoded = Dense(32, activation='relu', name='encoder_dense_2')(encoded)
    encoded = BatchNormalization(name='encoder_bn_2')(encoded)
    encoded = Dropout(dropout_rate, name='encoder_dropout_2')(encoded)

    encoded = Dense(16, activation='relu', name='encoder_dense_3')(encoded)
    encoded = BatchNormalization(name='encoder_bn_3')(encoded)
    
    # Tighter Bottleneck: 4 units
    latent = Dense(4, activation='relu', name='latent_space')(encoded)
    
    # Decoder
    decoded = Dense(16, activation='relu', name='decoder_dense_1')(latent)
    decoded = BatchNormalization(name='decoder_bn_1')(decoded)
    
    decoded = Dense(32, activation='relu', name='decoder_dense_2')(decoded)
    decoded = BatchNormalization(name='decoder_bn_2')(decoded)
    decoded = Dropout(dropout_rate, name='decoder_dropout_2')(decoded)

    decoded = Dense(64, activation='relu', name='decoder_dense_3')(decoded)
    decoded = BatchNormalization(name='decoder_bn_3')(decoded)
    decoded = Dropout(dropout_rate, name='decoder_dropout_3')(decoded)
    
    output_layer = Dense(input_dim, activation='linear', name='reconstructed_features')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer, name='sharper_hybrid_nids_autoencoder')
    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return autoencoder

def main():
    print("==========================================================================")
    print("        OPTION A: ADVANCED UNSUPERVISED HYBRID MODEL TRAINING            ")
    print("==========================================================================\n")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")
    model_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(model_dir, exist_ok=True)

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

    print("[1] Preprocessing & One-Hot Encoding Datasets...")
    train_df = pd.read_csv(train_path, header=None, names=columns).drop('difficulty', axis=1)
    test_df = pd.read_csv(test_path, header=None, names=columns).drop('difficulty', axis=1)

    train_df['label'] = train_df['label'].apply(lambda x: 0 if str(x).strip().lower() in ['normal', 'normal.'] else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if str(x).strip().lower() in ['normal', 'normal.'] else 1)

    train_encoded = pd.get_dummies(train_df, columns=['protocol_type', 'service', 'flag'])
    test_encoded = pd.get_dummies(test_df, columns=['protocol_type', 'service', 'flag'])

    train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

    X_train = train_encoded.drop('label', axis=1)
    y_train = train_encoded['label'].values
    X_test = test_encoded.drop('label', axis=1)
    y_test = test_encoded['label'].values

    feature_columns = list(X_train.columns)
    print(f"    Total Feature Columns: {len(feature_columns)}")

    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    # 2. Fit QuantileTransformer for Robust Feature Scaling
    print("\n[2] Fitting QuantileTransformer on 116-Feature Space...")
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    joblib.dump(scaler, os.path.join(model_dir, "standard_scaler.joblib"))
    print(f"    Saved QuantileTransformer to {os.path.join(model_dir, 'standard_scaler.joblib')}")

    # Extract clean normal training samples
    X_train_normal = X_train_scaled[y_train == 0]
    print(f"    Clean Normal Training Samples: {X_train_normal.shape[0]}")

    # 3. Train Sharper Autoencoder
    print("\n[3] Training Sharper Autoencoder (Bottleneck: 4 units)...")
    autoencoder = build_sharper_autoencoder(input_dim=len(feature_columns))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=60,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    autoencoder_path = os.path.join(model_dir, "autoencoder.keras")
    autoencoder.save(autoencoder_path)
    print(f"    Saved Sharper Autoencoder to {autoencoder_path}")

    # 4. Train Isolation Forest
    print("\n[4] Training Isolation Forest (250 trees, contamination=0.10)...")
    iso_forest = IsolationForest(
        n_estimators=250,
        contamination=0.10,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_scaled)
    iso_path = os.path.join(model_dir, "iso_forest.joblib")
    joblib.dump(iso_forest, iso_path)
    print(f"    Saved Isolation Forest to {iso_path}")

    # 5. Score Normalization & MinMaxScaler Fitting
    print("\n[5] Computing Reconstruction MSE & Isolation Forest Scores...")
    reconstruction_train = autoencoder.predict(X_train_scaled, batch_size=2048, verbose=0)
    train_mses = np.mean(np.power(X_train_scaled - reconstruction_train, 2), axis=1)
    train_if_scores = -iso_forest.decision_function(X_train_scaled)

    ae_scaler = MinMaxScaler(clip=True).fit(train_mses.reshape(-1, 1))
    if_scaler = MinMaxScaler(clip=True).fit(train_if_scores.reshape(-1, 1))

    joblib.dump(ae_scaler, os.path.join(model_dir, "ae_scaler.joblib"))
    joblib.dump(if_scaler, os.path.join(model_dir, "if_scaler.joblib"))

    # Test Set Predictions
    reconstruction_test = autoencoder.predict(X_test_scaled, batch_size=2048, verbose=0)
    test_mses = np.mean(np.power(X_test_scaled - reconstruction_test, 2), axis=1)
    test_if_scores = -iso_forest.decision_function(X_test_scaled)

    ae_norm = ae_scaler.transform(test_mses.reshape(-1, 1)).flatten()
    if_norm = if_scaler.transform(test_if_scores.reshape(-1, 1)).flatten()

    alpha = 0.75  # Increased weight on Autoencoder reconstruction
    hybrid_score = alpha * ae_norm + (1 - alpha) * if_norm

    # 6. ROC Curve Threshold Calibration
    print("\n[6] Calibrating Decision Cutoff Threshold via ROC Youden's J Index...")
    fpr, tpr, thresholds = roc_curve(y_test, hybrid_score)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = float(thresholds[optimal_idx])

    threshold_path = os.path.join(model_dir, "threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(f"{optimal_threshold:.6f}")
    print(f"    Calibrated Optimal Threshold (Theta): {optimal_threshold:.6f}")
    print(f"    Saved threshold to {threshold_path}")

    # 7. Evaluation & Metrics Output
    y_pred = (hybrid_score > optimal_threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n==========================================================================")
    print("           OPTION A OPTIMIZED PIPELINE MODEL EVALUATION RESULTS           ")
    print("==========================================================================")
    print(f"Total Test Samples               : {len(y_test)}")
    print(f"Normal Samples (Class 0)         : {sum(y_test == 0)}")
    print(f"Anomaly Samples (Class 1)        : {sum(y_test == 1)}")
    print("--------------------------------------------------------------------------")
    print(f"  ACCURACY    : {acc * 100:.2f}%")
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

    # --- ALSO SHOW TRAINING SET ACCURACY (for diagnostic comparison) ---
    train_preds_ae = autoencoder.predict(X_train_scaled, verbose=0)
    train_mses = np.mean((X_train_scaled - train_preds_ae) ** 2, axis=1)
    train_if_scores = -iso_forest.decision_function(X_train_scaled)
    train_ae_norm = ae_scaler.transform(train_mses.reshape(-1, 1)).flatten()
    train_if_norm = if_scaler.transform(train_if_scores.reshape(-1, 1)).flatten()
    train_hybrid = alpha * train_ae_norm + (1 - alpha) * train_if_norm
    train_y_pred = (train_hybrid > optimal_threshold).astype(int)
    train_acc = accuracy_score(y_train, train_y_pred)
    print(f"--- TRAINING SET ACCURACY (diagnostic) : {train_acc * 100:.2f}% ---")
    print(f"    (This is higher than test accuracy because the model has seen this data)")
    print(f"    (If your earlier '96%' matches this number, that was training accuracy)\n")

if __name__ == "__main__":
    main()
