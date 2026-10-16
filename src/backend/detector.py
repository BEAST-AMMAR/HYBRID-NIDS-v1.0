import numpy as np
import tensorflow as tf
import joblib
import os
import json
from collections import deque
import shap

def _minmax_normalize(values, lo=0.0, hi=1.0):
    span = hi - lo
    if span == 0:
        return np.zeros_like(values)
    return np.clip((values - lo) / span, 0.0, 1.0)

class HybridDetector:
    def __init__(self, model_dir):
        model_dir = os.path.abspath(model_dir)

        self.autoencoder = tf.keras.models.load_model(
            os.path.join(model_dir, "autoencoder.keras")
        )
        self.iso_forest = joblib.load(os.path.join(model_dir, "iso_forest.joblib"))
        
        # Load feature names
        with open(os.path.join(model_dir, "feature_columns.json"), "r") as f:
            self.feature_names = json.load(f)
            
        # Explainer for IF
        self.explainer = shap.TreeExplainer(self.iso_forest)

        ae_path = os.path.join(model_dir, "ae_scaler.joblib")
        if_path = os.path.join(model_dir, "if_scaler.joblib")
        self.ae_scaler = joblib.load(ae_path) if os.path.exists(ae_path) else None
        self.if_scaler = joblib.load(if_path) if os.path.exists(if_path) else None

        with open(os.path.join(model_dir, "threshold.txt"), "r") as f:
            self.base_threshold = float(f.read())
            
        self.alpha = 0.75
        
        # Rolling baseline for dynamic thresholding
        self.score_buffer = deque(maxlen=1000)
        self.dynamic_threshold = self.base_threshold

    def get_top_anomalous_features(self, scaled_features, reconstruction):
        # 1. MSE for Autoencoder (feature-wise reconstruction error)
        ae_importance = np.power(scaled_features[0] - reconstruction[0], 2)
        ae_norm_feat = _minmax_normalize(ae_importance, lo=ae_importance.min(), hi=ae_importance.max())

        try:
            # SHAP for Isolation Forest with check_additivity=False for performance and robustness
            shap_values = self.explainer.shap_values(scaled_features, check_additivity=False)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # binary classification
            shap_importance = np.abs(shap_values[0])
            shap_norm = _minmax_normalize(shap_importance, lo=shap_importance.min(), hi=shap_importance.max())
            combined_importance = shap_norm + ae_norm_feat
        except Exception:
            # Safe, instantaneous fallback to Autoencoder feature-level attribution
            combined_importance = ae_norm_feat
        
        # Get top 3 indices
        top_indices = combined_importance.argsort()[-3:][::-1]
        top_features = [self.feature_names[i] for i in top_indices if i < len(self.feature_names)]
        return top_features

    def predict(self, scaled_features, return_details=False, explain=False):
        reconstruction = self.autoencoder.predict(scaled_features, verbose=0)
        mse = np.mean(np.power(scaled_features - reconstruction, 2), axis=1)
        if_scores = -self.iso_forest.decision_function(scaled_features)

        if self.ae_scaler is not None:
            ae_norm = np.clip(self.ae_scaler.transform(mse.reshape(-1, 1)).flatten(), 0.0, 1.0)
        else:
            baseline_max = max(float(mse.max()), 0.1)
            ae_norm = _minmax_normalize(mse, lo=0.0, hi=baseline_max)

        if self.if_scaler is not None:
            if_norm = np.clip(self.if_scaler.transform(if_scores.reshape(-1, 1)).flatten(), 0.0, 1.0)
        else:
            lo_val = min(float(if_scores.min()), -0.1)
            hi_val = max(float(if_scores.max()), 0.1)
            if_norm = _minmax_normalize(if_scores, lo=lo_val, hi=hi_val)

        hybrid_score = float(self.alpha * ae_norm[0] + (1 - self.alpha) * if_norm[0])
        
        # Dynamic Threshold calculation & warm-up
        if len(self.score_buffer) >= 30:
            mean_score = float(np.mean(self.score_buffer))
            std_score = float(np.std(self.score_buffer))
            # Dynamic threshold reflects ambient network noise floor, with base_threshold as safety floor
            self.dynamic_threshold = max(self.base_threshold, mean_score + 3.0 * std_score)
        
        is_anomaly = bool(hybrid_score > self.dynamic_threshold)
        
        # Buffer update:
        # 1. During initial warm-up (< 100 samples), record non-extreme packets (< 0.35) to establish ambient baseline
        # 2. After warm-up, record benign packets below dynamic threshold to track drift
        if len(self.score_buffer) < 100:
            if hybrid_score < max(self.base_threshold * 2.0, 0.30):
                self.score_buffer.append(hybrid_score)
        elif not is_anomaly and hybrid_score < (self.dynamic_threshold * 1.25):
            self.score_buffer.append(hybrid_score)

        if return_details:
            return is_anomaly, hybrid_score, float(ae_norm[0]), float(if_norm[0])
            
        if explain and is_anomaly:
            top_features = self.get_top_anomalous_features(scaled_features, reconstruction)
            return is_anomaly, hybrid_score, top_features
            
        return is_anomaly, hybrid_score
