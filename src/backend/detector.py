"""
=============================================================================
  HybridDetector — Dedicated NF-UQ-NIDS-v2 Architecture
=============================================================================
  Combines:
    1. Deep Residual PyTorch Autoencoder (Reconstruction MSE) on RTX 5070 GPU
    2. Latent Isolation Forest (Partition Trees on 16-D Manifold)
    3. Latent Mahalanobis Distance (Benign Covariance Ellipsoid)
    4. Topological k-NN Distance (Manifold Neighborhood Density)
    5. Calibrated Non-Linear Gated Fusion
=============================================================================
"""

import os
import json
import numpy as np
import joblib
import torch
from collections import deque
from scipy.spatial.distance import mahalanobis
from .nf_uq_gpu_pipeline import NfUqAutoencoder


class HybridDetector:
    """
    Dedicated Hybrid NIDS Detector for NF-UQ-NIDS-v2 with GPU acceleration.
    """
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "saved_models")
        model_dir = os.path.abspath(model_dir)
        self.model_dir = model_dir

        # 1. Feature columns
        feat_path = os.path.join(model_dir, "feature_columns.json")
        with open(feat_path, "r") as f:
            self.feature_names = json.load(f)

        # 2. PyTorch GPU Autoencoder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch_model_path = os.path.join(model_dir, "autoencoder_torch.pt")
        self.autoencoder = NfUqAutoencoder(in_features=len(self.feature_names), latent_dim=16)
        
        if os.path.exists(torch_model_path):
            self.autoencoder.load_state_dict(
                torch.load(torch_model_path, map_location=self.device, weights_only=True)
            )
        self.autoencoder.to(self.device).eval()

        # 3. Latent Isolation Forest (200 trees)
        iso_path = os.path.join(model_dir, "iso_forest.joblib")
        self.iso_forest = joblib.load(iso_path) if os.path.exists(iso_path) else None

        # 4. Latent Mahalanobis & k-NN scoring artefacts
        artefacts_path = os.path.join(model_dir, "scoring_artefacts.joblib")
        if os.path.exists(artefacts_path):
            artefacts = joblib.load(artefacts_path)
            self.benign_latent_mean = artefacts["benign_latent_mean"]
            self.benign_latent_cov_inv = artefacts["benign_latent_cov_inv"]
            # Load calibrated normalization constants (computed from actual benign latent distribution)
            # These are the P99 distances of benign samples — used as the "unit" of normalization
            self.maha_norm_divisor = float(artefacts.get("maha_norm_divisor", 18.0))
            self.knn_norm_divisor = float(artefacts.get("knn_norm_divisor", 14.0))
        else:
            self.benign_latent_mean = np.zeros(16, dtype=np.float32)
            self.benign_latent_cov_inv = np.eye(16, dtype=np.float32)
            self.maha_norm_divisor = 18.0
            self.knn_norm_divisor = 14.0

        knn_path = os.path.join(model_dir, "knn_model.joblib")
        self.knn_model = joblib.load(knn_path) if os.path.exists(knn_path) else None

        # 5. Operating Thresholds
        thr_path = os.path.join(model_dir, "threshold.txt")
        if os.path.exists(thr_path):
            with open(thr_path, "r") as f:
                self.base_threshold = float(f.read().strip())
        else:
            self.base_threshold = 0.408528

        # Multiple calibrated policies
        self.threshold_policies = {
            "p95": 0.408528,      # Balanced low-FPR (~5%)
            "p99": 0.502945,      # Ultra-low FPR (<1%) - Industry standard for live production sniffing
            "youden": 0.414944,   # Maximum Sensitivity & Specificity
            "fpr_3pct": 0.435947  # High precision enterprise policy
        }
        self.current_policy = "p99"  # Default to p99 for live network capture

        # 6. Feature Adapter (Quantile Transformer + One-Hot definitions)
        adapter_meta_path = os.path.join(model_dir, "nf_uq_adapter_meta.joblib")
        adapter_qt_path = os.path.join(model_dir, "nf_uq_adapter_quantile.joblib")
        if os.path.exists(adapter_meta_path) and os.path.exists(adapter_qt_path):
            self.adapter_meta = joblib.load(adapter_meta_path)
            self.adapter_quantile = joblib.load(adapter_qt_path)
        else:
            self.adapter_meta = None
            self.adapter_quantile = None

        # Live score buffer for dynamic baseline estimation
        self.score_buffer = deque(maxlen=2000)
        self.dynamic_threshold = self.base_threshold

    def set_policy(self, policy_name: str):
        """Switch threshold policy dynamically (e.g. 'p95', 'p99', 'youden')."""
        if policy_name in self.threshold_policies:
            self.base_threshold = self.threshold_policies[policy_name]
            self.dynamic_threshold = self.base_threshold
            self.current_policy = policy_name
            return True
        return False

    def get_top_anomalous_features(self, scaled_features, reconstruction):
        """Identify top-3 most anomalous NetFlow features by squared error."""
        sq_errors = np.power(scaled_features[0] - reconstruction[0], 2)
        top_indices = sq_errors.argsort()[-3:][::-1]
        top_features = [self.feature_names[i] for i in top_indices if i < len(self.feature_names)]
        return top_features

    def predict(self, scaled_features, return_details=False, explain=False, threshold=None):
        """
        Score a sample array (1, n_features) or batch (N, n_features).

        Uses non-linear gated fusion combining Autoencoder, Latent Isolation Forest,
        Mahalanobis distance, and k-NN topological distance.

        Normalization is calibrated using actual benign latent P99 distances so that
        typical benign flows score well below the threshold.
        """
        active_thr = threshold if threshold is not None else self.base_threshold

        with torch.no_grad():
            tensor = torch.from_numpy(scaled_features).float().to(self.device)
            recon, latent_t = self.autoencoder(tensor)
            reconstruction = recon.cpu().numpy()
            latent = latent_t.cpu().numpy()

        # 1. Reconstruction MSE
        sq_diff = np.power(scaled_features - reconstruction, 2)
        mse = np.mean(sq_diff, axis=1)
        # Calibrated MSE: benign p95 is 0.078, attack p90 is 0.516
        mse_norm = np.clip(mse / 0.22, 0.0, 1.0)

        # 2. Latent Isolation Forest
        if self.iso_forest is not None:
            iso_raw = -self.iso_forest.decision_function(latent)
            # Calibrated: benign is ~-0.15 to -0.05 -> maps to 0.0–0.20
            # Attacks have iso_raw ~ 0.0 to +0.22 -> maps to 0.40–1.0
            iso_score = np.clip((iso_raw + 0.10) / 0.25, 0.0, 1.0)
        else:
            iso_score = np.zeros_like(mse)

        # 3. Latent Mahalanobis Distance
        diff = latent - self.benign_latent_mean
        maha_sq = np.clip(np.sum(diff @ self.benign_latent_cov_inv * diff, axis=1), 0.0, None)
        maha_dist = np.sqrt(maha_sq)
        maha_norm = np.clip(maha_dist / self.maha_norm_divisor, 0.0, 1.0)

        # 4. Topological k-NN Distance
        if self.knn_model is not None:
            dists, _ = self.knn_model.kneighbors(latent)
            knn_dist = np.mean(dists, axis=1)
            knn_norm = np.clip(knn_dist / self.knn_norm_divisor, 0.0, 1.0)
        else:
            knn_norm = np.zeros_like(mse)

        # 5. Calibrated Non-Linear Gated Fusion
        # Weights: AE (35%) + IF (30%) + k-NN (20%) + Mahalanobis (15%)
        base_fusion = 0.35 * mse_norm + 0.30 * iso_score + 0.20 * knn_norm + 0.15 * maha_norm

        # Gating: if both AE error and IF/KNN indicate anomaly, boost confidence
        dual_agreement = (mse_norm > 0.40) & ((iso_score > 0.40) | (knn_norm > 0.40))
        hybrid_scores = np.where(dual_agreement, np.clip(base_fusion * 1.15, 0.0, 1.0), base_fusion)

        is_anomalies = hybrid_scores > active_thr

        # Update dynamic buffer
        for s in hybrid_scores:
            self.score_buffer.append(float(s))

        # Single-sample return format matching UI caller expectations
        if scaled_features.shape[0] == 1:
            is_anomaly = bool(is_anomalies[0])
            score = float(hybrid_scores[0])
            cur_mse = float(mse[0])
            cur_maha = float(maha_dist[0])

            if return_details:
                return is_anomaly, score, cur_mse, cur_maha
            if explain:
                top_features = self.get_top_anomalous_features(scaled_features, reconstruction)
                return is_anomaly, score, top_features
            return is_anomaly, score

        return is_anomalies, hybrid_scores
