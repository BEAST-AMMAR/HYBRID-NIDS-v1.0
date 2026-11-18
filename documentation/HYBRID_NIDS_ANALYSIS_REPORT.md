# 🔍 Hybrid NIDS — Full Technical Analysis Report

> **Cross-examined across:** `nf_uq_pipeline.py`, `detector.py`, `sniffer.py`, `main.py`, `rag_service.py`, `kb_index.py`, `configs/nf_uq.yaml`, `feature_columns.json`, `knowledge_base.json`, `threshold.txt`
>
> **Last updated:** NF-UQ-NIDS-v2 Modernization — All legacy NSL-KDD codes expunged, full dataset GPU training on RTX 5070, and 4-tier FPR/FN minimization architecture deployed.

---

## 1. 📦 Dataset & Preprocessing

### Benchmark Dataset: NF-UQ-NIDS-v2
- The system is built on **NF-UQ-NIDS-v2**, comprising **75,987,976 NetFlow V2 flows** across 48 training partitions and held-out test partitions 50 & 51.
- Each flow represents realistic modern enterprise network communications across **20 distinct attack families** (DDoS, DoS, Botnets, Scanning, Brute Force, Exploits, Infiltration, etc.) and millions of diverse benign communications.

### Preprocessing & Normalization (`NfUqFeatureAdapter`)
- **log1p Non-Linear Transformation**: Network traffic byte counts and packet volumes are heavy-tailed. The pipeline applies $\text{sign}(x) \cdot \log(1 + |x|)$ to compress variance.
- **QuantileTransformer**: Uses `QuantileTransformer(output_distribution='uniform')` fitted strictly on benign flows to eliminate extreme outlier distortion without data leakage.
- **One-Hot Port Encoding**: Top enterprise service ports are one-hot encoded (`port_80`, `port_443`, `port_53`, `port_22`, etc.) while mapping unseen ports to `port_other`.
- **Final Feature Matrix**: Exactly **133 numerical columns**, serialized in `feature_columns.json` and mirrored in `configs/nf_uq.yaml`.

---

## 2. 🧠 Deep Autoencoder (PyTorch CUDA on RTX 5070)

### Neural Architecture
A deep, symmetric bottleneck autoencoder trained strictly on benign baseline flows:

| Layer Name | Type | Units | Activation | Regularization |
|---|---|---|---|---|
| `Input` | Input Layer | 133 | — | — |
| `Encoder_1` | Linear Dense | 64 | **ReLU** | BatchNorm1d + Dropout (10%) |
| `Encoder_2` | Linear Dense | 32 | **ReLU** | BatchNorm1d + Dropout (10%) |
| `Latent_Space` | Linear Dense | **16** | **Linear** | 16-D Bottleneck Manifold |
| `Decoder_1` | Linear Dense | 32 | **ReLU** | BatchNorm1d + Dropout (10%) |
| `Decoder_2` | Linear Dense | 64 | **ReLU** | BatchNorm1d + Dropout (10%) |
| `Reconstruction` | Linear Dense | 133 | **Linear** | Unclamped Reconstruction |

### Training Specifications
- **Hardware Acceleration**: Trained on an **NVIDIA GeForce RTX 5070 GPU** (CUDA 12.8 nightly runtime).
- **Dataset Scale**: 12 complete epochs across all 48 training partitions (~283 million benign flow passes).
- **Validation Loss**: Optimal Mean Squared Error (MSE) reached **0.00283**.

---

## 3. 🌲 Latent Space Ensembling & Manifold Verification

Rather than analyzing noisy raw 133-D space, secondary anomaly detectors operate directly on the regularized **16-dimensional bottleneck representation**:

1. **Latent Isolation Forest (200 Trees)**:
   - Evaluates tree isolation depth in latent space: anomalous flows isolate with significantly shorter path lengths.
2. **Latent Mahalanobis Distance**:
   - Compares latent points against the empirical covariance matrix $\Sigma^{-1}$ of benign training distributions:
     $$D_M(z) = \sqrt{(z - \mu)^T \Sigma^{-1} (z - \mu)}$$
3. **Topological 5-NN Reference Index**:
   - Measures local Euclidean density against 20,000 benign reference embeddings, preventing false positives along multi-modal cluster boundaries.
4. **Host-Level Temporal Tracker**:
   - Tracks a rolling 10-second traffic window per source IP (`flows_10s` and `unique_ports_10s`), eliminating false negatives on low-rate port scans and C2 beacon heartbeats.

---

## 4. ⚖️ Non-Linear Gated Fusion & Operating Policies

### Score Fusion
The composite anomaly score combines all 4 detection tiers:
$$S_{\text{base}} = 0.35 \times \text{MSE}_{\text{norm}} + 0.30 \times \text{IF}_{\text{score}} + 0.20 \times \text{Maha}_{\text{norm}} + 0.15 \times \text{kNN}_{\text{norm}}$$

- **Dual-Agreement Gating**: When both the Autoencoder and Isolation Forest exhibit high anomaly confidence ($> 0.45$), the composite score is boosted by $1.15\times$, locking in high precision on genuine attacks.

### Operating Policies (`/api/policy`)
- **`p95` (Default)**: 95th percentile validation threshold ($\Theta = 0.408528$). Balanced detection (88.33% accuracy, 5.08% FPR, 97.13% precision).
- **`p99` (High Precision)**: 99th percentile threshold ($\Theta = 0.536762$). Restricts False Positive Rate to **< 1.0%** for strict enterprise quotas.
- **`youden`**: Maximizes the Youden J-statistic ($J = \text{TPR} - \text{FPR}$).
- **`fpr_3pct`**: Calibrated 3% false alarm budget.

---

## 5. 📊 Held-Out Test Evaluation Benchmark

Evaluated across **1,702,252 held-out test flows** from partitions 50 & 51:

| Metric | Benchmark Result | SOC Operational Impact |
| :--- | :--- | :--- |
| **Overall Accuracy** | **88.33%** | 1,503,652 correct flow decisions |
| **Precision** | **97.13%** | Less than 3% false alarm incidence |
| **Recall (Sensitivity)** | **85.07%** | Intercepts 968,755 malicious flow vectors |
| **F1-Score** | **90.70%** | Exceptional harmonic balance |
| **ROC-AUC** | **93.40%** | Robust global discriminative power |
| **PR-AUC** | **97.16%** | High precision across broad recall |
| **False Positive Rate** | **5.08%** ($p_{95}$) / **< 1.0%** ($p_{99}$) | Dynamically selectable policy |

### Confusion Matrix (1,702,252 Flows)
```text
                 Predicted Normal (0)   Predicted Anomaly (1)
Actual Normal  (0)      TN = 534,897          FP = 28,610
Actual Anomaly (1)      FN = 169,990          TP = 968,755
```

---

## 6. 🤖 Explainable AI (XAI) & RAG Intelligence

When an alert is flagged:
1. **Feature Attribution**: Identifies the top 3 contributing NetFlow features based on reconstruction error variance (e.g., `DURATION_IN`, `TCP_WIN_MAX_IN`, `LONGEST_FLOW_PKT`).
2. **Semantic Knowledge Base**: Classifies threats against a 34-entry threat vector database across all 20 NF-UQ attack classes.
3. **Structured SOC Report**: Synthesizes telemetry into Markdown alerts with threat profiles, observed indicators, and recommended mitigation actions.
