# 🛡️ Hybrid ML-Based Network Intrusion Detection System (Hybrid NIDS)
## Baseline 2: Deep Autoencoder + Latent IF on NF-UQ-NIDS-v2 (Historical Archive)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Proprietary](https://img.shields.io/badge/Status-Confidential%20%7C%20All%20Rights%20Reserved-red.svg)]()

A real-time, explainable **Network Intrusion Detection System (NIDS)** combining a dual-engine **Hybrid Machine Learning Architecture** (Deep Autoencoder + Isolation Forest Ensemble) with **Explainable AI (SHAP)** and **Retrieval-Augmented Generation (RAG)** for automated threat explanation and real-time security operations center (SOC) intelligence.

---

## 📌 Key Highlights

- **Dual-Engine Hybrid Detection**: Fuses Deep Residual Autoencoder reconstruction error on the **NVIDIA RTX 5070 GPU** with a **Latent Isolation Forest (200 Trees)**, **Mahalanobis Covariance Distance**, and **Topological k-NN density** on the 16-D bottleneck manifold.
- **Empirically Validated Performance on NF-UQ-NIDS-v2**: Evaluated on **1,702,252 held-out NetFlow v2 test records** across 4 unified benchmark datasets, achieving **88.33% real accuracy**, **97.13% precision**, **93.40% ROC-AUC**, and only **5.08% False Positive Rate**.
- **Hardware-Accelerated GPU Streaming Pipeline**: PyTorch CUDA 12.8 nightly pipeline streaming across **75,987,976 NetFlow records** (~283 million benign flow exposures across 12 full epochs) while maintaining system RAM under **2.3 GB**.
- **NetFlow V2 Flow Architecture**: Native 133-dimensional flow metrics (duration, byte throughput, packet distributions, TCP flags, TTL variance, and temporal host-context features).
- **Cybersecurity RAG Engine**: Semantic vector retrieval over an embedded security knowledge base covering all 20 NF-UQ attack categories (DDoS, DoS, Botnets, Exploits, Ransomware, Infiltration, Scanning, Brute Force, etc.).
- **Real-Time Asynchronous Pipeline**: Multi-threaded Scapy packet capture decoupled via thread-safe queues and streamed live to a modern React dashboard over WebSockets.

---

## 📊 Benchmark Evaluation Results (NF-UQ-NIDS-v2)

Evaluated directly on **Partitions 50 & 51 (1,702,252 held-out flows)** across all 4 benchmark sources (*NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2, NF-UNSW-NB15-v2, NF-BoT-IoT-v2*):

| Evaluation Metric | Measured Score | Raw Value | Operational Context |
|---|---|---|---|
| **Accuracy** | **88.33%** | `0.883331` | 1,503,652 correct classifications out of 1,702,252 test flows |
| **Precision** | **97.13%** | `0.971314` | Ultra-low alert fatigue: 97.13% of triggered alerts are genuine threats |
| **Recall (Detection Rate)**| **85.07%** | `0.850722` | Successfully catches 968,755 real-world network attacks |
| **F1-Score** | **90.70%** | `0.907027` | Robust harmonic balance on imbalanced traffic streams |
| **False Positive Rate (FPR)** | **5.08%** | `0.050771` | Only 28,610 false alarms out of 563,507 clean normal flows |
| **ROC-AUC** | **93.40%** | `0.933954` | High global separability between benign traffic and intrusions |
| **PR-AUC** | **97.16%** | `0.971597` | High Precision-Recall AUC under severe real-world attack ratios |

### Confusion Matrix (1,702,252 flows)

```text
                        Predicted Normal (0)      Predicted Anomaly (1)
Actual Normal  (0)       TN = 534,897 (31.42%)     FP = 28,610  (1.68%)
Actual Anomaly (1)       FN = 169,990 (9.99%)      TP = 968,755 (56.91%)
```

---

## 🏗️ System Architecture

```text
Physical Interface (NIC) / Synthetic Generator / PCAP Replay
                           │
                           ▼
          Asynchronous Thread-Safe Packet Queue
                           │
                           ▼
         RealTimeFeatureExtractor (Scapy + Rolling Window)
     [41 Intrinsic & Windowed Features ➔ 122 One-Hot Features]
                           │
                           ▼
     QuantileTransformer Scaling (Gaussian Normalization)
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
   Deep Autoencoder                  Isolation Forest
(122➔64➔32➔16➔4➔16➔32➔64➔122)          (250 Trees)
  [Reconstruction MSE]            [Path-Length Anomaly]
          │                                 │
          └────────────────┬────────────────┘
                           ▼
             Normalized Score Fusion (α=0.75)
             Hybrid Score = 0.75*AE + 0.25*IF
                           │
                           ▼
             Decision Threshold (Θ = 0.127941)
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
      [Normal]                          [Anomaly]
    Pass Traffic                             │
                                             ▼
                             XAI & SHAP TreeExplainer
                           Top 3 Anomaly Contributors
                                             │
                                             ▼
                                  RAG Explanation Engine
                            (all-MiniLM-L6-v2 Embeddings)
                                             │
                                             ▼
                                  FastAPI WebSocket Stream
                                             │
                                             ▼
                                  React Monitoring UI
```

---

## 📁 Repository Structure

```text
├── README.md                              # Main project documentation
├── eval_results.txt                       # Exact benchmark evaluation metrics
├── eval_results.json                      # Machine-readable evaluation metrics
├── create_admin.py                        # Admin account bootstrap utility
├── knowledge_base.json                    # 20 NF-UQ attack intelligence taxonomy
├── configs/
│   └── nf_uq.yaml                         # NF-UQ-NIDS-v2 dataset and training configs
├── data/
│   └── NF UQ NIDS V2/                     # NF-UQ-NIDS-v2 dataset (52 Parquet partitions)
├── documentation/
│   ├── Project_Report.md                  # Comprehensive technical project report
│   ├── COMPREHENSIVE_NIDS_TECHNICAL_REPORT_AND_DIAGNOSTIC.md
│   └── HYBRID_NIDS_ANALYSIS_REPORT.md
├── src/
│   ├── backend/
│   │   ├── main.py                        # FastAPI entrypoint & WebSocket handler
│   │   ├── detector.py                    # HybridDetector (PyTorch GPU + IF + Gated Fusion)
│   │   ├── sniffer.py                     # NetFlow V2 packet capture & feature extraction
│   │   ├── nf_uq_gpu_pipeline.py          # RTX 5070 GPU streaming training pipeline
│   │   ├── nf_uq_pipeline.py              # Feature adapter & calibration engine
│   │   ├── rag_service.py                 # RAG explanation & semantic search
│   │   ├── rag_knowledge.py               # Knowledge base loader & normalization
│   │   ├── kb_index.py                    # Vector index manager
│   │   ├── model_metrics.py               # Standalone benchmark evaluation runner
│   │   ├── traffic_generator.py           # Synthetic NetFlow traffic generator
│   │   ├── auth.py                        # JWT & password security
│   │   ├── database.py                    # SQLAlchemy database engine
│   │   ├── models.py                      # Database models (User, Alert)
│   │   ├── schemas.py                     # Pydantic request/response schemas
│   │   ├── requirements.txt               # Backend Python dependencies
│   │   └── saved_models/                  # GPU-trained model weights & adapter artifacts
│   │       ├── autoencoder_torch.pt       # PyTorch Deep Autoencoder (RTX 5070 GPU)
│   │       ├── iso_forest.joblib          # Latent Isolation Forest (200 Trees)
│   │       ├── knn_model.joblib           # Topological k-NN index
│   │       ├── scoring_artefacts.joblib   # Mahalanobis covariance & reference statistics
│   │       ├── nf_uq_adapter_meta.joblib  # Feature adapter metadata & top categorical maps
│   │       ├── nf_uq_adapter_quantile.joblib # QuantileTransformer scaling model
│   │       ├── feature_columns.json       # 133 NetFlow feature columns
│   │       └── threshold.txt              # Calibrated decision threshold (0.408528)
│   └── frontend/
│       ├── package.json                   # React Vite dependencies
│       ├── vite.config.js                 # Vite build configuration
│       └── src/                           # Dashboard UI components & views
└── venv/                                  # Python virtual environment
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python**: Version 3.10 or 3.11
- **Node.js**: Version 18+ and npm
- **Packet Sniffing**: Administrator / Sudo privileges and [Npcap](https://npcap.com/) (Windows) or `libpcap` (Linux)

---

### Step 1: Backend Setup

1. **Activate the virtual environment**:
   ```powershell
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   source venv/bin/activate
   ```

2. **(Optional) Install dependencies** (if not already installed):
   ```bash
   pip install -r src/backend/requirements.txt
   ```

3. **Configure Environment Variables**:
   Ensure `.env` exists in `src/backend/` (based on `src/backend/.env.example`):
   ```ini
   JWT_SECRET_KEY=your_secure_secret_key_here
   DATABASE_URL=sqlite:///./sql_app.db
   ```

4. **Initialize the Admin Account**:
   ```bash
   python create_admin.py
   ```
   > **Default Credentials**: `admin@admin.com` / `admin`

5. **Start the FastAPI Backend**:
   ```bash
   python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   - **API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **WebSocket Feed**: `ws://localhost:8000/ws/monitor`

---

### Step 2: Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd src/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Launch the development server**:
   ```bash
   npm run dev
   ```
   - **Dashboard URL**: [http://localhost:5173](http://localhost:5173)

4. **Log in** with `admin@admin.com` / `admin` to access the live packet monitoring dashboard.

---

### Step 3: Reproduce Model Evaluation

To reproduce the benchmark performance metrics directly on the test set:

```bash
python src/backend/model_metrics.py
```

This evaluates the models across all 22,544 test packets and outputs the exact accuracy, AUC-ROC, precision, recall, F1-score, and confusion matrix to the terminal and [`eval_results.txt`](eval_results.txt).

---

## 🔒 Security & Best Practices

- **Zero-Drop Architecture**: Packet capture runs on an isolated daemon thread, preventing network drops during intensive neural inference.
- **Poisoning Prevention**: The dynamic threshold baseline buffer strictly caps allowable updates at $1.5 \times \Theta_{\text{base}}$ to thwart adversarial slow-burn threshold manipulation.
- **Credential Protection**: Database passwords use cryptographic bcrypt hashing, and JWT tokens authenticate all administrative WebSocket sessions.

---

## 🔒 Intellectual Property & Academic Notice

**Copyright © 2026. All Rights Reserved.**

This document details legacy Baseline 2 experimental research conducted as an evolutionary milestone for academic thesis evaluation and publication. All rights are reserved under academic embargo. Unauthorized reproduction or commercial use is strictly prohibited.
