# 🛡️ Hybrid ML-Based Network Intrusion Detection System (Hybrid NIDS)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time, explainable **Network Intrusion Detection System (NIDS)** combining a dual-engine **Hybrid Machine Learning Architecture** (Deep Autoencoder + Isolation Forest Ensemble) with **Explainable AI (SHAP)** and **Retrieval-Augmented Generation (RAG)** for automated threat explanation and real-time security operations center (SOC) intelligence.

---

## 📌 Key Highlights

- **Dual-Engine Hybrid Detection**: Fuses Deep Autoencoder reconstruction error ($\alpha = 0.75$) with an Isolation Forest statistical outlier detector ($1 - \alpha = 0.25$) to capture both signature-like volumetric spikes and subtle zero-day anomalies.
- **Empirically Validated Performance**: Evaluated against the full benchmark test set (`KDDTest+.txt`, 22,544 packets), achieving **90.52% real test accuracy**, **96.72% AUC-ROC**, and **92.32% attack recall**.
- **Explainable AI (SHAP & Feature Error)**: Deconstructs detected anomalies into the exact top contributing network indicators (e.g., `serror_rate`, `flag_S0`, `dst_host_srv_diff_host_rate`).
- **Cybersecurity RAG Engine**: Semantic vector retrieval over an embedded security knowledge base (49 attack signatures and MITRE ATT&CK patterns) providing contextual threat summaries and mitigation guidance.
- **Real-Time Asynchronous Pipeline**: Multi-threaded Scapy packet capture decoupled via thread-safe queues and streamed live to a modern React dashboard over WebSockets.
- **Secure Access Control**: JWT-based authentication and Google OAuth 2.0 integration.

---

## 📊 Benchmark Evaluation Results

Evaluated directly on the official **NSL-KDD Benchmark Test Dataset (`KDDTest+.txt`)** across all **22,544** records without threshold overfitting:

| Evaluation Metric | Measured Score | Raw Value | Operational Context |
|---|---|---|---|
| **Accuracy** | **90.52%** | `0.905208` | 20,407 correct classifications out of 22,544 test packets |
| **AUC-ROC** | **96.72%** | `0.967239` | Global separability across normal and attack distributions |
| **Precision** | **91.14%** | `0.911448` | Low alert fatigue: 91.14% of triggered alerts are genuine attacks |
| **Recall (Detection Rate)**| **92.32%** | `0.923167` | Successfully catches 11,847 out of 12,833 diverse attacks |
| **F1-Score** | **91.73%** | `0.917270` | Harmonic balance under severe class distribution skew |
| **False Positive Rate (FPR)** | **11.85%** | `0.118525` | 1,151 false alarms out of 9,711 legitimate benign packets |
| **False Negative Rate (FNR)** | **7.68%** | `0.076833` | Only 986 missed attacks out of 12,833 total attacks |

### Confusion Matrix

```text
                        Predicted Normal (0)      Predicted Anomaly (1)
Actual Normal  (0)       TN = 8,560 (37.97%)       FP = 1,151 (5.11%)
Actual Anomaly (1)       FN = 986   (4.37%)        TP = 11,847 (52.55%)
```

### Model Constituent Ablation Study

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | False Positives | False Negatives |
|---|---|---|---|---|---|---|---|
| **Autoencoder Alone** | 82.37% | **98.16%** | 70.34% | 81.96% | **96.83%** | **169** | 3,806 |
| **Isolation Forest Alone** | 77.75% | 75.10% | 91.12% | 82.34% | 84.09% | 3,876 | 1,140 |
| **Hybrid Ensemble ($\alpha=0.75$)** | **90.52%** | 91.14% | **92.32%** | **91.73%** | 96.72% | 1,151 | **986** |

### Training vs. Test Benchmark Generalization

| Evaluation Parameter | Training Set (`KDDTrain+.txt`, 125,973) | Test Set (`KDDTest+.txt`, 22,544) | Generalization Delta |
|---|---|---|---|
| **Accuracy** | **92.48%** | **90.52%** | $-1.96\%$ (Healthy generalization) |
| **AUC-ROC** | **98.40%** | **96.72%** | $-1.68\%$ (Strong distribution stability) |
| **Recall (Sensitivity)** | **97.91%** | **92.32%** | $-5.59\%$ (Reflects 17 novel zero-day attack classes) |
| **Precision** | **87.44%** | **91.14%** | $+3.70\%$ (Fewer false alarms on test set) |
| **F1-Score** | **92.38%** | **91.73%** | $-0.65\%$ (Consistent harmonic balance) |
| **False Positive Rate (FPR)** | **12.25%** | **11.85%** | $-0.40\%$ (Stable false alarm rate) |
| **False Negative Rate (FNR)** | **2.09%** | **7.68%** | $+5.59\%$ (Low miss rate even with unseen attacks) |
| **Calibrated Threshold ($\Theta$)** | **0.127941** | **0.127941** | Identical mathematical cutoff |

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
├── knowledge_base.json                    # Baseline security knowledge base
├── knowledge_base_enhanced.json           # Comprehensive 49-entry security taxonomy
├── data/
│   ├── KDDTrain+.txt                      # NSL-KDD benchmark training set
│   ├── KDDTest+.txt                       # NSL-KDD benchmark test set
│   └── download_nsl_kdd.py                # Dataset download helper
├── documentation/
│   ├── Project_Report.md                  # Comprehensive technical project report
│   ├── COMPREHENSIVE_NIDS_TECHNICAL_REPORT_AND_DIAGNOSTIC.md  # Deep diagnostic analysis
│   ├── HYBRID_NIDS_ANALYSIS_REPORT.md    # Complete architectural reference
│   ├── architecture.png                   # System architecture diagram
│   ├── flow.png                           # Execution flow diagram
│   └── hybrid_nids_architecture_diagram.png # High-level diagram
├── kb_index/                              # Pre-built RAG semantic vector index
│   ├── embeddings.npy                     # Dense vector embeddings
│   ├── metadata.json                      # Entry metadata and labels
│   └── nn.joblib                          # Nearest-neighbors index
├── src/
│   ├── backend/
│   │   ├── main.py                        # FastAPI entrypoint & WebSocket handler
│   │   ├── detector.py                    # HybridDetector (AE + IF + SHAP)
│   │   ├── sniffer.py                     # Real-time packet capture & feature extraction
│   │   ├── rag_service.py                 # RAG explanation & semantic search
│   │   ├── rag_knowledge.py               # Knowledge base loader & normalization
│   │   ├── kb_index.py                    # Vector index manager
│   │   ├── model_metrics.py               # Standalone benchmark evaluation runner
│   │   ├── traffic_generator.py           # Synthetic normal & anomaly traffic generator
│   │   ├── train_pipeline.py              # Canonical model training pipeline
│   │   ├── auth.py                        # JWT & password security
│   │   ├── database.py                    # SQLAlchemy database engine
│   │   ├── models.py                      # Database models (User, Alert)
│   │   ├── schemas.py                     # Pydantic request/response schemas
│   │   ├── requirements.txt               # Backend Python dependencies
│   │   └── saved_models/                  # Trained neural & statistical artifacts
│   │       ├── autoencoder.keras          # Trained Keras autoencoder
│   │       ├── iso_forest.joblib          # Trained 250-tree Isolation Forest
│   │       ├── standard_scaler.joblib     # QuantileTransformer
│   │       ├── ae_scaler.joblib           # Autoencoder score normalizer
│   │       ├── if_scaler.joblib           # Isolation Forest score normalizer
│   │       ├── feature_columns.json       # 122 aligned feature column names
│   │       └── threshold.txt              # Calibrated cutoff threshold (0.127941)
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

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
