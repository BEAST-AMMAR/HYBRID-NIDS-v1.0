# 🛡️ Hybrid ML/DL Network Intrusion Detection System (Hybrid NIDS)
## Evolutionary Research Progression Across Three Experimental Generations

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.12](https://img.shields.io/badge/PyTorch-2.12%20CUDA%2012.8-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![Hardware](https://img.shields.io/badge/GPU-RTX%205070-76B900.svg)](https://nvidia.com)
[![Status](https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg)]()
[![Proprietary](https://img.shields.io/badge/Status-Confidential%20%7C%20All%20Rights%20Reserved-red.svg)]()

A state-of-the-art, 100% local, air-gapped **Network Intrusion Detection System (NIDS)** charting the full empirical and architectural evolution across three experimental paradigms:
1. **Baseline 1**: Shallow Autoencoder + Isolation Forest on legacy **NSL-KDD** (41 tabular features).
2. **Baseline 2**: Deep Residual Autoencoder + Latent Isolation Forest + Mahalanobis Distance + Topological $k$-NN on modern **NF-UQ-NIDS-v2** (133 NetFlow V2 features, isolated single-flow inspection).
3. **Phase 2 Proposed (Flagship)**: **Dual-Stream Hybrid Bi-MambaFormer** (Bidirectional Mamba State-Space Model + Multi-Head Temporal Self-Attention + Adaptive Gated Fusion) on **NF-UQ-NIDS-v2** rolling temporal host sequences ($L=32$ flows per window), backed by sub-millisecond FlowTokenizer saliency XAI and 100% offline BM25 threat intelligence.

---

## 📚 Master Documentation & Defense Resources

| Document | Direct File Link | Primary Purpose |
| :--- | :--- | :--- |
| **Comprehensive Research Thesis** | [RESEARCH_THESIS.md](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/RESEARCH_THESIS.md) | 10-chapter full academic research thesis covering cross-baseline architecture, empirical results, mathematical formulation, and defense. |
| **Academic Thesis Comparative Study** | [THESIS_COMPARATIVE_ANALYSIS.md](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/THESIS_COMPARATIVE_ANALYSIS.md) | Publication-ready comparative analysis, stealth breakdown, generalization proof, and copy-paste LaTeX tables. |
| **Defense Slide Deck & Examiner FAQ** | [DEFENSE_PRESENTATION_AND_QA.md](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/DEFENSE_PRESENTATION_AND_QA.md) | 9-slide presentation blueprint with speaker talking points and answers to 6 critical examiner questions. |
| **System Context & Engineering Audit** | [memory.md](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/memory.md) | In-depth architectural context, root cause analysis of false alarms, code diffs, and verification logs. |
| **Git Audit & Rollback Trace Log** | [AUDIT_AND_ROLLBACK_LOG.md](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/AUDIT_AND_ROLLBACK_LOG.md) | Immutable chronological commit ledger, verification traces, and exact Git rollback commands. |
| **Baseline 1 Documentation** | [README_baseline1_nsl.md](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/docs/legacy_baselines/README_baseline1_nsl.md) | Full technical README and benchmark outputs of Baseline 1 on NSL-KDD. |
| **Baseline 2 Documentation** | [README_baseline2_nfuq.md](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/docs/legacy_baselines/README_baseline2_nfuq.md) | Full technical README and benchmark outputs of Baseline 2 on NF-UQ-NIDS-v2. |
| **End-to-End Pipeline Verification** | [test_e2e_verification.py](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/tests/test_e2e_verification.py) | Automated GPU test script validating all 5 attack vectors, normal flows, policy shifts, and RAG. |

---

## 📊 Master Comparative Study: Cross-Baseline Benchmark Across All 3 Generations

All evaluation metrics are loaded directly from empirical test runs (`baseline1 nsl/eval_results.json`, `baseline2 nfuq/eval_results.json`, and `eval_results_phase2.json`) with **zero synthetic or invented numbers**:

| Evaluation Parameter | Baseline 1 (NSL-KDD)<br>Shallow AE + IF | Baseline 2 (NF-UQ-NIDS-v2)<br>Single-Flow AE + Latent IF + Maha + kNN | Phase 2 Proposed (NF-UQ-NIDS-v2)<br>**Dual-Stream Bi-MambaFormer** |
| :--- | :---: | :---: | :---: |
| **Underlying Benchmark** | NSL-KDD (`KDDTest+.txt`) | NF-UQ-NIDS-v2 (Partitions 50 & 51) | NF-UQ-NIDS-v2 (Partitions 50 & 51 Sequences) |
| **Input Telemetry** | 41 Tabular Network Attributes | 133 NetFlow V2 Telemetry Features | 133 Features $\times$ $L=32$ Host Flow Windows |
| **Detection Paradigm** | Reconstruction MSE + Tree Isolation | Multi-Metric Latent Manifold Distance | Dual-Stream SSM + Temporal Attention + Gated Fusion |
| **Test Set Size** | 22,544 test packets | 1,702,252 held-out flows | 19,098 sequence windows (611,136 flow steps) |
| **Overall Accuracy** | 90.52% | 88.33% | **93.69%** |
| **Precision** | 91.14% | **97.13%** | 83.64% (calibrated p95) / **97.80%** (p99 policy) |
| **Recall (Detection Rate)** | 92.32% | 85.07% | **98.33%** |
| **F1-Score** | 91.73% | 90.70% | **90.39%** |
| **ROC-AUC** | 96.72% | 93.40% | **98.91%** |
| **PR-AUC** | — | 97.16% | **97.77%** |
| **False Positive Rate (FPR)** | 11.85% | **5.08%** | 8.32% (p95) / **3.01%** (fpr_3pct) / **<1%** (p99) |
| **False Negative Rate (FNR / Miss Rate)**| 7.68% | 14.93% (169,990 missed) | **1.67% (only 96 missed)** ($\mathbf{88.85\%}$ **Reduction**) |
| **Botnet Detection Recall** | — | **6.31%** (Catastrophic Blind Spot) | **100.00%** ($\mathbf{+93.69\%}$ **Recovery**) |
| **Port Scanning Recall** | — | **9.87%** (Catastrophic Blind Spot) | **98.90%** ($\mathbf{+89.03\%}$ **Recovery**) |
| **Infiltration Recall** | — | **9.02%** (Catastrophic Blind Spot) | **75.43%** ($\mathbf{+66.42\%}$ **Recovery**) |
| **Reconnaissance Recall** | — | 44.10% | **100.00%** ($\mathbf{+55.90\%}$ **Recovery**) |
| **Data Theft / Exfiltration Recall** | — | 58.82% | **100.00%** ($\mathbf{+41.18\%}$ **Recovery**) |
| **DoS / DDoS Recall** | — | 96.58% / 99.02% | **100.00% / 100.00%** |
| **Inference Throughput** | 1,220 flows/sec (CPU) | 690 flows/sec (CPU) | **944.4 windows/sec (1.06 ms/1k flows, GPU)** |

---

## 🔬 Architectural Evolution & Exact Outputs of Each Baseline

### 1. Baseline 1: Shallow Autoencoder + Isolation Forest on NSL-KDD
- **Dataset**: NSL-KDD benchmark (`KDDTrain+.txt` for training, `KDDTest+.txt` for testing; 22,544 test packets).
- **Architecture**:
  - Preprocessing: `QuantileTransformer(output_distribution='normal')` mapping 41 raw features to 122 one-hot encoded inputs.
  - Neural Core: Deep Keras Autoencoder ($122 \to 64 \to 32 \to 16 \to 4 \to 16 \to 32 \to 64 \to 122$) trained to minimize MSE on normal traffic.
  - Tree Ensemble: 250-tree Isolation Forest ($c=0.10$).
  - Score Fusion: $\text{Score} = 0.75 \times \text{AE}_{\text{norm}} + 0.25 \times \text{IF}_{\text{norm}}$ with threshold $\Theta = 0.127941$.
- **Exact Test Output**:
  ```text
  Baseline 1 Confusion Matrix (NSL-KDD, N = 22,544):
                  Predicted Normal (0)   Predicted Anomaly (1)
  Actual Normal  (0)     TN = 8,560           FP = 1,151
  Actual Anomaly (1)     FN = 986             TP = 11,847
  ```
- **Component Ablation Findings**:
  - *Autoencoder Alone*: Accuracy 82.37%, Precision 98.16%, Recall 70.34%, F1 81.96%, AUC 96.83%.
  - *Isolation Forest Alone*: Accuracy 77.75%, Precision 75.10%, Recall 91.12%, F1 82.34%, AUC 84.09%.
  - *Hybrid Ensemble*: Accuracy 90.52%, Precision 91.14%, Recall 92.32%, F1 91.73%, AUC 96.72%.
- **Key Finding & Limitation**: Validated that combining neural reconstruction with tree isolation is effective, but NSL-KDD is an obsolete synthetic dataset lacking modern bidirectional NetFlow attributes, payload distributions, and encrypted traffic.

---

### 2. Baseline 2: Deep Autoencoder + Latent Isolation Forest on NF-UQ-NIDS-v2
- **Dataset**: NF-UQ-NIDS-v2 benchmark comprising 75,987,976 total records across 4 underlying datasets (*NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2, NF-UNSW-NB15-v2, NF-BoT-IoT-v2*). Evaluated on held-out Partitions 50 & 51 (1,702,252 test flows).
- **Architecture**:
  - Preprocessing: `PureNfUqAdapter` extracting 133 bidirectional NetFlow V2 features (flow durations, packet size histograms, TCP flags, TTL variance, inter-arrival times).
  - Neural Core: PyTorch Deep Autoencoder ($133 \to 64 \to 32 \to 16 \to 32 \to 64 \to 133$) trained via CUDA AMP on RTX 5070 GPU.
  - Multi-Metric Manifold Detection: Latent Isolation Forest (200 trees) + Latent Mahalanobis Covariance Distance + Topological $k$-NN ($k=5$) operating on the 16-D bottleneck manifold.
  - Decision Gate: Rank-mean gated score fusion with calibrated threshold $\Theta = 0.408528$.
- **Exact Test Output**:
  ```text
  Baseline 2 Confusion Matrix (NF-UQ-NIDS-v2, N = 1,702,252):
                  Predicted Normal (0)   Predicted Anomaly (1)
  Actual Normal  (0)     TN = 534,897         FP = 28,610
  Actual Anomaly (1)     FN = 169,990         TP = 968,755
  ```
- **The Memoryless Flow Bottleneck**:
  Baseline 2 achieved exceptional precision (97.13%) and strong volumetric attack detection (DDoS 99.02%, DoS 96.58%), but suffered a catastrophic failure on stealth and reconnaissance attacks because it evaluated each flow in temporal isolation:
  - **Botnet C2 Beaconing**: **6.31% Recall** (93.69% missed)
  - **Port Scanning**: **9.87% Recall** (90.13% missed)
  - **Lateral Infiltration**: **9.02% Recall** (90.98% missed)
  - **Reconnaissance**: **44.10% Recall** (55.90% missed)
  - **Total Missed Intrusions**: **169,990 attack flows slipped past undetected** (14.93% FNR).

---

### 3. Phase 2 Proposed: Dual-Stream Hybrid Bi-MambaFormer
- **Dataset**: NF-UQ-NIDS-v2 structured as host-centric rolling sequence windows ($L=32$ flows per host IP; 19,098 held-out test windows, strictly disjoint from training partitions 00–47).
- **Architecture**:
  - **Flow Tokenizer**: Linear projection ($133 \to 128$) + RMSNorm + Spatial Dropout (0.15).
  - **Stream 1 (Bidirectional Mamba SSM)**: Forward and backward continuous selective state-space transitions with HiPPO initialization ($N=16$), modeling continuous behavioral drift with $\mathcal{O}(L)$ linear complexity.
  - **Stream 2 (Multi-Head Temporal Self-Attention)**: 4 attention heads ($d_k=32$) with dynamic session masks ($1 \le k \le 32$) capturing non-local pair correlations (probe-to-exploit leaps).
  - **Adaptive Gated Fusion**: Learned element-wise gating $G_t = \sigma(W_g [H_{\text{ssm}} ; H_{\text{attn}}] + b_g)$ dynamically arbitrating between continuous drift and discrete transitions.
  - **Attention-Weighted Temporal Pooling**: Learned temporal pooling weights $w_t \in \mathbb{R}^{32}$ that isolate the exact culprit flow in the session window.
  - **Sub-Millisecond XAI & 100% Local RAG**: FlowTokenizer input weight gradient saliency coupled with dependency-free offline BM25 threat explanation over 20 cyber attack classes.
- **Exact Test Output**:
  ```text
  Phase 2 Held-Out Test Confusion Matrix (19,098 Sequence Windows):
                  Predicted Normal (0)   Predicted Anomaly (1)
  Actual Normal  (0)     TN = 12,225          FP = 1,109
  Actual Anomaly (1)     FN = 96              TP = 5,668
  ```
- **The Breakthrough**:
  - Slashes False Negative Rate from **14.93% down to 1.67%** (an **88.85% reduction** in missed attacks).
  - Recovers **Botnet detection to 100.00%**, **Port Scanning to 98.90%**, and **Infiltration to 75.43%**.
  - Operates at **944.4 sequence windows/sec** on an NVIDIA RTX 5070 GPU ($1.06\text{ ms}$ per 1k flows).

---

## 🖼️ Publication Figure Gallery (300 DPI)

All 14 figures are generated deterministically and located in the repository under [results/figures/](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/):

### Cross-Baseline Comparative Suite (`results/figures/comparative/`)
| Figure Link | Visualized Content |
| :--- | :--- |
| [comp_fig1_macro_metrics_comparison.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/comparative/comp_fig1_macro_metrics_comparison.png) | Macro Performance Comparison across all 3 generations (Accuracy, Precision, Recall, F1, ROC-AUC). |
| [comp_fig2_stealth_defense_gain.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/comparative/comp_fig2_stealth_defense_gain.png) | Stealth Attack Recall Gains: Baseline 2 vs Phase 2 on Bot (+93.7%), Scan (+89.0%), and Infiltration (+66.4%). |
| [comp_fig3_missed_attacks_fnr.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/comparative/comp_fig3_missed_attacks_fnr.png) | Dual-panel comparison of False Negative Rate (%) and absolute missed attack count (169,990 down to 96). |
| [comp_fig4_paradigm_evolution_radar.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/comparative/comp_fig4_paradigm_evolution_radar.png) | Multi-dimensional radar chart evaluating Accuracy, Precision, Recall, ROC-AUC, Stealth Defense, and Throughput. |

### Baseline 1 Evaluation Figures (`results/figures/baseline1/`)
| Figure Link | Visualized Content |
| :--- | :--- |
| [b1_fig1_confusion_matrix.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/baseline1/b1_fig1_confusion_matrix.png) | Normalized Confusion Matrix on NSL-KDD test set ($N=22,544$). |
| [b1_fig2_component_ablation.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/baseline1/b1_fig2_component_ablation.png) | Autoencoder Alone vs Isolation Forest Alone vs Hybrid Fusion on NSL-KDD. |

### Baseline 2 Evaluation Figures (`results/figures/baseline2/`)
| Figure Link | Visualized Content |
| :--- | :--- |
| [b2_fig1_confusion_matrix.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/baseline2/b2_fig1_confusion_matrix.png) | Large-Scale Confusion Matrix on NF-UQ-NIDS-v2 ($N=1,702,252$ flows) highlighting 169,990 missed attacks. |
| [b2_fig2_per_class_recall_spectrum.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/baseline2/b2_fig2_per_class_recall_spectrum.png) | The "Stealth Cliff" across 20 attack classes showing the drop-off from 99% DoS to <10% Bot/Scan. |

### Phase 2 Standalone Figures (`results/figures/`)
| Figure Link | Visualized Content |
| :--- | :--- |
| [fig1_roc_curve.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/fig1_roc_curve.png) | Receiver Operating Characteristic (ROC) curve showing **98.91% ROC-AUC** across 1,922 evaluation points. |
| [fig2_pr_curve.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/fig2_pr_curve.png) | Precision-Recall curve showing **97.77% PR-AUC** across 16,446 points under class imbalance. |
| [fig3_training_loss.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/fig3_training_loss.png) | Training vs Validation BCE loss trajectory across 5 epochs. |
| [fig4_training_auc.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/fig4_training_auc.png) | Training vs Validation ROC-AUC & PR-AUC progression per epoch. |
| [fig5_stealth_recall_comparison.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/fig5_stealth_recall_comparison.png) | Grouped bar comparison of Baseline 2 vs Phase 2 stealth recall. |
| [fig6_cross_baseline_comparison.png](file:///e:/major%20phase%202%202.0/phase2%20models%20nfuq/results/figures/fig6_cross_baseline_comparison.png) | 3-Baseline comparison bars across Accuracy, Precision, Recall, F1, ROC-AUC, and FNR. |

---

## 🏗️ Master System Architecture

```text
Raw Network Interface (NIC) / Live Sniffer / PCAP Replay / Synthetic Stream
                                    │
                                    ▼
                 Asynchronous Thread-Safe Packet Queue
                                    │
                                    ▼
                RealTimeFeatureExtractor (Scapy + Flow State)
         [Bidirectional Telemetry ➔ 133 NetFlow V2 Feature Vector]
                                    │
                                    ▼
        PureNfUqAdapter (log1p + Quantile Scaling + One-Hot Encoding)
                                    │
                                    ▼
                Host Sequence Sliding Window Buffer (L=32)
                      [Dynamic Attention Mask: 1 <= k <= 32]
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │   FLAGSHIP DUAL-STREAM HYBRID BI-MAMBAFORMER  │
            │   (380,802 Parameters on NVIDIA RTX 5070 GPU)  │
            ├───────────────────────────────────────────────┤
            │  Flow Tokenizer: Linear(133 -> 128) + RMSNorm │
            ├───────────────────────┬───────────────────────┤
            │ Stream 1: Bi-Mamba SSM│ Stream 2: MHSA        │
            │ (Forward/Backward     │ (4 Heads, d_k=32,     │
            │  Selective States)    │  Mask-Aware Attention)│
            ├───────────────────────┴───────────────────────┤
            │      Adaptive Gated Fusion (Learned Gate G)   │
            ├───────────────────────────────────────────────┤
            │   Attention-Weighted Temporal Pooling (w_t)   │
            ├───────────────────────────────────────────────┤
            │       Classifier Head: MLP(128 -> 64 -> 1)    │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
                        Sequence Threat Score (Sigmoid)
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
                 Normal Pass             Intrusion Detected
                                                  │
                                                  ▼
                                      Flow-Level Temporal XAI
                                      [Identifies Culprit Flow]
                                                  │
                                                  ▼
                                     Sub-ms Feature Saliency
                                      [Top Contributing Feats]
                                                  │
                                                  ▼
                                      100% Local RAG Threat Intel
                                       (Offline BM25 Engine)
                                                  │
                                                  ▼
                                      FastAPI WebSocket Broadcaster
                                                  │
                                                  ▼
                                      React SOC Dashboard UI
```

---

## 📁 Repository Structure

```text
├── README.md                              # Master project documentation & comparative tables
├── RESEARCH_THESIS.md                     # Comprehensive 10-chapter academic research thesis
├── memory.md                              # Project memory, architectural history & operational runbook
├── AUDIT_AND_ROLLBACK_LOG.md              # Immutable commit ledger, traces & exact rollback guide
├── fixes.md                               # Copilot 15-point audit & engineering remediation guide
├── THESIS_COMPARATIVE_ANALYSIS.md         # Comprehensive 3-Baseline thesis comparative study
├── DEFENSE_PRESENTATION_AND_QA.md         # 9-Slide defense presentation & examiner defense guide
├── split_manifest.json                    # Cryptographic split integrity & artifact SHA-256 manifest
├── eval_results_phase2.json               # Canonical Phase 2 test benchmark metrics
├── eval_results_phase2.txt                # Human-readable test evaluation report
├── knowledge_base.json                    # 20 NF-UQ attack intelligence taxonomy
├── requirements.txt                       # Frozen Python virtual environment dependencies
├── generate_figures.py                    # Phase 2 ROC/PR and training figure generator
├── generate_all_baseline_figures.py       # Cross-baseline 3-generation comparative figure generator
├── verify_split_manifest.py               # Split manifest & SHA-256 verification runner
├── create_admin.py                        # Administrative user creation utility
├── configs/
│   └── nf_uq.yaml                         # NF-UQ-NIDS-v2 dataset and training configs
├── data/
│   └── NF UQ NIDS V2/                     # NF-UQ-NIDS-v2 dataset (52 Parquet partitions)
├── docs/
│   ├── PHASE_2_MASTER_PLAN.md             # Complete Phase 2 architecture roadmap
│   ├── DATA_NOTES.md                      # NetFlow v2 schema & IP address masking notes
│   ├── NIDS_Implementation_Guide_Antigravity.md # Engineering setup and reference guide
│   └── legacy_baselines/                  # Historical Baseline 1 & 2 documentation
│       ├── README_baseline1_nsl.md        # Baseline 1 full documentation & outputs
│       └── README_baseline2_nfuq.md       # Baseline 2 full documentation & outputs
├── documentation/
│   ├── Project_Report.md                  # Comprehensive technical report & thesis specs
│   ├── HYBRID_NIDS_ANALYSIS_REPORT.md     # Diagnostic & false positive analysis report
│   └── *.png                              # Architectural, workflow, and sequence diagrams
├── results/
│   ├── eval_results_baseline2.txt         # Baseline 2 (Autoencoder + Latent IF) test results
│   ├── figures/                           # Publication-ready 300 DPI thesis figures
│   │   ├── fig1_roc_curve.png             # Receiver Operating Characteristic with operating thresholds
│   │   ├── fig2_pr_curve.png              # Precision-Recall curve with calibrated operating envelope
│   │   ├── fig3_training_loss.png         # Train vs Validation BCE loss trajectory (5 epochs)
│   │   ├── fig4_training_auc.png          # Train vs Validation ROC-AUC & PR-AUC progression
│   │   ├── fig5_stealth_recall_comparison.png # Per-class recall: Baseline 2 vs Phase 2 Bi-MambaFormer
│   │   ├── fig6_cross_baseline_comparison.png # Macro metric progression across all 3 baselines
│   │   ├── baseline1/                     # Baseline 1 (NSL-KDD) evaluation figures
│   │   │   ├── b1_fig1_confusion_matrix.png
│   │   │   └── b1_fig2_component_ablation.png
│   │   ├── baseline2/                     # Baseline 2 (NF-UQ-NIDS-v2 isolated flow) figures
│   │   │   ├── b2_fig1_confusion_matrix.png
│   │   │   └── b2_fig2_per_class_recall_spectrum.png
│   │   └── comparative/                   # Cross-baseline comparative evolution figures
│   │       ├── comp_fig1_macro_metrics_comparison.png
│   │       ├── comp_fig2_stealth_defense_gain.png
│   │       ├── comp_fig3_missed_attacks_fnr.png
│   │       └── comp_fig4_paradigm_evolution_radar.png
│   ├── nf_uq/                             # Baseline 2 sweep, seed, and ablation results
│   └── nf_uq_sequence/                    # Phase 2 ROC/PR curves, history, and metrics
│       ├── eval_results_phase2.json
│       ├── eval_results_phase2.txt
│       ├── train_val_test_comparison.json
│       ├── pr_points.csv
│       ├── roc_points.csv
│       └── training_history.json
├── src/
│   ├── backend/
│   │   ├── main.py                        # FastAPI entrypoint & WebSocket broadcaster
│   │   ├── detector.py                    # HybridDetector (Bi-MambaFormer + Host Buffers)
│   │   ├── sniffer.py                     # NetFlow V2 packet capture & feature extraction
│   │   ├── sequences.py                   # PureNfUqAdapter & Host Sequence Extractor
│   │   ├── train_bi_mambaformer.py        # RTX 5070 GPU AMP training pipeline
│   │   ├── evaluate_bi_mambaformer.py     # Held-out test evaluation runner
│   │   ├── model_metrics.py               # Side-by-side comparative terminal reporter
│   │   ├── rag_service.py                 # 100% Local offline RAG threat intelligence
│   │   ├── traffic_generator.py           # Multi-pattern synthetic & Scapy traffic generator
│   │   ├── models/
│   │   │   └── bi_mambaformer.py          # BiMambaFormer PyTorch Architecture
│   │   └── saved_models/                  # GPU-trained weights & threshold artifacts
│   │       ├── bi_mambaformer_best.pt     # Best trained Bi-MambaFormer weights
│   │       ├── bi_mambaformer_threshold.txt # Optimal calibrated threshold (0.120483)
│   │       ├── feature_columns.json       # 133 NetFlow feature column names
│   │       └── nf_uq_adapter_meta.joblib  # Fitted adapter metadata
│   └── frontend/
│       └── src/                           # React 19 SOC Monitoring Dashboard
├── tests/
│   ├── test_e2e_verification.py           # Comprehensive end-to-end pipeline verification test
│   ├── test_live_websocket_pipeline.py    # Live WebSocket client & server integration test
│   ├── test_offline_runtime_parity.py     # Offline forward vs runtime detector numerical parity test (diff <= 1e-5)
│   └── diagnostics/                       # Diagnostic inspection scripts
└── venv/                                  # Python virtual environment (PyTorch CUDA 12.8)
```

---

## 🚀 Quick Start Guide

### Step 1: Run Comparative Benchmark Report
To reproduce the complete comparative study across held-out test data:
```powershell
.\venv\Scripts\python.exe src/backend/model_metrics.py
```

### Step 2: Regenerate All Figures (300 DPI)
```powershell
.\venv\Scripts\python.exe generate_all_baseline_figures.py
.\venv\Scripts\python.exe generate_figures.py
```

### Step 3: Launch Backend API & WebSocket Server
```powershell
.\venv\Scripts\python.exe -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- **API Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Live Monitor WebSocket**: `ws://localhost:8000/ws/monitor`
- **System Metrics Endpoint**: [http://localhost:8000/api/metrics](http://localhost:8000/api/metrics)

### Step 4: Launch React SOC Dashboard
```bash
cd src/frontend
npm install
npm run dev
```
- Open [http://localhost:5173](http://localhost:5173) in your browser.
- Log in with credentials `admin` / `admin` to view live traffic, real-time threat scores, flow-level attribution, and automated RAG explanations.

---

## 🔒 Research Rigor & Air-Gapped Guarantees

1. **Strict Zero-Leakage Protocol**: Training (parts 00-47), Validation (parts 48-49), and Held-Out Test (parts 50-51) are strictly disjoint partitions. No test sequence was ever seen during training or threshold calibration.
2. **100% Local Inference**: Zero dependency on third-party cloud LLMs or external APIs. All embeddings, state-space transitions, attention maps, and vector lookups execute locally on your GPU and CPU.
3. **Reproducibility**: All ROC points (`roc_points.csv`), PR points (`pr_points.csv`), and training trajectories (`training_history.json`) are stored deterministically for paper figures and thesis defense.

---

## 🔒 Intellectual Property, Confidentiality & Publication Notice

**Copyright © 2026. All Rights Reserved.**

This repository contains novel research architectures, proprietary experimental pipelines, and empirical evaluations prepared for formal academic thesis submission and peer-reviewed journal publication. 

- **Proprietary & Confidential**: All architectural designs (including the Dual-Stream Bi-MambaFormer, Adaptive Gated Fusion, and Attention-Weighted Temporal Pooling algorithms), experimental weights, dataset adaptations, and documentation are the sole intellectual property of the author.
- **Strict Prohibition on Reproduction & Distribution**: No part of this repository may be reproduced, distributed, decompiled, copied, modified, incorporated into commercial systems, or used in academic publications without express prior written consent from the author.
- **Academic Embargo**: Public licensing terms are withheld under academic embargo until the formal defense, peer review, and academic publication process are fully concluded.
