# 🛡️ Phase 2 Master Plan: Dual-Stream Hybrid Bi-MambaFormer for Sequence-Based Network Intrusion Detection

**Project**: Real-Time Explainable Hybrid NIDS  
**Dataset**: NF-UQ-NIDS-v2 ($75,987,976$ NetFlow v2 records across 4 unified benchmarks)  
**Hardware Target**: NVIDIA GeForce RTX 5070 Laptop GPU (8GB VRAM) + 16GB System RAM (CUDA 12.8 / PyTorch)  
**Execution Environment**: 100% Local, Air-Gapped, Zero External API Dependency  
**Branch**: `phase2/sequence-models`  
**Date**: November 2026  

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Phase 1 Baseline & Its Fundamental Ceiling
In Phase 1, we implemented and evaluated an isolated flow anomaly detection baseline (Deep Autoencoder + Latent Isolation Forest) on the full NF-UQ-NIDS-v2 benchmark ($1,702,252$ held-out test flows), reaching **88.33% accuracy**, **97.13% precision**, **93.40% ROC-AUC**, and **5.08% FPR**.

However, per-class recall analysis exposed a structural limitation inherent to all isolated-flow NIDS:

| Attack Category | Phase 1 Isolated-Flow Recall | Threat Character & Root Cause |
| :--- | :---: | :--- |
| **Volumetric Attacks** (DDoS, DoS, Brute Force, Fuzzers) | **96.5% – 100.0%** | Blatant single-packet volumetric deviations (gigabytes, packet bursts, flag anomalies). |
| **Stealth Port Scanning** | **9.87%** *(~90% missed)* | Individual scan probes look structurally identical to standard TCP SYN packets. |
| **Infiltration & C2 Beacons** | **9.02%** *(~91% missed)* | Periodic keep-alive heartbeats mimic ordinary background web/DNS requests. |
| **Botnet Traffic** | **6.31%** *(~93% missed)* | Low-rate query sequences hide within normal statistical noise. |
| **Reconnaissance** | **44.10%** *(~56% missed)* | Horizontal sweeps across IP subnets evade static per-flow anomaly bounds. |

### 1.2 The Phase 2 Vision
**Network intrusions are temporal processes, not static snapshots.** An individual SYN packet to port 445 is normal; 30 SYN packets to 30 sequential ports within 2 seconds is a stealth port scan.

**Phase 2 completely replaces the legacy Phase 1 models (Autoencoder + Isolation Forest) with a state-of-the-art Dual-Stream Hybrid Sequence Model: the Bi-MambaFormer.** By ingesting host-tracked temporal flow windows ($L=32$), the system directly resolves the stealth attack blind spot while preserving sub-millisecond inference speeds on our NVIDIA RTX 5070 GPU.

---

## 2. Complete End-to-End System Architecture

The overall system retains its modular, production-grade layout (Ingestion $\to$ Preprocessing $\to$ ML Engine $\to$ XAI $\to$ RAG $\to$ FastAPI $\to$ React), but upgrades the internal intelligence engine to handle stateful temporal sequences:

```text
====================================================================================================
STAGE 1: ASYNCHRONOUS INGESTION & ROLLING HOST BUFFER
====================================================================================================
Live Network (NIC / Npcap) / Synthetic NetFlow Generator / PCAP Replay
                                │
                                ▼
              Asynchronous Thread-Safe Packet Queue
                                │
                                ▼
           RealTimeFeatureExtractor & Stateful Reassembly
      (Extracts 43 base NetFlow v2 features: bytes, pkts, TTL, TCP flags)
                                │
                                ▼
            [Host-Centric Temporal Sliding Window Buffer]
     • Groups incoming flows by IPV4_SRC_ADDR into chronological deques
     • Window Length: L = 32 flows (configurable L ∈ {16, 32, 64})
     • Dynamic Attention Masking: Supports short sessions (1 ≤ L ≤ 32)
                                │
====================================================================================================
STAGE 2: TELEMETRY NORMALIZATION (133-D Schema)
====================================================================================================
                                │
                                ▼
            NfUqFeatureAdapter (Fitted strictly on benign train)
     • log1p non-linear transformation on heavy-tailed volume & rate metrics
     • QuantileTransformer(output_distribution='normal', n_quantiles=1000)
     • Top-20 L4 Destination Ports + Top-30 L7 Protocols One-Hot Encoded
     • 8-bit TCP Flag Decomposition (FIN, SYN, RST, PSH, ACK, URG, ECE, CWR)
     ➔ Output Tensor: X ∈ ℝ^(Batch × L × 133) with Mask M ∈ {0, 1}^(Batch × L)
                                │
====================================================================================================
STAGE 3: THE DETECTION CORE (Dual-Stream Hybrid Bi-MambaFormer)
====================================================================================================
                                │
                                ▼
                 [Learnable Flow Tokenizer / Projection]
                       Linear(133 ➔ D=128) + RMSNorm
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
        STREAM 1: State-Space         STREAM 2: Relational Attention
     [Bidirectional Mamba (Bi-SSM)] [Multi-Head Temporal Self-Attention]
     • Continuous state-space drift • Discrete pairwise flow correlation
     • Linear O(L) time complexity  • Non-local jump matching
     • Dynamic selective gating     • Flow-to-flow interaction matrix
     • Captures rate drift & scans  • Captures probe-to-exploit links
                 │                             │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 [Adaptive Gated Fusion Mechanism]
                 α = σ(Linear([H_ssm ; H_attn]))
                 H_fused = α ⊙ H_ssm + (1 - α) ⊙ H_attn
                                │
                                ▼
                 [Temporal Pooling & Threat Classifier]
                 Sequence Threat Score S ∈ [0, 1] + Attention Weights
                                │
====================================================================================================
STAGE 4: DECISION & REAL-WORLD ASYNCHRONOUS TRIAGE
====================================================================================================
                                │
                                ▼
                    Calibrated Threshold Policy
       (p95_val_normal = 0.4085 | p99_val_normal = 0.5029 | youden_val = 0.4149)
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
             [NORMAL]                      [THREAT DETECTED]
           Pass Traffic                            │
                                                   ▼
                                     [In-Memory Alert Deduplication]
                                     • Clusters 1,000s of burst alerts into 1 Incident
                                     • Prevents UI and database flood
                                                   │
                                                   ▼
                                     [Native XAI Attribution]
                                     • Temporal Attention Map: flags exact culprit flow t*
                                     • Gradient Saliency: flags top 3 contributing features
                                                   │
                                                   ▼
                                     [100% Local RAG Threat Intel Engine]
                                     • Local Vector Search (all-MiniLM-L6-v2 on disk)
                                     • Local 20-Category Knowledge Base (knowledge_base.json)
                                     • Zero external cloud API calls, zero cost, <2ms lookup
                                                   │
                                                   ▼
                                     [FastAPI WebSocket Broadcast (ws://localhost:8000)]
                                                   │
                                                   ▼
                                     [React 19 Security Operations Center Dashboard]
```

---

## 3. Mathematical Specification of the Dual-Stream Hybrid Bi-MambaFormer

```text
                        Sequence Input: X ∈ ℝ^(B × L × 133)
                                         │
                                         ▼
                            Linear(133 ➔ 128) + RMSNorm
                                         │
                                         ▼
                               H_0 ∈ ℝ^(B × L × 128)
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
                   ▼                                           ▼
      [STREAM 1: Bidirectional Mamba]               [STREAM 2: Multi-Head Attention]
   Forward SSM:                                     Q = H_0 W_Q, K = H_0 W_K, V = H_0 W_V
     h_t^f = A_t h_{t-1}^f + B_t x_t                A = Softmax((Q K^T) / √d_k + Mask)
     y_t^f = C_t h_t^f                              H_attn = A V
   Backward SSM:                                               │
     h_t^b = A_t h_{t+1}^b + B_t x_t                           ▼
     y_t^b = C_t h_t^b                              Dropout + LayerNorm
   H_ssm = Conv1D(Linear([y^f ; y^b]))                         │
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                       [ADAPTIVE GATED FUSION LAYER]
                       G = σ(Linear([H_ssm ; H_attn]))
                       H_fused = G ⊙ H_ssm + (1 - G) ⊙ H_attn
                                         │
                                         ▼
                       [ATTENTION-WEIGHTED TEMPORAL POOLING]
                       w = Softmax(Linear(H_fused))
                       z = ∑_{t=1}^L w_t · H_fused,t
                                         │
                                         ▼
                       [CLASSIFICATION & THREAT SCORING HEAD]
                       MLP(128 ➔ 64 ➔ 1) ➔ Sigmoid ➔ Score S ∈ [0, 1]
```

### 3.1 Flow Tokenizer
Each input flow vector $\mathbf{x}_t \in \mathbb{R}^{133}$ at time step $t \in \{1, \dots, L\}$ is projected into a $D$-dimensional latent token space ($D = 128$):
$$\mathbf{h}_t^{(0)} = \text{RMSNorm}\left(\mathbf{W}_{\text{in}} \mathbf{x}_t + \mathbf{b}_{\text{in}}\right)$$

### 3.2 Stream 1: Bidirectional Selective State Space (Bi-Mamba SSM)
Standard Mamba is strictly causal (unidirectional past $\to$ future). However, an intrusion window $X \in \mathbb{R}^{L \times D}$ is already buffered in memory. Backward temporal dependencies are equally informative (e.g., an abnormal data egress at $t=28$ contextualizes an anomalous credential test at $t=4$).

The continuous-time state-space equation maps input $x(t)$ to latent state $h(t) \in \mathbb{R}^N$ and output $y(t)$:
$$\frac{dh(t)}{dt} = \mathbf{A} h(t) + \mathbf{B} x(t), \quad y(t) = \mathbf{C} h(t)$$

Discretized with time-step parameter $\mathbf{\Delta}_t = \text{Softplus}\left(\text{Linear}(x_t)\right)$:
$$\overline{\mathbf{A}}_t = \exp\left(\mathbf{\Delta}_t \mathbf{A}\right), \quad \overline{\mathbf{B}}_t = \left(\mathbf{\Delta}_t \mathbf{A}\right)^{-1}\left(\exp(\mathbf{\Delta}_t \mathbf{A}) - \mathbf{I}\right) \cdot \left(\mathbf{\Delta}_t \mathbf{B}_t\right) \approx \mathbf{\Delta}_t \mathbf{B}_t$$
$$h_t = \overline{\mathbf{A}}_t h_{t-1} + \overline{\mathbf{B}}_t x_t, \quad y_t = \mathbf{C}_t h_t$$

**Bidirectional Processing**:
1. **Forward Branch**: Scans from $t=1 \to L$, producing $\mathbf{Y}_{\text{fwd}} \in \mathbb{R}^{B \times L \times D}$.
2. **Backward Branch**: Scans from $t=L \to 1$, producing $\mathbf{Y}_{\text{bwd}} \in \mathbb{R}^{B \times L \times D}$.
3. **Mamba Fusion**:
   $$\mathbf{H}_{\text{ssm}} = \text{SiLU}\left(\text{Conv1D}_{k=3}\left(\mathbf{W}_m [\mathbf{Y}_{\text{fwd}} \,\|\, \mathbf{Y}_{\text{bwd}}]\right)\right)$$

*Implementation*: Uses a clean, hardware-friendly PyTorch implementation leveraging associative prefix scans and tensor cores on CUDA 12.8, avoiding brittle Windows C++ compilation.

### 3.3 Stream 2: Multi-Head Temporal Self-Attention (MHSA)
While Stream 1 maintains continuous session memory, Stream 2 explicitly learns non-local pairwise interactions across all flows in the window:
$$\mathbf{Q} = \mathbf{H}^{(0)} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H}^{(0)} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H}^{(0)} \mathbf{W}_V$$
$$\mathbf{A}_{\text{attn}} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{mask}}\right)$$
$$\mathbf{H}_{\text{attn}} = \text{LayerNorm}\left(\mathbf{A}_{\text{attn}} \mathbf{V} + \mathbf{H}^{(0)}\right)$$
- Number of heads: $h = 4$, Head dimension: $d_k = 32$.
- $\mathbf{M}_{\text{mask}}$ masks padded positions for short sequences ($L < 32$).
- $\mathbf{A}_{\text{attn}}$ directly provides native XAI flow attribution without extra computational overhead.

### 3.4 Adaptive Gated Fusion Layer
Rather than simple addition or concatenation, a learned non-linear gate dynamically routes the signal:
$$\mathbf{G} = \sigma\left(\mathbf{W}_g [\mathbf{H}_{\text{ssm}} \,\|\, \mathbf{H}_{\text{attn}}] + \mathbf{b}_g\right) \in [0, 1]^{B \times L \times D}$$
$$\mathbf{H}_{\text{fused}} = \mathbf{G} \odot \mathbf{H}_{\text{ssm}} + (\mathbf{1} - \mathbf{G}) \odot \mathbf{H}_{\text{attn}}$$
- When continuous rate drift dominates (e.g., port scan sweep, DoS ramp), $\mathbf{G} \to 1$ (SSM dominates).
- When discrete pairwise flow correlations dominate (e.g., probe at $t=2 \to$ exploit at $t=28$), $\mathbf{G} \to 0$ (Attention dominates).

### 3.5 Attention-Weighted Pooling & Classification
$$\mathbf{w}_{\text{pool}} = \text{Softmax}\left(\mathbf{W}_p \mathbf{H}_{\text{fused}} + \mathbf{b}_p\right) \in \mathbb{R}^{B \times L \times 1}$$
$$\mathbf{z} = \sum_{t=1}^L w_{\text{pool}, t} \cdot \mathbf{H}_{\text{fused}, t} \in \mathbb{R}^{B \times D}$$
$$\text{Score } S = \sigma\left(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1) + b_2\right) \in [0, 1]$$

---

## 4. Sequence Data Engineering & Pipeline Design

### 4.1 Chronological Host Grouping (`src/backend/sequences.py`)
To prevent data leakage and ensure real temporal dynamics:
1. **Partition Grouping**: Read Parquet partitions sequentially (chronological arrival order).
2. **Host Keys**: Group by `IPV4_SRC_ADDR` (and destination subnet context).
3. **Sliding Windows**:
   - Window size: $L = 32$ flows (ablation benchmarks on $L=16$ and $L=64$).
   - Stride: $S = 16$ (50% overlap during training for data augmentation; non-overlapping $S=32$ for test evaluation).
4. **Dynamic Masking for Short Sessions**:
   - Any host with $1 \le k < 32$ flows is padded to length 32 with zeros, accompanied by a boolean mask vector where $\text{mask}[i] = 1$ for valid flows and $0$ for padding.
   - The model computes attention and SSM transitions only over active steps.
5. **Sequence Target Labeling**:
   - Binary Sequence Label: $Y_{\text{seq}} = 1$ if any flow in the window has `Label == 1`, else $0$.
   - Threat Ratio: $R_{\text{attack}} = \frac{1}{L} \sum_{i=1}^L y_i$ (stored for granular threshold analysis).
   - Dominant Attack Class: Most frequent non-benign category in the window.

### 4.2 Leak-Free Temporal Dataset Partitioning
We maintain the exact, leak-free partition splits established in Phase 1:
- **Training Set (Partitions 00–39)**: ~58.4M flows $\to$ Subsampled to balanced training sequence corpus.
- **Validation Set (Partitions 40–47)**: ~11.7M flows $\to$ Used strictly for early stopping and threshold policy calibration.
- **Held-Out Test Set (Partitions 48–51)**: ~5.8M flows (including benchmark partitions 50 & 51 with 1.70M flows) $\to$ Evaluated exactly once.

---

## 5. Model Ablation & Comparative Experimental Suite

To produce an academically rigorous, publication-ready research paper, we evaluate **4 standardized sequence models** under the exact same data pipeline, seeds (`[11, 22, 33, 44, 55]`), and evaluation protocols:

| Model ID | Architecture | Paradigm | Parameter Count | GPU Memory | Primary Strength |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **M1: BiLSTM-Attn** | 2-Layer BiLSTM ($133 \to 128 \to 64$) + Temporal Attention | Recurrent Baseline | ~280K | ~1.2 GB | Classic sequence reference; cuDNN-accelerated |
| **M2: Transformer** | 3-Layer Pre-LN Transformer Encoder ($d=128, 4\text{ heads}$) | Attention Baseline | ~310K | ~1.4 GB | Pure pairwise relational modeling |
| **M3: Pure Mamba** | 3-Layer Causal Mamba Selective SSM ($d=128, \text{state}=16$) | SSM Baseline | ~260K | ~1.1 GB | Ultra-fast linear temporal scaling |
| **M4: Bi-MambaFormer** | **Dual-Stream Hybrid Bi-Mamba + MHSA + Gated Fusion** | **Proposed Hybrid** | **~385K** | **~1.6 GB** | **Unified continuous drift + discrete jump detection** |

---

## 6. Real-World Production & Zero-API Operational Design

### 6.1 100% Local, Air-Gapped Operation
- **Zero Cloud API Reliance**: No OpenAI, no Anthropic, no Gemini cloud calls.
- **Embedded Threat Knowledge Base**: Pre-compiled `knowledge_base.json` covering all 20 NF-UQ threat families (DDoS, DoS, Port Scans, Reconnaissance, Brute Force, Infiltration, Botnets, Exploits, Worms, etc.).
- **Local Neural Embeddings**: `all-MiniLM-L6-v2` runs locally via `sentence-transformers` on CPU/GPU, executing semantic similarity lookups in $<2\text{ ms}$.
- **Zero Cost, Infinite Queries**: Immune to API rate limits (HTTP 429), subscription costs, or internet outages.

### 6.2 High-Throughput Alert Deduplication
In live network operations, an attack burst (e.g. SYN flood or fast port scan) can generate 5,000 alerts per second:
1. **Burst Ring Buffer**: Alerts are hashed by `(Source IP, Target Port, Attack Class)`.
2. **Incident Aggregation**: If a pattern repeats within a 10-second rolling window, it is grouped into a single unified incident counter.
3. **Clean WebSocket Feed**: The React UI displays the real-time flow rate without freezing the browser event loop.

---

## 7. Target Evaluation Metrics & Deliverables

Every experimental run writes standardized result artifacts to `results/nf_uq_sequence/<model_id>/<seed>/`:
- `metrics.json`: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR, Latency (ms/1k flows), Confusion Matrix.
- `per_class_recall.json`: Granular detection rate across all 20 attack classes, verifying the recall leap on:
  - **Scanning**: Target $\ge 85\%$ (vs. Phase 1: 9.87%)
  - **Infiltration**: Target $\ge 80\%$ (vs. Phase 1: 9.02%)
  - **Botnet**: Target $\ge 80\%$ (vs. Phase 1: 6.31%)
  - **Reconnaissance**: Target $\ge 85\%$ (vs. Phase 1: 44.10%)
- `pr_points.csv` and `roc_points.csv`: High-resolution curve data for paper figures.
- `ablation_table.csv`: Side-by-side comparison of BiLSTM vs. Transformer vs. Mamba vs. Bi-MambaFormer.

---

## 8. Implementation Roadmap (Task Order)

| Step | Component | Files Involved | Milestone Output |
| :---: | :--- | :--- | :--- |
| **S1** | **Sequence Data Engine** | `src/backend/sequences.py`, `configs/nf_uq_sequence.yaml` | Streaming sequence window builder ($L=32$), dynamic padding & PyTorch `DataLoader` |
| **S2** | **Model Architectures** | `src/backend/models/sequence_models.py` | Implementation of BiLSTM, Transformer, Pure Mamba, and **Hybrid Bi-MambaFormer** |
| **S3** | **GPU Training Pipeline** | `src/backend/train_sequence_gpu.py` | RTX 5070 training harness with mixed precision, cosine LR schedule, early stopping |
| **S4** | **Benchmark & Ablation** | `src/backend/evaluate_sequence.py`, `scripts/make_phase2_tables.py` | Full evaluation on held-out test partitions (50 & 51); generate ablation and per-class CSVs |
| **S5** | **Live Detector & UI Hook** | `src/backend/detector.py`, `src/backend/sniffer.py`, `src/backend/main.py` | Update detection engine and WebSocket stream to serve sequence-level alerts to the React dashboard |
| **S6** | **Report & Documentation** | `documentation/Phase2_Technical_Report.md`, `README.md` | Comprehensive Phase 2 technical report and updated repository documentation |

---

*Authored and Approved for Implementation.*  
*Target: State-of-the-art Deep Learning Hybrid NIDS on NF-UQ-NIDS-v2.*
