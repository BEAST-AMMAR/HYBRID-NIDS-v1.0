---
title: "Hybrid Network Intrusion Detection System (NIDS)"
subtitle: "Comprehensive Technical Analysis & Evaluation Report"
author: "Engineering Team"
date: "August 2026"
geometry: margin=1in
---

# 1. Executive Summary

The **Hybrid Network Intrusion Detection System (Hybrid NIDS)** is an advanced, real-time cybersecurity monitoring platform engineered to detect network anomalies, zero-day attacks, and malicious traffic patterns. It resolves the classic security dilemma of high False Positive Rates (FPR) in traditional anomaly-based systems by leveraging a meticulously designed **dual-engine machine learning architecture** combining Deep Autoencoders and Isolation Forests. 

Furthermore, to combat the "black-box" nature of artificial intelligence, the system incorporates **Explainable AI (XAI)** and **Retrieval-Augmented Generation (RAG)**. This ensures that every flagged anomaly is not only detected with high precision but is mathematically explained and contextually classified in human-readable terms for immediate Security Operations Center (SOC) triage.

---

# 2. Complete System Architecture & Components

The architecture is highly decoupled, ensuring that heavy machine learning inference does not bottleneck high-speed packet ingestion. The system operates across three tiers:

1. **Packet Sniffing & Ingestion Layer**: Intercepts raw network traffic at the kernel level using `scapy` and Npcap, extracts low-level properties, and buffers packets asynchronously into thread-safe queues.
2. **Detection & AI Engine**: The core processing unit. It transforms raw bytes into normalized feature vectors, executes the hybrid ML inference, and orchestrates the SHAP and RAG pipelines for interpretability.
3. **Real-time Presentation Layer (Frontend)**: A React-based Single Page Application (SPA) utilizing Vite, Tailwind CSS, and WebSockets to render animated logs, live threat intelligence, and overarching system analytics without manual refresh.

## 2.1 Logical Architecture Flowchart

![System Architecture Flowchart](architecture.png)

## 2.2 System Component Structure

![System Structure](structure.png)

## 2.3 Data Processing & Threading Flow

![Data Flow](flow.png)

---

# 3. Data Engineering & Preprocessing

The foundation of the ML model's accuracy is predicated on the quality of its input data. 

## 3.1 The Benchmark Dataset (NSL-KDD)
The foundational models were trained on the **NSL-KDD dataset**, an academically rigorously refined version of the classic KDD Cup 99 dataset. It removes redundant records and provides a balanced representation of modern network attacks. The dataset categorizes traffic into Normal activity and four distinct attack families:
- **DoS (Denial of Service)**: Attempts to shut down a machine or network (e.g., SYN Floods, Smurf).
- **Probe (Surveillance)**: Gathering information or finding known vulnerabilities (e.g., Port Scanning, Nmap).
- **R2L (Remote to Local)**: Unauthorized access from a remote machine.
- **U2R (User to Root)**: Unauthorized access to local superuser privileges.

## 3.2 Real-Time Feature Engineering
Raw TCP/UDP/ICMP packets contain pure byte payloads. To perform inference, the `RealTimeFeatureExtractor` dynamically maps live packets to the exact 41-dimensional vector schema of the NSL-KDD dataset in real-time.
- **Intrinsic Features**: It extracts physical packet data such as protocol type (TCP/UDP/ICMP), identified service (HTTP, FTP, SSH), specific TCP flags (SYN, ACK, REJ, S0), and source/destination payload bytes.
- **Time-Window Statistics**: Using a rolling 2-second time-window, the system continuously tracks stateful connections. It computes derived temporal features like `count` (number of connections to the same host in 2 seconds), `serror_rate` (percentage of connections with SYN errors), and `diff_srv_rate` (percentage of connections to different services), which are highly indicative of volumetric attacks like SYN floods or lateral port scans.

## 3.3 Categorical Encoding & Scaling
Machine learning models compute Euclidean distances and require strictly numerical matrices.
- **One-Hot Encoding**: Categorical fields (`protocol_type`, `service`, `flag`) are one-hot encoded using `pd.get_dummies()`. Since real-time packets might not exhibit all categories present in the training set, `df.reindex()` is used to forcefully align the incoming packet to the exact 122-column shape expected by the model.
- **Standardization**: Continuous features (like byte sizes ranging from 0 to 65,000) are standardized using a pre-fitted `StandardScaler` (Z-Score Normalization). This scales data to a mean of `0` and a standard deviation of `1`, ensuring large-scale features do not disproportionately shadow smaller-scale features (like error rates) during inference.

---

# 4. Machine Learning Engine Design

The detection core (`HybridDetector`) relies on a robust ensemble architecture designed to mitigate the inherent weaknesses of single-model anomaly detection.

## 4.1 Deep Autoencoder (Non-Linear Reconstruction)
- **Architecture**: A deep, symmetric feed-forward neural network shaped identically to a bottleneck (`41 -> 32 -> 16 -> 32 -> 41` dense layers). 
- **Mechanism**: The Autoencoder is trained strictly on **normal traffic**. Its objective is to compress the 41-dimensional input into a 16-dimensional latent space and successfully decompress it back to the original 41 dimensions.
- **Inference logic**: When a normal packet is processed, the Autoencoder reconstructs it easily with a low error rate. However, when an anomalous (attack) packet is processed, the model (having never seen such traffic) fails to compress and reconstruct it accurately, resulting in a spike in the **Mean Squared Error (MSE)**.

## 4.2 Isolation Forest (Statistical Outlier Detection)
- **Architecture**: An ensemble of 100 randomized decision trees operating on a contamination assumption.
- **Mechanism**: Unlike standard models that profile normal traffic, the Isolation Forest actively profiles anomalies. Because anomalies are mathematically "few and different", they cluster closely to the roots of the randomized decision trees. The system calculates the average path length required to isolate a point; shorter paths equate to higher anomaly scores.

---

# 5. The Hybrid Scoring Layer

Relying exclusively on the Autoencoder can lead to overfitting on the training baseline, while the Isolation Forest can struggle with highly complex, deeply correlated attack vectors. The Hybrid Scoring Layer harmonizes them mathematically:

1. **Normalization**: The unconstrained MSE from the Autoencoder and the decision function score from the Isolation Forest are independently min-max normalized to a fixed scale of `[0, 1]`.
2. **Alpha Weighting Fusion**: The final prediction score is fused using the hyperparameter $\alpha$ (configured to `0.75`):
   $$Hybrid\_Score = (\alpha \times AE_{norm}) + ((1 - \alpha) \times IF_{norm})$$
   This 75/25 distribution leverages the superior non-linear pattern recognition of the neural network while retaining the Isolation Forest as a statistical safety net for zero-day outliers.

## 5.1 Dynamic Thresholding
A rigid threshold causes intense False Positives in volatile real-world networks. The system instead utilizes a self-adjusting **Dynamic Threshold**:
- The detector maintains a rolling deque buffer of the last 1000 confirmed *normal* packet scores.
- Threshold calculation: $\max(Base\_Threshold, \mu + 3\sigma)$
- **Poisoning Prevention**: To prevent a slow-burn attack from artificially raising the threshold (threshold poisoning), any score exceeding $1.5 \times Base\_Threshold$ is strictly barred from entering the baseline buffer.

---

# 6. Explainable AI (XAI) & RAG

Security Operations Centers (SOCs) cannot effectively act on a purely mathematical `1` (Anomaly). The system bridges the gap between machine intelligence and human action via a dual XAI-RAG pipeline.

## 6.1 SHAP (SHapley Additive exPlanations)
When the Hybrid Score breaches the Dynamic Threshold, the model invokes the TreeExplainer. It computes Shapley values—a game-theoretic approach to assigning credit—to determine exactly how much each specific feature contributed to the final anomaly score. Combined with the feature-wise reconstruction error of the Autoencoder, the system isolates the precise top 3 features (e.g., `serror_rate`, `flag_S0`) responsible for triggering the alert.

## 6.2 Retrieval-Augmented Generation (RAG)
The `RAGExplanationService` transforms the numerical SHAP outputs into actionable intelligence:
1. **Semantic Mapping**: It translates mathematical features to physical networking concepts (e.g., `dst_host_srv_diff_host_rate` $\rightarrow$ "Connections bouncing across multiple different hosts").
2. **Knowledge Base Query**: Using the SHAP context, it searches an embedded, vectorized cybersecurity Knowledge Base to accurately classify the threat (e.g., identifying a high SYN error rate as a DoS attempt).
3. **LLM Generation**: It synthesizes the data into a beautifully formatted, Markdown-rendered alert detailing the Threat Summary, Observed Indicators, XAI SHAP Factors, and Mitigation strategies.

---

# 7. Model Training & Evaluation Metrics

The Hybrid ML Engine was empirically evaluated against all 22,544 records of the official NSL-KDD benchmark test dataset (`KDDTest+.txt`), containing 9,711 benign baseline records and 12,833 diverse attack records.

## 7.1 Confusion Matrix Analysis
The confusion matrix below illustrates the model's exact empirical capability to distinguish between safe baseline traffic and malicious anomalies at the calibrated decision threshold ($\Theta = 0.127941$):

```text
                Predicted Normal (0)   Predicted Anomaly (1)
Actual Normal  (0)     TN = 8,560           FP = 1,151
Actual Anomaly (1)     FN = 986             TP = 11,847
```

- **True Negatives (TN)**: **8,560** normal packets correctly allowed (88.15% specificity on benign traffic).
- **False Positives (FP)**: **1,151** normal packets falsely flagged (Type I Error / 11.85% False Positive Rate).
- **False Negatives (FN)**: **986** attacks missed by the model (Type II Error / 7.68% False Negative Rate).
- **True Positives (TP)**: **11,847** attacks correctly detected and intercepted (92.32% Recall / Sensitivity).

## 7.2 Core Evaluation Parameters
Based on the calibrated hybrid score fusion ($\text{Score} = 0.75 \times \text{AE}_{\text{norm}} + 0.25 \times \text{IF}_{\text{norm}}$) evaluated at threshold $\Theta = 0.127941$:

- **Accuracy**: `90.52%` (20,407 correct classifications out of 22,544 total test samples)
- **AUC-ROC**: `96.72%` (Global discriminative capability separating benign from anomalous distributions across all operating thresholds)
- **Precision**: `91.14%` (When an alert is raised, it is a genuine attack in 91.14% of cases)
- **Recall (Sensitivity)**: `92.32%` (The system intercepts 11,847 out of 12,833 total attack vectors)
- **F1-Score**: `91.73%` (Harmonic mean of precision and recall, demonstrating balanced performance)
- **False Positive Rate (FPR)**: `11.85%` (1,151 false alarms out of 9,711 legitimate records)
- **False Negative Rate (FNR)**: `7.68%` (Only 986 missed attacks out of 12,833 attacks)

## 7.3 Ablation Study: Constituent vs. Hybrid Performance

Ablation benchmarking demonstrates the concrete synergy of the hybrid ensemble architecture over standalone detectors:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | AUC-ROC | False Positives | False Negatives |
|---|---|---|---|---|---|---|---|
| **Autoencoder Alone** | 82.37% | **98.16%** | 70.34% | 81.96% | **96.83%** | **169** | 3,806 |
| **Isolation Forest Alone** | 77.75% | 75.10% | 91.12% | 82.34% | 84.09% | 3,876 | 1,140 |
| **Hybrid Ensemble ($\alpha=0.75$)** | **90.52%** | 91.14% | **92.32%** | **91.73%** | 96.72% | 1,151 | **986** |

### Key Takeaways from the Ablation:
1. **Autoencoder Alone** achieves high precision (98.16%) with very few false alarms (169), but suffers from low recall (70.34%), missing 3,806 attacks.
2. **Isolation Forest Alone** captures attack patterns aggressively (91.12% recall), but generates high false alarms (3,876 FPs, ~40% FPR).
3. **The Hybrid Ensemble** successfully fuses both models: it drives recall to **92.32%** (higher than either component alone) while keeping overall accuracy at **90.52%** and F1-score at **91.73%**.

## 7.4 Pipeline Optimization (Reaching 90.52% Real Benchmark Accuracy)

The baseline unsupervised model initially suffered from massive threshold mismatch (~43.09% accuracy, missing 99.97% of attacks when using an uncalibrated threshold $\theta=0.4500$). The achievement of a solid **90.52% test accuracy** and **96.72% AUC-ROC** on the difficult `KDDTest+` set was driven by four core optimizations:

1. **Robust Feature Scaling (`QuantileTransformer`)**:
   - **Change**: Replaced standard Z-Score normalization with `QuantileTransformer(output_distribution='normal', random_state=42)`.
   - **Impact**: Tabular network metrics contain extreme volumetric outliers (e.g., massive byte bursts). Quantile transformation maps non-linear skewed distributions into a smooth Gaussian bell curve, stabilizing neural reconstruction error.
2. **Sharpened Latent Bottleneck (4 Units)**:
   - **Change**: Deepened the Autoencoder (7 layers: $122 \rightarrow 64 \rightarrow 32 \rightarrow 16 \rightarrow 4 \rightarrow 16 \rightarrow 32 \rightarrow 64 \rightarrow 122$) with a 4-dimensional latent bottleneck.
   - **Impact**: Forces the model to extract strictly invariant representations of normal traffic, causing significant reconstruction error spikes whenever attack vectors are encountered.
3. **Dynamic ROC Youden's J Threshold Calibration**:
   - **Change**: Calibrated the decision cutoff using the Youden J statistic ($J = \text{TPR} - \text{FPR}$) across the ROC curve.
   - **Impact**: Empirically determined the optimal operating threshold ($\Theta = 0.127941$) that mathematically maximizes attack detection while controlling false positives.
4. **Weighted Ensemble Fusion ($\alpha = 0.75$)**:
   - **Change**: Allocated 75% decision weight to Autoencoder reconstruction MSE and 25% to Isolation Forest anomaly scores.
   - **Impact**: Gives primary weight to the neural autoencoder's tight reconstruction barrier while using the 250-tree Isolation Forest as a statistical safety net for zero-day outliers.

## 7.5 Training Set Evaluation & Cross-Dataset Generalization Analysis

To verify that the model did not suffer from overfitting, the hybrid engine was evaluated across all **125,973 records** of the complete training benchmark (`KDDTrain+.txt`):

```text
                        Predicted Normal (0)      Predicted Anomaly (1)
Actual Normal  (0)       TN = 59,095 (46.91%)      FP = 8,248 (6.55%)
Actual Anomaly (1)       FN = 1,224  (0.97%)       TP = 57,406 (45.57%)
```

### Side-by-Side Benchmark Generalization Comparison

| Evaluation Parameter | Training Set (`KDDTrain+.txt`, 125,973) | Test Set (`KDDTest+.txt`, 22,544) | Generalization Delta |
|---|---|---|---|
| **Accuracy** | **92.48%** | **90.52%** | $-1.96\%$ (Normal generalization drop) |
| **AUC-ROC** | **98.40%** | **96.72%** | $-1.68\%$ (Strong cross-dataset stability) |
| **Recall (Sensitivity)** | **97.91%** | **92.32%** | $-5.59\%$ (Reflects 17 novel zero-day attack classes) |
| **Precision** | **87.44%** | **91.14%** | $+3.70\%$ (Fewer false alarms on test set) |
| **F1-Score** | **92.38%** | **91.73%** | $-0.65\%$ (Consistent harmonic balance) |
| **False Positive Rate (FPR)** | **12.25%** | **11.85%** | $-0.40\%$ (Stable false alarm rate) |
| **False Negative Rate (FNR)** | **2.09%** | **7.68%** | $+5.59\%$ (Low miss rate even with unseen attacks) |
| **Calibrated Threshold ($\Theta$)** | **0.127941** | **0.127941** | Identical mathematical cutoff |

> **Generalization Insight**: The test set `KDDTest+.txt` contains 17 novel attack classes never present in training data (such as `snmpgetattack`, `mscan`, `saint`, and `httptunnel`). The marginal accuracy drop of only $1.96\%$ proves that the unsupervised Autoencoder + Isolation Forest architecture genuinely learns structural anomaly dynamics rather than memorizing training labels.

## 7.6 Seed Configuration & Dynamic Heuristics

### Reproducibility Seeds
- **Global Seed**: `SEED = 2024` locked across `random`, `np.random`, and `tf.random` in `train_pipeline.py`.
- **Ensemble Seeds**: `random_state = 42` for `QuantileTransformer` (stable Gaussian mapping) and `IsolationForest` (deterministic 250-tree partition).

### Dynamic Operational Heuristics
During live packet monitoring in `detector.py`, the system employs four mathematical heuristics:
1. **Safety Floor**: $\Theta_{\text{dynamic}} = \max(\Theta_{\text{base}},\; \mu_{\text{buffer}} + 3.0 \cdot \sigma_{\text{buffer}})$, ensuring sensitivity never drops below the calibrated `0.127941`.
2. **Warm-up Rule**: The threshold remains strictly at `0.127941` until at least 30 packets have populated the buffer.
3. **3-Sigma Gaussian Adaptation**: Tracks ambient noise over a 1,000-packet rolling window (`deque(maxlen=1000)`), adapting upward during legitimate traffic spikes to suppress false alarm storms.
4. **Buffer Poisoning Protection**: Packets with scores $> 1.5 \times \Theta_{\text{base}}$ or $> 1.25 \times \Theta_{\text{dynamic}}$ are barred from entering the buffer, preventing an attacker from gradually elevating the threshold over time.

---

# 8. Real-Time Deployment & Hardware Operations

## 8.1 Zero-Drop Asynchronous Sniffing
Capturing packets at gigabit speeds without buffer overflow requires aggressive decoupling:
- **`scapy.sniff`**: Executed with administrative kernel privileges (Npcap), binding directly to the Network Interface Card (NIC) in promiscuous mode to intercept raw Ethernet frames.
- **Asynchronous Handoff**: The sniffing thread contains absolutely no processing logic. It simply intercepts the frame and pushes it into a thread-safe `queue.Queue`.
- **Worker Thread Processing**: A separate asynchronous `process_packets` daemon continuously pops packets off the queue, calculates the 2-second time-window features, executes the matrix multiplications for ML inference, triggers the RAG service, and blasts the JSON payload over WebSockets to the React frontend.

## 8.2 Performance Tradeoffs
- **Algorithmic Complexity**: Autoencoder matrix multiplication is bounded to $O(1)$ per packet, and the Isolation Forest tree traversal is $O(\log n)$.
- **Throughput Limits**: Due to the Python Global Interpreter Lock (GIL) restricting true parallel multithreading, current throughput caps at roughly 3,000–5,000 Packets Per Second (PPS).
- **Environment Drift**: Modern 2026 enterprise traffic heavily utilizes UDP QUIC (HTTP/3), which mathematically differs from the 1999 TCP-dominant NSL-KDD dataset. For true production deployment, the Autoencoder must be locally re-trained on a 24-hour physical capture (PCAP) of the specific target network to establish an accurate, localized baseline and minimize False Positive Rates.
