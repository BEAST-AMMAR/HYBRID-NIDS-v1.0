# 🛡️ Comprehensive Deep-Dive Technical Report & Diagnostic Investigation: Hybrid NIDS

---

## 1. Executive Summary & Core Engineering Philosophy

The **Hybrid ML-based Network Intrusion Detection System (Hybrid NIDS)** is an advanced cyber defense platform designed to perform real-time, stateful packet inspection, unsupervised anomaly detection, feature-level root cause attribution via Explainable AI (XAI), and threat intelligence enrichment via Retrieval-Augmented Generation (RAG).

```
                      ┌─────────────────────────────────────────┐
                      │        Live Network Traffic / Scapy     │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │    Stateful Packet & Window Extractor   │
                      │  (41 Features → 116 One-Hot Dimensions) │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │   Quantile Normalization Transformation │
                      └────────────────────┬────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
     ┌─────────────────────────────┐               ┌─────────────────────────────┐
     │  Deep Autoencoder (Keras)   │               │  Isolation Forest (250 Ts)  │
     │  Reconstruction MSE Error   │               │   Subspace Tree Isolation   │
     └──────────────┬──────────────┘               └──────────────┬──────────────┘
                    │                                             │
                    │ Normalized [0, 1]                           │ Normalized [0, 1]
                    ▼                                             ▼
                    └──────────────┬──────────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────────┐
                    │  Score Fusion: 0.75 × AE + 0.25 × IF    │
                    │  Dynamic 3-Sigma Rolling Auto-Threshold │
                    └────────────────────┬────────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [ Normal Traffic ]              [ Threat Detected ]
                         │                               │
                         ▼                               ▼
                 Update Baseline               ┌─────────────────────────┐
                 Rolling Deque                 │ Dual-Signal XAI (SHAP)  │
                                               └────────────┬────────────┘
                                                            │
                                                            ▼
                                               ┌─────────────────────────┐
                                               │   Cascading RAG Engine  │
                                               │ (Vector KB / MiniLM-L6) │
                                               └────────────┬────────────┘
                                                            │
                                                            ▼
                                               ┌─────────────────────────┐
                                               │  FastAPI WebSocket Hub  │
                                               │  React 19 SOC Dashboard │
                                               └─────────────────────────┘
```

### The Dual-Model Philosophy
- **Deep Autoencoder (AE)**: Trained exclusively on verified normal traffic ($y=0$). It learns a compact 4-dimensional latent manifold of standard packet behaviors. When an attack packet arrives with abnormal payload proportions or invalid flags, the Autoencoder fails to reconstruct it, producing a high Mean Squared Error (MSE).
- **Isolation Forest (IF)**: An ensemble of 250 randomized isolation decision trees. Outliers in high-dimensional feature space require very few orthogonal cuts to isolate, yielding short path lengths and high anomaly scores.
- **Score Fusion ($\alpha = 0.75$)**: Merging the continuous manifold reconstruction error with tree-based isolation scores creates a hybrid signal that balances subtle temporal deviations with sharp structural outliers.

---

## 2. File-by-File Architectural Anatomy

### `src/backend/train_pipeline.py`
- **Role**: Canonical training and model artifact generation.
- **Key Pipeline Steps**:
  1. Ingests `data/KDDTrain+.txt` and `data/KDDTest+.txt`.
  2. Drops the DARPA metadata column `difficulty`.
  3. Binarizes target labels: `normal` $\rightarrow 0$, all 39 attack types $\rightarrow 1$.
  4. One-hot encodes `protocol_type`, `service`, and `flag` using `pd.get_dummies()`, aligning train and test sets to a consistent **116-dimensional feature vector** (`feature_columns.json`).
  5. Fits `QuantileTransformer(output_distribution='normal', random_state=42)` solely on the training matrix and persists it as `standard_scaler.joblib`.
  6. Filters clean normal samples ($y_{\text{train}} = 0$, 67,343 rows) to train the Deep Bottleneck Autoencoder.
  7. Fits the 250-tree Isolation Forest on the scaled training distribution (`contamination=0.10`).
  8. Calibrates two `MinMaxScaler` objects on training MSE and IF scores (`ae_scaler.joblib`, `if_scaler.joblib`).
  9. Evaluates out-of-sample data (`KDDTest+.txt`), computes the ROC curve, and optimizes the base decision cutoff $\Theta$ via **Youden's J Statistic** ($J = \text{TPR} - \text{FPR}$), saving $\Theta \approx 0.1279$ to `threshold.txt`.

### `src/backend/detector.py`
- **Role**: Online detection, scoring, dynamic thresholding, and XAI attribution.
- **`HybridDetector` Class**:
  - Loads serialized weights: `autoencoder.keras`, `iso_forest.joblib`, `standard_scaler.joblib`, `ae_scaler.joblib`, `if_scaler.joblib`, and `threshold.txt`.
  - Instantiates `shap.TreeExplainer` on the Isolation Forest.
  - Implements `predict(scaled_features)`: runs Keras forward pass to calculate MSE, extracts IF decision score, scales both via MinMax to $[0, 1]$, and computes $\text{Score} = 0.75 \cdot \text{AE}_{\text{norm}} + 0.25 \cdot \text{IF}_{\text{norm}}$.
  - Implements adaptive threshold logic via a rolling `deque(maxlen=1000)`:
    $$\Theta_{\text{dynamic}} = \max\left(\Theta_{\text{base}},\, \mu_{\text{buffer}} + 3\sigma_{\text{buffer}}\right)$$
  - Implements `get_top_anomalous_features()`: blends normalized absolute SHAP importances with Autoencoder feature-wise squared reconstruction errors to select the top 3 anomaly drivers.

### `src/backend/sniffer.py`
- **Role**: Raw packet sniffing, stateful session reassembly, rolling-window statistics, and live feature synthesis.
- **`ConnectionState`**:
  - Tracks individual 5-tuple flows: `(src, dst, sport, dport, proto)`.
  - Accumulates `src_bytes` and `dst_bytes`.
  - Stateful flag evaluator: extracts raw TCP flags (`S`, `A`, `F`, `R`) into NSL-KDD states (`SF`, `S0`, `REJ`, `S1`, etc.).
- **`RealTimeFeatureExtractor`**:
  - Maintains `active_connections` (clearing sessions idle $> 60$s).
  - Maintains `host_history` tracking the last 100 connections per destination IP.
  - Computes 2-second rolling temporal window metrics (`count`, `srv_count`, `serror_rate`, `same_srv_rate`, `diff_srv_rate`, `srv_diff_host_rate`).
  - Synthesizes 41 NSL-KDD features, one-hot encodes categoricals, reindexes to the 116 training columns, and applies the `QuantileTransformer`.

### `src/backend/main.py`
- **Role**: FastAPI backend, WebSocket streaming hub, and asynchronous coordinator.
- **Core Operations**:
  - Background asynchronous worker (`process_packets`) drains `packet_queue`, performs feature extraction, executes model prediction, and broadcasts JSON alerts to connected WebSockets (`/ws/monitor`).
  - `/api/capture/mode`: Switches traffic sources between live network sniffing, virtual synthetic injection, physical packet injection, and dataset replay.
  - `/api/upload_dataset`: Streams uploaded `.csv`, `.txt`, or `.pcap` files through the inference pipeline.
  - Auth routes (`/signup`, `/token`, `/login/google`, `/users/me`) backed by Bcrypt, PyJWT, and SQLite (`sql_app.db`).

### `src/backend/rag_service.py`
- **Role**: Contextual cybersecurity threat intelligence generation.
- **`RAGExplanationService`**:
  - Houses 49 attack profiles loaded from `knowledge_base_enhanced.json`.
  - Cascading 3-Tier Classification Engine:
    - *Tier 1*: Strict signature matching (e.g., `land == 1` $\rightarrow$ Land Attack, `flag_s0` or `serror_rate > 0.6` $\rightarrow$ Neptune SYN Flood, ICMP $> 60,000$ bytes $\rightarrow$ Ping of Death, high `diff_srv_rate` $\rightarrow$ Portsweep/Nmap).
    - *Tier 2*: Category fallbacks (DoS, Probe, R2L, U2R).
    - *Tier 3*: Statistical heuristic fallbacks.
  - Maps technical feature names to human-readable explanations via `_build_feature_map`.
  - Performs semantic vector queries against `KBIndex` to append matching CVE/threat profiles.

### `src/backend/kb_index.py` & `rag_knowledge.py`
- **Role**: Semantic vector index.
- Encodes threat profiles into 384-dimensional dense vectors using `SentenceTransformer('all-MiniLM-L6-v2')` (or OpenAI).
- Searches top-$k$ nearest threat descriptions using `NearestNeighbors(metric='cosine')`.

### `src/backend/traffic_generator.py`
- **Role**: Synthetic traffic and attack generation.
- Injects normal HTTP/HTTPS/SSH traffic as well as crafted attacks (SYN Floods, Land Attacks, Large UDP DoS, Horizontal Port Scans).

### `src/frontend/` (React 19 + Tailwind CSS)
- **`components/Monitor.jsx`**: Real-time SOC dashboard rendering live WebSocket packet feeds, threat badges, confidence metrics, Markdown RAG briefings, capture mode selectors, and duration timers.

---

## 3. Deep Preprocessing & Feature Engineering Mechanics

1. **Stripping Benchmark Metadata**: Column 43 (`difficulty`) is dropped to prevent label bias.
2. **Binary Transformation**:
   $$y = \begin{cases} 0 & \text{if } \text{label} \in \{\text{'normal'}, \text{'normal.'}\} \\ 1 & \text{otherwise (attack)} \end{cases}$$
3. **Categorical Expansion**:
   - `protocol_type` (3 values) $\rightarrow$ 3 binary columns.
   - `service` (~70 values) $\rightarrow$ ~70 binary columns.
   - `flag` (11 values) $\rightarrow$ 11 binary columns.
   - Enforces strict 116-dimensional structure via `reindex(columns=feature_columns, fill_value=0)`.
4. **Quantile Normalization**:
   - Network byte counts and rates follow extreme power-law distributions.
   - `QuantileTransformer(output_distribution='normal')` maps empirical rank quantiles to a Gaussian distribution ($\mu=0, \sigma=1$), preventing outlier explosion while preserving subtle variance in small-magnitude features.

---

## 4. Deep Autoencoder & Isolation Forest Architectures

### Autoencoder Graph (Keras)
```
Input Vector (116)
       │
Dense(64, ReLU) ──► BatchNorm ──► Dropout(0.15)
       │
Dense(32, ReLU) ──► BatchNorm ──► Dropout(0.15)
       │
Dense(16, ReLU) ──► BatchNorm
       │
[ Latent Space: Dense(4, ReLU) ]  ◄── 96.5% Dimensional Compression
       │
Dense(16, ReLU) ──► BatchNorm
       │
Dense(32, ReLU) ──► BatchNorm ──► Dropout(0.15)
       │
Dense(64, ReLU) ──► BatchNorm ──► Dropout(0.15)
       │
Output Layer: Dense(116, Linear)
```
- **Tightly Constrained 4D Bottleneck**: Forces the network to learn only fundamental normal traffic manifolds, making it highly sensitive to multi-feature anomalous combinations.
- **Linear Output**: Reconstructs unbounded real-valued Gaussian features produced by the QuantileTransformer without saturation.
- **Optimization**: Adam ($\eta=0.0005$), EarlyStopping (patience=12), ReduceLROnPlateau.

### Isolation Forest (Scikit-Learn)
- **Parameters**: 250 trees, contamination=0.10, `n_jobs=-1`.
- **Score**:
  $$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
- Negated via $-\text{decision\_function}(X)$ so higher scores signify anomalies.

### Score Fusion & Dynamic Thresholding
$$\text{Score}_{\text{hybrid}} = 0.75 \cdot \text{AE}_{\text{norm}} + 0.25 \cdot \text{IF}_{\text{norm}}$$
$$\Theta_{\text{dynamic}} = \max\left(\Theta_{\text{base}},\, \mu_{\text{buffer}} + 3\sigma_{\text{buffer}}\right)$$

---

## 5. Benchmark Performance Metrics (KDDTest+)

Evaluated against the official benchmark test set `KDDTest+.txt` (22,544 packets):

| Metric | Score | Analytical Interpretation |
|---|---|---|
| **AUC-ROC** | **96.72%** | Near-optimal global separability across all operating thresholds. |
| **Accuracy** | **90.52%** | Correct classification rate across combined normal and attack samples. |
| **Precision** | **91.14%** | Probability that a flagged alert is a genuine attack. |
| **Recall** | **92.32%** | Proportion of all attacks successfully intercepted. |
| **F1-Score** | **91.73%** | Harmonic mean confirming robust balance under class imbalance. |
| **Base Threshold ($\Theta$)** | **0.127941** | Optimized via Youden's J statistic ($J = \text{TPR} - \text{FPR}$). |
| **False Positive Rate (FPR)** | **11.85%** | 1,151 false alarms out of 9,711 legitimate benign packets. |
| **False Negative Rate (FNR)** | **7.68%** | 986 missed attacks out of 12,833 total attacks. |

### Confusion Matrix
$$\begin{array}{c|cc}
& \textbf{Predicted Normal (0)} & \textbf{Predicted Anomaly (1)} \\
\hline
\textbf{Actual Normal (0)} & \text{TN} = \mathbf{8,560} & \text{FP} = \mathbf{1,151} \\
\textbf{Actual Anomaly (1)} & \text{FN} = \mathbf{986} & \text{TP} = \mathbf{11,847} \\
\end{array}$$

---

# 🔬 FORENSIC INVESTIGATION: Why Almost Every Packet Was Flagged as an Anomaly

Despite an official test accuracy of 90.52% and a benchmark test FPR of 11.85%, live testing across Live Network, Synthetic Generator, Physical Injector, and Dataset Upload resulted in nearly 100% of packets being flagged as anomalies in the UI (e.g., 71/71, 50/52, 106/108, 126/128, 815/823).

The root cause is **not** conventional neural network overfitting or underfitting. It is caused by **five compounding architectural and algorithmic design flaws** identified through source code inspection:

---

### Root Cause 1: The Modern Network Domain Gap — The "Port 443 / HTTPS" Trap (Affects Live & Synthetic)

#### Evidence in Screenshots:
- **Screenshot 1 (Live Network)**: Packet to `8.219.122.25` on port 443 (Service: `http_443`). Marked Anomaly! Score: 0.1691. XAI: `service_http_443`.
- **Screenshot 2 (Virtual Synthetic)**: Traffic to `service: http_443`. Marked Anomaly! Score: 0.2051. XAI: `service_http_443`.
- **Screenshot 5 (Live Network)**: Packet from Google CDN `172.217.112.4` on port 443. Marked Anomaly! Score: 0.2143. XAI: `service_http_443`.

#### Code Analysis:
Inspection of `KDDTrain+.txt` reveals the exact sample distribution for port 443:
```
Total records with service == 'http_443' in KDDTrain+: 530
- neptune (SYN Flood attack): 523
- portsweep (Probe attack)  : 5
- nmap (Probe attack)       : 1
- satan (Probe attack)      : 1
- NORMAL                    : 0 (ZERO!)
```
In the 1999 DARPA dataset from which NSL-KDD is derived, HTTPS was rare, and the only recorded port 443 traffic consisted of SYN flood attacks.
Because the Autoencoder is trained **strictly on normal samples** ($y=0$):
- In `X_train_normal`, the column `service_http_443` is **0 in 100% of samples**.
- When live traffic hits port 443 (which constitutes 95%+ of modern internet web traffic), `service_http_443 = 1`.
- The Autoencoder attempts to reconstruct it to `0.0`, incurring a massive squared reconstruction error on that feature.
- Concurrently, the Isolation Forest treats `service_http_443 = 1` as an attack indicator because 100% of training samples with that feature were attacks.
- **Result**: Every HTTPS packet produces an anomaly score between 0.16 and 0.22, exceeding the base threshold of 0.1279.

---

### Root Cause 2: Cold-Start Rolling Window Distortion in `sniffer.py`

#### Evidence in Screenshots:
Screenshots 1 and 2 flag benign web requests as:
> **"Anomaly Detected: Portsweep Attack"**
> *Indicators*: `high diff_srv_rate (> 0.7)`, `high dst_host_count`, `dst_host_diff_srv_rate`

#### Code Analysis (`sniffer.py` & `rag_service.py`):
In `sniffer.py`, temporal window metrics are calculated as:
```python
diff_srv_rate = (count - same_srv_count) / count if count > 0 else 0.0
dst_host_diff_srv_rate = (
    (dst_host_count - dst_host_srv_count) / dst_host_count
    if dst_host_count > 0
    else 0.0
)
```
When monitoring begins or traffic is sparse:
- `count` is small (e.g., 1 or 2).
- When a user's browser opens a web page, it contacts an HTTPS host and a DNS server, or two different domains simultaneously.
- Because `count = 2` and `same_srv_count = 1`, `diff_srv_rate` immediately calculates to **0.50 or 1.00**!
- In `rag_service.py`, line 307:
```python
if (
    'diff_srv_rate' in features_str
    or diff_srv_rate > 0.5
    or 'srv_diff_host_rate' in features_str
):
  return 'portsweep_attack'
```
- **Result**: Legitimate multi-service connections on sparse rolling windows mathematically mimic port reconnaissance, triggering the "Portsweep" signature.

---

### Root Cause 3: The Deadlock in Dynamic Thresholding (`detector.py`)

#### Code Analysis:
```python
# Dynamic Threshold calculation
if len(self.score_buffer) > 100:
  mean_score = np.mean(self.score_buffer)
  std_score = np.std(self.score_buffer)
  self.dynamic_threshold = max(self.base_threshold, mean_score + 3 * std_score)

is_anomaly = bool(hybrid_score > self.dynamic_threshold)

# Add to buffer only if it's considered normal AND it's not excessively high
if not is_anomaly and hybrid_score < (self.base_threshold * 1.5):
  self.score_buffer.append(hybrid_score)
```

#### The Logical Trap:
1. The base threshold is set to `0.1279`.
2. Because of modern HTTPS traffic or cold-start window metrics, the initial packets score between `0.16` and `0.22`.
3. Because `0.1691 > 0.1279`, `is_anomaly` evaluates to `True`.
4. Because `is_anomaly == True`, the buffer update condition `if not is_anomaly` evaluates to `False`.
5. The packet score is discarded and **never added to `self.score_buffer`**.
6. `len(self.score_buffer)` remains at `0`.
7. `if len(self.score_buffer) > 100` **never executes**.
8. The threshold remains frozen at `0.1279` forever, trapping the system in a state where every subsequent packet is marked anomalous.

---

### Root Cause 4: Why Dataset Upload (`KDDTest+.txt`) Showed 126 Anomalies Out of 128 Packets

#### Evidence in Screenshot 4:
Dataset upload of `KDDTest+.txt` resulted in 128 Total Packets and 126 Anomalies.

#### Code Analysis (`main.py` lines 220–250):
1. **Attack Clustering at File Head**: In `KDDTest+.txt`, the dataset starts with heavy clusters of attacks (Rows 0, 1: `neptune`, Row 3: `saint`, Row 4: `mscan`, Row 7: `guess_passwd`, etc.).
2. **Missing Normal Baseline in Stream**: In `upload_dataset`, records are fed one-by-one into `detector.predict()`. Because the file begins with a dense cluster of attacks, `score_buffer` never gets a clean run of normal samples to calibrate a baseline, and subsequent borderline normal samples with rare test services are rejected.

---

### Root Cause 5: Flawed Assumptions in `traffic_generator.py`

In `traffic_generator.py`:
```python
def _generate_normal_packet(self):
  dport = random.choice([80, 443, 22, 53])  # 443 and 22 have 0% and 1.6% normal in KDD!
  flags = random.choice(['A', 'PA'])  # Raw packets without SYN handshake!
```
- Choosing port `443` or `22` directly triggers Root Cause 1.
- Generating isolated ACK packets (`A`, `PA`) without an initial SYN handshake causes `sniffer.py` to observe sessions with missing connection state transitions, driving up anomaly scores on "normal" synthetic traffic.

---

## 6. Actionable Remediation Plan

To eliminate false positives and align real-time performance with benchmark accuracy, apply the following targeted fixes:

### 1. Fix the Port 443 Domain Shift in `sniffer.py`
In `sniffer.py`, map `443` to `"http"` rather than `"http_443"`. Both are web traffic, and `"http"` has 38,049 normal samples in the training dataset:
```python
# sniffer.py: Map port 443 to 'http' so the model applies its deep web-traffic baseline
443: "http",  # Was: "http_443"
```

### 2. Fix the Dynamic Threshold Deadlock in `detector.py`
Allow the buffer to initialize during a short initial warm-up phase (e.g., the first 200 packets) or buffer scores that fall within a wider realistic baseline:
```python
# detector.py: Seed the buffer during the initial startup phase
if len(self.score_buffer) < 100:
  if hybrid_score < (self.base_threshold * 2.5):
    self.score_buffer.append(hybrid_score)
else:
  mean_score = np.mean(self.score_buffer)
  std_score = np.std(self.score_buffer)
  self.dynamic_threshold = max(self.base_threshold, mean_score + 3 * std_score)
  if not is_anomaly and hybrid_score < (self.dynamic_threshold * 1.2):
    self.score_buffer.append(hybrid_score)
```

### 3. Add Minimum Support Constraints to Window Metrics in `sniffer.py`
Prevent rate features from exploding when `count` is small:
```python
# sniffer.py: Do not compute volatile error/diff rates on tiny sample counts
diff_srv_rate = (
    (count - same_srv_count) / count if count >= 5 else 0.0
)  # Require at least 5 connections
dst_host_diff_srv_rate = (
    (dst_host_count - dst_host_srv_count) / dst_host_count
    if dst_host_count >= 5
    else 0.0
)
```

### 4. Correct `traffic_generator.py` Normal Traffic Generation
Target services with robust normal representations in the model (`http`, `smtp`):
```python
# traffic_generator.py: Select services with strong normal baseline representations
dport = random.choice([80, 25])  # Use well-represented normal ports
```

---

*Report prepared and integrated into the project documentation suite.*
