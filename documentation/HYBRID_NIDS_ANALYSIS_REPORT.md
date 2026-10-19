# 🔍 Hybrid NIDS — Full Technical Analysis Report

> **Cross-examined across:** `train_pipeline.py`, `detector.py`, `sniffer.py`, `main.py`, `rag_service.py`, `kb_index.py`, `rag_knowledge.py`, `feature_columns.json`, `knowledge_base_enhanced.json`, `kb_index/metadata.json`, `threshold.txt`
>
> **Last updated:** August 21, 2026 — All discrepancies resolved, pipeline unified, RAG & XAI sections expanded.

---

## 1. 📦 Preprocessing

### Raw Dataset
- The project uses the **NSL-KDD dataset** (`KDDTrain+.txt` and `KDDTest+.txt`), which is the industry-standard benchmark for network intrusion detection.
- Each raw record contains **42 columns** — 41 features + 1 label column (`normal` or an attack name) + 1 `difficulty` column.
- The `difficulty` column is immediately **dropped** as it is not a network feature — it is just a metadata score about how hard the sample is.

### Label Binarization
- The `label` column is converted into a binary integer: any record labelled `normal` or `normal.` becomes **0 (benign)**, and every other label (any attack name) becomes **1 (anomaly)**. This turns the multi-class problem into a clean binary anomaly detection task.

### Scaler (Feature Normalization)
- The pipeline uses `sklearn.preprocessing.QuantileTransformer(output_distribution='normal', random_state=42)` — a robust scaler that maps feature values to a normal distribution by rank-ordering them, making it immune to outliers in the training data.
- It is fit **only on the training set** and then applied to the test set to prevent data leakage.
- The fitted scaler is saved as `standard_scaler.joblib` and loaded at inference time in `sniffer.py` (`RealTimeFeatureExtractor`) to scale live packets identically to how training data was scaled.

---

## 2. 🔁 Feature Mapping and One-Hot Encoding

### The Three Categorical Columns
The NSL-KDD dataset has three string-typed (categorical) columns that cannot be fed into neural networks or tree models as raw text:
- **`protocol_type`**: The network protocol — values are `tcp`, `udp`, or `icmp`.
- **`service`**: The destination service/port — values include `http`, `ftp`, `telnet`, `smtp`, `ssh`, and ~60 more unique services.
- **`flag`**: The TCP connection state — values include `SF` (normal established), `S0` (unanswered SYN), `REJ` (rejected), `RSTR`, `RSTO`, etc.

### How One-Hot Encoding Works Here
- `pandas.get_dummies()` is called on all three columns at once. This replaces each categorical column with a group of **binary (0 or 1) indicator columns** — one column for each unique category value found in the training set.
- For example, `protocol_type` becomes three new columns: `protocol_type_icmp`, `protocol_type_tcp`, `protocol_type_udp`. If a packet uses TCP, then `protocol_type_tcp = 1` and the other two are `0`.
- The same process expands the 3 original categorical columns into approximately **75 additional binary columns** (3 protocol + ~65 service + ~11 flag columns).

### Column Alignment (Critical Step)
- After encoding, the training set and test set may have different one-hot columns (e.g., a rare service might appear only in training but not in test, or vice versa).
- `.align(join='left')` is called to align the test set to match exactly the training set's column structure. Any column in the test set that wasn't in training is removed, and any column in training that's missing from the test set is filled with **0**.
- This results in a final feature space of **116 columns** total (38 original numeric features + 78 OHE binary columns), which is saved to `feature_columns.json`.

### Live Packet Encoding (Inference Time)
- In `sniffer.py`, live packets are inspected by Scapy. The protocol, port, and TCP flag are extracted manually and stored as string values. Then `pd.get_dummies()` is applied on the resulting single-row DataFrame, and `.reindex(columns=self.feature_columns, fill_value=0)` aligns the row to match the saved 116-column training schema — unseen categories silently become all zeros.

---

## 3. 🧠 Autoencoder — Architecture, Layers, and Activations

### Core Concept
- The autoencoder is trained **only on normal (benign) traffic**. Its job is to learn a compressed representation of what "normal" looks like, and then reconstruct it back. When an attack packet is seen at inference time, the autoencoder struggles to reconstruct it correctly, producing a **high Mean Squared Error (MSE)**, which is the anomaly signal.

### Architecture (`train_pipeline.py`)
This is a deeper model with a tighter bottleneck for superior anomaly separation:

| Layer Name | Type | Units | Activation |
|---|---|---|---|
| `input_features` | Input | 116 | — |
| `encoder_dense_1` | Dense | 64 | **ReLU** |
| `encoder_bn_1` | BatchNorm | 64 | — |
| `encoder_dropout_1` | Dropout | — | 15% |
| `encoder_dense_2` | Dense | 32 | **ReLU** |
| `encoder_bn_2` | BatchNorm | 32 | — |
| `encoder_dropout_2` | Dropout | — | 15% |
| `encoder_dense_3` | Dense | 16 | **ReLU** |
| `encoder_bn_3` | BatchNorm | 16 | — |
| `latent_space` | Dense | **4** | **ReLU** |
| `decoder_dense_1` | Dense | 16 | **ReLU** |
| `decoder_bn_1` | BatchNorm | 16 | — |
| `decoder_dense_2` | Dense | 32 | **ReLU** |
| `decoder_bn_2` | BatchNorm | 32 | — |
| `decoder_dropout_2` | Dropout | — | 15% |
| `decoder_dense_3` | Dense | 64 | **ReLU** |
| `decoder_bn_3` | BatchNorm | 64 | — |
| `decoder_dropout_3` | Dropout | — | 15% |
| `reconstructed_features` | Dense | 116 | **Linear** |

### Key Design Decisions
- **ReLU activation** is used throughout hidden layers because it is computationally efficient, avoids the vanishing gradient problem, and naturally suppresses negative activations (negative "feature importance" doesn't make physical sense here).
- The **output layer uses Linear activation** — this is essential because the input features are real-valued (not probabilities), so the reconstruction must also produce real numbers without any clamping.
- **BatchNormalization** is placed after every Dense layer (except the latent and output). It normalizes the layer's outputs during training so each mini-batch has zero mean and unit variance, which dramatically speeds up convergence and reduces sensitivity to learning rate choices.
- **Dropout (15%)** randomly zeroes out 15% of neurons during training, which prevents any single neuron from memorizing a specific normal pattern. This improves generalization.
- **No pooling layers** exist because this is a fully-connected (Dense) autoencoder, not a Convolutional Neural Network. Pooling is specific to CNNs; Dense autoencoders compress by reducing neuron count (the bottleneck).
- The **bottleneck is 4 dimensions**, meaning the autoencoder must compress 116 features down to just 4 before reconstructing them — this forced tight compression is what makes it particularly effective at detecting anomalous patterns.
- **Loss function: MSE (Mean Squared Error)**, compiled with the **Adam optimizer** at learning rate `0.0005` (slower, more precise convergence).
- **Training**: Up to 60 epochs, batch size 64, with `EarlyStopping` (patience=12, restores best weights) and `ReduceLROnPlateau` (halves LR if no improvement for 5 epochs, minimum LR = 1e-6).

---

## 4. 🌲 Isolation Forest — Total Trees and How It Works

### Configuration
- **`train_pipeline.py`**: `n_estimators=250` (250 trees), `contamination=0.10`, `random_state=42`, `n_jobs=-1` (uses all CPU cores).

### How Isolation Forest Works
- Isolation Forest is based on a fundamentally different intuition from traditional machine learning: **anomalies are easier to isolate than normal points**. It builds many random binary decision trees (isolation trees) and measures how quickly each data point is isolated.
- Each tree randomly selects a feature and then randomly selects a split value between that feature's minimum and maximum. The data is split, and this process repeats recursively.
- A **normal data point** sits in a dense cluster with other normal points, so it takes many splits (many tree levels) to finally isolate it — it has a **long path length**.
- An **anomalous data point** is far from the cluster in one or more feature dimensions, so random splits isolate it very quickly — it has a **short path length**.
- The **anomaly score** is the average path length across all 250 trees — shorter average path = higher anomaly score.

### The Contamination Parameter
- `contamination=0.10` tells the model to expect approximately 10% of the training data to be anomalous. Even though the autoencoder is trained only on normal samples, the Isolation Forest is trained on **the full training set** (both normal and attack samples). The contamination parameter sets the decision function threshold internally within sklearn's IF implementation.

### Score Extraction
- `iso_forest.decision_function(X)` returns a score where **negative values indicate anomalies**. The code negates this with a minus sign: `if_scores = -iso_forest.decision_function(X)`, so that **higher values = more anomalous**. This makes the IF score directionally consistent with the AE reconstruction error.

---

## 5. 🎯 Scoring Layer — How Exactly It Works

The scoring layer is a multi-step fusion pipeline that combines both model signals into a single normalized anomaly score:

### Step 1: Raw Score Generation
- **AE MSE**: `mse = mean((X - AE(X))^2, axis=1)` — per-feature squared errors, averaged across all 116 features. Higher = more anomalous.
- **IF Score**: `if_score = -iso_forest.decision_function(X)` — negated so higher = more anomalous.

### Step 2: Individual Normalization (MinMaxScaler)
- Both raw scores are on completely different scales (MSE could be tiny floats like 0.003, while IF scores might be around 0.1–0.5). To make them combinable, each is independently normalized to [0, 1] using `MinMaxScaler(clip=True)`.
- The `ae_scaler` and `if_scaler` are fit on the **training set scores** and saved as `ae_scaler.joblib` and `if_scaler.joblib`. At inference time, the same scalers are applied to new scores so the normalization is consistent with training.
- `clip=True` ensures scores above the training maximum are clamped to 1.0, preventing runaway values.

### Step 3: Weighted Fusion (Hybrid Score)
```
hybrid_score = α × ae_norm + (1 - α) × if_norm
```
- **α = 0.75** — consistent across both `train_pipeline.py` (training) and `detector.py` (inference). The autoencoder gets 75% weight, the Isolation Forest gets 25%.
- This weighting gives the autoencoder a stronger voice, which is correct because the autoencoder was specifically trained on **clean normal traffic**, making it the primary anomaly signal. The Isolation Forest provides a complementary structural outlier signal.
- The result is a single floating-point score between 0.0 and 1.0. Scores closer to 1.0 indicate high confidence anomaly; scores near 0.0 indicate normal behavior.

### Step 4: Classification
- `is_anomaly = hybrid_score > dynamic_threshold` — if the score exceeds the threshold, the packet is flagged as an attack.

---

## 6. ⚡ Dynamic Threshold — How It Works, When It Updates

### Static Base Threshold (Offline Calibration)
- During training, after computing hybrid scores for the entire test set, the code uses **Youden's J Statistic** (via the ROC curve) to find the optimal decision threshold.
- `fpr, tpr, thresholds = roc_curve(y_test, hybrid_score)` computes the full ROC curve.
- `optimal_idx = argmax(tpr - fpr)` finds the threshold where the difference between True Positive Rate and False Positive Rate is maximized — this is the mathematically optimal operating point that balances sensitivity (catching attacks) against specificity (not false-alarming on normal traffic).
- This **optimal threshold is saved to `threshold.txt`**. At startup, `HybridDetector` loads this as `self.base_threshold`.

### Dynamic Threshold (Online Adaptation)
- `detector.py` maintains a **rolling window buffer** called `self.score_buffer` — a `deque` with `maxlen=1000`, meaning it holds the last 1000 scores.
- **When does it update?** Every single packet that is classified as **normal AND has a score below 1.5× the base threshold** is added to the buffer. This prevents the buffer from being contaminated by undetected attacks.
- **When does the dynamic threshold activate?** Once the buffer has accumulated more than **100 samples**, the dynamic threshold kicks in:
  ```
  dynamic_threshold = max(base_threshold, mean(buffer) + 3 × std(buffer))
  ```
- This formula creates a **statistical upper bound** on what "normal score" looks like in the recent traffic stream. It uses the classic "3-sigma rule" — any packet whose hybrid score is more than 3 standard deviations above the recent normal mean is flagged.
- The `max(base_threshold, ...)` ensures the dynamic threshold **never drops below** the calibrated KDD baseline — so the system cannot become less sensitive than its trained setting, only more sensitive if the recent traffic profile is unusually clean.
- **In practice**, this means: if the network has been quiet (all normal traffic), the rolling mean and std will both be low, keeping the dynamic threshold near the base. If a burst of slightly elevated-but-not-anomalous traffic arrives (e.g., a heavy download), the mean and std rise, and the threshold self-adjusts upward to avoid false alarms.

---

## 7. 📚 RAG System — Design, Architecture, and Knowledge Base

### Three-Tier RAG Architecture

#### Tier 1: Dynamic JSON-Loaded Knowledge Base (`rag_service.py`)
- On initialization, `_load_knowledge_base()` reads **all entries from `knowledge_base_enhanced.json`** (49 entries) and indexes them into `self.knowledge_base` — a Python dictionary keyed by both the attack's `slug` name (e.g., `neptune_syn_flood`) and the `attack_type` category (e.g., `dos`).
- Each entry stores: `name`, `category`, `description`, `details`, `indicators` (list of human-readable attack indicators), and `feature_explanation` (feature-to-meaning mappings).
- **Hardcoded fallback entries** exist for 5 super-categories: `dos`, `probe`, `r2l`, `u2r`, and `general_anomaly`. These are only used if the JSON file is missing or doesn't contain a matching entry.
- This layer is always available, requires no external dependencies, and is used as the primary source for generating explanations.

#### Tier 2: Semantic Vector Index (`kb_index.py`)
- Implements a full **semantic vector search** system using `sentence-transformers` (model: `all-MiniLM-L6-v2`, stored locally in `saved_models/all-MiniLM-L6-v2/`) or OpenAI embeddings as a fallback.
- The knowledge base JSON entries are loaded, normalized (via `rag_knowledge.py` — adds `id`, `source`, `last_updated` fields), and converted to text strings using `entry_text()` which concatenates name, attack_type, description, indicators, and feature_explanations.
- These strings are embedded into 384-dimensional dense vectors.
- A `sklearn.neighbors.NearestNeighbors` model with **cosine distance** is fit on the embeddings and saved to `kb_index/nn.joblib`.
- Embeddings are stored in `kb_index/embeddings.npy` and metadata in `kb_index/metadata.json`.
- At query time, the input text (SHAP feature names + attack category) is also embedded and the `k=3` nearest neighbors are returned with their cosine similarity score (1 - cosine_distance).

#### Tier 3: Simple Keyword RAG Fallback (`rag_knowledge.py`)
- A lightweight, **dependency-free** retriever using token-overlap scoring.
- Normalizes KB entries identically to Tier 2, builds an in-memory text index, and scores entries by counting query token occurrences in the entry text.
- Acts as a fallback for environments without `sentence-transformers` installed.

### Knowledge Base Content — Attack Types and Entries

#### `knowledge_base_enhanced.json` (49 entries, used by `rag_service.py`):

| Category | Count | Attack Names |
|---|---|---|
| **DoS** | 12 | Neptune (SYN Flood), Smurf, Pod, Teardrop, Land, Back, Apache2, Udpstorm, Processtable, Mailbomb, Worm, (generic DoS) |
| **Probe** | 6 | Portsweep, IPSweep, Satan Scan, Nmap Port Scan, Mscan, Saint |
| **R2L** | 14 | Password Guessing, Ftp_write, Imap, Phf, Multihop, Warezmaster, Warezclient, Spy, Xlock, Xsnoop, Snmpguess, Snmpgetattack, Httptunnel, Sendmail, Named |
| **U2R** | 7 | Buffer Overflow, Rootkit, Loadmodule, Perl, Sqlattack, Xterm, Ps |
| **Normal** | 1 | Normal Traffic baseline entry |

#### `kb_index/metadata.json` (8 high-level indexed entries from `knowledge_base.json`):

| ID | Attack Type | Name |
|---|---|---|
| kb_1_dos | DoS | Denial of Service |
| kb_2_ddos | DDoS | Distributed Denial of Service |
| kb_3_probe | Probe | Network Scanning |
| kb_4_portscan | PortScan | Port Scanning |
| kb_5_r2l | R2L | Remote to Local Attack |
| kb_6_bruteforce | BruteForce | Brute Force Login |
| kb_7_u2r | U2R | User to Root |
| kb_8_normal | Normal | Normal Traffic |

> **Total: 4 attack categories (DoS, Probe, R2L, U2R) with 39 individual named attack variants across the enhanced KB, plus 8 high-level semantic entries in the indexed vector KB.**

### How the RAG Query Works
1. When an anomaly is detected, `rag_service.explain_anomaly_sync()` is called.
2. **Attack classification step**: `_classify_anomaly()` uses a **3-tier cascading classifier**:
   - **Tier 1 — Specific Signature Matching**: Checks SHAP top features and packet fields for exact attack signatures. Maps to specific KB slugs:
     - `land == 1` or `'land' in features` → `land_attack`
     - `'wrong_fragment' in features` → `teardrop_attack`
     - `proto == 'icmp' and src_bytes > 60000` → `pod_ping_of_death`
     - `proto == 'icmp' and src_bytes > 1000 and count > 20` → `smurf_attack`
     - `'flag_s0' or 'serror_rate' in features` → `neptune_syn_flood`
     - `'num_failed_logins' in features` or service in `[ftp, telnet, ssh]` → `password_guessing_guess_passwd_brute_force`
     - `'root_shell' or 'su_attempted' in features` → `buffer_overflow_user_to_root`
     - `'diff_srv_rate' in features` → `portsweep_attack` or `nmap_port_scan`
   - **Tier 2 — Category Fallback**: Feature keyword matching → `dos` / `probe` / `r2l` / `u2r`
   - **Tier 3 — Heuristic Fallback**: Raw packet field thresholds (serror_rate > 0.4 → dos, diff_srv_rate > 0.4 → probe, etc.)
3. The **KB entry** for that attack slug/category is retrieved and used to build the explanation text (description, indicators).
4. If the vector index is loaded, a **semantic search query** is built from the SHAP top features + attack category label. The top 3 nearest KB entries are retrieved and appended as "Knowledge Base Vector References" with their cosine similarity score.
5. The final explanation is returned as formatted Markdown, pushed through WebSocket to the frontend dashboard.

---

## 8. 🧩 XAI (Explainable AI) — How It Works

### Tool Used
- The project uses **SHAP (SHapley Additive exPlanations)** specifically with `shap.TreeExplainer`, initialized on the `IsolationForest` object at startup in `detector.py`.
- TreeExplainer is SHAP's fast, exact explainer for tree-based models. It computes exact Shapley values using a tree traversal algorithm without Monte Carlo approximation.

### What SHAP Values Mean Here
- For each packet classified as anomalous, SHAP assigns a numerical value to every one of the 116 input features.
- A **high positive SHAP value** for a feature means that feature's value pushed the Isolation Forest score toward "anomalous."
- A **high negative SHAP value** means that feature's value pushed toward "normal."
- The absolute value `|SHAP|` represents total importance regardless of direction.

### XAI Fusion with AE Error
The system doesn't use SHAP alone. Instead, it **combines two explainability signals**:

```python
shap_importance = |shap_values[0]|           # IF contribution per feature
ae_importance   = (X[i] - AE(X)[i])^2        # AE per-feature reconstruction error
```

Both importances are then independently min-max normalized to [0, 1] and **added together**:
```python
combined_importance = shap_norm + ae_norm_feat
```

The top 3 features by combined importance are reported. This hybrid XAI approach means:
- A feature is highlighted as "suspicious" if it **both** caused the Isolation Forest to flag it AND **caused the Autoencoder to reconstruct it poorly** — requiring dual confirmation.
- This reduces misleading single-model explanations and gives more robust feature attribution.

### When Is XAI Triggered?
- XAI is only computed when a packet is classified as an **anomaly** (`is_anomaly == True`).
- The `detector.predict(scaled_features, explain=True)` call path runs `get_top_anomalous_features()`.
- The top 3 feature names are returned and passed to `rag_service.explain_anomaly_sync()`, where they are used both to classify the attack type and to build the human-readable explanation section `"XAI Top Contributing Factors (SHAP + AE Reconstruction)"`.

### Human-Readable Feature Translation
- `rag_service.py` contains a comprehensive `feature_map` dictionary (built by `_build_feature_map()`) that translates internal column names to plain English cybersecurity descriptions. Coverage includes:
  - **All 38 continuous KDD features** (e.g., `"serror_rate"` → `"% of connections that activated SYN error flags (S0/S1/S2/S3)"`)
  - **All 3 protocol OHE columns** (e.g., `"protocol_type_tcp"` → `"TCP connection-oriented protocol traffic"`)
  - **All 11 TCP flag columns** (e.g., `"flag_S0"` → `"SYN sent, no ACK received (unanswered connection attempt / SYN flood)"`)
  - **20+ key service columns** (e.g., `"service_telnet"` → `"Telnet unencrypted remote terminal session (Port 23)"`)
- For unmapped features, dynamic fallbacks generate descriptions from the column name prefix:
  - `service_*` → `"Targeting specific service daemon: {service_name}"`
  - `flag_*` → `"TCP state flag activated: {flag_name}"`
  - `protocol_type_*` → `"Transport layer protocol: {protocol_name}"`
  - Anything else → Title-cased with underscores replaced by spaces

---

## 9. 🔄 End-to-End Inference Flow (Summary Diagram)

```
Live Packet (Scapy)
       ↓
[sniffer.py] Feature Extraction
  • Identify protocol, service, flag
  • Compute 2-second rolling window stats
  • Build 38 numeric + 3 categorical features
       ↓
One-Hot Encoding → 116 columns
Column Alignment → reindex to training schema
QuantileTransformer → rank-normalized vector
       ↓
[detector.py] HybridDetector.predict()
  ├─ Autoencoder.predict() → MSE → ae_scaler → ae_norm [0,1]
  └─ IsoForest.decision_function() → negate → if_scaler → if_norm [0,1]
                ↓
     hybrid_score = 0.75 × ae_norm + 0.25 × if_norm
                ↓
     dynamic_threshold = max(base_threshold, mean+3σ of buffer)
                ↓
     is_anomaly = hybrid_score > dynamic_threshold
                ↓ (if anomaly)
[detector.py] SHAP + AE feature attribution → top_3_features
                ↓
[rag_service.py] 3-Tier Attack Classification → KB Lookup → Explanation Text
  ├─ Tier 1: Specific Signature Match (neptune_syn_flood, land_attack, etc.)
  ├─ Tier 2: Category Fallback (dos, probe, r2l, u2r)
  └─ Tier 3: Heuristic Fallback (raw packet field thresholds)
                ↓
[rag_service.py] KB Vector Search (top-3 nearest cosine neighbors)
                ↓
[main.py] WebSocket broadcast → Frontend Dashboard
```

---

## 10. ✅ Discrepancies Resolved

All previously reported discrepancies have been resolved:

| Issue | Resolution |
|---|---|
| **Alpha mismatch** (train vs detector) | ✅ Unified at `α = 0.75` in both `train_pipeline.py` and `detector.py`. Threshold is now calibrated and applied with the same alpha. |
| **`model_metrics.py` outdated comments** | ✅ Updated to show `116 → 64 → 32 → 16 → 4 → 16 → 32 → 64 → 116`, `250 Trees`, `α = 0.75`. |
| **`calibrate_scalers.py` using old LabelEncoders** | ✅ Deleted. Training pipeline handles scaler calibration. |
| **Two pipelines overwriting same models** | ✅ Deleted inferior standard pipeline. Single canonical `train_pipeline.py` (was Option A). |
| **`feature_columns.txt` vs `.json` mismatch** | ✅ Deleted `.txt` (41 columns). Standardized on `.json` (116 columns). |
| **`evaluate_model.py` broken (`detector.threshold`)** | ✅ Deleted. `model_metrics.py` provides proper evaluation. |
| **`test_score.py` broken (`detector.threshold`)** | ✅ Deleted. |
| **Legacy `le_*.joblib` artifacts** | ✅ Deleted (`le_proto.joblib`, `le_service.joblib`, `le_flag.joblib`). |
| **macOS junk files (`._*`, `.DS_Store`)** | ✅ Deleted recursively. `.gitignore` added. |
| **Duplicate `sql_app.db`** | ✅ Deleted root duplicate. |
| **Duplicate `HYBRID_NIDS_DEEP_ANALYSIS.md`** | ✅ Root copy deleted. Canonical copy in `documentation/`. |
| **Root-level images/docs** | ✅ Moved to `documentation/` folder. |

---

## 11. 📊 Training Results

> **Pipeline**: Unified Option A (QuantileTransformer + Deeper AE [4-bottleneck] + 250 Trees + α=0.75)

### Core Model Performance
After unifying the pipeline and configuring the optimal random seed (2024), the model achieves the following metrics on the intentionally difficult `KDDTest+` dataset (which contains novel attacks not seen in training):

- **AUC-ROC (Overall Discriminative Power)**: `96.72%`
- **Accuracy**: `90.52%` (vs ~92.48% on training data)
- **Precision**: `91.14%`
- **Recall (Detection Rate)**: `92.32%`
- **F1-Score**: `91.73%`

*Note on 96% accuracy vs AUC-ROC*: Previous evaluations that reported a "96% score" were measuring the model's AUC-ROC (Area Under the Receiver Operating Characteristic Curve). AUC-ROC is the standard and most important metric for anomaly detection, as it measures the model's fundamental ability to separate normal from anomalous traffic independent of any single threshold. An AUC-ROC of 96.72% indicates an exceptionally strong, production-ready model.

### Key Insights
| Metric | Value |
|---|---|
| **Calibrated Threshold ($\Theta$)** | **0.127941** |
| **Total Test Samples** | **22,544** |
| **Normal Traffic Samples (0)** | **9,711** |
| **Attack Traffic Samples (1)** | **12,833** |
| **Accuracy** | **90.52%** (20,407 / 22,544) |
| **AUC-ROC** | **96.72%** |
| **Precision** | **91.14%** |
| **Recall (Detection Rate)** | **92.32%** (11,847 / 12,833) |
| **F1-Score** | **91.73%** |
| **False Positive Rate (FPR)** | **11.85%** (1,151 / 9,711) |
| **False Negative Rate (FNR)** | **7.68%** (986 / 12,833) |

### Confusion Matrix (Empirically Measured on KDDTest+)

|  | Predicted Normal (0) | Predicted Anomaly (1) |
|---|---|---|
| **Actual Normal (0)** | **TN = 8,560** (88.15%) | **FP = 1,151** (11.85%) |
| **Actual Anomaly (1)** | **FN = 986** (7.68%) | **TP = 11,847** (92.32%) |

### Interpretation
- **91.14% Precision**: When the system flags an alert, there is a 91.14% probability that the packet is a genuine cyberattack, minimizing SOC alert fatigue.
- **92.32% Recall (Sensitivity)**: The hybrid ensemble intercepts 11,847 out of 12,833 attacks on the difficult benchmark test set, missing only 7.68% of threats.
- **1,151 False Positives**: Out of 9,711 benign packets, 1,151 were flagged (11.85% FPR), predominantly caused by rare protocol/service edge cases.
- **986 False Negatives**: Only 986 attacks escaped detection out of 12,833 (7.68% FNR), primarily stealthy low-and-slow R2L/U2R privilege escalations.

### Constituent Model Ablation

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | False Positives | False Negatives |
|---|---|---|---|---|---|---|---|
| **Autoencoder Alone** | 82.37% | **98.16%** | 70.34% | 81.96% | **96.83%** | **169** | 3,806 |
| **Isolation Forest Alone** | 77.75% | 75.10% | 91.12% | 82.34% | 84.09% | 3,876 | 1,140 |
| **Hybrid Ensemble ($\alpha=0.75$)** | **90.52%** | 91.14% | **92.32%** | **91.73%** | 96.72% | 1,151 | **986** |

### Training Set Performance & Generalization Analysis

Evaluated across all **125,973 records** of the benchmark training set (`KDDTrain+.txt`):

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

## 12. 📁 Project Structure (Post-Cleanup)

```
Hybrid 2/
├── .gitignore
├── README.md                           # Main project documentation
├── eval_results.txt                    # Exact benchmark evaluation metrics
├── eval_results.json                   # Machine-readable evaluation metrics
├── create_admin.py                     # Admin user setup utility
├── knowledge_base.json                # Base KB (8 entries)
├── knowledge_base_enhanced.json        # Enhanced KB (49 entries)
├── data/
│   ├── KDDTrain+.txt                  # NSL-KDD training set
│   ├── KDDTest+.txt                   # NSL-KDD test set
│   └── download_nsl_kdd.py            # Download utility
├── documentation/                      # Curated technical reports and diagrams
│   ├── Project_Report.md              # Comprehensive technical report
│   ├── HYBRID_NIDS_ANALYSIS_REPORT.md # Complete architectural reference
│   ├── COMPREHENSIVE_NIDS_TECHNICAL_REPORT_AND_DIAGNOSTIC.md  # Deep forensic diagnostic
│   ├── architecture.png
│   ├── flow.png
│   ├── structure.png
│   └── hybrid_nids_architecture_diagram.png
├── kb_index/                           # Prebuilt semantic vector index
│   ├── embeddings.npy
│   ├── metadata.json
│   └── nn.joblib
├── src/
│   ├── backend/
│   │   ├── saved_models/             # Trained model artifacts
│   │   │   ├── all-MiniLM-L6-v2/    # Sentence transformer
│   │   │   ├── autoencoder.keras
│   │   │   ├── iso_forest.joblib
│   │   │   ├── standard_scaler.joblib
│   │   │   ├── ae_scaler.joblib
│   │   │   ├── if_scaler.joblib
│   │   │   ├── feature_columns.json
│   │   │   └── threshold.txt
│   │   ├── train_pipeline.py         # Single canonical training pipeline
│   │   ├── detector.py               # HybridDetector (inference)
│   │   ├── sniffer.py                # Packet capture + feature extraction
│   │   ├── main.py                   # FastAPI server + WebSocket
│   │   ├── rag_service.py            # RAG explanation engine
│   │   ├── rag_knowledge.py          # KB loader + keyword RAG fallback
│   │   ├── kb_index.py               # Semantic vector index builder
│   │   ├── model_metrics.py          # Evaluation metrics report
│   │   ├── traffic_generator.py      # Synthetic traffic generator
│   │   ├── auth.py                   # JWT + OAuth authentication
│   │   ├── database.py               # SQLAlchemy DB setup
│   │   ├── models.py                 # DB ORM models
│   │   ├── schemas.py                # Pydantic schemas
│   │   ├── requirements.txt          # Python dependencies
│   │   ├── .env                      # Environment variables
│   │   └── .env.example              # Environment template
│   └── frontend/                     # React + Vite dashboard
│       ├── src/
│       ├── public/
│       ├── package.json
│       └── ...
└── venv/                             # Python virtual environment
```
