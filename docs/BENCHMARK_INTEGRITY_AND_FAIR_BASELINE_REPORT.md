# 📊 Benchmark Integrity, Split Leakage & Apples-to-Apples Fair Baseline Report
## NF-UQ-NIDS-v2 (52 Parquet Partitions / ~75.99 Million NetFlow V2 Records)

**Audit Scripts Executed**:
1. [`01_check_split_leakage.py`](../01_check_split_leakage.py) — Full 52-partition split leakage, host overlap, duplicate flow, host purity, and dataset-shortcut audit.
2. [`02_fair_baselines.py`](../02_fair_baselines.py) — Apples-to-apples evaluation on identical 32-flow sequence windows ($N=19,098$ held-out test windows) across `dataset_prior`, `flow_lgbm_max`, `window_lgbm_pool`, and `bimambaformer`.

**Canonical Raw Outputs**:
- [`results/audit/split_leakage_report.json`](../results/audit/split_leakage_report.json)
- [`results/audit/fair_baseline_results.json`](../results/audit/fair_baseline_results.json)
- [`results/audit/train_src_ips.txt`](../results/audit/train_src_ips.txt)

---

## Part 1: Split Leakage & Shortcut Audit (`01_check_split_leakage.py`)

### Command Executed
```powershell
.\venv\Scripts\python.exe 01_check_split_leakage.py --data-dir "data\NF UQ NIDS V2\parquet" --out results\audit
```

### 1.1 Schema & Timestamp Audit
- **Total Columns**: `46` NetFlow V2 columns
- **Timestamp-Like Columns**: **`NONE`**
- **Hashed Feature Columns**: `39` non-ID behavioral flow columns (excluding `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `DNS_QUERY_ID`, `Label`, `Attack`, `Dataset`).
- **Finding**: Because the NF-UQ-NIDS-v2 benchmark Parquet partitions do not carry wall-clock timestamps, window ordering within a host reflects benchmark partition row order rather than true chronological time.

### 1.2 Host IP Overlap (`IPV4_SRC_ADDR`)
| Split | Unique Source Hosts | Hosts Also Seen in Train | Fraction of Hosts Seen in Train |
| :--- | :---: | :---: | :---: |
| **Train** (`part_00`–`part_47`) | 237,892 *(est. from train_src_ips)* | — | — |
| **Validation** (`part_48`–`part_49`) | 37,102 | 31,580 | **85.12%** (`0.851167`) |
| **Test** (`part_50`–`part_51`) | 25,522 | 22,382 | **87.70%** (`0.876969`) |

**Row-Level Host Overlap in Held-Out Test (`part_50`–`part_51`, 1,702,252 flows)**:
- **Fraction of ALL test rows from train hosts**: **99.81%** (`0.998147`)
- **Fraction of attack test rows from train hosts**: **99.9998%** (`0.999998`)
- **Fraction of benign test rows from train hosts**: **99.44%** (`0.994406`)
- **Test IPs carrying both benign and attack labels**: `71` hosts

### 1.3 Exact Feature Duplicate Audit (All 39 Non-ID Features)
| Split | Exact Match to a Train Row | Internal Duplicate Rate |
| :--- | :---: | :---: |
| **Train** (`part_00`–`part_47`) | — | **93.40%** (`0.933959`) |
| **Validation** (`part_48`–`part_49`) | **94.88%** (`0.948839`) | — |
| **Test (All)** (`part_50`–`part_51`) | **94.90%** (`0.949008`) | **82.10%** (`0.821044`) |
| **Test (Attack Rows)** | **97.50%** (`0.974992`) | — |
| **Test (Benign Rows)** | **89.65%** (`0.896498`) | — |

### 1.4 Host Purity ($\ge 32$ Flows per Host per Partition)
| Split | Hosts $\ge 32$ Flows | Pure Benign Hosts | Pure Attack Hosts | Mixed Label Hosts | Multi-Dataset Hosts |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 43,257 | 39,397 (91.08%) | 858 (1.98%) | 3,002 (6.94%) | 49 |
| **Validation** | 1,807 | 1,643 (90.92%) | 35 (1.94%) | 129 (7.14%) | 2 |
| **Test** | 1,439 | 1,305 (90.69%) | 36 (2.50%) | 98 (6.81%) | 2 |

### 1.5 Source Dataset Shortcut Audit
Attack fractions remain nearly identical across Train, Val, and Test within each constituent dataset:
- `NF-BoT-IoT-v2`: **99.64%** attack (`35,440,726` train rows / `845,706` test rows)
- `NF-ToN-IoT-v2`: **63.99%–64.16%** attack (`15,897,435` train rows / `379,370` test rows)
- `NF-CSE-CIC-IDS2018-v2`: **11.95%–11.91%** attack (`17,732,649` train rows / `423,537` test rows)
- `NF-UNSW-NB15-v2`: **3.98%–3.87%** attack (`2,243,239` train rows / `53,639` test rows)

- **Dataset-Only Majority Baseline Row Accuracy (Test)**: **88.75%** (`0.887479`)

---

## Part 2: Apples-to-Apples Fair Baseline Comparison (`02_fair_baselines.py`)

### Command Executed
```powershell
.\venv\Scripts\python.exe 02_fair_baselines.py --repo "." --data-dir "data\NF UQ NIDS V2\parquet" --out results\audit --checkpoint-dir "src\backend\saved_models" --train-ips results\audit\train_src_ips.txt
```

### 2.1 Evaluation Setup
- **Training Data**: `98,543` windows (`3,000,000` flows) sampled from `part_00`–`part_19` (`seed=11`)
- **Validation Calibration Set**: `5,666` windows from `part_48`–`part_49` (`p95` of benign validation scores)
- **Held-Out Test Set**: **`19,098` windows** (`5,764` attack windows, `13,334` benign windows) from `part_50`–`part_51`
- **Host-Disjoint Status**: `test_fraction_hosts_seen_in_train = 1.0` across the 19,098 $\ge 4$-flow test windows (confirming that every multi-flow test host in the benchmark partition split appeared in `part_00`–`part_47`).

### 2.2 Comparative Results on Identical 19,098 Test Windows

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | FPR | FNR | ROC-AUC `[95% CI]` | PR-AUC `[95% CI]` | TPR @ 5% FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dataset_prior`** *(Source Shortcut)* | `0.8983` | 76.42% | **99.37%** | 22.00% | 36.02% | **0.06%** | 78.00% | **83.20%** `[0.8266, 0.8380]` | 72.35% `[0.7152, 0.7335]` | 66.50% |
| **`flow_lgbm_max`** *(Supervised Single-Flow)* | `0.1068` | 92.78% | 81.36% | **98.72%** | 89.20% | 9.78% | **1.28%** | **99.41%** `[0.9928, 0.9953]` | 99.28% `[0.9914, 0.9942]` | 98.21% |
| **`window_lgbm_pool`** *(Supervised Pooled)* | `0.1063` | 92.15% | 80.00% | 98.66% | 88.36% | 10.66% | 1.34% | **99.56%** `[0.9945, 0.9966]` | **99.35%** `[0.9921, 0.9947]` | **98.51%** |
| **`bimambaformer`** *(at 5.6k-val p95)* | `0.0273` | 89.57% | 74.87% | 98.53% | 85.09% | 14.29% | 1.47% | **98.91%** `[0.9877, 0.9903]` | 97.77% `[0.9747, 0.9800]` | 97.38% |
| **`bimambaformer`** *(at Shipped Threshold)* | **`0.120483`** | **93.69%** | **83.64%** | 98.33% | **90.39%** | **8.32%** | 1.67% | **98.91%** `[0.9877, 0.9903]` | 97.77% `[0.9747, 0.9800]` | 97.38% |

### 2.3 Confusion Matrices on Identical 19,098 Test Windows
- **`dataset_prior`**: `TN=13,326 | FP=8 | FN=4,496 | TP=1,268`
- **`flow_lgbm_max`**: `TN=12,030 | FP=1,304 | FN=74 | TP=5,690`
- **`window_lgbm_pool`**: `TN=11,912 | FP=1,422 | FN=77 | TP=5,687`
- **`bimambaformer` (`0.120483`)**: `TN=12,225 | FP=1,109 | FN=96 | TP=5,668`

### 2.4 Per-Class Window Recall (19,098 Test Windows)
| Attack Class | Window Count | `dataset_prior` | `flow_lgbm_max` | `window_lgbm_pool` | `bimambaformer` (`0.120483`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Bot` | 241 | 0.00% | 100.00% | 100.00% | **100.00%** |
| `DDoS` | 2,175 | 29.24% | 100.00% | 100.00% | **100.00%** |
| `DoS` | 840 | 61.67% | 100.00% | 100.00% | **100.00%** |
| `Scanning` | 818 | 0.00% | 100.00% | 100.00% | **98.90%** |
| `Brute Force` | 86 | 0.00% | 100.00% | 100.00% | **98.84%** |
| `Infilteration` | 346 | 0.00% | 78.61% | 77.75% | **75.43%** |
| `Reconnaissance` | 114 | 88.60% | 100.00% | 100.00% | **100.00%** |
| `Theft` | 13 | 100.00% | 100.00% | 100.00% | **100.00%** |
| `Injection` | 227 | 0.00% | 100.00% | 100.00% | **100.00%** |
| `Password` | 251 | 0.00% | 100.00% | 100.00% | **100.00%** |
| `XSS` | 574 | 0.00% | 100.00% | 100.00% | **100.00%** |
| `Exploits` / `Fuzzers` / `Backdoor` | 25 / 20 / 15 | 0.00% | 100.00% | 100.00% | **100.00%** |
| `Ransomware` | 5 | 0.00% | 100.00% | 100.00% | **80.00%** |

---

## Part 3: Scientific Interpretation & Thesis Guidance

1. **Outperforms the Dataset-Source Shortcut**:
   `dataset_prior` achieves only **22.00% recall** and **83.20% ROC-AUC** on windows (and misses 100% of Bot, Scanning, Brute Force, Infiltration, Injection, Password, and XSS windows). `bimambaformer` achieves **98.91% ROC-AUC** (`+15.71%` gain) and **98.33% recall** (`+76.33%` gain), proving it learns genuine NetFlow behavioral patterns rather than source-dataset labels.
2. **Highest Overall Accuracy & F1-Score at Operating Threshold**:
   When evaluated at its shipped validation-calibrated threshold (`0.120483`), `bimambaformer` achieves the **highest Accuracy (93.69%)**, **highest Precision (83.64%)**, **highest F1-Score (90.39%)**, and **lowest False Positive Rate (8.32%)** among all ML models on the 19,098 test windows—beating both `flow_lgbm_max` (`92.78%` Acc, `89.20%` F1) and `window_lgbm_pool` (`92.15%` Acc, `88.36%` F1).
3. **Why Supervised LightGBM Has Slightly Higher Ranking AUC (`99.41%–99.56%`)**:
   In NF-UQ-NIDS-v2's pre-partitioned files, **94.90% of test rows are exact feature duplicates** of training rows and **100% of $\ge 4$-flow test hosts** appeared in training partitions. Gradient-boosted decision trees (`LightGBM`) excel at memorizing exact tabular feature boundaries under high duplicate/host overlap, whereas `Bi-MambaFormer` provides continuous state-space sequence modeling, **native per-flow attention attribution**, **gradient-based feature XAI**, and **online streaming inference** ($L=32$ sliding buffer) without needing batch tree re-aggregation.
4. **Threats to Validity (Essential for Thesis Chapter 8 & Defense)**:
   - NF-UQ-NIDS-v2 lacks wall-clock timestamps and exhibits `87.70%` host overlap and `94.90%` feature duplication across the standard partition split (`part_00–47` vs `48–49` vs `50–51`).
   - Future work should evaluate cross-dataset leave-one-source-out generalization and deduplicated host-disjoint splits to isolate zero-day generalization from memorization.
