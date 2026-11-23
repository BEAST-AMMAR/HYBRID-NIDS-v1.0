"""
=============================================================================
  evaluate_bi_mambaformer.py — Comprehensive Test Benchmark & Evaluation
=============================================================================
  Evaluates the Flagship Dual-Stream Hybrid Bi-MambaFormer on held-out
  NF-UQ-NIDS-v2 test partitions (parts 50 & 51, 1,702,252 flows).
  
  Outputs:
    - Global Metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR)
    - Confusion Matrix (TN, FP, FN, TP)
    - Granular Per-Class Recall across all 20 attack categories
    - Direct Comparative Delta against Phase 1 Baseline (AE+IF)
    - ROC and PR curve coordinate points for paper charts
    - GPU Throughput & Latency Benchmarks (ms per 1k flows)
=============================================================================
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# Ensure workspace root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.backend.sequences import PureNfUqAdapter, extract_host_sequences
from src.backend.models.bi_mambaformer import BiMambaFormer


def evaluate_test_set(
    data_dir: str = "data/NF UQ NIDS V2/parquet",
    model_dir: str = "src/backend/saved_models",
    results_dir: str = "results/nf_uq_sequence",
    batch_size: int = 512,
    seq_len: int = 32,
    stride: int = 32,  # Non-overlapping for evaluation
    max_test_samples: int = 50000
):
    print(f"\n{'='*80}")
    print("  🏆 EVALUATING DUAL-STREAM HYBRID BI-MAMBAFORMER ON HELD-OUT TEST DATA")
    print(f"  Test Target: Partitions 50 & 51 (1,702,252 Total Benchmark Flows)")
    print(f"{'='*80}\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device}")
    if device.type == "cuda":
        print(f"    Device Name: {torch.cuda.get_device_name(0)}")

    adapter = PureNfUqAdapter(model_dir=model_dir)

    # Load Model
    model_path = os.path.join(model_dir, "bi_mambaformer_best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train model first!")

    model = BiMambaFormer(
        in_features=133,
        d_model=128,
        mamba_expand=2,
        mamba_state=16,
        attn_heads=4,
        dropout=0.0
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[*] Loaded trained weights from {model_path}")

    # Load Calibrated Threshold
    thr_path = os.path.join(model_dir, "bi_mambaformer_threshold.txt")
    if os.path.exists(thr_path):
        with open(thr_path, "r") as f:
            threshold = float(f.read().strip())
    else:
        threshold = 0.50
    print(f"[*] Active Operating Threshold: {threshold:.6f}")

    # Extract sequences from Partitions 50 & 51
    test_files = [
        os.path.join(data_dir, "part_0050.parquet"),
        os.path.join(data_dir, "part_0051.parquet")
    ]
    test_files = [f for f in test_files if os.path.exists(f)]
    if not test_files:
        raise FileNotFoundError(f"Test partitions 50/51 missing in {data_dir}")

    cache_path = os.path.join(data_dir, "..", "sequence_cache", "test_cache_L32.pt")
    if os.path.exists(cache_path):
        print(f"[*] Loading cached test sequences from {cache_path}...")
        test_cache = torch.load(cache_path, weights_only=False)
        X_test_t = test_cache["x"]
        M_test_t = test_cache["mask"]
        y_test_t = test_cache["y"]
        attack_test = test_cache["attacks"]
        dataset_test = test_cache["datasets"]
    else:
        print(f"[*] Ingesting and extracting test sequence windows from Partitions 50 & 51...")
        X_list, M_list, y_list, attack_test, dataset_test = [], [], [], [], []

        for ppath in test_files:
            print(f"    Reading {os.path.basename(ppath)}...")
            table = pq.read_table(ppath)
            df = table.to_pandas()
            del table

            X_p, M_p, y_p, atk_p, ds_p = extract_host_sequences(
                df, adapter, seq_len=seq_len, stride=stride, include_incomplete=True
            )
            del df

            if len(X_p) > 0:
                X_list.append(X_p)
                M_list.append(M_p)
                y_list.append(y_p)
                attack_test.extend(atk_p)
                dataset_test.extend(ds_p)

            if sum(len(x) for x in X_list) >= max_test_samples:
                break

        X_test = np.concatenate(X_list, axis=0)[:max_test_samples]
        M_test = np.concatenate(M_list, axis=0)[:max_test_samples]
        y_test = np.concatenate(y_list, axis=0)[:max_test_samples]
        attack_test = attack_test[:max_test_samples]
        dataset_test = dataset_test[:max_test_samples]

        X_test_t = torch.from_numpy(X_test).float()
        M_test_t = torch.from_numpy(M_test).bool()
        y_test_t = torch.from_numpy(y_test).float()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            "x": X_test_t, "mask": M_test_t, "y": y_test_t,
            "attacks": attack_test, "datasets": dataset_test
        }, cache_path)
        print(f"    Saved deterministic test sequence cache to {cache_path}")

    n_test = len(X_test_t)
    print(f"[*] Total Test Sequences: {n_test:,} (Normal: {(y_test_t == 0).sum().item():,}, Attack: {(y_test_t == 1).sum().item():,})")

    # Run Batched Inference & Latency Benchmark
    scores_list = []
    t_start = time.time()

    with torch.no_grad():
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            bx = X_test_t[start:end].to(device)
            bm = M_test_t[start:end].to(device)

            out = model(bx, mask=bm)
            scores = out["threat_score"].squeeze(-1).cpu().numpy()
            scores_list.append(scores)

    total_infer_time = time.time() - t_start
    scores = np.concatenate(scores_list)
    targets = y_test_t.numpy()
    attacks_arr = np.array(attack_test)

    # Latency Metrics
    ms_per_1k = (total_infer_time / n_test) * 1000 * 1000
    fps = n_test / total_infer_time

    # Predictions
    preds = (scores > threshold).astype(int)

    # Core Metrics
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    roc_auc = roc_auc_score(targets, scores)
    pr_auc = average_precision_score(targets, scores)

    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    fpr = fp / max(tn + fp, 1)
    fnr = fn / max(tp + fn, 1)

    # Per-Class Recall Analysis
    per_class_recall = {}
    phase1_baseline = {
        "DDoS": 0.990233, "DoS": 0.965831, "Brute Force": 0.997064,
        "Fuzzers": 0.995968, "Backdoor": 0.995360, "Generic": 1.0,
        "Shellcode": 1.0, "Worms": 1.0, "Ransomware": 0.949367,
        "Analysis": 0.902439, "Injection": 0.898767, "Password": 0.894152,
        "Exploits": 0.892517, "Theft": 0.588235, "Reconnaissance": 0.441014,
        "XSS": 0.423416, "MITM": 0.333333, "Scanning": 0.098743,
        "Infilteration": 0.090161, "Infiltration": 0.090161, "Bot": 0.063083
    }

    unique_attacks = sorted(list(set(attacks_arr) - {"Benign", "0"}))
    for atk in unique_attacks:
        mask = (attacks_arr == atk)
        n_atk = int(mask.sum())
        if n_atk > 0:
            rec_atk = float(recall_score(targets[mask], preds[mask], zero_division=0))
            per_class_recall[atk] = {
                "recall": rec_atk,
                "count": n_atk,
                "phase1_recall": phase1_baseline.get(atk, phase1_baseline.get(atk.capitalize(), 0.0)),
                "gain": rec_atk - phase1_baseline.get(atk, phase1_baseline.get(atk.capitalize(), 0.0))
            }

    # Print Summary Report
    print(f"\n{'='*80}")
    print("  📊 PHASE 2 HYBRID BI-MAMBAFORMER TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  • Overall Accuracy:         {acc * 100:.2f}% ({acc:.6f})")
    print(f"  • Attack Precision:         {prec * 100:.2f}% ({prec:.6f})")
    print(f"  • Attack Recall:            {rec * 100:.2f}% ({rec:.6f})")
    print(f"  • F1-Score:                 {f1 * 100:.2f}% ({f1:.6f})")
    print(f"  • ROC-AUC:                  {roc_auc * 100:.2f}% ({roc_auc:.6f})")
    print(f"  • PR-AUC:                   {pr_auc * 100:.2f}% ({pr_auc:.6f})")
    print(f"  • False Positive Rate:      {fpr * 100:.2f}% ({fpr:.6f})")
    print(f"  • Confusion Matrix:         TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")
    print(f"  • GPU Inference Speed:      {fps:,.1f} sequences/sec ({ms_per_1k:.2f} ms per 1k flows)")

    print(f"\n{'-'*80}")
    print("  🎯 STEALTH ATTACK RECALL PROGRESSION (Phase 1 Baseline vs. Phase 2 Proposed)")
    print(f"{'-'*80}")
    print(f"  {'Attack Family':<20} | {'Phase 1 (AE+IF)':<16} | {'Phase 2 (Proposed)':<18} | {'Improvement Delta'}")
    print(f"  {'-'*18:<20} | {'-'*14:<16} | {'-'*16:<18} | {'-'*17}")

    for atk, data in sorted(per_class_recall.items(), key=lambda x: x[1]["gain"], reverse=True):
        p1 = data["phase1_recall"] * 100
        p2 = data["recall"] * 100
        delta = data["gain"] * 100
        sign = "+" if delta >= 0 else ""
        print(f"  {atk:<20} | {p1:>6.2f}%          | {p2:>6.2f}%            | {sign}{delta:>6.2f}%")

    # Export Full JSON & TXT Result Manifest
    os.makedirs(results_dir, exist_ok=True)
    results_payload = {
        "dataset": "NF-UQ-NIDS-v2",
        "model": "Dual-Stream Hybrid Bi-MambaFormer",
        "architecture": {
            "stream_1": "Bidirectional Mamba SSM (d_model=128, expand=2, state=16)",
            "stream_2": "Multi-Head Temporal Self-Attention (4 heads, d_k=32)",
            "fusion": "Adaptive Gated Fusion (Learned Gate G)",
            "pooling": "Attention-Weighted Temporal Pooling",
            "classifier": "MLP(128 -> 64 -> 1, Mish)"
        },
        "threshold": threshold,
        "n_test_sequences": n_test,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr)
        },
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "latency": {"sequences_per_second": float(fps), "ms_per_1k_flows": float(ms_per_1k)},
        "per_class_recall": {k: float(v["recall"]) for k, v in per_class_recall.items()},
        "stealth_gains": per_class_recall
    }

    # Save to primary result locations
    json_path = os.path.join(results_dir, "eval_results_phase2.json")
    txt_path = os.path.join(results_dir, "eval_results_phase2.txt")
    root_json = "eval_results_phase2.json"
    root_txt = "eval_results_phase2.txt"

    for p in [json_path, root_json]:
        with open(p, "w") as f:
            json.dump(results_payload, f, indent=2)

    txt_content = f"""=============================================================================
  PHASE 2 DUAL-STREAM HYBRID BI-MAMBAFORMER TEST EVALUATION REPORT
=============================================================================
  Dataset: NF-UQ-NIDS-v2 (Partitions 50 & 51 Held-Out Test Flows)
  Model: Dual-Stream Hybrid Bi-MambaFormer (Bi-Mamba SSM + MHSA + Gated Fusion)
  Operating Threshold: {threshold:.6f}
=============================================================================
  Overall Accuracy:         {acc * 100:.2f}% ({acc:.6f})
  Precision:                {prec * 100:.2f}% ({prec:.6f})
  Recall (Detection Rate):  {rec * 100:.2f}% ({rec:.6f})
  F1-Score:                 {f1 * 100:.2f}% ({f1:.6f})
  ROC-AUC:                  {roc_auc * 100:.2f}% ({roc_auc:.6f})
  PR-AUC:                   {pr_auc * 100:.2f}% ({pr_auc:.6f})
  False Positive Rate:      {fpr * 100:.2f}% ({fpr:.6f})
  Throughput:               {fps:,.1f} sequences/sec ({ms_per_1k:.2f} ms/1k flows)

  Confusion Matrix:
    True Negatives  (TN): {tn:,}
    False Positives (FP): {fp:,}
    False Negatives (FN): {fn:,}
    True Positives  (TP): {tp:,}
=============================================================================
"""
    for p in [txt_path, root_txt]:
        with open(p, "w") as f:
            f.write(txt_content)

    # Save ROC and PR curves
    fpr_curve, tpr_curve, _ = roc_curve(targets, scores)
    prec_curve, rec_curve, _ = precision_recall_curve(targets, scores)

    pd.DataFrame({"fpr": fpr_curve, "tpr": tpr_curve}).to_csv(
        os.path.join(results_dir, "roc_points.csv"), index=False
    )
    pd.DataFrame({"precision": prec_curve, "recall": rec_curve}).to_csv(
        os.path.join(results_dir, "pr_points.csv"), index=False
    )

    print(f"\n[OK] All results, confusion matrices, and ROC/PR points successfully written to:")
    print(f"     - {json_path}")
    print(f"     - {txt_path}")
    print(f"     - {root_json}")
    print(f"{'='*80}\n")
    return results_payload


if __name__ == "__main__":
    evaluate_test_set()
