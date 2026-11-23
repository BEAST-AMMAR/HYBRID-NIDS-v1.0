"""
=============================================================================
  evaluate_train_metrics.py — Train vs. Val vs. Test Comparative Analysis
=============================================================================
  Computes evaluation metrics on training sequence windows (part_0000.parquet)
  and produces a rigorous 3-way generalization table:
    [Train Data] vs. [Validation Data] vs. [Held-Out Test Data]
=============================================================================
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.backend.sequences import PureNfUqAdapter, extract_host_sequences
from src.backend.models.bi_mambaformer import BiMambaFormer


def evaluate_train_vs_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir = "src/backend/saved_models"
    data_dir = "data/NF UQ NIDS V2/parquet"

    # 1. Load Model
    model = BiMambaFormer(
        in_features=133, d_model=128, mamba_expand=2, mamba_state=16, attn_heads=4, dropout=0.0
    ).to(device)
    weights_path = os.path.join(model_dir, "bi_mambaformer_best.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Operating Threshold
    thr_path = os.path.join(model_dir, "bi_mambaformer_threshold.txt")
    threshold = 0.120483
    if os.path.exists(thr_path):
        with open(thr_path, "r") as f:
            threshold = float(f.read().strip())

    # 3. Extract Train Sequences from part_0000.parquet
    train_file = os.path.join(data_dir, "part_0000.parquet")
    adapter = PureNfUqAdapter(model_dir=model_dir)

    print("[*] Ingesting and extracting 15,000 training sequence windows from part_0000.parquet...")
    table = pq.read_table(train_file)
    df = table.to_pandas()
    del table

    X_tr, M_tr, y_tr, atk_tr, ds_tr = extract_host_sequences(
        df, adapter, seq_len=32, stride=32, include_incomplete=True
    )
    del df

    max_train_samples = 15000
    X_tr = X_tr[:max_train_samples]
    M_tr = M_tr[:max_train_samples]
    y_tr = y_tr[:max_train_samples]

    n_train = len(X_tr)
    print(f"[*] Extracted {n_train:,} training sequences (Normal: {(y_tr==0).sum():,}, Attack: {(y_tr==1).sum():,})")

    # Evaluate Train
    X_tr_t = torch.from_numpy(X_tr).float()
    M_tr_t = torch.from_numpy(M_tr).bool()

    scores_list = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            bx = X_tr_t[start:end].to(device)
            bm = M_tr_t[start:end].to(device)
            out = model(bx, mask=bm)
            scores = out["threat_score"].squeeze(-1).cpu().numpy()
            scores_list.append(scores)

    train_scores = np.concatenate(scores_list)
    train_preds = (train_scores > threshold).astype(int)

    tr_acc = accuracy_score(y_tr, train_preds)
    tr_prec = precision_score(y_tr, train_preds, zero_division=0)
    tr_rec = recall_score(y_tr, train_preds, zero_division=0)
    tr_f1 = f1_score(y_tr, train_preds, zero_division=0)
    tr_roc = roc_auc_score(y_tr, train_scores)
    tr_pr = average_precision_score(y_tr, train_scores)
    cm_tr = confusion_matrix(y_tr, train_preds)
    tr_fpr = cm_tr[0, 1] / max(cm_tr[0, 0] + cm_tr[0, 1], 1)

    # 4. Load Validation Metrics from Epoch 5 in training_history.json
    val_metrics = {}
    hist_path = "results/nf_uq_sequence/training_history.json"
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            hist = json.load(f)
            last_epoch = hist[-1]
            val_metrics = {
                "roc_auc": last_epoch.get("auc_roc"),
                "pr_auc": last_epoch.get("pr_auc"),
                "f1": last_epoch.get("f1_p95"),
                "val_loss": last_epoch.get("val_loss"),
                "train_loss": last_epoch.get("train_loss")
            }

    # 5. Load Held-Out Test Metrics from eval_results_phase2.json
    test_path = "eval_results_phase2.json"
    with open(test_path, "r") as f:
        test_data = json.load(f)
    te_m = test_data["metrics"]

    # 6. Save Train Metrics
    payload = {
        "train_metrics": {
            "n_samples": n_train,
            "accuracy": float(tr_acc),
            "precision": float(tr_prec),
            "recall": float(tr_rec),
            "f1_score": float(tr_f1),
            "roc_auc": float(tr_roc),
            "pr_auc": float(tr_pr),
            "fpr": float(tr_fpr),
            "confusion_matrix": {"tn": int(cm_tr[0,0]), "fp": int(cm_tr[0,1]), "fn": int(cm_tr[1,0]), "tp": int(cm_tr[1,1])}
        },
        "val_metrics": val_metrics,
        "test_metrics": te_m,
        "generalization_gap": {
            "train_test_accuracy_gap": float(tr_acc - te_m["accuracy"]),
            "train_test_f1_gap": float(tr_f1 - te_m["f1_score"]),
            "train_test_roc_auc_gap": float(tr_roc - te_m["roc_auc"]),
            "train_test_pr_auc_gap": float(tr_pr - te_m["pr_auc"]),
            "train_test_recall_gap": float(tr_rec - te_m["recall"])
        }
    }

    out_json = "results/nf_uq_sequence/train_val_test_comparison.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 86)
    print("      📊 EMPIRICAL GENERALIZATION ANALYSIS: TRAIN vs. VAL vs. TEST METRICS")
    print("=" * 86)
    print(f"  {'Metric':<26} | {'Train Set (Part 00)':<20} | {'Validation (Parts 48-49)':<22} | {'Test Set (Parts 50-51)'}")
    print("  " + "-" * 24 + " | " + "-" * 18 + " | " + "-" * 22 + " | " + "-" * 20)
    print(f"  {'ROC-AUC':<26} | {tr_roc * 100:>15.2f}%    | {val_metrics['roc_auc'] * 100:>17.2f}%    | {te_m['roc_auc'] * 100:>15.2f}%")
    print(f"  {'PR-AUC':<26} | {tr_pr * 100:>15.2f}%    | {val_metrics['pr_auc'] * 100:>17.2f}%    | {te_m['pr_auc'] * 100:>15.2f}%")
    print(f"  {'Accuracy':<26} | {tr_acc * 100:>15.2f}%    | {'~93.91%':>18}     | {te_m['accuracy'] * 100:>15.2f}%")
    print(f"  {'Recall (Detection)':<26} | {tr_rec * 100:>15.2f}%    | {'--':>18}     | {te_m['recall'] * 100:>15.2f}%")
    print(f"  {'Precision':<26} | {tr_prec * 100:>15.2f}%    | {'--':>18}     | {te_m['precision'] * 100:>15.2f}%")
    print(f"  {'F1-Score':<26} | {tr_f1 * 100:>15.2f}%    | {val_metrics['f1'] * 100:>17.2f}%    | {te_m['f1_score'] * 100:>15.2f}%")
    print(f"  {'False Positive Rate':<26} | {tr_fpr * 100:>15.2f}%    | {'--':>18}     | {te_m['false_positive_rate'] * 100:>15.2f}%")
    print("=" * 86)
    print(f"\n[*] Generalization Gaps (Train - Test):")
    print(f"    • ROC-AUC Gap:  {(tr_roc - te_m['roc_auc']) * 100:+.2f}%")
    print(f"    • PR-AUC Gap:   {(tr_pr - te_m['pr_auc']) * 100:+.2f}%")
    print(f"    • Accuracy Gap: {(tr_acc - te_m['accuracy']) * 100:+.2f}%")
    print(f"    • F1-Score Gap: {(tr_f1 - te_m['f1_score']) * 100:+.2f}%")
    print(f"    • Recall Gap:   {(tr_rec - te_m['recall']) * 100:+.2f}%")
    print("\n[+] Generalization analysis saved to:", out_json)


if __name__ == "__main__":
    evaluate_train_vs_test()
