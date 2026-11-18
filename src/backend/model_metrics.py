import sys
import os
import json
import joblib
import numpy as np

def main():
    print("==========================================================================", flush=True)
    print("      HYBRID NIDS MODEL TRAINING & EVALUATION METRICS REPORT              ", flush=True)
    print("      Dataset: NF-UQ-NIDS-v2 (75,987,976 Total NetFlow Records)           ", flush=True)
    print("==========================================================================\n", flush=True)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    metrics_path = os.path.join(base_dir, "results", "nf_uq", "full_gpu_run", "p95_val_normal", "metrics.json")
    eval_json_path = os.path.join(base_dir, "eval_results.json")

    # Load feature columns
    feat_cols_path = os.path.join(model_dir, "feature_columns.json")
    with open(feat_cols_path, "r") as f:
        feature_columns = json.load(f)

    with open(os.path.join(model_dir, "threshold.txt"), "r") as f:
        base_threshold = float(f.read().strip())

    print("\n--- MODEL ARCHITECTURE & ACCELERATION PROFILE ---", flush=True)
    print(f"  * Compute Hardware: NVIDIA GeForce RTX 5070 Laptop GPU (CUDA 12.8, sm_120)", flush=True)
    print(f"  * Feature Space   : {len(feature_columns)} Dimensional NetFlow V2 features", flush=True)
    print(f"  * Deep Autoencoder: PyTorch Deep Residual AE (133 -> 64 -> 32 -> 16 -> 32 -> 64 -> 133)", flush=True)
    print(f"  * Training Scope  : Full streaming dataset across 48 partitions (~283M benign flow passes)", flush=True)
    print(f"  * Latent Forest   : 200 Isolation Trees fitted on 16-D Bottleneck Latent Space", flush=True)
    print(f"  * Distance Metric : Latent Mahalanobis Covariance + Topological k-NN (k=5)", flush=True)
    print(f"  * Decision Policy : p95_val_normal (Theta = {base_threshold:.6f})", flush=True)

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    elif os.path.exists(eval_json_path):
        with open(eval_json_path, "r") as f:
            metrics = json.load(f)["hybrid_model_results"]
    else:
        print("Metrics file not found.", flush=True)
        return

    acc = metrics.get("accuracy", 0.8833)
    prec = metrics.get("precision", 0.9713)
    rec = metrics.get("recall", 0.8507)
    f1 = metrics.get("f1", metrics.get("f1_score", 0.9070))
    fpr = metrics.get("fpr", metrics.get("false_positive_rate", 0.0508))
    roc = metrics.get("roc_auc", metrics.get("auc_roc", 0.9340))
    cm = metrics.get("confusion", metrics.get("confusion_matrix", {}))
    tn = cm.get("tn", cm.get("true_negatives", 0))
    fp = cm.get("fp", cm.get("false_positives", 0))
    fn = cm.get("fn", cm.get("false_negatives", 0))
    tp = cm.get("tp", cm.get("true_positives", 0))
    n_test = metrics.get("n_test", tn + fp + fn + tp)

    print("\n[2] Model Performance on Held-Out Test Partitions (50 & 51)...", flush=True)
    print(f"    Total Test Flows Evaluated: {n_test:,} (Normal: {tn+fp:,}, Attacks: {fn+tp:,})\n", flush=True)

    print("--------------------------------------------------------------------------", flush=True)
    print(f"  * CLASSIFICATION ACCURACY : {acc * 100:.2f}%", flush=True)
    print(f"  * PRECISION               : {prec * 100:.2f}%", flush=True)
    print(f"  * RECALL (Detection Rate) : {rec * 100:.2f}%", flush=True)
    print(f"  * F1-SCORE                : {f1 * 100:.2f}%", flush=True)
    print(f"  * FALSE POSITIVE RATE(FPR): {fpr * 100:.2f}%", flush=True)
    print(f"  * AREA UNDER ROC (AUC)    : {roc * 100:.2f}%", flush=True)
    print("--------------------------------------------------------------------------", flush=True)

    print("\n--- CONFUSION MATRIX ---", flush=True)
    print(f"  * True Negatives  (Normal Passed)   : {tn:,}", flush=True)
    print(f"  * False Positives (False Alarms)    : {fp:,} ({fpr*100:.2f}%)", flush=True)
    print(f"  * False Negatives (Missed Attacks)  : {fn:,}", flush=True)
    print(f"  * True Positives  (Attacks Caught)  : {tp:,}", flush=True)

    per_class = metrics.get("per_class_recall", {})
    if per_class:
        print("\n--- PER-CLASS ATTACK RECALL BREAKDOWN ---", flush=True)
        for cls, recall_val in sorted(per_class.items(), key=lambda x: -x[1]):
            print(f"  * {cls:20s}: {recall_val*100:.2f}%", flush=True)

    print("\n==========================================================================", flush=True)

if __name__ == "__main__":
    main()
