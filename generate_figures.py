"""
=============================================================================
  generate_figures.py — Publication-Quality Thesis Figures Generator
=============================================================================
  Generates 6 high-resolution figures for research thesis and paper submission:
    1. ROC Curve (Phase 2 Bi-MambaFormer)
    2. Precision-Recall Curve (Phase 2 Bi-MambaFormer)
    3. Training Loss History (Train vs Validation)
    4. Training ROC-AUC & PR-AUC History
    5. Stealth Attack Recall: Baseline 2 vs Phase 2 (Grouped Bar Chart)
    6. Cross-Baseline Performance Radar/Comparison Bar Chart
  All output saved to results/figures/ as 300 DPI PNG files.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_DIR = os.path.join(BASE_DIR, "results", "nf_uq_sequence")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

DARK_BLUE = "#1a5276"
TEAL = "#148f77"
RED = "#c0392b"
ORANGE = "#e67e22"
PURPLE = "#8e44ad"
GREY = "#7f8c8d"


# ===== Figure 1: ROC Curve =====
def fig_roc_curve():
    roc = pd.read_csv(os.path.join(SEQ_DIR, "roc_points.csv"))
    fig, ax = plt.subplots(figsize=(6, 5.5))

    ax.plot(roc["fpr"], roc["tpr"], color=DARK_BLUE, linewidth=2.0,
            label="Bi-MambaFormer (AUC = 0.9891)")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.0,
            label="Random Classifier")

    # Mark operating points
    ax.scatter([0.0832], [0.9833], color=RED, s=80, zorder=5, marker="D",
               label=f"p95 (FPR=8.3%, TPR=98.3%)")
    ax.scatter([0.03], [0.9615], color=ORANGE, s=80, zorder=5, marker="s",
               label=f"fpr_3pct (FPR≈3%, TPR=96.2%)")

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve\nPhase 2 Dual-Stream Bi-MambaFormer on NF-UQ-NIDS-v2")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    path = os.path.join(FIG_DIR, "fig1_roc_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [1/6] Saved: {path}")


# ===== Figure 2: Precision-Recall Curve =====
def fig_pr_curve():
    pr = pd.read_csv(os.path.join(SEQ_DIR, "pr_points.csv"))
    fig, ax = plt.subplots(figsize=(6, 5.5))

    ax.plot(pr["recall"], pr["precision"], color=TEAL, linewidth=2.0,
            label="Bi-MambaFormer (PR-AUC = 0.9777)")

    ax.scatter([0.9833], [0.8364], color=RED, s=80, zorder=5, marker="D",
               label="p95 Operating Point")

    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve\nPhase 2 Dual-Stream Bi-MambaFormer on NF-UQ-NIDS-v2")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.0, 1.05])

    path = os.path.join(FIG_DIR, "fig2_pr_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [2/6] Saved: {path}")


# ===== Figure 3: Training Loss History =====
def fig_training_loss():
    with open(os.path.join(SEQ_DIR, "training_history.json")) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(epochs, train_loss, "o-", color=DARK_BLUE, linewidth=2.0, markersize=7, label="Training Loss")
    ax.plot(epochs, val_loss, "s--", color=RED, linewidth=2.0, markersize=7, label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss History\nBi-MambaFormer (AdamW + Cosine Annealing, AMP FP16)")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    path = os.path.join(FIG_DIR, "fig3_training_loss.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [3/6] Saved: {path}")


# ===== Figure 4: ROC-AUC & PR-AUC per Epoch =====
def fig_training_auc():
    with open(os.path.join(SEQ_DIR, "training_history.json")) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    roc_auc = [h["auc_roc"] for h in history]
    pr_auc = [h["pr_auc"] for h in history]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(epochs, roc_auc, "o-", color=DARK_BLUE, linewidth=2.0, markersize=7, label="Validation ROC-AUC")
    ax.plot(epochs, pr_auc, "s-", color=TEAL, linewidth=2.0, markersize=7, label="Validation PR-AUC")

    ax.axhline(y=0.9891, color=DARK_BLUE, linestyle=":", alpha=0.5, label="Final Test ROC-AUC (0.9891)")
    ax.axhline(y=0.9777, color=TEAL, linestyle=":", alpha=0.5, label="Final Test PR-AUC (0.9777)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC Score")
    ax.set_title("Validation ROC-AUC & PR-AUC per Epoch\nBi-MambaFormer on NF-UQ-NIDS-v2")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_ylim([0.92, 1.0])
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    path = os.path.join(FIG_DIR, "fig4_training_auc.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [4/6] Saved: {path}")


# ===== Figure 5: Stealth Attack Recall Comparison =====
def fig_stealth_recall():
    categories = [
        "Bot\n(C2)", "Scanning", "Infiltration", "Recon.", "Theft",
        "MITM", "XSS", "Exploits", "Injection", "Password",
        "DoS", "DDoS", "Brute\nForce"
    ]
    baseline2 = [6.31, 9.87, 9.02, 44.10, 58.82, 33.33, 42.34, 89.25, 89.88, 89.42, 96.58, 99.02, 99.71]
    phase2 = [100.0, 98.90, 75.43, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 98.84]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars1 = ax.bar(x - width/2, baseline2, width, label="Baseline 2 (AE + Ensemble)", color="#e74c3c", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, phase2, width, label="Phase 2 (Bi-MambaFormer)", color="#2ecc71", alpha=0.85, edgecolor="white")

    ax.set_xlabel("Attack Category")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Per-Class Attack Recall: Baseline 2 (Memoryless) vs. Phase 2 (Temporal Bi-MambaFormer)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim([0, 110])
    ax.axhline(y=50, color=GREY, linestyle="--", alpha=0.3)

    # Annotate the most dramatic gains
    for i, (b, p) in enumerate(zip(baseline2, phase2)):
        gain = p - b
        if gain > 40:
            ax.annotate(f"+{gain:.0f}%", xy=(x[i] + width/2, p + 1), ha="center",
                        fontsize=7, fontweight="bold", color="#27ae60")

    path = os.path.join(FIG_DIR, "fig5_stealth_recall_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [5/6] Saved: {path}")


# ===== Figure 6: Cross-Baseline Performance Comparison =====
def fig_cross_baseline():
    metrics = ["Accuracy", "Recall", "ROC-AUC", "F1-Score", "Precision"]
    baseline1 = [90.52, 92.32, 96.72, 91.73, 91.14]
    baseline2 = [88.33, 85.07, 93.40, 90.70, 97.13]
    phase2 = [93.69, 98.33, 98.91, 90.39, 83.64]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, baseline1, width, label="Baseline 1\n(NSL-KDD, AE+IF)", color="#3498db", alpha=0.85, edgecolor="white")
    ax.bar(x, baseline2, width, label="Baseline 2\n(NF-UQ, AE+Ensemble)", color="#e67e22", alpha=0.85, edgecolor="white")
    ax.bar(x + width, phase2, width, label="Phase 2\n(NF-UQ, Bi-MambaFormer)", color="#2ecc71", alpha=0.85, edgecolor="white")

    ax.set_xlabel("Evaluation Metric")
    ax.set_ylabel("Score (%)")
    ax.set_title("Cross-Baseline Performance Comparison\nAll Three Experimental Architectures")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax.set_ylim([75, 105])

    # Add value labels on top of bars
    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    path = os.path.join(FIG_DIR, "fig6_cross_baseline_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [6/6] Saved: {path}")


# ===== Main =====
if __name__ == "__main__":
    print("=" * 70)
    print("  GENERATING PUBLICATION-QUALITY THESIS FIGURES (300 DPI)")
    print("=" * 70)

    fig_roc_curve()
    fig_pr_curve()
    fig_training_loss()
    fig_training_auc()
    fig_stealth_recall()
    fig_cross_baseline()

    print("=" * 70)
    print(f"  ALL 6 FIGURES SAVED TO: {FIG_DIR}")
    print("=" * 70)
