"""
=============================================================================
  train_bi_mambaformer.py — RTX 5070 GPU Streaming Training Pipeline
=============================================================================
  Trains the Dual-Stream Hybrid Bi-MambaFormer on NF-UQ-NIDS-v2 telemetry.
  
  Features:
    - PyTorch CUDA 12.8 with Automatic Mixed Precision (AMP)
    - Out-of-core streaming from partitioned Parquet files
    - Cosine Annealing learning rate schedule with AdamW
    - Gradient clipping for absolute numerical stability
    - Validation-calibrated decision thresholds (p95, p99, youden)
    - Full unbuffered real-time logging
=============================================================================
"""

import os
import sys
import time
import glob
import json
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Ensure workspace root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.backend.sequences import PureNfUqAdapter, StreamingSequenceDataset, extract_host_sequences
from src.backend.models.bi_mambaformer import BiMambaFormer


def set_seed(seed: int = 11):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def prepare_validation_cache(
    val_files: list,
    adapter: PureNfUqAdapter,
    cache_path: str,
    max_val_samples: int = 12000,
    seq_len: int = 32,
    stride: int = 32
) -> tuple:
    """Pre-extracts and caches a deterministic validation set for fast epoch evaluation."""
    if os.path.exists(cache_path):
        print(f"[*] Loading cached validation sequences from {cache_path}...", flush=True)
        data = torch.load(cache_path, weights_only=False)
        return data["x"], data["mask"], data["y"], data["attacks"]

    print(f"[*] Building deterministic validation sequence cache (target {max_val_samples:,} windows)...", flush=True)
    val_x_list = []
    val_m_list = []
    val_y_list = []
    val_attacks = []

    for f in val_files:
        if not os.path.exists(f):
            continue
        print(f"    Reading validation partition {os.path.basename(f)}...", flush=True)
        table = pq.read_table(f)
        if len(table) > 250000:
            table = table.slice(0, 250000)
        df = table.to_pandas()
        del table

        X_s, M_s, y_s, attacks, _ = extract_host_sequences(
            df, adapter, seq_len=seq_len, stride=stride, min_flows=4, include_incomplete=False, max_windows_per_host=100
        )
        del df

        if len(X_s) > 0:
            val_x_list.append(X_s)
            val_m_list.append(M_s)
            val_y_list.append(y_s)
            val_attacks.extend(attacks)
            print(f"    Accumulated {sum(len(x) for x in val_x_list):,} validation windows...", flush=True)

        curr_total = sum(len(x) for x in val_x_list)
        if curr_total >= max_val_samples:
            break

    val_x = np.concatenate(val_x_list, axis=0)[:max_val_samples]
    val_m = np.concatenate(val_m_list, axis=0)[:max_val_samples]
    val_y = np.concatenate(val_y_list, axis=0)[:max_val_samples]
    val_attacks = val_attacks[:max_val_samples]

    val_x_t = torch.from_numpy(val_x).float()
    val_m_t = torch.from_numpy(val_m).bool()
    val_y_t = torch.from_numpy(val_y).float()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"x": val_x_t, "mask": val_m_t, "y": val_y_t, "attacks": val_attacks}, cache_path)
    print(f"    Validation cache saved: {len(val_x_t):,} windows (Benign: {(val_y_t==0).sum().item():,}, Attack: {(val_y_t==1).sum().item():,})", flush=True)
    return val_x_t, val_m_t, val_y_t, val_attacks


def evaluate_validation(
    model: nn.Module,
    val_x: torch.Tensor,
    val_m: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    batch_size: int = 512
) -> dict:
    """Evaluates model on validation sequences in batches."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    scores_list = []
    n_samples = len(val_x)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            bx = val_x[start:end].to(device)
            bm = val_m[start:end].to(device)
            by = val_y[start:end].to(device)

            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                out = model(bx, mask=bm)
                logits = out["logits"].squeeze(-1)
                loss = criterion(logits, by)

            total_loss += loss.item() * (end - start)
            scores_list.append(out["threat_score"].squeeze(-1).cpu().numpy())

    val_scores = np.concatenate(scores_list)
    val_targets = val_y.numpy()

    val_loss = total_loss / n_samples
    auc_roc = roc_auc_score(val_targets, val_scores)
    pr_auc = average_precision_score(val_targets, val_scores)

    # Threshold calibration on validation set
    benign_scores = val_scores[val_targets == 0]
    p95_thr = float(np.percentile(benign_scores, 95))
    p99_thr = float(np.percentile(benign_scores, 99))

    # Youden threshold
    from sklearn.metrics import roc_curve
    fprs, tprs, thrs = roc_curve(val_targets, val_scores)
    j_scores = tprs - fprs
    youden_thr = float(thrs[np.argmax(j_scores)])

    # Precision at p95
    preds_p95 = (val_scores > p95_thr).astype(int)
    f1_p95 = f1_score(val_targets, preds_p95)

    return {
        "loss": val_loss,
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "f1_p95": f1_p95,
        "p95_threshold": p95_thr,
        "p99_threshold": p99_thr,
        "youden_threshold": youden_thr,
        "scores": val_scores
    }


def train_pipeline(
    data_dir: str = "data/NF UQ NIDS V2/parquet",
    model_dir: str = "src/backend/saved_models",
    results_dir: str = "results/nf_uq_sequence",
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    seed: int = 11,
    max_parts_per_epoch: int = 4
):
    print(f"\n{'='*80}", flush=True)
    print("  🚀 TRAINING FLAGSHIP DUAL-STREAM HYBRID BI-MAMBAFORMER (RTX 5070 GPU)", flush=True)
    print(f"{'='*80}\n", flush=True)
    set_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device}", flush=True)
    if device.type == "cuda":
        print(f"    GPU Device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"    VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)

    adapter = PureNfUqAdapter(model_dir=model_dir)
    print(f"[*] Loaded 133-Feature Preprocessor from {model_dir}", flush=True)

    # Discover Parquet partitions
    all_parts = sorted(glob.glob(os.path.join(data_dir, "part_*.parquet")))
    if not all_parts:
        raise FileNotFoundError(f"No parquet partitions found in {data_dir}")

    print(f"[*] Found {len(all_parts)} Parquet partitions total.", flush=True)

    # Strict Leak-Free Partitions:
    # 00-47: Train | 48-49: Val | 50-51: Held-out Test
    train_files = all_parts[:48]
    val_files = all_parts[48:50]
    test_files = all_parts[50:52]

    print(f"    Training files:   {len(train_files)} (parts 00 to 47)", flush=True)
    print(f"    Validation files: {len(val_files)} (parts 48 & 49)", flush=True)
    print(f"    Held-Out Test:    {len(test_files)} (parts 50 & 51)", flush=True)

    # Build validation cache
    cache_path = os.path.join(data_dir, "..", "sequence_cache", "val_cache_L32.pt")
    val_x, val_m, val_y, val_attacks = prepare_validation_cache(
        val_files, adapter, cache_path=cache_path, max_val_samples=12000, seq_len=32, stride=32
    )

    # Instantiate BiMambaFormer
    model = BiMambaFormer(
        in_features=133,
        d_model=128,
        mamba_expand=2,
        mamba_state=16,
        attn_heads=4,
        dropout=0.15
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] BiMambaFormer initialized with {total_params:,} parameters", flush=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    best_pr_auc = 0.0
    best_thr = 0.5
    history = []

    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "bi_mambaformer_best.pt")
    thr_file_path = os.path.join(model_dir, "bi_mambaformer_threshold.txt")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0
        n_samples = 0

        # Subsample partitions per epoch to rotate through the dataset efficiently
        start_part = ((epoch - 1) * max_parts_per_epoch) % len(train_files)
        selected_parts = [train_files[(start_part + i) % len(train_files)] for i in range(max_parts_per_epoch)]

        print(f"\n--- Epoch {epoch}/{epochs} (Streaming {len(selected_parts)} partitions, parts {start_part:02d} to {start_part+max_parts_per_epoch-1:02d}) ---", flush=True)

        dataset = StreamingSequenceDataset(
            parquet_files=selected_parts,
            adapter=adapter,
            seq_len=32,
            stride=16,
            batch_size=batch_size,
            shuffle_files=True,
            max_rows_per_partition=250000,
            max_sequences_per_partition=6000
        )

        for bx, bm, by in dataset:
            bx = bx.to(device, non_blocking=True)
            bm = bm.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                out = model(bx, mask=bm)
                logits = out["logits"].squeeze(-1)
                loss = criterion(logits, by)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(bx)
            n_samples += len(bx)
            n_batches += 1

            if n_batches % 25 == 0:
                print(f"    Batch {n_batches:,} | Sequences: {n_samples:,} | Current Batch Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        epoch_loss = running_loss / max(n_samples, 1)
        epoch_duration = time.time() - epoch_start

        # Validation evaluation
        print(f"[*] Validating Epoch {epoch} on {len(val_x):,} held-out sequences...", flush=True)
        val_res = evaluate_validation(model, val_x, val_m, val_y, device, batch_size=512)

        print(f"    [Epoch {epoch} Summary] Time: {epoch_duration:.1f}s | Train Loss: {epoch_loss:.4f} | Val Loss: {val_res['loss']:.4f}", flush=True)
        print(f"    ROC-AUC: {val_res['auc_roc']:.4f} | PR-AUC: {val_res['pr_auc']:.4f} | F1@p95: {val_res['f1_p95']:.4f}", flush=True)
        print(f"    Calibrated Thresholds: p95={val_res['p95_threshold']:.4f}, p99={val_res['p99_threshold']:.4f}, youden={val_res['youden_threshold']:.4f}", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": float(epoch_loss),
            "val_loss": float(val_res["loss"]),
            "auc_roc": float(val_res["auc_roc"]),
            "pr_auc": float(val_res["pr_auc"]),
            "f1_p95": float(val_res["f1_p95"]),
            "thresholds": {
                "p95": float(val_res["p95_threshold"]),
                "p99": float(val_res["p99_threshold"]),
                "youden": float(val_res["youden_threshold"])
            },
            "duration_s": float(epoch_duration)
        })

        # Checkpoint if best PR-AUC
        if val_res["pr_auc"] > best_pr_auc:
            best_pr_auc = val_res["pr_auc"]
            best_thr = val_res["p95_threshold"]
            torch.save(model.state_dict(), best_model_path)
            with open(thr_file_path, "w") as f:
                f.write(f"{best_thr:.6f}\n")
            print(f"    ✨ New best model saved to {best_model_path} (PR-AUC: {best_pr_auc:.4f})", flush=True)

    # Save training history
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*80}", flush=True)
    print(f"  ✅ TRAINING COMPLETE! Best PR-AUC: {best_pr_auc:.4f} | Optimal Threshold: {best_thr:.6f}", flush=True)
    print(f"  Model Weights: {best_model_path}", flush=True)
    print(f"  Training History: {history_path}", flush=True)
    print(f"{'='*80}\n", flush=True)
    return model, best_thr


if __name__ == "__main__":
    train_pipeline()
