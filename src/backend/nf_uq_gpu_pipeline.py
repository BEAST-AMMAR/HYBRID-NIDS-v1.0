"""
=============================================================================
  NF-UQ-NIDS-v2 — Full-Scale Streaming PyTorch GPU Pipeline
  Hardware: NVIDIA GeForce RTX 5070 (8GB VRAM) + 16GB System RAM
=============================================================================
  Streams all 52 Parquet partitions without loading the 76M dataset into RAM.
  Trains the Autoencoder epoch-by-epoch on RTX 5070 using PyTorch + CUDA 12.8,
  extracts 16-D latent representations, fits Latent Isolation Forest,
  computes multi-view fusion (Recon MSE + Mahalanobis + k-NN), calibrates
  thresholds on validation, and evaluates on held-out test data.
=============================================================================
"""

import os
import sys
import time
import json
import yaml
import glob
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from scipy.spatial.distance import mahalanobis

# Ensure workspace root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import feature adapter from nf_uq_pipeline
from src.backend.nf_uq_pipeline import (
    NfUqFeatureAdapter, load_config, set_all_seeds,
    compute_thresholds, evaluate
)


# =============================================================================
#  1. PYTORCH AUTOENCODER (Mish Activations, Latent Bottleneck 16-D)
# =============================================================================
class NfUqAutoencoder(nn.Module):
    def __init__(self, in_features: int = 133, latent_dim: int = 16):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Mish(),
            nn.Linear(64, 32),
            nn.Mish(),
            nn.Linear(32, latent_dim),
            nn.Mish(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Mish(),
            nn.Linear(32, 64),
            nn.Mish(),
            nn.Linear(64, in_features),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

    def encode(self, x):
        return self.encoder(x)


# =============================================================================
#  2. STREAMING BENIGN FLOW GENERATOR (Bounded Memory)
# =============================================================================
def stream_parquet_benign(parquet_files, adapter, batch_size=2048, device="cuda"):
    """
    Generator that iterates through parquet partitions, filters strictly benign
    traffic, transforms features using the adapter, and yields PyTorch CUDA batches.
    System RAM stays strictly < 2 GB.
    """
    for ppath in parquet_files:
        try:
            # Read single partition (approx 1.4M rows)
            table = pq.read_table(ppath)
            df = table.to_pandas()
        except Exception as e:
            print(f"    [WARN] Failed reading {ppath}: {e}")
            continue

        # Keep strictly benign
        benign = df[df["Label"] == 0]
        del df
        del table

        if len(benign) == 0:
            continue

        # Transform using pre-fitted adapter
        try:
            X = adapter.transform(benign)
        except Exception as e:
            print(f"    [WARN] Transform error on {ppath}: {e}")
            continue

        del benign

        # Yield in GPU-ready mini-batches
        n_rows = X.shape[0]
        indices = np.arange(n_rows)
        np.random.shuffle(indices)

        for start_idx in range(0, n_rows, batch_size):
            end_idx = min(start_idx + batch_size, n_rows)
            batch = X[indices[start_idx:end_idx]]
            tensor = torch.from_numpy(batch).float().to(device, non_blocking=True)
            yield tensor


# =============================================================================
#  3. FULL GPU TRAINING LOOP
# =============================================================================
def train_gpu_autoencoder(
    train_files,
    val_df,
    adapter,
    device,
    epochs=15,
    batch_size=4096,
    lr=1e-3,
    save_path="src/backend/saved_models/autoencoder_torch.pt"
):
    print(f"\n{'='*80}")
    print(f"  RTX 5070 GPU AUTOENCODER TRAINING (Full Streaming Scale)")
    print(f"  Device: {torch.cuda.get_device_name(device)}  |  Batch Size: {batch_size}")
    print(f"  Partitions for Training: {len(train_files)} Parquet Files")
    print(f"{'='*80}\n")

    in_features = len(adapter.final_columns)
    model = NfUqAutoencoder(in_features=in_features, latent_dim=16).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    # Prepare validation tensor
    val_benign = val_df[val_df["Label"] == 0]
    X_val = adapter.transform(val_benign)
    val_tensor = torch.from_numpy(X_val).float().to(device)
    print(f"    [OK] Prepared Validation Set: {X_val.shape[0]:,} normal flows")

    best_val_loss = float("inf")
    best_weights = None
    patience = 4
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        n_samples = 0

        # Stream through training partitions
        for batch in stream_parquet_benign(train_files, adapter, batch_size=batch_size, device=device):
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                recon, _ = model(batch)
                loss = criterion(recon, batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * batch.size(0)
            n_samples += batch.size(0)
            n_batches += 1

            if n_batches % 500 == 0:
                print(f"      ... Epoch {epoch:02d} | Step {n_batches:,} | Loss: {loss.item():.5f} | Samples: {n_samples:,}")

        avg_train_loss = total_loss / max(n_samples, 1)

        # Validation step
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                val_recon, _ = model(val_tensor)
                val_loss = criterion(val_recon, val_tensor).item()

        scheduler.step()
        elapsed = time.time() - t0

        print(f"  [Epoch {epoch:02d}/{epochs:02d}] {elapsed:.1f}s | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | Samples: {n_samples:,}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(best_weights, save_path)
            print(f"    [★] New best validation loss: {val_loss:.5f} -> Model saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    [!] Early stopping triggered at epoch {epoch}")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    return model


# =============================================================================
#  4. LATENT EXTRACTION & ISOLATION FOREST + FUSION
# =============================================================================
def extract_latent_gpu(model, X_np, device, batch_size=8192):
    """Extract latent 16-D representations using RTX 5070 GPU."""
    model.eval()
    latents = []
    n = X_np.shape[0]

    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch = torch.from_numpy(X_np[start_idx:end_idx]).float().to(device)
            with torch.amp.autocast('cuda'):
                z = model.encode(batch)
            latents.append(z.cpu().numpy().astype(np.float32))

    return np.vstack(latents)


def compute_gpu_recon_error(model, X_np, device, batch_size=8192):
    """Compute per-flow MSE reconstruction error on GPU."""
    model.eval()
    errors = []
    n = X_np.shape[0]

    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch = torch.from_numpy(X_np[start_idx:end_idx]).float().to(device)
            with torch.amp.autocast('cuda'):
                recon, _ = model(batch)
                mse = torch.mean((recon - batch) ** 2, dim=1)
            errors.append(mse.cpu().numpy().astype(np.float32))

    return np.concatenate(errors)


# =============================================================================
#  5. MAIN ORCHESTRATION PIPELINE
# =============================================================================
def run_full_gpu_pipeline():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(project_root, "configs", "nf_uq.yaml")
    cfg = load_config(config_path)

    seed = 11
    set_all_seeds(seed)

    parquet_dir = os.path.join(project_root, cfg["data"]["parquet_dir"])
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    assert len(parquet_files) == 52, f"Expected 52 parquet files, found {len(parquet_files)}"

    model_dir = os.path.join(project_root, "src", "backend", "saved_models")
    results_dir = os.path.join(project_root, "results", "nf_uq", "full_gpu_run")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Compute Device: {device} ({torch.cuda.get_device_name(device)})")

    # Partitions:
    # 0 to 47: Training stream (~70M rows total)
    # 48, 49: Validation set
    # 50, 51: Held-out Test set
    train_files = parquet_files[:48]
    val_files = parquet_files[48:50]
    test_files = parquet_files[50:]

    # Load Adapter
    adapter = NfUqFeatureAdapter(cfg)
    adapter_meta_path = os.path.join(model_dir, "nf_uq_adapter_meta.joblib")
    if os.path.exists(adapter_meta_path):
        adapter.load(os.path.join(model_dir, "nf_uq_adapter"))
        print(f"[OK] Loaded pre-fitted Feature Adapter ({len(adapter.final_columns)} features)")
    else:
        print("[*] Fitting adapter on initial benign flows...")
        p0 = pq.read_table(parquet_files[0]).to_pandas()
        benign_p0 = p0[p0["Label"] == 0]
        adapter.fit(benign_p0)
        adapter.save(os.path.join(model_dir, "nf_uq_adapter"))
        del p0, benign_p0

    # Load Validation Set
    print("[*] Loading Validation Partitions (48-49)...")
    val_tables = [pq.read_table(f).to_pandas() for f in val_files]
    val_df = pd.concat(val_tables, ignore_index=True)
    del val_tables
    print(f"    Validation flows: {len(val_df):,} (Benign: {(val_df['Label']==0).sum():,}, Attack: {(val_df['Label']==1).sum():,})")

    # Step 1: Train Autoencoder on RTX 5070 across all 48 training partitions
    model_save_path = os.path.join(model_dir, "autoencoder_torch.pt")
    in_features = len(adapter.final_columns)
    model = NfUqAutoencoder(in_features=in_features, latent_dim=16).to(device)

    if os.path.exists(model_save_path):
        print(f"[*] Found trained model checkpoint at {model_save_path}. Loading weights...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("    [OK] Loaded trained weights onto RTX 5070 GPU.")
    else:
        model = train_gpu_autoencoder(
            train_files=train_files,
            val_df=val_df,
            adapter=adapter,
            device=device,
            epochs=12,
            batch_size=4096,
            lr=1e-3,
            save_path=model_save_path
        )

    # Step 2: Fit or Load Latent Isolation Forest & Distance Artefacts
    iso_path = os.path.join(model_dir, "iso_forest.joblib")
    knn_path = os.path.join(model_dir, "knn_model.joblib")
    artefacts_path = os.path.join(model_dir, "scoring_artefacts.joblib")

    if os.path.exists(iso_path) and os.path.exists(knn_path) and os.path.exists(artefacts_path):
        print(f"[*] Found existing Isolation Forest and distance artefacts. Loading...")
        iso_forest = joblib.load(iso_path)
        knn = joblib.load(knn_path)
        artefacts = joblib.load(artefacts_path)
        benign_mean = artefacts["benign_latent_mean"]
        cov_inv = artefacts["benign_latent_cov_inv"]
        print("    [OK] Loaded Latent Isolation Forest and Distance Artefacts.")
    else:
        print(f"\n{'='*80}")
        print("  Fitting Latent Isolation Forest & Distance Artefacts on GPU Latents")
        print(f"{'='*80}\n")

        val_benign = val_df[val_df["Label"] == 0]
        X_val_benign = adapter.transform(val_benign)
        latent_benign = extract_latent_gpu(model, X_val_benign, device)

        # Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination=0.01,
            random_state=seed,
            n_jobs=-1
        )
        iso_forest.fit(latent_benign)
        joblib.dump(iso_forest, iso_path)
        print("    [OK] Isolation Forest fitted and saved")

        # Mahalanobis reference
        benign_mean = np.mean(latent_benign, axis=0)
        cov = np.cov(latent_benign, rowvar=False) + np.eye(latent_benign.shape[1]) * 1e-4
        cov_inv = np.linalg.pinv(cov)

        # k-NN reference
        knn_sample = latent_benign[:min(50000, len(latent_benign))]
        knn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1)
        knn.fit(knn_sample)
        joblib.dump(knn, knn_path)

        artefacts = {
            "benign_latent_mean": benign_mean,
            "benign_latent_cov_inv": cov_inv,
            "n_reference_samples": len(latent_benign)
        }
        joblib.dump(artefacts, artefacts_path)
        print("    [OK] Scoring artefacts saved")

    # Step 3: Calibrate Thresholds on Validation Set
    print("\n[*] Calibrating Thresholds on Validation Set...")
    X_val = adapter.transform(val_df)
    y_val = val_df["Label"].values.astype(int)

    val_mse = compute_gpu_recon_error(model, X_val, device)
    val_latent = extract_latent_gpu(model, X_val, device)

    # Vectorized Mahalanobis
    diff_val = val_latent - benign_mean
    val_maha = np.sqrt(np.clip(np.sum(diff_val @ cov_inv * diff_val, axis=1), 0.0, None))

    val_knn, _ = knn.kneighbors(val_latent)
    val_knn = np.mean(val_knn, axis=1)

    # Rank-mean fusion
    from scipy.stats import rankdata
    n_val = len(val_df)
    r_mse = rankdata(val_mse) / n_val
    r_maha = rankdata(val_maha) / n_val
    r_knn = rankdata(val_knn) / n_val
    fused_val = (r_mse + r_maha + r_knn) / 3.0

    thresholds = compute_thresholds(fused_val, y_val)
    print(f"    Threshold policies calibrated: {thresholds}")

    primary_thr = thresholds["p95_val_normal"]
    with open(os.path.join(model_dir, "threshold.txt"), "w") as f:
        f.write(f"{primary_thr:.6f}\n")

    del val_df, X_val, val_latent, val_mse, val_maha, val_knn

    # Step 4: Evaluate on Held-Out Test Partitions (50-51)
    print(f"\n{'='*80}")
    print("  HELD-OUT TEST SET EVALUATION (Partitions 50-51)")
    print(f"{'='*80}\n")

    test_tables = [pq.read_table(f).to_pandas() for f in test_files]
    test_df = pd.concat(test_tables, ignore_index=True)
    del test_tables

    print(f"    Test flows: {len(test_df):,} (Benign: {(test_df['Label']==0).sum():,}, Attack: {(test_df['Label']==1).sum():,})")

    X_test = adapter.transform(test_df)
    y_test = test_df["Label"].values.astype(int)
    attack_test = test_df["Attack"].values
    source_test = test_df["Dataset"].values

    test_mse = compute_gpu_recon_error(model, X_test, device)
    test_latent = extract_latent_gpu(model, X_test, device)

    diff_test = test_latent - benign_mean
    test_maha = np.sqrt(np.clip(np.sum(diff_test @ cov_inv * diff_test, axis=1), 0.0, None))

    test_knn, _ = knn.kneighbors(test_latent)
    test_knn = np.mean(test_knn, axis=1)

    n_test = len(test_df)
    t_mse = rankdata(test_mse) / n_test
    t_maha = rankdata(test_maha) / n_test
    t_knn = rankdata(test_knn) / n_test
    fused_test = (t_mse + t_maha + t_knn) / 3.0

    # Evaluate across all calibrated policies
    results_summary = {}
    for policy_name, thr in thresholds.items():
        metrics = evaluate(
            y_true=y_test,
            fused_scores=fused_test,
            threshold=thr,
            attack_classes=attack_test,
            sources=source_test,
            dataset="nf_uq",
            experiment="gpu_full_scale",
            seed=seed,
            policy=policy_name
        )

        pdir = os.path.join(results_dir, policy_name)
        os.makedirs(pdir, exist_ok=True)
        with open(os.path.join(pdir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        results_summary[policy_name] = {
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "FPR": metrics["fpr"],
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"]
        }

    # Print Primary & Youden Results
    p95 = results_summary["p95_val_normal"]
    youden = results_summary["youden_val"]

    print(f"\n{'='*80}")
    print("  FINAL ACCURACY & PERFORMANCE METRICS (RTX 5070 GPU Full Scale)")
    print(f"{'='*80}")
    print(f"  Total Test Flows Evaluated: {n_test:,}\n")
    print(f"  [Primary Policy: p95_val_normal (Threshold={thresholds['p95_val_normal']:.4f})]")
    print(f"    - Accuracy  : {p95['Accuracy']*100:.2f}%")
    print(f"    - Precision : {p95['Precision']*100:.2f}%")
    print(f"    - Recall    : {p95['Recall']*100:.2f}%")
    print(f"    - F1-Score  : {p95['F1']*100:.2f}%")
    print(f"    - FPR       : {p95['FPR']*100:.2f}%")
    print(f"    - ROC-AUC   : {p95['ROC-AUC']*100:.2f}%")
    print(f"    - PR-AUC    : {p95['PR-AUC']*100:.2f}%\n")

    print(f"  [Balanced Policy: Youden's Index (Threshold={thresholds['youden_val']:.4f})]")
    print(f"    - Accuracy  : {youden['Accuracy']*100:.2f}%")
    print(f"    - Precision : {youden['Precision']*100:.2f}%")
    print(f"    - Recall    : {youden['Recall']*100:.2f}%")
    print(f"    - F1-Score  : {youden['F1']*100:.2f}%")
    print(f"    - FPR       : {youden['FPR']*100:.2f}%")
    print(f"    - ROC-AUC   : {youden['ROC-AUC']*100:.2f}%\n")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_full_gpu_pipeline()
