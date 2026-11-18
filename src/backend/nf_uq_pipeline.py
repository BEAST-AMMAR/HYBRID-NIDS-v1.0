#!/usr/bin/env python
"""
=============================================================================
 NF-UQ-NIDS-v2  —  Hybrid NIDS Training, Testing & Evaluation Pipeline
=============================================================================
 Frozen pipeline: AE(64->32->16->32->64), IF(200 trees, latent, benign-only),
 rank_mean fusion, 5 threshold policies, 5 seeds.

 Usage:
   python nf_uq_pipeline.py                     # full run, seed 11
   python nf_uq_pipeline.py --seed 22           # single seed
   python nf_uq_pipeline.py --all-seeds         # all 5 seeds
   python nf_uq_pipeline.py --skip-parquet      # skip CSV->Parquet if done
=============================================================================
"""

import os, sys, json, time, argparse, hashlib, warnings, platform, gc
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import yaml
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from scipy.spatial.distance import mahalanobis

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
#  0.  REPRODUCIBILITY
# ============================================================================
def set_all_seeds(seed: int):
    """Lock every RNG to `seed` for full reproducibility."""
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================================
#  1.  CONFIG LOADER
# ============================================================================
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================
#  2.  CSV -> PARQUET  (chunked, one-time)
# ============================================================================
def csv_to_parquet(raw_csv: str, parquet_dir: str, chunk_size: int = 500_000):
    """Convert the 13.7 GB CSV to partitioned Parquet files (chunked read)."""
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Check if already converted
    existing = list(parquet_dir.glob("part_*.parquet"))
    if existing:
        print(f"    [SKIP] {len(existing)} parquet parts already exist in {parquet_dir}")
        return

    print(f"    Converting CSV -> Parquet (chunk_size={chunk_size:,}) ...")
    reader = pcsv.open_csv(raw_csv, read_options=pcsv.ReadOptions(block_size=256 * 1024 * 1024))

    part_idx = 0
    total_rows = 0
    for batch in reader:
        table = pa.Table.from_batches([batch])
        out_path = parquet_dir / f"part_{part_idx:04d}.parquet"
        pq.write_table(table, out_path, compression="snappy")
        total_rows += len(batch)
        part_idx += 1
        if part_idx % 10 == 0:
            print(f"        ... {total_rows:,} rows written ({part_idx} parts)")

    print(f"    Done: {total_rows:,} rows in {part_idx} parquet parts.")


# ============================================================================
# ============================================================================
#  3.  STREAMING SUBSAMPLING FROM PARQUET (Memory-safe, bounds to <1.5GB RAM)
# ============================================================================
def stream_subsample_parquet(parquet_dir: str, cfg: dict, seed: int) -> pd.DataFrame:
    """
    Stream 52 parquet parts one by one, extracting a stratified subsample.
    Peak memory is strictly bounded to a single part (~300MB) + the accumulated sample (~600MB).
    Never loads the full 76M rows into memory.
    """
    parquet_dir = Path(parquet_dir)
    parts = sorted(parquet_dir.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    rng = np.random.RandomState(seed)
    benign_cap = cfg["sampling"]["benign_train_cap"]
    attack_cap = cfg["sampling"]["attack_class_cap"]

    # Ground-truth total counts verified from scan
    total_benign = 25_165_295
    attack_totals = {
        'DoS': 17875585, 'scanning': 3781419, 'DDoS': 21748351, 'xss': 2455020,
        'Bot': 143097, 'Reconnaissance': 2633778, 'password': 1153323, 'Fuzzers': 22310,
        'injection': 684897, 'Theft': 2431, 'Brute Force': 123982, 'Infilteration': 116361,
        'Exploits': 31551, 'Generic': 16560, 'Analysis': 2299, 'Backdoor': 18978,
        'mitm': 7723, 'Shellcode': 1427, 'ransomware': 3425, 'Worms': 164
    }

    benign_rate = min(1.0, (benign_cap * 1.05) / total_benign)

    sampled_chunks = []
    print(f"    Streaming subsampling from {len(parts)} parquet parts (Option B: {benign_cap:,} benign cap, {attack_cap:,} atk cap)...")

    for i, p in enumerate(parts):
        part_df = pq.read_table(p).to_pandas()

        # Downcast floats to float32 immediately to minimize RAM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for c in part_df.select_dtypes(include=['float64']).columns:
                part_df[c] = part_df[c].astype(np.float32)

        part_df = part_df[part_df["Label"].isin([0, 1])]

        part_b = part_df[part_df["Label"] == 0]
        part_a = part_df[part_df["Label"] == 1]

        # Sample benign proportionally across source datasets
        part_sampled_b = []
        for src, grp in part_b.groupby("Dataset"):
            n_take = int(np.ceil(len(grp) * benign_rate))
            if n_take > 0:
                part_sampled_b.append(grp.sample(n=min(len(grp), n_take), random_state=rng))

        # Sample attacks (preserve 100% of rare attacks)
        part_sampled_a = []
        for atk, grp in part_a.groupby("Attack"):
            tot = attack_totals.get(atk, len(grp))
            if tot <= attack_cap:
                part_sampled_a.append(grp)
            else:
                atk_rate = min(1.0, (attack_cap * 1.05) / tot)
                n_take = int(np.ceil(len(grp) * atk_rate))
                if n_take > 0:
                    part_sampled_a.append(grp.sample(n=min(len(grp), n_take), random_state=rng))

        to_combine = []
        if part_sampled_b:
            to_combine.extend(part_sampled_b)
        if part_sampled_a:
            to_combine.extend(part_sampled_a)

        if to_combine:
            sampled_chunks.append(pd.concat(to_combine, ignore_index=True))

        if (i + 1) % 10 == 0 or (i + 1) == len(parts):
            rows_so_far = sum(len(c) for c in sampled_chunks)
            print(f"        ... processed {i+1}/{len(parts)} parts ({rows_so_far:,} rows sampled)")

        del part_df, part_b, part_a, to_combine
        gc.collect()

    df = pd.concat(sampled_chunks, ignore_index=True)
    del sampled_chunks
    gc.collect()

    print(f"    Raw sampled: {len(df):,} rows. Applying exact caps...")

    # Final exact capping
    benign = df[df["Label"] == 0].copy()
    attack = df[df["Label"] == 1].copy()

    if len(benign) > benign_cap:
        sampled_b = []
        for src, grp in benign.groupby("Dataset"):
            n_take = min(len(grp), int(np.ceil(benign_cap * len(grp) / len(benign))))
            if n_take > 0:
                sampled_b.append(grp.sample(n=min(len(grp), n_take), random_state=rng))
        benign = pd.concat(sampled_b, ignore_index=True)
        if len(benign) > benign_cap:
            benign = benign.sample(n=benign_cap, random_state=rng).reset_index(drop=True)

    sampled_attacks = []
    for atk, grp in attack.groupby("Attack"):
        if len(grp) > attack_cap:
            grp = grp.sample(n=attack_cap, random_state=rng)
        sampled_attacks.append(grp)

    if sampled_attacks:
        attack = pd.concat(sampled_attacks, ignore_index=True)

    assert "Dataset" in benign.columns, "'Dataset' column missing from benign!"
    assert "Dataset" in attack.columns, "'Dataset' column missing from attack!"

    result = pd.concat([benign, attack], ignore_index=True)
    result = result.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    print(f"    Subsampling complete: {len(result):,} rows (Benign: {len(benign):,}, Attack: {len(attack):,})")
    return result


# ============================================================================
#  4.  DATA VERIFICATION & NOTES (B0)
# ============================================================================
def verify_and_document(df: pd.DataFrame, docs_dir: str):
    """Write docs/DATA_NOTES.md with verified full dataset and subsampled dataset stats."""
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    total_full_rows = 75_987_976
    full_labels = {0: 25_165_295, 1: 50_822_681}
    full_datasets = {
        'NF-BoT-IoT-v2': 37763497,
        'NF-CSE-CIC-IDS2018-v2': 18893708,
        'NF-ToN-IoT-v2': 16940496,
        'NF-UNSW-NB15-v2': 2390275
    }
    full_attacks = {
        'Benign': 25165295, 'DDoS': 21748351, 'DoS': 17875585, 'scanning': 3781419,
        'Reconnaissance': 2633778, 'xss': 2455020, 'password': 1153323, 'injection': 684897,
        'Bot': 143097, 'Brute Force': 123982, 'Infilteration': 116361, 'Exploits': 31551,
        'Fuzzers': 22310, 'Backdoor': 18978, 'Generic': 16560, 'mitm': 7723,
        'ransomware': 3425, 'Theft': 2431, 'Analysis': 2299, 'Shellcode': 1427, 'Worms': 164
    }

    notes = []
    notes.append("# NF-UQ-NIDS-v2 Data Notes\n\n")
    notes.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    notes.append("## Full Dataset Summary\n")
    notes.append(f"- **Total Rows:** {total_full_rows:,}\n")
    notes.append(f"- **Total Columns:** {len(df.columns)}\n")
    notes.append(f"- **Parquet Storage:** 52 parts (Snappy compressed, ~2.7 GB total)\n\n")

    notes.append("### Full Dataset Label Distribution\n| Label | Count | Percentage |\n|---|---|---|\n")
    for val, cnt in full_labels.items():
        notes.append(f"| {val} | {cnt:,} | {cnt/total_full_rows*100:.2f}% |\n")

    notes.append("\n### Source Datasets\n| Source Dataset | Flow Count | Percentage |\n|---|---|---|\n")
    for ds, cnt in full_datasets.items():
        notes.append(f"| {ds} | {cnt:,} | {cnt/total_full_rows*100:.2f}% |\n")

    notes.append("\n### Attack Classes (Full Dataset)\n| Attack Class | Count | Percentage |\n|---|---|---|\n")
    for atk, cnt in full_attacks.items():
        notes.append(f"| {atk} | {cnt:,} | {cnt/total_full_rows*100:.4f}% |\n")

    notes.append(f"\n## Subsampled Dataset (Option B Protocol)\n")
    notes.append(f"- **Subsampled Rows:** {len(df):,}\n")
    for val, cnt in df["Label"].value_counts().items():
        notes.append(f"- Label={val}: {cnt:,} ({cnt/len(df)*100:.2f}%)\n")

    notes.append("\n### Subsampled Attack Breakdown\n| Attack Class | Subsampled Count |\n|---|---|\n")
    for atk, cnt in df["Attack"].value_counts().items():
        notes.append(f"| {atk} | {cnt:,} |\n")

    notes_path = docs_dir / "DATA_NOTES.md"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.writelines(notes)
    print(f"    Data notes saved to {notes_path}")


# ============================================================================
#  5.  CLEANING & DEDUP  (B2)
# ============================================================================
def clean_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf with NaN (imputation later), drop invalid labels, dedup."""
    n_before = len(df)

    # Replace inf
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        mask = np.isinf(df[c].values)
        if mask.any():
            df.loc[mask, c] = np.nan

    # Drop rows with invalid labels
    df = df[df["Label"].isin([0, 1])].copy()

    # Deduplicate on all feature columns + label (before split!)
    feat_cols = [c for c in df.columns if c not in ["Dataset"]]
    df = df.drop_duplicates(subset=feat_cols, keep="first").reset_index(drop=True)

    n_after = len(df)
    print(f"    Cleaning: {n_before:,} -> {n_after:,} rows (removed {n_before - n_after:,})")
    return df


# ============================================================================
#  7.  SPLITTING  (B4 — arrival-order split as primary)
# ============================================================================
def split_data(df: pd.DataFrame, cfg: dict, protocol: str = "arrival_order"):
    """
    Split into train/val/test per protocol.
    Returns (train_df, val_df, test_df).
    """
    train_r = cfg["splits"]["train_ratio"]
    val_r   = cfg["splits"]["val_ratio"]

    if protocol == "arrival_order":
        # Within each source, use row order (arrival proxy) for temporal split
        trains, vals, tests = [], [], []
        for src, grp in df.groupby("Dataset"):
            grp = grp.reset_index(drop=True)  # row order = arrival order
            n = len(grp)
            t1 = int(n * train_r)
            t2 = int(n * (train_r + val_r))
            trains.append(grp.iloc[:t1])
            vals.append(grp.iloc[t1:t2])
            tests.append(grp.iloc[t2:])
        return (pd.concat(trains, ignore_index=True),
                pd.concat(vals, ignore_index=True),
                pd.concat(tests, ignore_index=True))

    elif protocol == "stratified_random":
        from sklearn.model_selection import train_test_split
        # Stratify by (Label, Dataset)
        strat_col = df["Label"].astype(str) + "_" + df["Dataset"]
        train_df, temp = train_test_split(df, test_size=1-train_r,
                                          stratify=strat_col, random_state=42)
        strat_temp = temp["Label"].astype(str) + "_" + temp["Dataset"]
        val_frac = val_r / (val_r + cfg["splits"]["test_ratio"])
        val_df, test_df = train_test_split(temp, test_size=1-val_frac,
                                           stratify=strat_temp, random_state=42)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    else:
        raise ValueError(f"Unknown protocol: {protocol}")


# ============================================================================
#  8.  FEATURE ADAPTER  (B5 — fitted on benign train only)
# ============================================================================
TCP_FLAG_BITS = ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]

def _decompose_tcp_flags(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Decompose integer bitmask into 8 binary flag columns."""
    vals = series.fillna(0).astype(int).values
    flags = {}
    for i, name in enumerate(TCP_FLAG_BITS):
        flags[f"{prefix}_{name}"] = ((vals >> i) & 1).astype(np.float32)
    return pd.DataFrame(flags)


class NfUqFeatureAdapter:
    """
    Feature engineering fitted STRICTLY on benign training data.
    Persists all fitted artefacts for reproducible inference.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.top_dst_ports = None
        self.top_l7_protos = None
        self.quantile_transformer = None
        self.imputer_medians = None
        self.final_columns = None
        self._numeric_cols = None
        self._binary_cols = None

    # ---- identifiers for what to drop / keep ----
    _DROP = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "DNS_QUERY_ID",
             "Label", "Attack", "Dataset",
             "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
             "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
             "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE"}

    _TCP_FLAG_COLS = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]

    _SPECIAL_CATEGORICAL = {
        "ICMP_TYPE": 5,
        "ICMP_IPV4_TYPE": 5,
        "DNS_QUERY_TYPE": 5,
        "FTP_COMMAND_RET_CODE": 5,
    }

    def fit(self, benign_train: pd.DataFrame):
        """Fit all transforms on benign training data only."""
        # 1) Top-K destination ports
        port_counts = benign_train["L4_DST_PORT"].value_counts()
        self.top_dst_ports = list(port_counts.head(self.cfg["features"]["dst_port_top_k"]).index)

        # 2) Top-K L7 protocols
        l7_counts = benign_train["L7_PROTO"].value_counts()
        self.top_l7_protos = list(l7_counts.head(self.cfg["features"]["l7_proto_top_k"]).index)

        # 3) Identify numeric columns (everything not categorical/identifier/label)
        all_cols = set(benign_train.columns)
        non_numeric = self._DROP | set(self._TCP_FLAG_COLS) | set(self._SPECIAL_CATEGORICAL.keys())
        raw_numeric = sorted([c for c in all_cols - non_numeric
                              if benign_train[c].dtype in (np.float64, np.int64, np.float32, np.int32, float, int)])
        self._numeric_cols = raw_numeric

        # 4) Build feature matrix on benign train to fit imputer + quantile
        X_benign = self._transform_raw(benign_train, fitting=True)

        # Identify binary vs numeric for scaling
        self._binary_cols = [c for c in X_benign.columns
                             if set(X_benign[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
        scale_cols = [c for c in X_benign.columns if c not in self._binary_cols]

        # 5) Impute NaN with benign-train median (fitted here)
        self.imputer_medians = X_benign[scale_cols].median().to_dict()
        X_benign[scale_cols] = X_benign[scale_cols].fillna(X_benign[scale_cols].median())

        # 6) log1p transform on numeric columns before quantile
        X_benign[scale_cols] = np.log1p(np.abs(X_benign[scale_cols].values)) * np.sign(X_benign[scale_cols].values)

        # 7) QuantileTransformer (output=normal) fitted on benign train
        self.quantile_transformer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(self.cfg["features"]["quantile_n_quantiles"], len(X_benign)),
            random_state=42
        )
        X_benign[scale_cols] = self.quantile_transformer.fit_transform(X_benign[scale_cols].values)

        # Handle any remaining NaN/inf from quantile transform
        X_benign = X_benign.fillna(0.0)
        X_benign.replace([np.inf, -np.inf], 0.0, inplace=True)

        self.final_columns = list(X_benign.columns)
        self._scale_cols = scale_cols
        print(f"    Feature adapter fitted: {len(self.final_columns)} features")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform any DataFrame using fitted artefacts. Returns float32 array."""
        X = self._transform_raw(df, fitting=False)

        # Apply same column order and impute missing features with training medians
        for c in self.final_columns:
            if c not in X.columns:
                X[c] = self.imputer_medians.get(c, 0.0)
        X = X[self.final_columns]

        # Impute
        scale_cols = self._scale_cols
        for c in scale_cols:
            if c in self.imputer_medians:
                X[c] = X[c].fillna(self.imputer_medians[c])

        # log1p
        sc = [c for c in scale_cols if c in X.columns]
        X[sc] = np.nan_to_num(X[sc].values, nan=0.0, posinf=1e12, neginf=-1e12)
        X[sc] = np.log1p(np.abs(X[sc].values)) * np.sign(X[sc].values)
        X[sc] = np.nan_to_num(X[sc].values, nan=0.0, posinf=1e6, neginf=-1e6)

        # Quantile transform
        X[sc] = self.quantile_transformer.transform(X[sc].values)

        # Final cleanup
        X = X.fillna(0.0)
        X.replace([np.inf, -np.inf], 0.0, inplace=True)
        return X.values.astype(np.float32)

    def _transform_raw(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        """Build the raw feature matrix (before scaling) in a single fast allocation."""
        data = {}

        # 1) Destination port one-hot
        port_vals = df["L4_DST_PORT"].fillna(0).astype(int).values
        top_ports_int = [int(p) for p in self.top_dst_ports]
        for p in top_ports_int:
            data[f"port_{p}"] = (port_vals == p).astype(np.float32)
        data["port_other"] = (~np.isin(port_vals, top_ports_int)).astype(np.float32)

        # 2) Protocol one-hot (TCP=6, UDP=17, ICMP=1)
        proto = df["PROTOCOL"].fillna(0).astype(int).values
        for p, name in [(6, "proto_TCP"), (17, "proto_UDP"), (1, "proto_ICMP")]:
            data[name] = (proto == p).astype(np.float32)
        data["proto_other"] = (~np.isin(proto, [6, 17, 1])).astype(np.float32)

        # 3) L7 protocol one-hot (with float32 precision tolerance)
        l7 = df["L7_PROTO"].fillna(0.0).values.astype(np.float32)
        top_l7_f32 = [np.float32(lp) for lp in self.top_l7_protos]
        matched_any = np.zeros(len(l7), dtype=bool)
        for lp, lp_f32 in zip(self.top_l7_protos, top_l7_f32):
            m = np.isclose(l7, lp_f32, atol=1e-3)
            data[f"l7_{lp}"] = m.astype(np.float32)
            matched_any |= m
        data["l7_other"] = (~matched_any).astype(np.float32)

        # 4) TCP flag decomposition (3 columns * 8 bits = 24 binary features)
        for col in self._TCP_FLAG_COLS:
            if col in df.columns:
                vals = df[col].fillna(0).astype(int).values
                for i, name in enumerate(TCP_FLAG_BITS):
                    data[f"{col}_{name}"] = ((vals >> i) & 1).astype(np.float32)

        # 5) Special categorical with presence flag + top-K one-hot
        for col, k in self._SPECIAL_CATEGORICAL.items():
            if col in df.columns:
                vals = df[col].fillna(-999).values
                data[f"{col}_present"] = ((vals != -999) & (vals != 0)).astype(np.float32)
                if fitting:
                    top_vals = pd.Series(vals).value_counts().head(k).index.tolist()
                    if not hasattr(self, "_special_top"):
                        self._special_top = {}
                    self._special_top[col] = top_vals
                top_vals = getattr(self, "_special_top", {}).get(col, [])
                for tv in top_vals:
                    data[f"{col}_{tv}"] = (vals == tv).astype(np.float32)

        # 6) All remaining numeric columns
        for c in self._numeric_cols:
            if c in df.columns:
                data[c] = np.nan_to_num(df[c].values, nan=0.0, posinf=1e12, neginf=-1e12).astype(np.float32)

        return pd.DataFrame(data)

    def save(self, path: str):
        """Persist all fitted artefacts."""
        artefacts = {
            "top_dst_ports": self.top_dst_ports,
            "top_l7_protos": self.top_l7_protos,
            "imputer_medians": self.imputer_medians,
            "final_columns": self.final_columns,
            "numeric_cols": self._numeric_cols,
            "binary_cols": self._binary_cols,
            "scale_cols": self._scale_cols,
            "special_top": getattr(self, "_special_top", {}),
        }
        joblib.dump(artefacts, path + "_meta.joblib")
        joblib.dump(self.quantile_transformer, path + "_quantile.joblib")
        print(f"    Feature adapter saved to {path}_*.joblib")

    def load(self, path: str):
        """Restore fitted artefacts."""
        artefacts = joblib.load(path + "_meta.joblib")
        self.top_dst_ports = artefacts["top_dst_ports"]
        self.top_l7_protos = artefacts["top_l7_protos"]
        self.imputer_medians = artefacts["imputer_medians"]
        self.final_columns = artefacts["final_columns"]
        self._numeric_cols = artefacts["numeric_cols"]
        self._binary_cols = artefacts["binary_cols"]
        self._scale_cols = artefacts["scale_cols"]
        self._special_top = artefacts.get("special_top", {})
        self.quantile_transformer = joblib.load(path + "_quantile.joblib")
        return self


# ============================================================================
#  9.  AUTOENCODER  (FROZEN: 64 -> 32 -> 16 -> 32 -> 64)
# ============================================================================
def build_autoencoder(input_dim: int, cfg: dict):
    """Build the frozen autoencoder architecture per guide spec."""
    lr = cfg["ae"]["learning_rate"]
    dr = cfg["ae"]["dropout_rate"]
    hidden = cfg["ae"]["hidden"]  # [64, 32, 16]

    inp = Input(shape=(input_dim,), name="input")

    # Encoder
    x = Dense(hidden[0], activation="relu", name="enc_1")(inp)
    x = BatchNormalization(name="enc_bn_1")(x)
    x = Dropout(dr, name="enc_do_1")(x)

    x = Dense(hidden[1], activation="relu", name="enc_2")(x)
    x = BatchNormalization(name="enc_bn_2")(x)
    x = Dropout(dr, name="enc_do_2")(x)

    latent = Dense(hidden[2], activation="relu", name="latent")(x)
    x = BatchNormalization(name="lat_bn")(latent)

    # Decoder (mirror)
    x = Dense(hidden[1], activation="relu", name="dec_1")(x)
    x = BatchNormalization(name="dec_bn_1")(x)
    x = Dropout(dr, name="dec_do_1")(x)

    x = Dense(hidden[0], activation="relu", name="dec_2")(x)
    x = BatchNormalization(name="dec_bn_2")(x)
    x = Dropout(dr, name="dec_do_2")(x)

    output = Dense(input_dim, activation="linear", name="output")(x)

    ae = Model(inputs=inp, outputs=output, name="hybrid_nids_ae")
    ae.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])

    # Encoder-only model for latent extraction
    encoder = Model(inputs=inp, outputs=latent, name="encoder")
    return ae, encoder


# ============================================================================
#  10. SCORE COMPONENTS
# ============================================================================
def compute_scores(X: np.ndarray, reconstruction: np.ndarray,
                   latent: np.ndarray, iforest_model,
                   benign_latent_mean: np.ndarray,
                   benign_latent_cov_inv: np.ndarray,
                   knn_model, n_features: int) -> dict:
    """Compute all 5 anomaly score components for each sample."""

    # MSE: mean squared reconstruction error
    sq_errors = np.power(X - reconstruction, 2)
    mse = np.mean(sq_errors, axis=1)

    # Top-8: mean of the 8 largest per-feature squared errors
    top_k = min(8, n_features)
    top8 = np.mean(np.sort(sq_errors, axis=1)[:, -top_k:], axis=1)

    # Mahalanobis distance in latent space (vectorized for speed)
    diff = latent - benign_latent_mean
    maha = np.sqrt(np.clip(np.sum(diff @ benign_latent_cov_inv * diff, axis=1), 0.0, None))

    # k-NN distance in latent space
    dists, _ = knn_model.kneighbors(latent)
    knn = np.mean(dists, axis=1)

    # Isolation Forest score on latent
    if_score = -iforest_model.decision_function(latent)

    return {"mse": mse, "top8": top8, "maha": maha, "knn": knn, "iforest": if_score}


# ============================================================================
#  11. FUSION  (rank_mean — label-free, scale-free)
# ============================================================================
def rank_mean_fusion(scores: dict, val_normal_scores: dict,
                     components: list) -> np.ndarray:
    """
    For each component, convert score to its percentile among validation
    normals, then average across components.
    """
    fused = np.zeros(len(next(iter(scores.values()))))
    for comp in components:
        vals = scores[comp]
        ref = val_normal_scores[comp]
        # Percentile mapping: fraction of ref values <= each val
        percentiles = np.searchsorted(np.sort(ref), vals) / max(len(ref), 1)
        fused += percentiles
    fused /= len(components)
    return fused


# ============================================================================
#  12. THRESHOLD POLICIES
# ============================================================================
def compute_thresholds(fused_val_scores: np.ndarray, y_val: np.ndarray) -> dict:
    """Compute 5 threshold policies from validation data."""
    normal_scores = fused_val_scores[y_val == 0]
    policies = {}

    # p95 and p99 of validation normal scores
    policies["p95_val_normal"] = float(np.percentile(normal_scores, 95))
    policies["p99_val_normal"] = float(np.percentile(normal_scores, 99))

    # Youden's J on validation
    fpr_arr, tpr_arr, thr_arr = roc_curve(y_val, fused_val_scores)
    j_idx = np.argmax(tpr_arr - fpr_arr)
    policies["youden_val"] = float(thr_arr[j_idx])

    # FPR-controlled policies
    for target_fpr, name in [(0.05, "fpr_5pct"), (0.03, "fpr_3pct")]:
        # Find largest threshold where FPR <= target
        candidates = thr_arr[fpr_arr <= target_fpr]
        if len(candidates) > 0:
            policies[name] = float(candidates[-1])
        else:
            policies[name] = float(thr_arr[0])

    return policies


# ============================================================================
#  13. EVALUATION  (single pass on test — never tune here)
# ============================================================================
def evaluate(y_true: np.ndarray, fused_scores: np.ndarray, threshold: float,
             attack_classes: np.ndarray, sources: np.ndarray,
             dataset: str, experiment: str, seed: int, policy: str) -> dict:
    """Full evaluation producing metrics.json compatible output."""
    y_pred = (fused_scores > threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = float(fp / max(fp + tn, 1))
    roc = float(roc_auc_score(y_true, fused_scores))
    pr = float(average_precision_score(y_true, fused_scores))

    # Per-class recall
    per_class = {}
    for cls in np.unique(attack_classes):
        if cls == "Benign":
            continue
        mask = attack_classes == cls
        if mask.sum() > 0:
            cls_recall = float(recall_score(y_true[mask], y_pred[mask], zero_division=0))
            per_class[cls] = cls_recall

    # Per-source FPR
    per_source = {}
    for src in np.unique(sources):
        src_mask = sources == src
        src_normal = (y_true[src_mask] == 0)
        if src_normal.sum() > 0:
            src_fp = ((y_pred[src_mask] == 1) & src_normal).sum()
            src_fpr = float(src_fp / src_normal.sum())
            per_source[src] = {"fpr": src_fpr, "n_normal": int(src_normal.sum())}

    return {
        "dataset": dataset,
        "experiment": experiment,
        "seed": seed,
        "policy": policy,
        "threshold": float(threshold),
        "n_test": int(len(y_true)),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc,
        "pr_auc": pr,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_class_recall": per_class,
        "per_source": per_source,
    }


# ============================================================================
#  14. MAIN PIPELINE
# ============================================================================
def run_pipeline(cfg: dict, seed: int, project_root: str, skip_parquet: bool = False):
    """Execute the full pipeline for one seed."""
    t_start = time.time()
    set_all_seeds(seed)
    print(f"\n{'='*80}")
    print(f"  NF-UQ-NIDS-v2 HYBRID PIPELINE  |  Seed: {seed}")
    print(f"{'='*80}\n")

    raw_csv = os.path.join(project_root, cfg["data"]["raw_csv"])
    parquet_dir = os.path.join(project_root, cfg["data"]["parquet_dir"])
    results_dir = os.path.join(project_root, "results", "nf_uq", f"seed_{seed}")
    model_dir = os.path.join(project_root, "src", "backend", "saved_models")
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ---- Step 1: CSV -> Parquet ----
    print("[1/10] CSV -> Parquet Conversion")
    if not skip_parquet:
        csv_to_parquet(raw_csv, parquet_dir, cfg["data"]["chunk_size"])
    else:
        print("    [SKIP] --skip-parquet flag set")

    # ---- Step 2: Streaming Subsampling ----
    print("\n[2/10] Streaming Subsampling from Parquet")
    df = stream_subsample_parquet(parquet_dir, cfg, seed)

    # ---- Step 3: Verify & document ----
    print("\n[3/10] Data verification & documentation")
    verify_and_document(df, docs_dir)

    # ---- Step 4: Clean & dedup ----
    print("\n[4/10] Cleaning & deduplication")
    df = clean_and_dedup(df)

    # ---- Step 5: Save sampling distribution ----
    print("\n[5/10] Saving sampling distribution")
    sampling_table = df.groupby(["Dataset", "Attack"]).size().reset_index(name="count")
    sampling_path = os.path.join(project_root, "results", "nf_uq", "sampling.csv")
    sampling_table.to_csv(sampling_path, index=False)
    print(f"    Sampling table saved to {sampling_path}")

    # ---- Step 6: Split ----
    print("\n[6/10] Splitting (arrival-order primary)")
    train_df, val_df, test_df = split_data(df, cfg, protocol="arrival_order")
    print(f"    Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # Extract labels and metadata before feature transform
    y_train = train_df["Label"].values.astype(int)
    y_val   = val_df["Label"].values.astype(int)
    y_test  = test_df["Label"].values.astype(int)
    attack_train = train_df["Attack"].values
    attack_val   = val_df["Attack"].values
    attack_test  = test_df["Attack"].values
    source_test  = test_df["Dataset"].values

    # Benign-only training data
    benign_train_df = train_df[train_df["Label"] == 0].copy()
    print(f"    Benign training rows: {len(benign_train_df):,}")

    # ---- Step 7: Feature engineering ----
    print("\n[7/10] Feature adapter (fitted on benign train only)")
    adapter = NfUqFeatureAdapter(cfg)
    adapter.fit(benign_train_df)

    X_train_benign = adapter.transform(benign_train_df)
    X_train_all    = adapter.transform(train_df)
    X_val          = adapter.transform(val_df)
    X_test         = adapter.transform(test_df)

    input_dim = X_train_benign.shape[1]
    print(f"    Input dimension: {input_dim}")

    # Sanity checks
    assert not np.any(np.isnan(X_train_benign)), "NaN in benign train features!"
    assert not np.any(np.isinf(X_train_benign)), "Inf in benign train features!"
    print("    [OK] No NaN/Inf in transformed features")

    # Save adapter artefacts
    adapter_path = os.path.join(model_dir, "nf_uq_adapter")
    adapter.save(adapter_path)

    # Save feature columns
    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump(adapter.final_columns, f, indent=2)

    # ---- Step 8: Train AE & IF ----
    print("\n[8/10] Training Autoencoder & Isolation Forest")

    # 8a. Autoencoder on benign train only
    print(f"    Training AE: {input_dim} -> 64 -> 32 -> 16 -> 32 -> 64 -> {input_dim}")
    ae, encoder = build_autoencoder(input_dim, cfg)
    ae.summary(print_fn=lambda x: None)  # suppress summary

    # Split benign train into AE fit and AE val
    n_ae_val = int(len(X_train_benign) * cfg["ae"]["val_fraction"])
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X_train_benign))
    ae_fit_idx = indices[n_ae_val:]
    ae_val_idx = indices[:n_ae_val]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg["ae"]["early_stop_patience"],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    history = ae.fit(
        X_train_benign[ae_fit_idx], X_train_benign[ae_fit_idx],
        validation_data=(X_train_benign[ae_val_idx], X_train_benign[ae_val_idx]),
        epochs=cfg["ae"]["epochs"],
        batch_size=cfg["ae"]["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    ae.save(os.path.join(model_dir, "autoencoder.keras"))
    print("    [OK] Autoencoder saved")

    # 8b. Extract latent representations
    print("    Extracting latent vectors ...")
    latent_benign_train = encoder.predict(X_train_benign, batch_size=2048, verbose=0)
    latent_val          = encoder.predict(X_val, batch_size=2048, verbose=0)
    latent_test         = encoder.predict(X_test, batch_size=2048, verbose=0)

    # 8c. Isolation Forest on latent benign only (200 trees, no contamination)
    print(f"    Training IF: {cfg['iforest']['n_estimators']} trees on {latent_benign_train.shape[0]:,} × {latent_benign_train.shape[1]}D latent")
    iforest = IsolationForest(
        n_estimators=cfg["iforest"]["n_estimators"],
        max_samples=cfg["iforest"]["max_samples"],
        random_state=seed,
        n_jobs=-1,
    )
    iforest.fit(latent_benign_train)
    joblib.dump(iforest, os.path.join(model_dir, "iso_forest.joblib"))
    print("    [OK] Isolation Forest saved")

    # 8d. Prepare Mahalanobis reference (benign latent mean + covariance)
    benign_latent_mean = np.mean(latent_benign_train, axis=0)
    benign_latent_cov  = np.cov(latent_benign_train, rowvar=False) + 1e-3 * np.eye(latent_benign_train.shape[1])
    benign_latent_cov_inv = np.linalg.inv(benign_latent_cov)

    # 8e. k-NN reference (random 20k benign latent points)
    knn_ref_size = min(cfg["scores"]["knn_ref_size"], len(latent_benign_train))
    knn_ref_idx = rng.choice(len(latent_benign_train), size=knn_ref_size, replace=False)
    knn_ref = latent_benign_train[knn_ref_idx]
    knn_model = NearestNeighbors(n_neighbors=cfg["scores"]["knn_k"], n_jobs=-1)
    knn_model.fit(knn_ref)

    # Save scoring artefacts
    scoring_artefacts = {
        "benign_latent_mean": benign_latent_mean,
        "benign_latent_cov_inv": benign_latent_cov_inv,
        "knn_ref": knn_ref,
    }
    joblib.dump(scoring_artefacts, os.path.join(model_dir, "scoring_artefacts.joblib"))
    joblib.dump(knn_model, os.path.join(model_dir, "knn_model.joblib"))

    # ---- Step 9: Score & fuse ----
    print("\n[9/10] Computing scores & fusion")

    # Reconstructions
    recon_val  = ae.predict(X_val, batch_size=2048, verbose=0)
    recon_test = ae.predict(X_test, batch_size=2048, verbose=0)

    # Scores
    print("    Computing validation scores ...")
    scores_val = compute_scores(X_val, recon_val, latent_val, iforest,
                                benign_latent_mean, benign_latent_cov_inv,
                                knn_model, input_dim)
    print("    Computing test scores ...")
    scores_test = compute_scores(X_test, recon_test, latent_test, iforest,
                                 benign_latent_mean, benign_latent_cov_inv,
                                 knn_model, input_dim)

    # Validation-normal reference for rank_mean
    val_normal_mask = (y_val == 0)
    val_normal_scores = {k: v[val_normal_mask] for k, v in scores_val.items()}

    # Fuse
    fusion_components = cfg["fusion"]["components"]
    fused_val  = rank_mean_fusion(scores_val, val_normal_scores, fusion_components)
    fused_test = rank_mean_fusion(scores_test, val_normal_scores, fusion_components)

    # ---- Step 10: Calibrate & Evaluate ----
    print("\n[10/10] Calibration & evaluation")

    # Thresholds from validation
    thresholds = compute_thresholds(fused_val, y_val)
    print(f"    Threshold policies: {thresholds}")

    # Save threshold
    primary_policy = cfg["thresholds"]["primary_policy"]
    primary_threshold = thresholds[primary_policy]
    with open(os.path.join(model_dir, "threshold.txt"), "w") as f:
        f.write(f"{primary_threshold:.6f}")

    # Evaluate on test (ONCE per policy)
    all_metrics = []
    for policy_name, thr in thresholds.items():
        metrics = evaluate(
            y_test, fused_test, thr,
            attack_test, source_test,
            cfg["dataset"], cfg["experiment"], seed, policy_name
        )
        all_metrics.append(metrics)

        # Save metrics.json
        policy_dir = os.path.join(results_dir, policy_name)
        os.makedirs(policy_dir, exist_ok=True)
        with open(os.path.join(policy_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        if policy_name == primary_policy:
            print(f"\n    -- PRIMARY RESULTS ({policy_name}, threshold={thr:.6f}) --")
            print(f"    Accuracy  : {metrics['accuracy']*100:.2f}%")
            print(f"    Precision : {metrics['precision']*100:.2f}%")
            print(f"    Recall    : {metrics['recall']*100:.2f}%")
            print(f"    F1-Score  : {metrics['f1']*100:.2f}%")
            print(f"    FPR       : {metrics['fpr']*100:.2f}%")
            print(f"    ROC-AUC   : {metrics['roc_auc']*100:.2f}%")
            print(f"    PR-AUC    : {metrics['pr_auc']*100:.2f}%")
            cm = metrics["confusion"]
            print(f"    Confusion : TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}  TP={cm['tp']}")

    # Save ROC & PR curves
    fpr_arr, tpr_arr, thr_arr = roc_curve(y_test, fused_test)
    roc_df = pd.DataFrame({"fpr": fpr_arr, "tpr": tpr_arr, "threshold": thr_arr})
    roc_df.to_csv(os.path.join(results_dir, "roc_points.csv"), index=False)

    prec_arr, rec_arr, thr_pr = precision_recall_curve(y_test, fused_test)
    pr_df = pd.DataFrame({"precision": prec_arr[:-1], "recall": rec_arr[:-1], "threshold": thr_pr})
    pr_df.to_csv(os.path.join(results_dir, "pr_points.csv"), index=False)

    # Save raw scores for later analysis
    scores_df = pd.DataFrame({
        "label": y_test,
        "attack_class": attack_test,
        "source": source_test,
        "fused": fused_test,
        **{f"score_{k}": v for k, v in scores_test.items()},
    })
    scores_df.to_parquet(os.path.join(results_dir, "scores_test.parquet"), index=False)

    # Save environment info
    env_info = {
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "seed": seed,
    }
    with open(os.path.join(results_dir, "env.json"), "w") as f:
        json.dump(env_info, f, indent=2)

    # Save config used
    with open(os.path.join(results_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"  PIPELINE COMPLETE  |  Seed {seed}  |  {elapsed/60:.1f} minutes")
    print(f"  Results: {results_dir}")
    print(f"{'='*80}\n")

    return all_metrics


# ============================================================================
#  15. ABLATION RUNNER (individual score components)
# ============================================================================
def run_ablation(cfg: dict, seed: int, project_root: str):
    """Quick ablation: evaluate each score component alone."""
    results_dir = os.path.join(project_root, "results", "nf_uq", f"seed_{seed}")
    scores_path = os.path.join(results_dir, "scores_test.parquet")

    if not os.path.exists(scores_path):
        print(f"    [SKIP] No scores file found at {scores_path}")
        return

    df = pd.read_parquet(scores_path)
    y_test = df["label"].values

    ablation_results = []
    for comp in ["mse", "top8", "maha", "knn", "iforest", "fused"]:
        col = f"score_{comp}" if comp != "fused" else "fused"
        if col not in df.columns:
            continue
        scores = df[col].values
        auc = float(roc_auc_score(y_test, scores))
        pr_auc = float(average_precision_score(y_test, scores))
        ablation_results.append({"component": comp, "roc_auc": auc, "pr_auc": pr_auc})
        print(f"    {comp:12s}  ROC-AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(os.path.join(results_dir, "ablation.csv"), index=False)


# ============================================================================
#  16. SUPERVISED BASELINE (LightGBM reality check)
# ============================================================================
def run_supervised_baseline(cfg: dict, seed: int, project_root: str):
    """Train supervised LightGBM as Baseline B (reality check)."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("    [SKIP] lightgbm not installed")
        return

    set_all_seeds(seed)
    print(f"\n{'='*80}")
    print(f"  BASELINE B: SUPERVISED LightGBM  |  Seed: {seed}")
    print(f"{'='*80}\n")

    parquet_dir = os.path.join(project_root, cfg["data"]["parquet_dir"])
    results_dir = os.path.join(project_root, "results", "nf_uq", f"seed_{seed}", "baseline_lgbm")
    model_dir = os.path.join(project_root, "src", "backend", "saved_models")
    os.makedirs(results_dir, exist_ok=True)

    # Load, clean, split
    df = stream_subsample_parquet(parquet_dir, cfg, seed)
    df = clean_and_dedup(df)
    train_df, val_df, test_df = split_data(df, cfg, protocol="arrival_order")

    y_train = train_df["Label"].values.astype(int)
    y_val   = val_df["Label"].values.astype(int)
    y_test  = test_df["Label"].values.astype(int)
    attack_test = test_df["Attack"].values
    source_test = test_df["Dataset"].values

    # Load adapter (fitted by main pipeline)
    adapter = NfUqFeatureAdapter(cfg)
    adapter.load(os.path.join(model_dir, "nf_uq_adapter"))

    X_train = adapter.transform(train_df)
    X_val   = adapter.transform(val_df)
    X_test  = adapter.transform(test_df)

    # Train LightGBM with class weight balancing
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale = n_neg / max(n_pos, 1)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False)])

    # Predict probabilities
    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    # Evaluate with same threshold policies
    thresholds = compute_thresholds(proba_val, y_val)
    for policy_name, thr in thresholds.items():
        y_pred = (proba_test > thr).astype(int)
        metrics = evaluate(y_test, proba_test, thr, attack_test, source_test,
                           cfg["dataset"], "baseline_lgbm", seed, policy_name)

        policy_dir = os.path.join(results_dir, "baseline_lgbm", policy_name)
        os.makedirs(policy_dir, exist_ok=True)
        with open(os.path.join(policy_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Primary policy results
    primary = cfg["thresholds"]["primary_policy"]
    thr = thresholds[primary]
    y_pred = (proba_test > thr).astype(int)
    print(f"\n    -- BASELINE B ({primary}) --")
    print(f"    Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"    Precision : {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"    Recall    : {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"    F1-Score  : {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"    ROC-AUC   : {roc_auc_score(y_test, proba_test)*100:.2f}%")
    print(f"    PR-AUC    : {average_precision_score(y_test, proba_test)*100:.2f}%")


# ============================================================================
#  17. SOURCE-SHORTCUT CHECK (B8 — is the model learning testbed identity?)
# ============================================================================
def source_shortcut_check(cfg: dict, seed: int, project_root: str):
    """Check if source dataset is predictable from features (benign only)."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("    [SKIP] lightgbm not installed")
        return

    set_all_seeds(seed)
    print(f"\n{'='*80}")
    print(f"  SOURCE-SHORTCUT CHECK  |  Seed: {seed}")
    print(f"{'='*80}\n")

    parquet_dir = os.path.join(project_root, cfg["data"]["parquet_dir"])
    model_dir = os.path.join(project_root, "src", "backend", "saved_models")
    results_dir = os.path.join(project_root, "results", "nf_uq", f"seed_{seed}")

    df = stream_subsample_parquet(parquet_dir, cfg, seed)
    df = clean_and_dedup(df)

    # Benign only
    benign = df[df["Label"] == 0].copy()

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_src = le.fit_transform(benign["Dataset"].values)

    adapter = NfUqFeatureAdapter(cfg)
    adapter.load(os.path.join(model_dir, "nf_uq_adapter"))
    X = adapter.transform(benign)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_src, test_size=0.2, random_state=seed, stratify=y_src)

    model = lgb.LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=seed, verbose=-1)
    model.fit(X_tr, y_tr)
    src_acc = float(accuracy_score(y_te, model.predict(X_te)))

    result = {"source_predict_accuracy": src_acc, "n_sources": len(le.classes_),
              "sources": list(le.classes_)}
    print(f"    Source-prediction accuracy (benign only): {src_acc*100:.2f}%")
    if src_acc > 0.8:
        print(f"    [!] WARNING: Source is highly predictable — model may learn testbed identity!")

    with open(os.path.join(results_dir, "source_shortcut.json"), "w") as f:
        json.dump(result, f, indent=2)


# ============================================================================
#  18. ENTRY POINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="NF-UQ-NIDS-v2 Hybrid Pipeline")
    parser.add_argument("--config", default="configs/nf_uq.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=11, help="Single seed to run")
    parser.add_argument("--all-seeds", action="store_true", help="Run all 5 seeds")
    parser.add_argument("--skip-parquet", action="store_true", help="Skip CSV->Parquet if already done")
    parser.add_argument("--baseline", action="store_true", help="Also run supervised baseline")
    parser.add_argument("--shortcut", action="store_true", help="Also run source-shortcut check")
    parser.add_argument("--ablation", action="store_true", help="Also run ablation analysis")
    parser.add_argument("--only-aux", action="store_true", help="Skip main pipeline training and run only baseline/shortcut/ablation")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(project_root, args.config)
    cfg = load_config(config_path)

    seeds = cfg["seeds"] if args.all_seeds else [args.seed]

    for seed in seeds:
        # Main hybrid pipeline
        if not args.only_aux:
            metrics = run_pipeline(cfg, seed, project_root, skip_parquet=args.skip_parquet)

        # Optional: ablation
        if args.ablation:
            print("\n-- ABLATION --")
            run_ablation(cfg, seed, project_root)

        # Optional: supervised baseline
        if args.baseline:
            run_supervised_baseline(cfg, seed, project_root)

        # Optional: source shortcut check
        if args.shortcut:
            source_shortcut_check(cfg, seed, project_root)

        # Skip parquet for subsequent seeds (already converted)
        args.skip_parquet = True

    # Summary across seeds
    if len(seeds) > 1:
        print(f"\n{'='*80}")
        print(f"  MULTI-SEED SUMMARY")
        print(f"{'='*80}")
        primary_policy = cfg["thresholds"]["primary_policy"]
        all_f1, all_auc, all_prauc, all_fpr = [], [], [], []
        for s in seeds:
            mpath = os.path.join(project_root, "results", "nf_uq", f"seed_{s}", primary_policy, "metrics.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    m = json.load(f)
                all_f1.append(m["f1"])
                all_auc.append(m["roc_auc"])
                all_prauc.append(m["pr_auc"])
                all_fpr.append(m["fpr"])

        if all_f1:
            print(f"  Policy: {primary_policy}")
            print(f"  F1     : {np.mean(all_f1)*100:.2f}% ± {np.std(all_f1)*100:.2f}%")
            print(f"  ROC-AUC: {np.mean(all_auc)*100:.2f}% ± {np.std(all_auc)*100:.2f}%")
            print(f"  PR-AUC : {np.mean(all_prauc)*100:.2f}% ± {np.std(all_prauc)*100:.2f}%")
            print(f"  FPR    : {np.mean(all_fpr)*100:.2f}% ± {np.std(all_fpr)*100:.2f}%")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
