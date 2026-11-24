#!/usr/bin/env python
"""
01_check_split_leakage.py
=========================
Real leakage / shortcut audit for the NF-UQ-NIDS-v2 parquet partitions used by
HYBRID-NIDS (train = parts 00-47, val = 48-49, test = 50-51).

It answers, with numbers you can put in the thesis:
  1. Which columns exist? Is there a timestamp (needed for "chronological" claims)?
  2. How many source IPs (hosts) of the test split also appear in train?
  3. How many test flows are exact duplicates (all non-ID features) of train flows?
  4. How pure are host windows (all-benign / all-attack / mixed / single-source-dataset)?
  5. How strongly does the source dataset alone predict the label (shortcut)?

Run from anywhere:
  python 01_check_split_leakage.py --data-dir "E:/.../data/NF UQ NIDS V2/parquet" --out results_audit

Outputs: <out>/split_leakage_report.json, <out>/train_src_ips.txt (used by script 02).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SPLITS = {"train": (0, 48), "val": (48, 50), "test": (50, 52)}
ID_COLS = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "DNS_QUERY_ID",
           "Label", "Attack", "Dataset"}
TIME_HINTS = ("TIME", "START", "END", "TIMESTAMP")


def part_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "part_*.parquet")))
    if len(files) < 52:
        print(f"[WARN] expected 52 partitions, found {len(files)}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results_audit")
    ap.add_argument("--max-train-parts", type=int, default=48,
                    help="use fewer train parts for a quick run (default all 48)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = part_files(args.data_dir)
    report = {"n_partitions": len(files)}

    # ---- 1. schema -------------------------------------------------------
    schema_cols = pq.read_schema(files[0]).names
    report["columns"] = schema_cols
    time_cols = [c for c in schema_cols if any(h in c.upper() for h in TIME_HINTS)]
    report["timestamp_like_columns"] = time_cols
    hash_cols = [c for c in schema_cols if c not in ID_COLS]
    report["hash_columns_count"] = len(hash_cols)
    print(f"[1] columns={len(schema_cols)} timestamp-like={time_cols or 'NONE'}")

    # ---- 2/3/4/5: single pass over partitions -----------------------------
    hashes = {s: [] for s in SPLITS}
    ips = {s: set() for s in SPLITS}
    test_rows = []          # (ip, label, dataset) per test row
    ds_label = {s: {} for s in SPLITS}   # split -> dataset -> [n, n_attack]
    host_stats = {s: {"hosts_ge32": 0, "pure_benign": 0, "pure_attack": 0, "mixed": 0,
                      "multi_dataset": 0} for s in SPLITS}
    order_check = {"hosts_checked": 0, "hosts_time_sorted": 0}

    for split, (a, b) in SPLITS.items():
        sel = files[a:b] if split != "train" else files[a:min(b, a + args.max_train_parts)]
        for f in sel:
            t0 = time.time()
            df = pq.read_table(f).to_pandas()
            h = pd.util.hash_pandas_object(df[hash_cols], index=False).values.astype(np.uint64)
            hashes[split].append(h)
            ips[split].update(df["IPV4_SRC_ADDR"].unique().tolist())

            for ds, g in df.groupby("Dataset", sort=False):
                rec = ds_label[split].setdefault(str(ds), [0, 0])
                rec[0] += len(g)
                rec[1] += int((g["Label"] == 1).sum())

            if split == "test":
                test_rows.append(df[["IPV4_SRC_ADDR", "Label", "Dataset", "Attack"]].copy())

            # host purity inside this partition (same grouping the model uses)
            g = df.groupby("IPV4_SRC_ADDR", sort=False)
            sizes = g.size()
            big = sizes[sizes >= 32].index
            if len(big):
                sub = df[df["IPV4_SRC_ADDR"].isin(big)]
                agg = sub.groupby("IPV4_SRC_ADDR").agg(
                    n_att=("Label", lambda x: int((x == 1).sum())),
                    n=("Label", "size"),
                    n_ds=("Dataset", "nunique"))
                st = host_stats[split]
                st["hosts_ge32"] += len(agg)
                st["pure_attack"] += int((agg.n_att == agg.n).sum())
                st["pure_benign"] += int((agg.n_att == 0).sum())
                st["mixed"] += int(((agg.n_att > 0) & (agg.n_att < agg.n)).sum())
                st["multi_dataset"] += int((agg.n_ds > 1).sum())

            if time_cols and order_check["hosts_checked"] < 2000:
                tc = time_cols[0]
                for ip in list(big[:200]):
                    s = df.loc[df["IPV4_SRC_ADDR"] == ip, tc].values
                    order_check["hosts_checked"] += 1
                    order_check["hosts_time_sorted"] += int(np.all(np.diff(s) >= 0))
            print(f"    {split} {os.path.basename(f)} rows={len(df):,} ({time.time()-t0:.1f}s)")
            del df

    # ---- 2. host overlap --------------------------------------------------
    tr_ips = ips["train"]
    report["unique_src_ips"] = {s: len(v) for s, v in ips.items()}
    overlap = {}
    for s in ("val", "test"):
        inter = ips[s] & tr_ips
        overlap[s] = {"unique_hosts": len(ips[s]), "hosts_also_in_train": len(inter),
                      "fraction_hosts_seen_in_train": len(inter) / max(len(ips[s]), 1)}
    tdf = pd.concat(test_rows, ignore_index=True)
    seen = tdf["IPV4_SRC_ADDR"].isin(tr_ips)
    overlap["test"]["fraction_rows_from_train_hosts"] = float(seen.mean())
    overlap["test"]["fraction_attack_rows_from_train_hosts"] = float(seen[tdf.Label == 1].mean())
    overlap["test"]["fraction_benign_rows_from_train_hosts"] = float(seen[tdf.Label == 0].mean())
    # attack hosts that are also benign hosts (IP reuse across sources)
    att_ips = set(tdf.loc[tdf.Label == 1, "IPV4_SRC_ADDR"])
    ben_ips = set(tdf.loc[tdf.Label == 0, "IPV4_SRC_ADDR"])
    overlap["test"]["ips_with_both_labels"] = len(att_ips & ben_ips)
    report["host_overlap"] = overlap
    print("[2] host overlap:", json.dumps(overlap["test"], indent=1))

    # ---- 3. exact duplicates ---------------------------------------------
    H = {s: (np.concatenate(v) if v else np.zeros(0, np.uint64)) for s, v in hashes.items()}
    dup = {}
    for s in ("val", "test"):
        dup[s] = {"fraction_rows_identical_to_a_train_row": float(np.isin(H[s], H["train"]).mean())}
    tr_u = np.unique(H["train"])
    dup["train_internal_duplicate_fraction"] = float(1 - len(tr_u) / max(len(H["train"]), 1))
    m = np.isin(H["test"], H["train"])
    lab = tdf["Label"].values
    dup["test"]["dup_fraction_attack_rows"] = float(m[lab == 1].mean())
    dup["test"]["dup_fraction_benign_rows"] = float(m[lab == 0].mean())
    dup["test"]["internal_duplicate_fraction"] = float(1 - len(np.unique(H["test"])) / max(len(H["test"]), 1))
    report["duplicates"] = dup
    print("[3] duplicates:", json.dumps(dup, indent=1))

    # ---- 4. host purity ---------------------------------------------------
    report["host_purity_hosts_ge32_flows_per_partition"] = host_stats
    if order_check["hosts_checked"]:
        report["row_order_vs_time"] = order_check
    print("[4] host purity:", json.dumps(host_stats["test"]))

    # ---- 5. dataset shortcut ---------------------------------------------
    sc = {}
    for s in SPLITS:
        sc[s] = {d: {"rows": v[0], "attack_fraction": v[1] / max(v[0], 1)}
                 for d, v in ds_label[s].items()}
    report["attack_fraction_by_source_dataset"] = sc
    # accuracy of "predict majority label of the source dataset" (learned on train)
    maj = {d: int(v["attack_fraction"] >= 0.5) for d, v in sc["train"].items()}
    tdf["maj"] = tdf["Dataset"].astype(str).map(maj).fillna(1)
    report["dataset_only_baseline_test_row_accuracy"] = float((tdf["maj"] == tdf["Label"]).mean())
    print("[5] dataset-only baseline accuracy (rows):", report["dataset_only_baseline_test_row_accuracy"])

    with open(os.path.join(args.out, "train_src_ips.txt"), "w") as f:
        f.write("\n".join(map(str, sorted(tr_ips))))
    with open(os.path.join(args.out, "split_leakage_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nDONE -> {args.out}/split_leakage_report.json")


if __name__ == "__main__":
    main()
