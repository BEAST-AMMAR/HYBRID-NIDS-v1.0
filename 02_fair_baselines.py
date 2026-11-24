#!/usr/bin/env python
"""
02_fair_baselines.py
====================
Apples-to-apples comparison on the SAME windows, SAME splits and SAME threshold
policy (p95 of validation-benign scores) as the Bi-MambaFormer evaluation.

Models compared
  A. dataset_prior     : predicts attack-probability of the window's source dataset
                         (measures how much a pure shortcut explains)
  B. flow_lgbm_max     : supervised LightGBM on SINGLE flows; window score = max over its flows
  C. window_lgbm_pool  : supervised LightGBM on mean/max/std-pooled window features
  D. bimambaformer     : your checkpoint (optional, needs torch + --repo + --checkpoint-dir)

Every model gets: accuracy/precision/recall/F1/FPR/FNR, ROC-AUC, PR-AUC (+95% bootstrap CI),
TPR@5%FPR, per-class recall WITH counts. Optionally a host-disjoint test subset
(--train-ips from script 01) so you can show results when no test host was seen in training.

Example:
  python 02_fair_baselines.py --repo "E:/.../phase2 models nfuq" ^
      --data-dir "E:/.../data/NF UQ NIDS V2/parquet" --out results_baselines ^
      --checkpoint-dir "E:/.../src/backend/saved_models" --train-ips results_audit/train_src_ips.txt

Requires: pandas numpy pyarrow scikit-learn lightgbm  (+ torch for model D)
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

L = 32


# --------------------------------------------------------------------------- windows
def make_windows(df, adapter, stride, min_flows=4, include_incomplete=False, max_windows_per_host=150):
    """Same grouping/labelling rule as src/backend/sequences.py, but also keeps per-flow labels
    and the host IP of each window. Windows are returned as index arrays (-1 = padding)."""
    n = len(df)
    X_all = adapter.transform(df)
    labels = df["Label"].values.astype(int)
    attacks = df["Attack"].astype(str).values if "Attack" in df.columns else np.array(["Unknown"] * n)
    datasets = df["Dataset"].astype(str).values if "Dataset" in df.columns else np.array(["NF-UQ"] * n)
    src = df["IPV4_SRC_ADDR"].values
    host_idx = pd.Series(np.arange(n)).groupby(src, sort=False).indices

    widx, y, yflow, atk, ds, hosts = [], [], [], [], [], []

    def add(ip, idx):
        w = np.full(L, -1, dtype=np.int64)
        w[:len(idx)] = idx
        lab = labels[idx]
        is_att = int((lab == 1).any())
        widx.append(w)
        yf = np.zeros(L, dtype=np.int8)
        yf[:len(idx)] = lab
        yflow.append(yf)
        y.append(is_att)
        atk.append(str(attacks[idx[lab == 1][0]]) if is_att else "Benign")
        ds.append(datasets[idx[0]])
        hosts.append(ip)

    for ip, idx in host_idx.items():
        k = len(idx)
        if k < min_flows:
            continue
        if k >= L:
            added = 0
            for s in range(0, k - L + 1, stride):
                if added >= max_windows_per_host:
                    break
                add(ip, idx[s:s + L])
                added += 1
        elif include_incomplete:
            add(ip, idx)
    if not widx:
        return None
    return dict(X_all=X_all, widx=np.stack(widx), y=np.array(y, dtype=np.int8),
                yflow=np.stack(yflow), attack=np.array(atk), dataset=np.array(ds),
                host=np.array(hosts))


def gather(X_all, widx):
    M = widx >= 0
    X = np.zeros((len(widx), L, X_all.shape[1]), dtype=np.float32)
    X[M] = X_all[widx[M]]
    return X, M


def pool(X, M):
    m = M[:, :, None].astype(np.float32)
    n = m.sum(1).clip(1)
    mean = (X * m).sum(1) / n
    mx = np.where(M[:, :, None], X, -1e9).max(1)
    std = np.sqrt((((X - mean[:, None, :]) ** 2) * m).sum(1) / n)
    return np.concatenate([mean, mx, std, n], axis=1).astype(np.float32)


def read_part(path, max_rows=None):
    t = pq.read_table(path)
    if max_rows and len(t) > max_rows:
        t = t.slice(0, max_rows)
    return t.to_pandas()


# --------------------------------------------------------------------------- data builders
def build_train(files, adapter, n_parts, max_win, max_flows_per_part, seed):
    rng = np.random.default_rng(seed)
    P, Y, FX, FY, DS = [], [], [], [], []
    for f in files[:n_parts]:
        t0 = time.time()
        w = make_windows(read_part(f, 250000), adapter, stride=16)
        if w is None:
            continue
        sel = np.arange(len(w["y"]))
        if len(sel) > max_win:
            sel = rng.choice(sel, max_win, replace=False)
        X, M = gather(w["X_all"], w["widx"][sel])
        P.append(pool(X, M)); Y.append(w["y"][sel]); DS.append(w["dataset"][sel])
        fx, fy = X[M], w["yflow"][sel][M]
        if len(fx) > max_flows_per_part:
            k = rng.choice(len(fx), max_flows_per_part, replace=False)
            fx, fy = fx[k], fy[k]
        FX.append(fx); FY.append(fy)
        print(f"  train {os.path.basename(f)} windows={len(sel):,} ({time.time()-t0:.0f}s)", flush=True)
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(FX), np.concatenate(FY),
            np.concatenate(DS))


def build_eval(files, adapter, stride, include_incomplete, cap, slice_rows, max_wph):
    out, total = [], 0
    for f in files:
        w = make_windows(read_part(f, slice_rows), adapter, stride=stride,
                         include_incomplete=include_incomplete, max_windows_per_host=max_wph)
        if w is None:
            continue
        X, M = gather(w["X_all"], w["widx"])
        out.append(dict(X=X, M=M, y=w["y"], attack=w["attack"], dataset=w["dataset"], host=w["host"]))
        total += len(w["y"])
        print(f"  eval {os.path.basename(f)} windows={len(w['y']):,}", flush=True)
        if total >= cap:
            break
    cat = {k: np.concatenate([o[k] for o in out])[:cap] for k in out[0]}
    return cat


# --------------------------------------------------------------------------- metrics
def evaluate(y, s, thr, attack, n_boot=200, seed=0):
    pred = (s > thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    fpr_c, tpr_c, _ = roc_curve(y, s)
    res = dict(n=int(len(y)), n_attack=int(y.sum()), threshold=float(thr),
               accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
               fpr=fp / max(tn + fp, 1), fnr=fn / max(tp + fn, 1),
               roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
               tpr_at_5pct_fpr=float(tpr_c[np.searchsorted(fpr_c, 0.05, side="right") - 1]),
               confusion=dict(tn=tn, fp=fp, fn=fn, tp=tp))
    rng = np.random.default_rng(seed)
    ra, pa = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        ra.append(roc_auc_score(y[i], s[i])); pa.append(average_precision_score(y[i], s[i]))
    res["roc_auc_ci95"] = [float(np.percentile(ra, 2.5)), float(np.percentile(ra, 97.5))]
    res["pr_auc_ci95"] = [float(np.percentile(pa, 2.5)), float(np.percentile(pa, 97.5))]
    pc = {}
    for a in sorted(set(attack) - {"Benign", "0"}):
        m = attack == a
        pc[a.lower()] = dict(count=int(m.sum()), recall=float(pred[m].mean()))
    res["per_class_recall"] = pc
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items()}


def p95(val_s, val_y):
    if (val_y == 0).sum() == 0:
        raise SystemExit("No benign validation windows -> cannot calibrate threshold. Check partitions/paths.")
    return float(np.percentile(val_s[val_y == 0], 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the HYBRID-NIDS repo root")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results_baselines")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--train-ips", default=None, help="train_src_ips.txt from script 01")
    ap.add_argument("--train-parts", type=int, default=20, help="20 = what the original model saw")
    ap.add_argument("--max-win-per-part", type=int, default=6000)
    ap.add_argument("--max-flows-per-part", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.repo)

    import lightgbm as lgb
    from src.backend.sequences import PureNfUqAdapter
    model_dir = args.checkpoint_dir or os.path.join(args.repo, "src", "backend", "saved_models")
    adapter = PureNfUqAdapter(model_dir=model_dir)
    files = sorted(glob.glob(os.path.join(args.data_dir, "part_*.parquet")))
    assert len(files) >= 52, f"need 52 partitions, found {len(files)}"

    print("[1/4] building train windows"); t0 = time.time()
    trP, trY, trFX, trFY, trDS = build_train(files[:48], adapter, args.train_parts,
                                              args.max_win_per_part, args.max_flows_per_part, args.seed)
    print("[2/4] building val / test windows")
    val = build_eval(files[48:50], adapter, stride=32, include_incomplete=False,
                     cap=12000, slice_rows=250000, max_wph=100)
    test = build_eval(files[50:52], adapter, stride=32, include_incomplete=True,
                      cap=50000, slice_rows=None, max_wph=150)
    print(f"    train windows={len(trY):,} flows={len(trFY):,} val={len(val['y']):,} test={len(test['y']):,}"
          f"  ({time.time()-t0:.0f}s)")

    # ----- fit + score ------------------------------------------------------
    print("[3/4] fitting baselines")
    scores = {}      # name -> (val_scores, test_scores)

    prior = pd.Series(trY).groupby(trDS).mean().to_dict()
    scores["dataset_prior"] = (np.array([prior.get(d, 0.5) for d in val["dataset"]]),
                               np.array([prior.get(d, 0.5) for d in test["dataset"]]))

    vF, vM = val["X"], val["M"]
    kw = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=args.seed, n_jobs=-1, verbose=-1)
    cb = [lgb.early_stopping(30, verbose=False)]

    flow_clf = lgb.LGBMClassifier(**kw)
    # flow-level labels for validation windows are not stored in `val`; use window label broadcast
    # only for early stopping, using a small held-out slice of the training flows instead:
    k = len(trFY); perm = np.random.default_rng(args.seed).permutation(k); cut = int(0.9 * k)
    flow_clf.fit(trFX[perm[:cut]], trFY[perm[:cut]], eval_set=[(trFX[perm[cut:]], trFY[perm[cut:]])],
                 eval_metric="auc", callbacks=cb)

    def flow_window_score(X, M):
        out = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), 2000):
            xb, mb = X[i:i + 2000], M[i:i + 2000]
            p = flow_clf.predict_proba(xb.reshape(-1, xb.shape[-1]))[:, 1].reshape(mb.shape)
            out[i:i + 2000] = np.where(mb, p, -1).max(1)
        return out
    scores["flow_lgbm_max"] = (flow_window_score(vF, vM), flow_window_score(test["X"], test["M"]))

    pool_clf = lgb.LGBMClassifier(**kw)
    pool_clf.fit(trP, trY, eval_set=[(pool(vF, vM), val["y"])], eval_metric="auc", callbacks=cb)
    scores["window_lgbm_pool"] = (pool_clf.predict_proba(pool(vF, vM))[:, 1],
                                  pool_clf.predict_proba(pool(test["X"], test["M"]))[:, 1])

    shipped_thr = None
    if args.checkpoint_dir:
        try:
            import torch
            from src.backend.models.bi_mambaformer import BiMambaFormer
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = BiMambaFormer(in_features=133, d_model=128, mamba_expand=2, mamba_state=16,
                                attn_heads=4, dropout=0.0).to(dev)
            net.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "bi_mambaformer_best.pt"),
                                           map_location=dev, weights_only=True))
            net.eval()

            def bm(X, M):
                o = []
                with torch.no_grad():
                    for i in range(0, len(X), 512):
                        r = net(torch.from_numpy(X[i:i + 512]).to(dev), mask=torch.from_numpy(M[i:i + 512]).to(dev))
                        o.append(r["threat_score"].squeeze(-1).cpu().numpy())
                return np.concatenate(o)
            scores["bimambaformer"] = (bm(vF, vM), bm(test["X"], test["M"]))
            tp = os.path.join(args.checkpoint_dir, "bi_mambaformer_threshold.txt")
            if os.path.exists(tp):
                shipped_thr = float(open(tp).read().strip())
        except Exception as e:
            print(f"[WARN] bimambaformer scoring skipped: {e}")

    # ----- evaluate ---------------------------------------------------------
    print("[4/4] evaluating")
    seen_mask = None
    if args.train_ips and os.path.exists(args.train_ips):
        tr_ips = set(open(args.train_ips).read().split("\n"))
        seen_mask = np.array([str(h) in tr_ips for h in test["host"]])
    results = {"setup": dict(train_parts=args.train_parts, n_train_windows=int(len(trY)),
                             n_val=int(len(val["y"])), n_test=int(len(test["y"])), seed=args.seed,
                             test_fraction_hosts_seen_in_train=(None if seen_mask is None else float(seen_mask.mean())))}
    for name, (vs, ts) in scores.items():
        thr = p95(vs, val["y"])
        entry = {"full_test": evaluate(test["y"], ts, thr, test["attack"])}
        if name == "bimambaformer" and shipped_thr is not None:
            entry["full_test_at_shipped_threshold"] = evaluate(test["y"], ts, shipped_thr, test["attack"])
        if seen_mask is not None:
            for tag, m in (("host_disjoint_subset", ~seen_mask), ("seen_host_subset", seen_mask)):
                if m.sum() > 200 and len(np.unique(test["y"][m])) == 2:
                    entry[tag] = evaluate(test["y"][m], ts[m], thr, test["attack"][m])
        results[name] = entry
        r = entry["full_test"]
        print(f"  {name:18s} acc={r['accuracy']:.4f} prec={r['precision']:.4f} rec={r['recall']:.4f} "
              f"fpr={r['fpr']:.4f} rocAUC={r['roc_auc']:.4f} {r['roc_auc_ci95']} prAUC={r['pr_auc']:.4f}")

    with open(os.path.join(args.out, "fair_baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE -> {args.out}/fair_baseline_results.json")


if __name__ == "__main__":
    main()
