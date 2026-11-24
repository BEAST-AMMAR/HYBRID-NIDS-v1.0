#!/usr/bin/env python
"""
03_strict_protocol.py
=====================
Honest evaluation protocols for the HYBRID-NIDS thesis (NF-UQ-NIDS-v2).

  standard : the original partition split (train parts 00-19 sampled, val 48-49, test 50-51).
             Directly comparable with your earlier numbers / the literature.
  strict   : (1) GLOBAL exact-duplicate removal on the 39 non-ID feature columns,
             (2) HOST-GROUPED split: every source IP lives in exactly one of
                 train (80%) / val (10%) / test (10%)   -> no host leakage,
             (3) padded+masked windows (>=4 flows) in ALL splits, so train == deployment.
  loso     : leave-one-source-out on the deduplicated data (train on 3 source datasets,
             test on the 4th; val = host-held-out slice of the training sources).

Models (same windows, same threshold rule = p95 of validation-benign scores):
  dataset_prior, flow_lgbm_max, window_lgbm_pool                       (always)
  bimambaformer, mamba_only, attn_only, bilstm, transformer            (needs torch; --models)

Everything is reported with mean +- std over --seeds, threshold-free metrics first
(ROC-AUC, PR-AUC, TPR@5%FPR), then p95-threshold metrics, per-source and per-class
(with window count AND number of distinct hosts).

Typical use (Windows PowerShell, from the repo root):
  python 03_strict_protocol.py --repo . --data-dir "data\\NF UQ NIDS V2\\parquet" ^
         --out results\\protocols --protocol both --seeds 11,22,33
  python 03_strict_protocol.py --repo . --data-dir "..." --out results\\protocols ^
         --protocol loso --seeds 11 --models bimambaformer

Rough cost: dedup pass ~15-40 min (reads all 52 parts once, cached in <out>/strict_cache.npz);
each Mamba-variant run ~10 min per seed at --max-train-windows 40000, others ~1-3 min.
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
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

L = 32
ID_COLS = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "DNS_QUERY_ID", "Label", "Attack", "Dataset"}
SUMMARY_KEYS = ["roc_auc", "pr_auc", "tpr_at_5pct_fpr", "accuracy", "precision", "recall", "f1", "fpr"]


# =============================================================================== data prep
class Data:
    """Flat, in-memory flow table: features + metadata, no DataFrame overhead."""
    def __init__(self):
        self.X = None
        self.y = None          # int8 flow label
        self.attack = None     # uint8 code
        self.ds = None         # uint8 code
        self.host = None       # uint64 host key
        self.part = None       # int16
        self.attack_names = ["Benign"]
        self.ds_names = []


def _codes(values, names):
    out = np.empty(len(values), dtype=np.uint8)
    for i, v in enumerate(values):
        v = str(v)
        if v not in names:
            names.append(v)
        out[i] = names.index(v)
    return out


def _codes_fast(series, names):
    """Vectorised version for low-cardinality columns."""
    cats, inv = np.unique(series.astype(str).values, return_inverse=True)
    lut = np.empty(len(cats), dtype=np.uint8)
    for i, c in enumerate(cats):
        if c not in names:
            names.append(c)
        lut[i] = names.index(c)
    return lut[inv]


def _host_hash(ip_series):
    return pd.util.hash_array(ip_series.astype(str).values.astype(object))


def _append(parts, df, adapter, D, part_id):
    X = adapter.transform(df)
    parts["X"].append(X)
    parts["y"].append(df["Label"].values.astype(np.int8))
    atk = df["Attack"].astype(str)
    atk = atk.where(df["Label"].values == 1, "Benign")
    parts["attack"].append(_codes_fast(atk, D.attack_names))
    parts["ds"].append(_codes_fast(df["Dataset"], D.ds_names))
    parts["host"].append(_host_hash(df["IPV4_SRC_ADDR"]))
    parts["part"].append(np.full(len(df), part_id, dtype=np.int16))


def _finish(parts, D):
    D.X = np.concatenate(parts["X"]); D.y = np.concatenate(parts["y"])
    D.attack = np.concatenate(parts["attack"]); D.ds = np.concatenate(parts["ds"])
    D.host = np.concatenate(parts["host"]); D.part = np.concatenate(parts["part"])
    return D


def _new_parts():
    return {k: [] for k in ("X", "y", "attack", "ds", "host", "part")}


def prep_standard(files, adapter, train_parts):
    D, parts = Data(), _new_parts()
    D.attack_names = ["Benign"]
    sel = [(i, "train", 250000) for i in range(train_parts)] + \
          [(48, "val", 250000), (49, "val", 250000), (50, "test", None), (51, "test", None)]
    for i, tag, cap in sel:
        t = pq.read_table(files[i])
        if cap and len(t) > cap:
            t = t.slice(0, cap)
        _append(parts, t.to_pandas(), adapter, D, i)
        print(f"  [standard] part {i:02d} ({tag}) rows={len(t):,}", flush=True)
    D = _finish(parts, D)
    split = np.full(len(D.y), 255, dtype=np.uint8)
    split[D.part < train_parts] = 0
    split[(D.part >= 48) & (D.part <= 49)] = 1
    split[D.part >= 50] = 2
    # a host key is (ip, partition): windows never cross partitions (as in the original pipeline)
    D.host = D.host + D.part.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    return D, split


def prep_strict(files, adapter, cache_path, max_rows_per_part):
    if cache_path and os.path.exists(cache_path):
        print(f"[strict] loading cache {cache_path}")
        z = np.load(cache_path, allow_pickle=True)
        D = Data()
        D.X, D.y, D.attack, D.ds, D.host, D.part = (z[k] for k in ("X", "y", "attack", "ds", "host", "part"))
        D.attack_names, D.ds_names = list(z["attack_names"]), list(z["ds_names"])
        stats = json.loads(str(z["stats"]))
        return D, stats
    D, parts = Data(), _new_parts()
    seen = np.zeros(0, dtype=np.uint64)
    stats = {"rows_in": 0, "rows_kept": 0, "per_part": []}
    hash_cols = None
    for i, f in enumerate(files):
        t0 = time.time()
        t = pq.read_table(f)
        if max_rows_per_part and len(t) > max_rows_per_part:
            t = t.slice(0, max_rows_per_part)
        df = t.to_pandas()
        if hash_cols is None:
            hash_cols = [c for c in df.columns if c not in ID_COLS]
        h = pd.util.hash_pandas_object(df[hash_cols], index=False).values.astype(np.uint64)
        keep = ~pd.Series(h).duplicated().values
        keep &= ~np.isin(h, seen)
        seen = np.union1d(seen, h[keep])
        stats["rows_in"] += len(df); stats["rows_kept"] += int(keep.sum())
        stats["per_part"].append([i, len(df), int(keep.sum())])
        _append(parts, df[keep].reset_index(drop=True), adapter, D, i)
        print(f"  [strict] part {i:02d}: {len(df):,} -> {int(keep.sum()):,} unique rows "
              f"(total kept {stats['rows_kept']:,}, {time.time()-t0:.0f}s)", flush=True)
        del df, t
    D = _finish(parts, D)
    stats["dedup_fraction_removed"] = 1 - stats["rows_kept"] / max(stats["rows_in"], 1)
    if cache_path:
        np.savez(cache_path, X=D.X, y=D.y, attack=D.attack, ds=D.ds, host=D.host, part=D.part,
                 attack_names=np.array(D.attack_names), ds_names=np.array(D.ds_names),
                 stats=json.dumps(stats))
    return D, stats


def host_fold_split(D):
    fold = (D.host % np.uint64(10)).astype(np.int64)
    split = np.full(len(D.y), 0, dtype=np.uint8)
    split[fold == 8] = 1
    split[fold == 9] = 2
    return split


# =============================================================================== windows
def build_windows(D, rows, stride, min_flows, include_incomplete, max_wph):
    key = D.host[rows]
    groups = pd.Series(np.arange(len(rows))).groupby(key, sort=False).indices
    widx, wy, wa, wd, wh = [], [], [], [], []

    def add(idx):
        w = np.full(L, -1, dtype=np.int64)
        w[:len(idx)] = idx
        lab = D.y[idx]
        a = int((lab == 1).any())
        widx.append(w); wy.append(a)
        wa.append(D.attack[idx[lab == 1][0]] if a else 0)
        wd.append(D.ds[idx[0]]); wh.append(D.host[idx[0]])

    for _, loc in groups.items():
        idx = rows[loc]
        k = len(idx)
        if k < min_flows:
            continue
        if k >= L:
            added = 0
            for s in range(0, k - L + 1, stride):
                if added >= max_wph:
                    break
                add(idx[s:s + L]); added += 1
        elif include_incomplete:
            add(idx)
    if not widx:
        return None
    return dict(widx=np.stack(widx), y=np.array(wy, np.int8), attack=np.array(wa, np.uint8),
                ds=np.array(wd, np.uint8), host=np.array(wh, np.uint64))


def sub(W, sel):
    return {k: v[sel] for k, v in W.items()}


def gather_np(X, widx):
    M = widx >= 0
    out = np.zeros((len(widx), L, X.shape[1]), dtype=np.float32)
    out[M] = X[widx[M]]
    return out, M


def pool_np(X, widx, chunk=4000):
    res = []
    for i in range(0, len(widx), chunk):
        xb, mb = gather_np(X, widx[i:i + chunk])
        m = mb[:, :, None].astype(np.float32)
        n = m.sum(1).clip(1)
        mean = (xb * m).sum(1) / n
        mx = np.where(mb[:, :, None], xb, -1e9).max(1)
        std = np.sqrt((((xb - mean[:, None, :]) ** 2) * m).sum(1) / n)
        res.append(np.concatenate([mean, mx, std, n], 1).astype(np.float32))
    return np.concatenate(res)


# =============================================================================== metrics
def p95(vs, vy):
    if (vy == 0).sum() == 0:
        raise SystemExit("No benign validation windows -> cannot calibrate the threshold.")
    return float(np.percentile(vs[vy == 0], 95))


def evaluate(y, s, thr, n_boot=0, seed=0):
    pred = (s > thr).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    fpr_c, tpr_c, _ = roc_curve(y, s)
    r = dict(n=int(len(y)), n_attack=int(y.sum()), threshold=float(thr),
             accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
             recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
             fpr=fp / max(tn + fp, 1), roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
             tpr_at_5pct_fpr=float(tpr_c[np.searchsorted(fpr_c, 0.05, side="right") - 1]),
             confusion=dict(tn=tn, fp=fp, fn=fn, tp=tp))
    if n_boot:
        rng = np.random.default_rng(seed)
        ra = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                ra.append(roc_auc_score(y[i], s[i]))
        r["roc_auc_ci95"] = [float(np.percentile(ra, 2.5)), float(np.percentile(ra, 97.5))]
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}


def breakdowns(W, s, thr, D):
    y = W["y"]
    pred = s > thr
    per_src, per_cls = {}, {}
    for d in np.unique(W["ds"]):
        m = W["ds"] == d
        e = dict(n=int(m.sum()), attack_fraction=float(y[m].mean()),
                 fpr=float(pred[m & (y == 0)].mean()) if (m & (y == 0)).any() else None,
                 recall=float(pred[m & (y == 1)].mean()) if (m & (y == 1)).any() else None)
        if len(np.unique(y[m])) == 2:
            e["roc_auc"] = float(roc_auc_score(y[m], s[m]))
        per_src[D.ds_names[d]] = e
    for a in np.unique(W["attack"]):
        if a == 0:
            continue
        m = W["attack"] == a
        per_cls[D.attack_names[a].lower()] = dict(windows=int(m.sum()), hosts=int(len(np.unique(W["host"][m]))),
                                                  recall=float(pred[m].mean()))
    return per_src, per_cls


# =============================================================================== models
def lgbm_models(D, Wtr, Wva, Wte, seed, args):
    import lightgbm as lgb
    out = {}
    prior = pd.Series(Wtr["y"]).groupby(Wtr["ds"]).mean().to_dict()
    out["dataset_prior"] = (np.array([prior.get(d, .5) for d in Wva["ds"]]),
                            np.array([prior.get(d, .5) for d in Wte["ds"]]))
    kw = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
    cb = [lgb.early_stopping(30, verbose=False)]

    # flow-level supervised model, window score = max flow probability
    rng = np.random.default_rng(seed)
    flows = np.unique(Wtr["widx"][Wtr["widx"] >= 0])
    if len(flows) > args.max_flows:
        flows = rng.choice(flows, args.max_flows, replace=False)
    perm = rng.permutation(len(flows)); cut = int(0.9 * len(flows))
    tr_i, ev_i = flows[perm[:cut]], flows[perm[cut:]]
    fclf = lgb.LGBMClassifier(**kw).fit(D.X[tr_i], D.y[tr_i], eval_set=[(D.X[ev_i], D.y[ev_i])],
                                        eval_metric="auc", callbacks=cb)

    def fscore(W):
        o = np.empty(len(W["y"]), np.float32)
        for i in range(0, len(o), 2000):
            wi = W["widx"][i:i + 2000]
            m = wi >= 0
            p = np.full(wi.shape, -1.0, np.float32)
            p[m] = fclf.predict_proba(D.X[wi[m]])[:, 1]
            o[i:i + 2000] = p.max(1)
        return o
    out["flow_lgbm_max"] = (fscore(Wva), fscore(Wte))

    ptr, pva, pte = pool_np(D.X, Wtr["widx"]), pool_np(D.X, Wva["widx"]), pool_np(D.X, Wte["widx"])
    pclf = lgb.LGBMClassifier(**kw).fit(ptr, Wtr["y"], eval_set=[(pva, Wva["y"])], eval_metric="auc",
                                        callbacks=cb)
    out["window_lgbm_pool"] = (pclf.predict_proba(pva)[:, 1], pclf.predict_proba(pte)[:, 1])
    return out


def make_net(kind, repo):
    import torch
    import torch.nn as nn
    sys.path.insert(0, repo)
    from src.backend.models import bi_mambaformer as bm

    if kind == "bimambaformer":
        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = bm.BiMambaFormer(in_features=133, d_model=128, mamba_expand=2, mamba_state=16,
                                          attn_heads=4, dropout=0.15)

            def forward(self, x, mask):
                return self.m(x, mask=mask)["logits"].squeeze(-1)
        return Wrap()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            d = 128
            self.tok = nn.Sequential(nn.Linear(133, d), bm.RMSNorm(d), nn.Dropout(0.15))
            if kind == "mamba_only":
                self.core = bm.BidirectionalMambaBlock(d, 2, 16)
            elif kind == "attn_only":
                self.core = bm.MultiHeadFlowAttention(d, 4, 0.15)
            elif kind == "bilstm":
                self.core = nn.LSTM(d, d // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
            elif kind == "transformer":
                self.core = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d, 4, 2 * d, 0.15, batch_first=True, norm_first=True,
                                               activation="gelu"), 3)
            else:
                raise ValueError(kind)
            self.pool = bm.AttentionTemporalPooling(d)
            self.cls = nn.Sequential(nn.Linear(d, d // 2), nn.Mish(), nn.Dropout(0.15), nn.Linear(d // 2, 1))

        def forward(self, x, mask):
            h = self.tok(x)
            if kind == "mamba_only":
                h = self.core(h)
            elif kind == "attn_only":
                h = self.core(h, mask=mask)[0]
            elif kind == "bilstm":
                lens = mask.sum(1).clamp(min=1).cpu()
                pk = nn.utils.rnn.pack_padded_sequence(h, lens, batch_first=True, enforce_sorted=False)
                h, _ = self.core(pk)
                h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
            else:
                h = self.core(h, src_key_padding_mask=~mask)
            z, _ = self.pool(h, mask=mask)
            return self.cls(z).squeeze(-1)
    return Net()


def train_neural(kind, D, Wtr, Wva, Wte, seed, args, dev, Xg):
    import copy
    import torch
    import torch.nn as nn
    from torch.amp import GradScaler, autocast
    torch.manual_seed(seed); np.random.seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = make_net(kind, args.repo).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = GradScaler("cuda" if dev.type == "cuda" else "cpu")
    crit = nn.BCEWithLogitsLoss()
    Wt = torch.from_numpy(Wtr["widx"]).to(dev)
    yt = torch.from_numpy(Wtr["y"].astype(np.float32)).to(dev)

    def batch(W, ix):
        idx = W[ix]
        m = idx >= 0
        return Xg[idx.clamp(min=0)] * m.unsqueeze(-1).float(), m

    def score(W):
        Wg = torch.from_numpy(W["widx"]).to(dev)
        net.eval(); o = []
        with torch.no_grad():
            for i in range(0, len(Wg), 1024):
                ix = torch.arange(i, min(i + 1024, len(Wg)), device=dev)
                x, m = batch(Wg, ix)
                with autocast(device_type=dev.type):
                    o.append(torch.sigmoid(net(x, m).float()).cpu().numpy())
        return np.concatenate(o)

    best, best_state = -1, None
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(Wt), device=dev); t0 = time.time(); tot = 0
        for i in range(0, len(perm), 256):
            ix = perm[i:i + 256]
            x, m = batch(Wt, ix)
            opt.zero_grad()
            with autocast(device_type=dev.type):
                loss = crit(net(x, m).float(), yt[ix])
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); tot += loss.item() * len(ix)
        sched.step()
        vs = score(Wva)
        pr = average_precision_score(Wva["y"], vs)
        print(f"    {kind} seed={seed} ep{ep+1}/{args.epochs} loss={tot/len(perm):.4f} valPR={pr:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if pr > best:
            best, best_state = pr, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    return score(Wva), score(Wte), n_par


# =============================================================================== runner
def run_protocol(name, D, split, args, seeds, models, win_cfg, out_dir):
    import importlib.util
    has_torch = importlib.util.find_spec("torch") is not None
    dev = Xg = None
    neural = [m for m in models if m not in ("dataset_prior", "flow_lgbm_max", "window_lgbm_pool")]
    if neural and has_torch:
        import torch
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Xg = torch.from_numpy(D.X).to(dev)
    elif neural:
        print("[WARN] torch not installed -> neural models skipped")
        neural = []

    rows = {s: np.where(split == k)[0] for s, k in (("train", 0), ("val", 1), ("test", 2))}
    W = {}
    for s in ("train", "val", "test"):
        c = win_cfg[s]
        W[s] = build_windows(D, rows[s], c["stride"], 4, c["incomplete"], c["wph"])
        if W[s] is None:
            raise SystemExit(f"[{name}] no windows for {s}")
        if s == "val" and len(W[s]["y"]) > args.max_val_windows:
            W[s] = sub(W[s], np.random.default_rng(0).choice(len(W[s]["y"]), args.max_val_windows, replace=False))
    info = {s: dict(windows=int(len(W[s]["y"])), attack_windows=int(W[s]["y"].sum()),
                    hosts=int(len(np.unique(W[s]["host"])))) for s in W}
    cls = {s: {D.attack_names[a].lower(): int((W[s]["attack"] == a).sum()) for a in np.unique(W[s]["attack"]) if a}
           for s in W}
    print(f"[{name}] windows:", json.dumps(info))
    res = {"info": info, "class_counts": cls, "models": {}}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        n = len(W["train"]["y"])
        Wtr = sub(W["train"], rng.choice(n, min(n, args.max_train_windows), replace=False))
        scores = lgbm_models(D, Wtr, W["val"], W["test"], seed, args)
        for k in neural:
            vs, ts, npar = train_neural(k, D, Wtr, W["val"], W["test"], seed, args, dev, Xg)
            scores[k] = (vs, ts)
            res.setdefault("n_params", {})[k] = npar
        for mname, (vs, ts) in scores.items():
            thr = p95(vs, W["val"]["y"])
            e = evaluate(W["test"]["y"], ts, thr, n_boot=(200 if seed == seeds[0] else 0), seed=seed)
            ps, pc = breakdowns(W["test"], ts, thr, D)
            e["per_source"], e["per_class"] = ps, pc
            res["models"].setdefault(mname, []).append(dict(seed=seed, **e))
            print(f"  [{name}] seed={seed} {mname:17s} rocAUC={e['roc_auc']:.4f} prAUC={e['pr_auc']:.4f} "
                  f"TPR@5%={e['tpr_at_5pct_fpr']:.4f} F1={e['f1']:.4f} FPR={e['fpr']:.4f}", flush=True)
    summ = {}
    for m, runs in res["models"].items():
        summ[m] = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
                   for k in SUMMARY_KEYS}
    res["summary_mean_std"] = summ
    with open(os.path.join(out_dir, f"{name}_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[{name}] SUMMARY (mean +- std over seeds {seeds})")
    for m, s in summ.items():
        print(f"  {m:17s} " + " ".join(f"{k}={s[k][0]:.4f}+-{s[k][1]:.4f}" for k in SUMMARY_KEYS[:4]))
    del Xg
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/protocols")
    ap.add_argument("--protocol", default="both", choices=["standard", "strict", "loso", "both", "all"])
    ap.add_argument("--seeds", default="11")
    ap.add_argument("--models", default="dataset_prior,flow_lgbm_max,window_lgbm_pool,bimambaformer,"
                                        "mamba_only,attn_only,bilstm,transformer")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-train-windows", type=int, default=40000)
    ap.add_argument("--max-val-windows", type=int, default=12000)
    ap.add_argument("--max-flows", type=int, default=1500000)
    ap.add_argument("--standard-train-parts", type=int, default=20)
    ap.add_argument("--max-rows-per-part", type=int, default=0, help="debug: cap rows read per partition")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.repo)
    from src.backend.sequences import PureNfUqAdapter
    adapter = PureNfUqAdapter(model_dir=os.path.join(args.repo, "src", "backend", "saved_models"))
    files = sorted(glob.glob(os.path.join(args.data_dir, "part_*.parquet")))
    seeds = [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",")
    protos = {"both": ["standard", "strict"], "all": ["standard", "strict", "loso"]}.get(
        args.protocol, [args.protocol])

    if "standard" in protos:
        assert len(files) >= 52
        D, split = prep_standard(files, adapter, args.standard_train_parts)
        cfg = {"train": dict(stride=16, incomplete=False, wph=150),
               "val": dict(stride=32, incomplete=False, wph=100),
               "test": dict(stride=32, incomplete=True, wph=150)}
        run_protocol("standard", D, split, args, seeds, models, cfg, args.out)
        del D

    if "strict" in protos or "loso" in protos:
        D, stats = prep_strict(files, adapter, os.path.join(args.out, "strict_cache.npz"),
                               args.max_rows_per_part)
        with open(os.path.join(args.out, "strict_dedup_stats.json"), "w") as f:
            json.dump({k: v for k, v in stats.items() if k != "per_part"} |
                      {"class_counts_after_dedup": {D.attack_names[a]: int((D.attack == a).sum())
                                                    for a in np.unique(D.attack)},
                       "source_counts_after_dedup": {D.ds_names[a]: int((D.ds == a).sum())
                                                     for a in np.unique(D.ds)}}, f, indent=2)
        cfg = {s: dict(stride=(16 if s == "train" else 32), incomplete=True, wph=150)
               for s in ("train", "val", "test")}
        if "strict" in protos:
            run_protocol("strict", D, host_fold_split(D), args, seeds, models, cfg, args.out)
        if "loso" in protos:
            base = host_fold_split(D)
            for k, held in enumerate(D.ds_names):
                split = np.full(len(D.y), 255, dtype=np.uint8)
                split[(D.ds != k) & (base == 0)] = 0
                split[(D.ds != k) & (base == 1)] = 1
                split[D.ds == k] = 2
                run_protocol(f"loso_heldout_{held}", D, split, args, seeds[:1], models, cfg, args.out)
    print("\nALL DONE ->", args.out)


if __name__ == "__main__":
    main()
