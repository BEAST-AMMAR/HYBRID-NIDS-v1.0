# 🛡️ AUDIT & ROLLBACK TRACE LOG
## Hybrid NIDS — Phase 2 Dual-Stream Bi-MambaFormer (`phase2/mambaformer-nfuq-v2`)

This document maintains a complete, immutable engineering audit trail of every commit, architectural fix, test verification, and exact Git rollback procedure on the `phase2/mambaformer-nfuq-v2` branch.

> **Branch Protection Note**: `origin/main`, `origin/baseline/nsl-kdd`, and `origin/baseline/nfuq-v2` remain 100% untouched. All Phase 2 work and hardening commits reside exclusively on `phase2/mambaformer-nfuq-v2`.

---

## 1. Chronological Commit & Trace Ledger

| Order | Commit Hash | Git Tag / Ref | Summary of Changes & Files Affected | Verification Status |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `fdd3f96` | `v1.0-baseline-nsl-kdd`<br>(`origin/main`, `origin/baseline/nsl-kdd`) | **Baseline 1 (NSL-KDD)**: Shallow Autoencoder + Isolation Forest (41 features $\to$ 122), 90.52% accuracy on `KDDTest+.txt`. | Baseline 1 Verified |
| **2** | `927eb18` | `v1.1-baseline-nfuq-v2`<br>(`origin/baseline/nfuq-v2`) | **Baseline 2 (NF-UQ-NIDS-v2)**: Deep PyTorch AE + Latent IF + Mahalanobis + $k$-NN (133 features, 1,702,252 flows), 88.33% acc, 97.13% prec. | Baseline 2 Verified |
| **3** | `fc4d7d6` | `phase2/submission-ready` | **Phase 2 Flagship Submission**: Dual-Stream Bi-MambaFormer (`bi_mambaformer.py`, `bi_mambaformer_best.pt`), `RESEARCH_THESIS.md`, `THESIS_COMPARATIVE_ANALYSIS.md`, `DEFENSE_PRESENTATION_AND_QA.md`, 14 300-DPI figures, frozen `requirements.txt`, `tests/test_e2e_verification.py`, `tests/test_live_websocket_pipeline.py`. | E2E & WS Tests Passed |
| **4** | `fbef54d` | — | **Comparative README & IP Protection**: Enriched `README.md` with 3-baseline comparison, integrated `docs/legacy_baselines/README_baseline2_nfuq.md`, replaced MIT license notice with Proprietary / All-Rights-Reserved notice. | Verified |
| **5** | `6a1c600` | — | **Copilot Audit Remediation (Phase 1 & 2)**:<br>• `docs/legacy_baselines/*.md`: Purged remaining MIT badges.<br>• `detector.py`: Enforced training-runtime mask-aware zero-padding parity.<br>• `auth.py` & `main.py`: Added cryptographic 256-bit runtime JWT secret, optional WS token auth (`?token=`), `/api/health`, `/api/readiness`, and `Alert` DB persistence (`/api/alerts`).<br>• `verify_split_manifest.py` & `split_manifest.json`: SHA-256 artifact checksums & partition disjointness assertions. | E2E & WS Tests Passed |
| **6** | `3d8ee25` | — | **Runtime Parity & Safety Hardening**:<br>• `sniffer.py`: Replaced `NfUqFeatureAdapter` with `PureNfUqAdapter` (eliminated TensorFlow startup dependency).<br>• `detector.py`: Added 133-feature schema guard & fail-loud missing-weights check.<br>• `main.py`: Bounded queue (`maxsize=10000`), 100 MB upload limit, `telemetry_source` tagging.<br>• `rag_service.py`: Added `PROVISIONAL AI EVALUATION` disclaimer.<br>• `tests/test_offline_runtime_parity.py`: Proved `0.00e+00` float parity ($\le 10^{-5}$). | Parity (`0.00e+00`), E2E, & WS Tests Passed |
| **7** | `314cdf4` | — | **SOC UI Responsiveness & Labelling**:<br>• `Monitor.jsx`: Capped visible DOM packet feed to 100 entries (reduced DOM snapshot from >343 KB to ~39 KB) and updated engine card text to Phase 2 Bi-MambaFormer. | Verified in Chrome DOM |
| **8** | `da63e5e` | `v2.0-phase2-verified`<br>(`HEAD`, `origin/phase2/mambaformer-nfuq-v2`) | **Live Pipeline Scoring, RAG, XAI & Event-Loop Fixes**:<br>• `sniffer.py`: Fixed `0x08` (PSH) vs `0x04` (RST) bitmask bug in `is_legitimate_handshake` (dropped normal HTTP/HTTPS FPR from 17.88% to **0.00%**).<br>• `rag_service.py`: Reordered `_classify_anomaly()` so Brute Force, Botnet C2, and Port Scanning classify accurately before generic DoS/Shellcode rules.<br>• `detector.py`: Upgraded XAI to exact neural input-gradient saliency $\left|\frac{\partial \text{logit}}{\partial x}\right|$.<br>• `models/__init__.py`: Added `Alert` SQLAlchemy model to resolve package shadow.<br>• `main.py`: Fixed FastAPI event-loop starvation (`explain=False` on normal flows, queue draining on mode switches, `await asyncio.sleep(0)`).<br>• `Monitor.jsx`: Added 401 session-expiration auto-logout redirect.<br>• Added `tests/diagnose_1000_packets.py` & `tests/verify_live_localhost_1000.py`. | Verified Live in Chrome DOM (1,328+ packets, 0% normal FPR, distinct RAG/XAI) |

---

## 2. Exact Rollback Procedures

If any issue arises later, you can inspect or roll back cleanly using Git without risking data loss:

### A. Inspect Differences Between Any Two Checkpoints
```powershell
# Compare current HEAD against initial Phase 2 submission
git diff fc4d7d6..HEAD --stat

# Inspect a specific commit's exact code diff
git show da63e5e
```

### B. Surgical Rollback of a Single Commit (Safe, Preserves History)
To undo a specific fix while keeping all other commits intact:
```powershell
# Example: revert only commit da63e5e
git revert da63e5e

# Or revert UI change 314cdf4 only
git revert 314cdf4
```

### C. Roll Back a Single File to an Earlier Commit
```powershell
# Example: restore sniffer.py to commit 3d8ee25
git checkout 3d8ee25 -- src/backend/sniffer.py
```

### D. Create a Clean Recovery Branch from Any Milestone Tag
```powershell
# Switch back to Baseline 1
git checkout -b recovery/baseline1 v1.0-baseline-nsl-kdd

# Switch back to Baseline 2
git checkout -b recovery/baseline2 v1.1-baseline-nfuq-v2

# Switch back to initial Phase 2 submission (before hardening)
git checkout -b recovery/phase2-initial fc4d7d6

# Switch back to verified Phase 2 hardened state
git checkout -b recovery/phase2-verified v2.0-phase2-verified
```

---

## 3. Standalone Archive Backup
- **Archive File**: `../Phase2_BiMambaFormer_Submission_Package.zip`
- **Export Method**: `git archive --format=zip HEAD` (strictly tracked files, zero temporary/uncommitted artifacts, excludes raw multi-GB Parquet partitions, `venv/`, and `node_modules/`).
