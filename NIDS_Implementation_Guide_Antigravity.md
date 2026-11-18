# NIDS Implementation Guide (for Antigravity)

Scope: harden the Phase 1 Hybrid NIDS (NSL-KDD), then build a second baseline on NF-UQ-NIDS-v2 using the same pipeline. Phase 2 models (Mamba, BiLSTM) come later and reuse the harness built here.

Companion files: `NIDS_Phase1_Review_and_Fixes.md` (what is wrong and why), `nids-console.html` (UI reference).

---

## 0. Rules for the coding agent (read first)

1. Work in the task order of Section 14. One task at a time. After each task: run its acceptance check, commit with the task ID in the message (`M1: leak-free threshold`).
2. Do not change anything outside the task. Do not rewrite working code for style.
3. **Never invent numbers.** Every metric in a doc, README or report must be read from a results file written by the code. If a result is missing, write "not yet run".
4. **Never tune on the test set.** Test data is read once, at the final evaluation step, by `evaluate.py` only.
5. If a column name, count or path here does not match reality, stop, record the real value in `docs/DATA_NOTES.md`, and continue with the real value. Do not guess.
6. Every experiment is driven by a YAML config and a seed. No hardcoded paths or constants in code.
7. Keep the algorithms fixed (see Section 1). Only the data adapter changes between datasets.

---

## 1. Goals, frozen parts, adapted parts

**Goal.** Two reproducible baselines with the same pipeline and the same evaluation protocol:

- Baseline A1: NSL-KDD (Phase 1, fixed).
- Baseline A2: NF-UQ-NIDS-v2 (new).
- Baseline B: supervised tree model on the same features and splits, as a reality check.

**Frozen (identical across datasets):** autoencoder architecture (input → 64 → 32 → 16 → 32 → 64 → input), optimiser and epochs, Isolation Forest settings, fusion method, threshold policies, metrics, seeds, result file format.

**Adapted per dataset (the "schema adapter"):** column selection, categorical encoding, numeric transforms, label mapping, split strategy. Input dimension changes, nothing else in the model.

**Why:** a reviewer accepts "same pipeline" if stages, hyperparameters and protocol are identical. Feature schemas cannot be identical because NSL-KDD has content and window features and NetFlow has neither.

**Non-goals now:** Mamba/BiLSTM training, new attack-type classifiers, live packet capture accuracy claims.

---

## 2. Target repository layout

```
nids/
  configs/            nsl_kdd.yaml  nf_uq.yaml  base.yaml
  src/nids/
    data/             nsl_kdd.py  nf_uq.py  splits.py  sequences.py
    features/         encoders.py  transforms.py
    models/           autoencoder.py  iforest.py  scores.py  fusion.py
    calibrate.py      thresholds.py  evaluate.py  report.py
    baselines/        trees.py
  scripts/            run_experiment.py  make_tables.py
  tests/
  results/            <dataset>/<experiment_id>/<seed>/
  docs/               DATA_NOTES.md  DECISIONS.md
  app/                backend/ (FastAPI)  frontend/ (React)
```

The existing `src/backend` and `src/frontend` are moved under `app/` in task G0. The ML code becomes an importable package so the API and the experiments use the same code.

---

## 3. Global protocol (applies to every experiment)

### 3.1 Leakage rules

| Rule | Why |
|---|---|
| Fit scalers, quantile transformers, encoders, AE, IF **only on training benign data**. | Anything fitted on test or attack data leaks. |
| Thresholds and fusion weights come from the **validation** split only. | Phase 1 tuned on test (inflated results). |
| Deduplicate full rows **before** splitting. | Duplicates across train and test inflate scores. |
| If any resampling is used, apply it after the split and to training only. | AE-DTNN-style resampling before split leaks. |
| Test set is evaluated once per (config, seed). | Prevents implicit tuning. |

### 3.2 Seeds

Run seeds `[11, 22, 33, 44, 55]`. Report mean ± std. Seed controls: split sampling, subsampling, AE init, IF, batching. Set `PYTHONHASHSEED`, NumPy, TensorFlow/PyTorch seeds, and `TF_DETERMINISTIC_OPS=1` where available.

### 3.3 Config (example `base.yaml`)

```yaml
seeds: [11, 22, 33, 44, 55]
ae: {hidden: [64, 32, 16], activation: relu, epochs: 40, batch_size: 256, early_stop_patience: 5, val_fraction: 0.1}
iforest: {n_estimators: 200, max_samples: auto, on: latent}
fusion: {method: rank_mean, components: [mse, maha, knn], knn_k: 5, knn_ref_size: 20000}
thresholds: {policies: [p95_val_normal, p99_val_normal, youden_val, fpr_5pct, fpr_3pct]}
```

**Why these defaults:** they are the settings used in the review experiments (40 epochs, early stopping, 200 trees). Keep them so results stay comparable.

### 3.3.1 Hyperparameters are not tuned on test

If you want to tune (epochs, bottleneck), do it on the validation split and record every trial in `results/.../tuning.csv`. Do not tune per dataset beyond input dimension in the frozen comparison. A separate "tuned" run may be added and clearly labelled.

---

## 4. Evaluation contract (what every run must write)

For each `(dataset, experiment, seed, threshold_policy)` write `metrics.json`:

```json
{
  "dataset": "nsl_kdd", "experiment": "hybrid_rank_mean", "seed": 11, "policy": "p95_val_normal",
  "threshold": 0.0, "n_test": 0,
  "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "fpr": 0,
  "roc_auc": 0, "pr_auc": 0,
  "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
  "latency_ms_per_1k": 0, "model_size_mb": 0,
  "per_class_recall": {"DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0},
  "per_source": {}
}
```

Also write: `roc_points.csv` (fpr, tpr, threshold), `pr_points.csv`, `scores_test.parquet` (id, label, class, source, each score), `env.json` (library versions, CPU/GPU, git commit), `config_used.yaml`.

**Why:** the later papers need ROC curves, per-class tables and exact reproducibility. Saving raw scores means plots and new thresholds can be produced without retraining. These files are also what you send back for paper writing (Section 15).

**Metric notes**
- Positive class = attack for both datasets.
- Always report FPR next to accuracy.
- PR-AUC is mandatory on NF-UQ (attack majority).
- `latency_ms_per_1k`: time to score 1,000 flows, batch size 1,024, CPU, warm model.

---

## 5. Part A: Phase 1 fixes on NSL-KDD

Files in the existing repo: `src/backend/train_pipeline.py`, `detector.py`, `model_metrics.py`, `saved_models/*`, `data/KDDTrain+.txt`, `data/KDDTest+.txt`.

### A0. Freeze the old results as a reference
Run the existing saved models on KDDTest+ and save the numbers as `results/nsl_kdd/phase1_legacy/`. Expected (from review): accuracy 90.52%, FPR 11.85%, AUC 0.9672 at threshold 0.127941. If they differ, record the actual and continue.
**Why:** a before/after record. The legacy numbers are what the old documents claimed, so the paper needs a clean reference.
**Accept:** `metrics.json` exists with the re-run values and a note that the threshold was test-tuned.

### A1 (M1, M2). Leak-free threshold
Split KDDTrain+ into `fit` (70%) and `val` (30%), stratified by label, seeded. AE, IF and preprocessing are fitted on `fit` benign rows only. Thresholds are computed on `val`. Evaluate once on KDDTest+.
**Why:** removes test tuning. This is the most important fix.
**Accept:** grep shows no code path that reads `y_test` before `evaluate.py`. A unit test fails if `thresholds.py` receives test data.

### A2 (M3). Isolation Forest on latent, normal-only
Train IF on AE latent vectors (16-D) of `fit` benign rows. Do not use `contamination`; scores come from `decision_function`.
**Why:** in testing, IF AUC rose from 0.841 (raw, mixed) to 0.943 (latent, normal-only). Latent space is smaller and denser, and normal-only matches how the AE is trained.
**Accept:** `scores.py` exposes `iforest_score(latent)`; the training code never sees attack rows.

### A3. Score components
Compute per sample:
- `mse`: mean squared reconstruction error.
- `top8`: mean of the 8 largest per-feature squared errors.
- `maha`: Mahalanobis distance of the latent vector to the benign latent mean (covariance + 1e-3·I).
- `knn`: mean distance to the 5 nearest benign latent points (reference set 20,000 random fit-benign points).
- `iforest`: latent IF score.

**Why:** they measure different things (reconstruction, density shape, local density, isolation). In testing, `rank_mean(mse, maha, knn)` reached AUC 0.9726 vs 0.9672 for the old fusion.

### A4 (M5). Fusion
Implement three fusions and keep all for ablation:
1. `alpha_blend` (legacy): `alpha·mse_norm + (1−alpha)·iforest_norm`, with alpha tuned on validation (grid 0 to 1, step 0.1, maximise validation AUC).
2. `rank_mean` (default): for each component, convert score to its percentile among **validation normals**, then average.
3. `lr_fusion` (optional, labelled semi-supervised): logistic regression over component scores fitted on validation.

**Why default `rank_mean`:** it needs no attack labels and does not depend on score scales. `lr_fusion` uses validation labels, so it must not be called unsupervised.
**Accept:** unit test: percentile mapping of validation normals is uniform on [0, 1].

### A5 (M4, M9, M10). Ablation, seeds, per-class results
Experiments: `ae_mse`, `ae_top8`, `iforest_latent`, `maha`, `knn`, `alpha_blend`, `rank_mean`, `lr_fusion`. For each: 5 seeds × 5 threshold policies.
Per-class recall uses this mapping of `KDDTest+` labels (anything not listed is R2L):
- DoS: back, land, neptune, pod, smurf, teardrop, apache2, udpstorm, processtable, worm, mailbomb
- Probe: ipsweep, nmap, portsweep, satan, mscan, saint
- U2R: buffer_overflow, loadmodule, perl, rootkit, ps, sqlattack, xterm, httptunnel
- Normal: `normal`

Also report recall on attacks not present in KDDTrain+ ("novel") vs present ("known").
**Why:** the review found hybrid ≈ AE-alone, R2L recall 74%, and novel-attack recall 83.6%. These must be visible in the paper.
**Accept:** `make_tables.py` produces `ablation.csv` and `per_class.csv` from the result files only.

### A6 (M7). UDP/private analysis
Report: share of `udp & service=private` among train normals vs test normals (review: 1.45% vs 8.65%), and FPR with and without that group. Save as `analysis/udp_private.json`.
**Why:** about 58% of Phase 1 false positives came from this group. It is a documented dataset shift and a genuine finding.

### A7 (M8). Dynamic threshold
Either evaluate it offline on replayed test data (compare fixed vs dynamic FPR and recall) or remove it from the detector. Do not ship it unevaluated.
**Why:** its buffer only keeps non-anomalies, so it drifts upward and reinforces itself.

### A8. Operating-point policy
Report these policies for every run: `p95_val_normal`, `p99_val_normal`, `youden_val`, `fpr_5pct`, `fpr_3pct` (largest threshold with validation FPR ≤ target). Mark one as the **primary** in the config (default `p95_val_normal`).
**Why:** FPR and recall trade off. The paper should show the curve and name an operating point, not one lucky number.

### A9. Secondary protocol
Add an experiment on a stratified 60/20/20 random split of KDDTrain+ (labelled "internal split"). Never mix its numbers with KDDTest+ numbers in one table without the protocol column.
**Why:** many papers report this protocol (95%+). It is needed for literature comparison, but it is not comparable to KDDTest+.

**Expected ranges (sanity checks, not targets):** KDDTest+ leak-free accuracy roughly 87–89%, FPR roughly 3–5%, AUC roughly 0.97. Internal split roughly 96% accuracy, ~4% FPR. If your results are far outside, check for leakage or bugs. **Do not tune to hit these.**

---

## 6. Part B: NF-UQ-NIDS-v2 baseline (A2)

### B0. Verify the dataset first (write `docs/DATA_NOTES.md`)
Record: file names and formats, row count, column list, dtypes, label columns (`Label`, `Attack`, and a source-dataset column if present), benign/attack counts, per-source counts, number of attack classes, null/inf counts, duplicate rate.
**Why:** the guide lists the standard NetFlow v2 feature set from the literature (43 features). The actual file may differ in column names or extras. Everything downstream depends on this table.

### B1. Chunked loading
Read in chunks (Parquet via polars/pyarrow if available, otherwise CSV chunks of 1M rows). Convert to Parquet once and work from Parquet.
**Why:** about 76M rows does not fit in memory on a student machine.

### B2. Cleaning
1. Replace ±inf with NaN. Impute NaN with the benign-train median, fitted on train only. Record NaN counts per column.
2. Drop rows with invalid labels.
3. **Deduplicate on all feature columns plus label** before splitting. Record how many rows were removed per source and class.

**Why:** NetFlow data has many exact duplicates (scans, retransmits). Without dedup, identical rows end up in both train and test.

### B3. Subsampling (deterministic)
Stratify by `(source, attack_class)`. Defaults:
- Benign train pool: up to 1.5M rows, proportional across sources.
- Each attack class: up to 150k rows (all rows if fewer).
- Validation and test: drawn from their own splits with the same caps.

Record the sampling table in `results/nf_uq/sampling.csv`.
**Why:** full data is unnecessary for AE+IF and prohibitive for iteration. Capping attack classes prevents a few huge classes (e.g., DoS-like) from dominating metrics. A fixed seed and a recorded table keep it reproducible.

### B4. Splits (three protocols)
1. **Primary: per-source temporal split.** Within each source, sort by `FLOW_START_MILLISECONDS` and assign the first 70% to train/fit, the next 10% to validation, the last 20% to test.
2. **Secondary: stratified random split** after dedup (same ratios).
3. **Generalisation: leave-one-source-out (LOSO).** Train and validate on three sources, test on the fourth. Repeat for each source.

**Why primary is temporal:** flows from the same session or scan land close in time, so random splits leak near-duplicates. Temporal splits are harder and more honest. LOSO tests whether the model learns attacks or learns "which testbed".
**Caveat:** some attack classes may be absent from the early part of a source. This is fine for the unsupervised model (trained on benign only). Record missing classes per split.

### B5. Feature adapter

**Verify each column name in B0 first.** Actions below follow the standard NetFlow v2 feature set.

| Group | Columns (as in the standard set) | Action |
|---|---|---|
| Identifiers | `IPV4_SRC_ADDR`, `IPV4_DST_ADDR` | **Drop from features.** Keep only as grouping keys for sequences (Part D). |
| Time | `FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS` | Not features. Used for temporal split and ordering. |
| Ports | `L4_SRC_PORT` | Drop (ephemeral). |
| | `L4_DST_PORT` | Top 20 ports by benign-train frequency as one-hot, others to `port_other`. |
| Protocol | `PROTOCOL` | One-hot (TCP, UDP, ICMP, other). |
| L7 | `L7_PROTO` | Top 30 by benign-train frequency as one-hot, others to `l7_other`. Categorical, never ordinal. |
| TCP flags | `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS` | Decompose each bitmask into 8 binary columns (FIN, SYN, RST, PSH, ACK, URG, ECE, CWR). |
| ICMP / DNS / FTP | `ICMP_TYPE`, `ICMP_IPV4_TYPE`, `DNS_QUERY_TYPE`, `FTP_COMMAND_RET_CODE` | Presence flag + one-hot of top few values. |
| Random IDs | `DNS_QUERY_ID` | Drop. |
| Numeric | bytes, packets, durations, TTLs, packet lengths, IATs, throughputs, retransmissions, packet-size bins, TCP window | `log1p`, then QuantileTransformer(output=normal, n_quantiles=1000) fitted on **benign train**. |
| Binary columns | flags above | No scaling. |
| Labels / source | `Label`, `Attack`, source column | Not features. Keep for targets and per-source reporting. |

**Why:**
- Dropping IPs and ephemeral ports prevents memorising testbed hosts.
- One-hot with a top-K cap controls dimensionality (high-cardinality L7 would explode it).
- `log1p` handles the heavy tails of bytes and packet counts. The QuantileTransformer then matches what Phase 1 used for NSL-KDD, so preprocessing logic stays the same in spirit.
- Fit on benign train only keeps the leak rules.

Persist everything fitted (`top_ports.json`, `top_l7.json`, quantile transformer, imputer medians, final column list) to `artifacts/nf_uq/<seed>/`.
**Accept:** a test that transforms 1,000 rows and checks: no NaN/inf, column order equals saved list, binary columns unscaled, no identifier columns.

### B6. Model and training
Same AE (64→32→16, 40 epochs, early stop), same IF on latent, same scores, fusion and policies as Part A. Train on benign rows of the train split only.
**Why same architecture:** the frozen-pipeline claim. If a reviewer asks about capacity, add a clearly labelled variant with a wider AE (128→64→32), but keep the frozen one as the headline.

### B7. Metrics specific to this dataset
In addition to Section 4:
- PR-AUC (attack positive).
- FPR **per source** and recall per attack class.
- Results on all three protocols (primary temporal, random, LOSO).

**Why:** attacks are the majority (~67% by the UQ documentation), so accuracy alone is misleading. Per-source FPR exposes the multi-environment "normal" problem.

### B8. Source-shortcut check (required)
Train a small classifier (LightGBM, 100 trees) to predict the **source dataset** from the same features on benign rows only. Report accuracy and AUC. Also train a classifier to predict benign vs attack using only the source ID.
**Why:** the dataset merges four testbeds. If source is easily predictable, models can score well by recognising the testbed. This must be disclosed in the paper.

---

## 7. Part C: Baseline B (supervised trees)

Implement `baselines/trees.py` with LightGBM (default) and Random Forest (small, for reference).

- Same features, same splits, same seeds as Part B (and Part A for NSL-KDD).
- Train on train split labelled data with `class_weight="balanced"` (or `scale_pos_weight`). **No SMOTE/ADASYN.**
- Choose the decision threshold on validation (same five policies).
- Same `metrics.json` output.

**Why no resampling:** it is a common leakage source and not needed for trees. Keeping it simple makes this a credible, strong reference.
**Why this baseline exists:** on NetFlow data tree models are often very strong. If AE+IF or Mamba does not beat them, the paper states it. If they do, the claim is stronger.
**Label this paradigm as "supervised" in every table.**

---

## 8. Part D: Sequence builder for Phase 2 (prepare only)

Implement `data/sequences.py` that, for NF-UQ, builds per-source-IP sequences:
- Group by `IPV4_SRC_ADDR` (and optionally destination), sort by start time.
- Windows: fixed length L (default 32, also 16 and 64) with stride L/2, or time windows of 60 s.
- Sequence label: attack if any flow in the window is an attack, plus the fraction of attack flows (for analysis).
- Split by the same temporal boundaries as B4 so no window crosses a split boundary.
- Save as Parquet of indices into the feature matrix (not copies).

**Why:** Mamba/BiLSTM need meaningful sequences. A row-as-sequence design removes any advantage of sequence models. This also recreates the "window" information (counts per source) that NSL-KDD had and NetFlow lacks.
Do not train sequence models in this task. Produce the builder, a unit test, and `sequence_stats.json` (count, mean attack fraction, label balance).

---

## 9. Part E: RAG / explanation layer fixes

Files: `rag_service.py`, `kb_index.py`, `detector.py`, knowledge base JSON files.

| Task | What to do | Why |
|---|---|---|
| E1 (R3) | Replace substring checks with exact feature-name set membership. Group one-hot columns back to their parent (`service_*` → `service`). | `'count' in joined_string` matches `srv_count` and `dst_host_count`, so most anomalies become "DoS". |
| E2 (R5) | Rank features by AE per-feature error contribution aggregated by parent feature. Drop per-alert SHAP or compute it only on demand. | One-hot dims dominate; per-alert SHAP on the forest is slow. |
| E3 (R4) | In live mode, skip rules that need content features (zeroed by the sniffer) and say so in the explanation metadata (`"content_features_available": false`). | Those rules can never fire live. |
| E4 (R6) | Build the retrieval query as a sentence from parent features and values, not a concatenation of column names. | Sentence-like queries embed better. |
| E5 (R7) | Merge the two KBs into one schema (`id, family, title, description, indicators[], mitigation[], source`). Balance entries across DoS, Probe, R2L, U2R. Store a content hash in the index file; rebuild if it changes. | Casing/schema mismatch and a stale index silently break retrieval. |
| E6 (R8) | Pin one embedding model. Store its name and dimension in the index metadata and refuse to query on mismatch. Remove the `openai.Embedding.create` path. Never put exception text in user-visible output. Update docs from "FAISS" to what is actually used. | Provider switching caused dimension errors and leaked exceptions. |
| E7 (R9) | Add a minimum cosine similarity; below it return "no close reference found". | Top-k is currently always shown, even when irrelevant. |
| E8 (R1, R2) | Decide and implement one: (a) rename to "retrieval-grounded explanation" and keep it template-based, or (b) add an optional LLM generation step that may only use retrieved passages and must return structured JSON. Default to (a). | The current code is not RAG. Option (a) is honest and cheap; (b) is optional because it adds cost and risk. |
| E9 (R11, R12) | Output structured JSON (schema in Section 12). Cache explanations per (family, top features) and aggregate bursts. Remove ground-truth labels from alerts. | Markdown-with-emoji coupling to the UI; label leakage in dataset mode. |
| E10 (R10) | Evaluate on KDDTest+: map labels to families, report explainer family accuracy and macro-F1, retrieval hit@1/hit@3 against an annotated subset (50–100 alerts annotated by you), and a faithfulness check (cited features must be within the top-10 error features). Save as `results/rag/`. | There is currently no evaluation of this layer, although labels exist. |

**Default choice and why:** option (a) in E8 because the research claim is about detection and explanation quality, and the explainer is rule-based. Calling it RAG without generation would be challenged in review.

---

## 10. Part F: Backend / API fixes

| Task | What to do | Why |
|---|---|---|
| F1 (B1) | CORS: explicit origins from `ALLOWED_ORIGINS` env var; no regex `http://.*`. | Any http origin could call the API with credentials. |
| F2 (B2) | WebSocket auth: client first calls `/ws-ticket` (JWT-protected) and receives a 60 s one-time ticket; connects to `/ws/monitor?ticket=...`. | The socket is currently open to anyone. |
| F3 (B3) | `queue.Queue(maxsize=N)`; count and expose dropped packets. | The queue is unbounded; floods exhaust memory. |
| F4 (B4) | Fail on startup if `JWT_SECRET_KEY` is unset. | Hardcoded fallback secret. |
| F5 (B5) | OAuth: exchange a one-time code for a token, or use an httpOnly cookie. Frontend URL from env. | Token in URL leaks via history and referrer. |
| F6 (B6) | One detector per session, or a lock around `predict`. Keep dynamic threshold per session (if A7 keeps it). | Shared buffer and non-thread-safe predict cause races. |
| F7 (B7) | Upload: size limit, streamed parsing, a `/mode/stop` that actually stops replay (threading.Event), vectorised one-hot. | Whole file read into memory; replay cannot be stopped; per-row `get_dummies` is slow. |
| F8 (B8) | `alerts` table: id, ts, session, score, threshold, family, features JSON, status (new/acknowledged/dismissed). Endpoints: list, ack, export CSV. | No history exists. |
| F9 (B9) | Use FastAPI lifespan; make sniffer optional via env; rate-limit login; DB path from env; async DB or run sync DB in a threadpool. | Deprecated hooks; blocking calls; root requirement at startup. |
| F10 (B10) | Tests: auth flow, WS ticket, upload limits, detector scoring on a fixed fixture, parser. | No tests exist. |

---

## 11. Part G: Frontend (React port of the new design)

Reference: `nids-console.html` (design) and Section 6 of `NIDS_Phase1_Review_and_Fixes.md`.

- G0: Move to `app/frontend`. Use **one** Tailwind setup (v4 with `@tailwindcss/postcss`), remove v3. Move `playwright` to devDependencies. Set title and favicon.
- G1: Config through `import.meta.env` (`VITE_API_URL`, `VITE_WS_URL`). No hardcoded hosts.
- G2: Components: `Login`, `Signup`, `ConsolePage`, `ScoreTape` (canvas), `FlowTable`, `Inspector`, `ThresholdControl`, `StatusBar`; hooks: `useAuth`, `useAlertSocket` (ticket flow, reconnect with backoff), `useFlowBuffer`.
- G3: State: ring buffer of at most 600 flows, rendering throttled with `requestAnimationFrame`, table shows the last 80 matching rows, stable keys by flow id (not index).
- G4: Errors: a failed start or upload shows what failed. No swallowed errors. Stop calls the backend stop endpoint.
- G5: Auth: handle 401 and expiry; store the token in memory or an httpOnly cookie, not `localStorage` (if backend F5 supports cookies).
- G6: Accessibility: status by text and shape, visible focus, `aria-live` for the verdict sentence, no `alert()`.
- G7: Render explanation fields as text (never `dangerouslySetInnerHTML`).
- G8: Tests: component tests for table filtering and threshold re-flagging; one Playwright smoke test using the demo stream.

**Design tokens:** paper `#e9edf0`, panel `#f7f8f9`, ink `#17222d`, normal `#1e7f74`, flagged `#d9402a`, warning `#b57a0a`; dark variants in the reference file. Fonts: Instrument Sans (UI), Spline Sans Mono (addresses and scores only).

---

## 12. WebSocket message schema (backend → frontend)

```json
{
  "type": "flow",
  "id": 123,
  "ts": "2026-10-21T10:15:00Z",
  "score": 0.213,
  "threshold": 0.128,
  "is_anomaly": true,
  "flow": {"src": "10.0.1.5", "dst": "10.0.9.7", "proto": "tcp", "service": "http", "bytes": 512},
  "explanation": {
    "family": "SYN flood",
    "family_confidence": "rule-based hint",
    "content_features_available": true,
    "features": [{"name": "serror_rate", "contribution": 0.41, "meaning": "Share of connections with SYN errors"}],
    "references": [{"id": "kb-017", "title": "...", "similarity": 0.62}]
  }
}
```

Never include ground-truth labels in live alerts. Evaluation views read labels from result files only.

---

## 13. Tests and docs

**Tests (pytest):** leak guard (thresholds never see test data), preprocessing invariants (NSL-KDD and NF-UQ), dedup before split, split determinism per seed, score component shapes, fusion percentile mapping, metrics against hand-computed confusion matrices, result schema validation, sequence builder boundaries.
**Docs sync:** generate the feature list and architecture tables from code/config into `docs/GENERATED.md`. Delete stale statements (41 vs 116 vs 122 features, 100 vs 250 trees, FAISS). Every number in README and reports must come from `results/` via `make_tables.py`.
**Why:** documentation drift caused the contradictory 96.13% / 90.52% figures.

---

## 14. Execution order

| Step | Tasks | Output |
|---|---|---|
| 1 | Repo layout, config system, env pinning, result writer, metrics module | skeleton + tests |
| 2 | A0, A1, A2, A3, A4 | leak-free NSL-KDD hybrid |
| 3 | A5, A6, A8, A9 | ablation, per-class, policies, internal split |
| 4 | A7, E1–E3, E5–E7 (quick RAG fixes) | stable explainer |
| 5 | B0–B3 | verified data notes, Parquet, sampling table |
| 6 | B4–B8 | NF-UQ baseline A2 with 3 protocols + shortcut check |
| 7 | C (trees on both datasets) | Baseline B |
| 8 | D (sequence builder) | Phase 2 ready |
| 9 | F1–F10, G0–G8, E4, E8–E10 | secured backend, new UI, evaluated explainer |
| 10 | Docs sync, `make_tables.py`, `RESULTS_MANIFEST.md` | paper-ready tables |

Steps 1–3 and 5–7 are the research-critical path. Steps 4 and 9 are engineering and can run in parallel after step 3.

---

## 15. What to send back for the research and paper work

Zip and send: `results/` (all `metrics.json`, `roc_points.csv`, `scores_test.parquet` if size allows, `per_class.csv`, `ablation.csv`), `configs/`, `docs/DATA_NOTES.md`, `docs/DECISIONS.md`, `env.json`, training logs, and the source code zip. Also send screenshots of the new UI and any errors you could not resolve.
`RESULTS_MANIFEST.md` should list every experiment id, config, seed, runtime, and any deviation from this guide.

---

## 16. Decision log

| Decision | Alternatives | Why this one |
|---|---|---|
| Frozen model, adapted schema | Force NSL-KDD columns onto NetFlow | Impossible (no content/window features), and would be meaningless. |
| `rank_mean` fusion default | Learned LR fusion, hand-set alpha | Label-free and scale-free; LR uses validation labels so is not unsupervised. |
| Temporal split primary on NF-UQ | Random split primary | Random leaks near-duplicate flows; temporal is stricter. |
| Dedup before split | Dedup after | After-split dedup leaves identical rows in both sets. |
| Top-K one-hot for ports/L7 | Ordinal or hashing | Ordinal implies false order; hashing hurts interpretability for the explainer. |
| No SMOTE/ADASYN | Resample to balance | Not needed for anomaly models or trees; common leakage source. |
| LightGBM as Baseline B | Deep supervised model | Strong, fast, and a fair "reality check". |
| 5 seeds | 1 seed | Variance is required for credible claims. |
| Explainer renamed (E8a) | Add LLM generation | Honest and cheap; LLM generation adds cost and hallucination risk. |
| Token via ticket / httpOnly cookie | Token in URL or localStorage | Avoids leaks via history, referrers and XSS. |

---

## 17. Known pitfalls

- Attack majority on NF-UQ: do not use accuracy as the headline.
- Source shortcut: always run B8 and report it.
- Quantile transformer fitted on mixed data: fit on benign train only.
- Duplicate-heavy flows: dedup, and report how many rows were removed.
- Class absent in a temporal slice: record it, do not drop silently.
- Mixing KDDTest+ and internal-split numbers in one table: always include the protocol column.
- Any statement of the form "95%+" must name the dataset, protocol and threshold policy next to it.
- Live sniffer features are approximations of KDD features (content features zeroed). Do not claim live accuracy.

---

## 18. Open items to confirm during implementation

- Exact NF-UQ-NIDS-v2 column names, counts and source column (B0).
- Whether the file named `standard_scaler.joblib` is a StandardScaler or the QuantileTransformer used in the code (docs and file name disagree). Check and rename accordingly.
- Available hardware: AE and IF run on CPU; GPU only matters for Phase 2 models.
- Whether per-alert SHAP is required for the paper. If yes, keep it as an on-demand call (E2).
