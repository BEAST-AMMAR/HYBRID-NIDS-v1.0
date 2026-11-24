# HYBRID-NIDS-v1.0 - Issues and Exact Remediation Guide
*(Sourced from GitHub Copilot Audit)*

---

## 1. Benchmark leakage and split integrity

**Issue:** NF-UQ partitions may contain duplicate flows and repeated hosts or sources. Baseline 1 calibrates against test data.

**Solution:**
- Deduplicate exact flows before splitting.
- Assert strict disjointness for rows, hosts, source datasets, and capture sessions.
- Never use source or destination IP as a model feature.
- Calibrate Baseline 1 only from validation data.
- Repeat with multiple seeds and source leave-out tests.
- Produce a split manifest with hashes, row counts, host counts, source counts, and assertions.

```python
assert set(train_hosts).isdisjoint(val_hosts)
assert set(train_hosts).isdisjoint(test_hosts)
assert set(val_hosts).isdisjoint(test_hosts)
assert not train_df.duplicated(all_feature_columns).any()
```
**Acceptance:** Every split assertion passes.

---

## 2. Metric units are incomparable

**Issue:** Baseline 1 uses packets, Baseline 2 uses flows, and Phase 2 uses sequence windows.

**Solution:**
- Publish separate packet, flow, host, and sequence metrics.
- Report unique missed flows, windows, and hosts separately.
- Normalize all attack-class names.
- Never present cross-baseline misses as a direct numerical reduction.

```json
{
  "unit": "flow",
  "n_test": 1702252,
  "accuracy": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "roc_auc": 0.0,
  "pr_auc": 0.0,
  "false_positive_rate": 0.0,
  "false_negative_rate": 0.0,
  "confusion_matrix": {}
}
```

---

## 3. Runtime and benchmark logic differ

**Issue:** Baseline 2 benchmark rank fusion differs from deployed gated fusion. Phase 2 live padding differs from offline sequence extraction.

**Solution:**
- Share feature adapters, sequence builders, threshold logic, score functions, and metrics.
- Use the same mask-aware padding in training and inference.
- Do not replicate short histories.
- Add fixed-vector offline/runtime parity tests.

**Acceptance:** Offline and runtime scores differ by no more than $1\times 10^{-5}$.

---

## 4. Live packet conversion is approximate

**Issue:** Scapy packets are heuristically converted into NF-UQ fields.

**Solution:**
- Integrate a real NetFlow/IPFIX exporter or Zeek/Suricata pipeline.
- Export timestamps, bidirectional counters, TCP state, packet-length histograms, TTL, retransmissions, and protocol fields.
- Reject incomplete telemetry instead of inventing benchmark assumptions.
- Label live capture, synthetic replay, and dataset replay separately.

---

## 5. Sequence construction is ambiguous

**Issue:** Windows are positive if any flow is malicious. Host/source grouping is not proven leak-free.

**Solution:**
- Store both `sequence_label` and optional `anchor_label`.
- Sort by timestamp within a capture session.
- Group by host plus capture session/source.
- Reject host overlap across splits.
- Define minimum-flow and padding behavior in configuration.
- Report sequence recall separately from culprit-flow recall.

---

## 6. Data and artifacts are not self-validating

**Issue:** NF-UQ partitions are absent and model/feature compatibility is not cryptographically checked.

**Solution:**
- Add dataset download and checksum scripts.
- Store partition row counts and label distributions.
- Add manifests for checkpoints, adapters, feature columns, thresholds, and splits.
- Verify SHA-256 hashes at startup.
- Fail on missing artifacts or unknown feature counts.

---

## 7. Dependencies are incomplete

**Issue:** The NF-UQ branch requires PyTorch, PyArrow, SciPy, PyYAML, and CUDA packages absent from older manifests.

**Solution:**
- Use one locked environment.
- Separate CPU, CUDA, runtime, training, test, and development dependencies.
- Remove unused TensorFlow/SHAP runtime dependencies.
- Pin Python, PyTorch, CUDA, NumPy, pandas, PyArrow, scikit-learn, FastAPI, and SQLAlchemy.
- Add Docker or a Dev Container.

---

## 8. Authentication and authorization are unsafe

**Issue:** JWT has an insecure fallback, credentials are hard-coded, CORS is permissive, and WebSockets are unauthenticated.

**Solution:**
- Require `JWT_SECRET_KEY` with a strong minimum length.
- Remove hard-coded usernames and passwords.
- Provision the first administrator through an environment secret or CLI prompt.
- Add viewer, analyst, operator, and admin roles.
- Protect capture, upload, policy, model, and alert actions.
- Authenticate WebSockets with short-lived bearer tokens.
- Restrict CORS to configured origins.
- Use PostgreSQL and Alembic for production.

---

## 9. Upload and queue processing are unsafe

**Issue:** Upload limits, schemas, processing timeouts, job status, cancellation, queue depth, and failures are incomplete.

**Solution:**
- Enforce upload size, row count, decompression size, and processing time limits.
- Validate filename, MIME type, CSV/Parquet schema, and row count.
- Use tracked background jobs with status and cancellation.
- Bound queue depth and expose queue-full behavior.
- Use structured client errors and server-side exception logs.

---

## 10. Real-time observability is missing

**Issue:** No formal health, queue, drop, latency, model-readiness, or alert-feedback metrics exist.

**Solution:**
- Add liveness and readiness endpoints.
- Export queue depth, drop count, processing latency, inference latency, WebSocket count, and feedback metrics.
- Add structured logs and trace IDs.
- Add stop, pause, resume, drain, and buffer-reset controls.
- Never silently drop live packets.

---

## 11. RAG explanations require caution

**Issue:** RAG and heuristic classification can be presented as definitive evidence.

**Solution:**
- Separate observed telemetry, retrieved knowledge, model wording, and recommendations.
- Add knowledge sources and review status.
- Mark explanations provisional.
- Add confirmed, false-positive, and unknown feedback.

---

## 12. Alert lifecycle is missing

**Issue:** Alerts are transient WebSocket messages with no investigation workflow.

**Solution:**
- Persist model version, unit, score, threshold, features, explanation, host/session IDs, and status.
- Implement: `new` -> `acknowledged` -> `investigating` -> `confirmed` -> `resolved` / `false_positive`.
- Add SIEM/Syslog or webhook integration.
- Add alert severity and analyst resolution.

---

## 13. Tests and CI are incomplete

**Issue:** Custom scripts are not an independent test and CI gate.

**Solution:**
- Add pytest unit, integration, benchmark-integrity, security, API, WebSocket, upload, and end-to-end suites.
- Run compilation, lint, type checks, frontend build, secret/dependency scans, split assertions, schema checks, and metric reconciliation in CI.
- CI must fail on split leakage or artifact/metric mismatch.

---

## 14. Licensing is contradictory

**Issue:** The latest branch says proprietary/all-rights-reserved, while legacy README files claim MIT and no license file exists.

**Solution:**
- Use proprietary/all-rights-reserved consistently across every branch.
- Remove MIT badges and claims.
- Add an actual copyright notice and terms document.

---

## 15. Documentation contains unsupported claims

**Issue:** Documentation includes stale diagrams, unsupported cross-baseline comparisons, absolute file links, and “100% tests passed” language.

**Solution:**
- State units and denominators beside every metric.
- Replace local absolute links with relative paths.
- Add model-card, data-card, threat-model, limitation, reproducibility, and deployment-status documents.

---

## Specific Baseline Remediation Directives

### Main / Baseline 1
- Calibrate thresholds from validation, never test data.
- Test 41-to-122 feature alignment and artifact round-trips.
- Reconcile training, validation, test, confusion matrix, and novel-attack metrics.

### Baseline 2
- Run duplicate and host/source split assertions.
- Reconcile rank-fused benchmark and runtime gated detector behavior.
- Verify the 1,702,252-flow confusion matrix.
- Add leave-one-source-out evaluation.

### Phase 2
- Verify 32-flow windows, masks, labels, and padding.
- Assert disjoint hosts, sources, sessions, and deduplicated flows.
- Report 19,098 sequence metrics separately from flow coverage.
- Verify every threshold policy independently.
- Match offline and runtime sequence scores within tolerance.

---

## Priority Order for Implementation
1. Split integrity and metric reconciliation.
2. Benchmark/runtime parity.
3. Artifact and environment reproducibility.
4. Authentication and WebSocket security.
5. Upload and real-time queue safety.
6. Alert lifecycle and observability.
7. Tests, CI, licensing, and documentation.

*No branch should be called production-ready until the first four phases pass independently.*
