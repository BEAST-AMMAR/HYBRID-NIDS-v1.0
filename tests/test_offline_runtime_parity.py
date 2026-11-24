"""
tests/test_offline_runtime_parity.py
====================================
Validates exact numerical parity between offline benchmark evaluation and
runtime HybridDetector inference (Copilot Issue 3).

Acceptance Criteria:
  The absolute numerical difference between direct BiMambaFormer forward pass
  and runtime HybridDetector.predict() / predict_sequence() must not exceed 1e-5.
"""

import os
import sys
import torch
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.backend.detector import HybridDetector
from src.backend.models.bi_mambaformer import BiMambaFormer

def test_full_sequence_runtime_parity():
    """Verify that a full 32-flow sequence matches offline forward pass <= 1e-5."""
    models_dir = os.path.join(_PROJECT_ROOT, "src", "backend", "saved_models")
    detector = HybridDetector(models_dir)

    # 1. Deterministic test sequence
    np.random.seed(42)
    seq_np = np.random.randn(1, 32, 133).astype(np.float32)
    mask_np = np.ones((1, 32), dtype=bool)

    # 2. Offline forward pass
    x_t = torch.from_numpy(seq_np).to(detector.device)
    m_t = torch.from_numpy(mask_np).to(detector.device)
    detector.bi_mambaformer.eval()
    with torch.no_grad():
        out_offline = detector.bi_mambaformer(x_t, mask=m_t)
        score_offline = float(out_offline["threat_score"].item())
        attn_offline = out_offline["flow_attribution"].cpu().numpy()

    # 3. Runtime predict_sequence
    res_seq = detector.predict_sequence(x_t, mask=m_t)
    score_seq = float(res_seq["threat_score"][0])
    attn_seq = res_seq["flow_attribution"][0]

    # 4. Runtime predict (batched 3D mode)
    _, score_batch = detector.predict(seq_np, mask=mask_np)

    # 5. Parity Assertions (Tolerance: 1e-5)
    diff_seq = abs(score_offline - score_seq)
    diff_batch = abs(score_offline - float(score_batch[0]))
    attn_diff = np.max(np.abs(attn_offline[0] - attn_seq))

    print(f"\n[*] Full Sequence Parity Check:")
    print(f"    Offline Score:       {score_offline:.8f}")
    print(f"    predict_sequence:    {score_seq:.8f} (diff: {diff_seq:.2e})")
    print(f"    predict(3D):         {score_batch[0]:.8f} (diff: {diff_batch:.2e})")
    print(f"    Attribution MaxDiff: {attn_diff:.2e}")

    assert diff_seq <= 1e-5, f"predict_sequence score drift {diff_seq} exceeds 1e-5!"
    assert diff_batch <= 1e-5, f"predict(3D) score drift {diff_batch} exceeds 1e-5!"
    assert attn_diff <= 1e-5, f"Attribution drift {attn_diff} exceeds 1e-5!"

def test_masked_partial_sequence_parity():
    """Verify that partial sequences (k < 32) maintain mask-aware zero-padding parity."""
    models_dir = os.path.join(_PROJECT_ROOT, "src", "backend", "saved_models")
    detector = HybridDetector(models_dir)

    k = 12
    np.random.seed(1337)
    flows = np.random.randn(k, 133).astype(np.float32)

    # Zero-padded window with boolean mask
    win_np = np.zeros((1, 32, 133), dtype=np.float32)
    win_np[0, :k] = flows
    mask_np = np.zeros((1, 32), dtype=bool)
    mask_np[0, :k] = True

    # Offline forward
    x_t = torch.from_numpy(win_np).to(detector.device)
    m_t = torch.from_numpy(mask_np).to(detector.device)
    with torch.no_grad():
        out_offline = detector.bi_mambaformer(x_t, mask=m_t)
        score_offline = float(out_offline["threat_score"].item())

    # Simulate streaming through host buffer
    detector.reset_host_buffer("test_host_masked")
    final_score = None
    for f in flows:
        is_anom, final_score = detector.predict(f.reshape(1, -1), host_ip="test_host_masked")

    diff_streaming = abs(score_offline - final_score)
    print(f"\n[*] Masked Partial Sequence (k={k}) Parity Check:")
    print(f"    Offline Masked Score: {score_offline:.8f}")
    print(f"    Streaming Buffer Score: {final_score:.8f} (diff: {diff_streaming:.2e})")

    assert diff_streaming <= 1e-5, f"Streaming masked drift {diff_streaming} exceeds 1e-5!"

if __name__ == "__main__":
    test_full_sequence_runtime_parity()
    test_masked_partial_sequence_parity()
    print("\n[+] ALL RUNTIME & OFFLINE PARITY CHECKS PASSED (Tolerance <= 1e-5)!")
