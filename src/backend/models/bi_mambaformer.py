"""
=============================================================================
  bi_mambaformer.py — Dual-Stream Hybrid Bi-MambaFormer Architecture
=============================================================================
  State-of-the-Art Deep Hybrid Network for Sequential Network Intrusion Detection
  on NF-UQ-NIDS-v2 Telemetry.

  Architecture:
    1. Flow Feature Tokenizer: Linear(133 -> D) + RMSNorm
    2. Stream 1: Bidirectional Selective State Space (Bi-Mamba SSM)
       - Continuous temporal rate drift, cumulative byte monitoring, O(L) scaling
       - Forward and Backward selective state transitions
    3. Stream 2: Multi-Head Temporal Self-Attention (MHSA)
       - Discrete pairwise non-local flow correlation (probe-to-exploit links)
       - Mask-aware attention scoring
    4. Adaptive Gated Fusion:
       - Learned gating G = sigmoid(W_g · [H_ssm ; H_attn])
       - Dynamic routing of continuous drift vs. discrete jump threats
    5. Attention-Weighted Temporal Pooling:
       - Learned pooling weights w = softmax(W_p · H_fused)
       - Pinpoints exact culprit flow in the 32-flow window
    6. Threat Classification Head:
       - MLP(D -> D/2 -> 1) with Mish activations and calibrated output
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for fast, stable training."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SelectiveSSM(nn.Module):
    """
    Pure PyTorch Selective State Space (SSM) Block.
    Implements input-dependent continuous-to-discrete state transitions:
      h_t = exp(Δ_t * A) * h_{t-1} + (Δ_t * B_t) * x_t
      y_t = C_t * h_t
    """
    def __init__(self, d_inner: int, d_state: int = 16, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # Initialize A to log-spaced negative eigenvalues (HiPPO-inspired stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # A = -exp(A_log)

        # Discretization delta projection (input-dependent step size)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        # Initialize delta bias to span [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Input-dependent B and C projections
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_len, d_inner)
        Returns:
            y: Tensor of shape (Batch, Seq_len, d_inner)
        """
        B, L, D = x.shape
        N = self.d_state

        # Compute dynamic step size Δ_t > 0
        delta = F.softplus(self.dt_proj(x))  # (B, L, D)

        # Dynamic B_t and C_t
        b_t = self.B_proj(x)  # (B, L, N)
        c_t = self.C_proj(x)  # (B, L, N)

        # Continuous A matrix (always strictly negative for contractive stability)
        A = -torch.exp(self.A_log)  # (D, N)

        # Discretized A_bar = exp(Δ_t * A): shape (B, L, D, N)
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # Discretized B_bar * x_t: shape (B, L, D, N)
        # x: (B, L, D, 1), b_t: (B, L, 1, N) -> outer product
        delta_B_x = (delta.unsqueeze(-1) * x.unsqueeze(-1)) * b_t.unsqueeze(2)

        # Sequential state scan (fast across L=32 on modern tensor cores)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        y_list = []

        for t in range(L):
            h = delta_A[:, t] * h + delta_B_x[:, t]  # (B, D, N)
            # y_t = h_t · C_t: sum over N state dimension
            y_t = torch.sum(h * c_t[:, t].unsqueeze(1), dim=-1)  # (B, D)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)  # (B, L, D)
        return y


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block for Telemetry Sequences.
    Processes sequences in both forward (t=1->L) and backward (t=L->1) directions,
    fusing them via a causal 1D convolution and SiLU gated projection.
    """
    def __init__(self, d_model: int = 128, expand: int = 2, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        # Input projection & gating branch
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Depthwise Convolution (captures local multi-flow burst context)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
            bias=True
        )

        # Forward and Backward SSM cores
        self.ssm_fwd = SelectiveSSM(d_inner=self.d_inner, d_state=d_state)
        self.ssm_bwd = SelectiveSSM(d_inner=self.d_inner, d_state=d_state)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Seq_len, d_model)
        Returns:
            out: (Batch, Seq_len, d_model)
        """
        residual = x
        x_norm = self.norm(x)

        # Project and split into signal and gate
        projected = self.in_proj(x_norm)  # (B, L, 2 * d_inner)
        u, gate = projected.chunk(2, dim=-1)

        # Local 1D depthwise convolution
        u_conv = self.conv1d(u.transpose(1, 2)).transpose(1, 2)
        u_conv = F.silu(u_conv)

        # Forward SSM
        y_fwd = self.ssm_fwd(u_conv)

        # Backward SSM: reverse along sequence dimension, run SSM, reverse back
        u_rev = torch.flip(u_conv, dims=[1])
        y_bwd_rev = self.ssm_bwd(u_rev)
        y_bwd = torch.flip(y_bwd_rev, dims=[1])

        # Symmetrize and gate
        y = 0.5 * (y_fwd + y_bwd)
        y = y * F.silu(gate)

        out = self.out_proj(y) + residual
        return out


class MultiHeadFlowAttention(nn.Module):
    """
    Multi-Head Temporal Self-Attention (MHSA) Stream.
    Discovers discrete, non-local relational jump correlations across flows
    (e.g., initial port scan probe at t=2 triggering an exploit at t=28).
    Exposes attention weights for native Explainable AI (XAI).
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (Batch, Seq_len, d_model)
            mask: (Batch, Seq_len) bool tensor (True for valid, False for padding)
        Returns:
            out: (Batch, Seq_len, d_model)
            attn_weights: (Batch, n_heads, Seq_len, Seq_len)
        """
        residual = x
        x_norm = self.norm(x)
        B, L, _ = x.shape

        # Linear projections
        Q = self.q_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, L, d_k)
        K = self.k_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, h, L, L)

        # Apply attention mask for padded flows
        if mask is not None:
            # mask: (B, L) -> (B, 1, 1, L)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            # Use -1e4 instead of -1e9 to prevent float16 (c10::Half) overflow in AMP
            scores = scores.masked_fill(~mask_expanded, -1e4)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        context = torch.matmul(attn_weights_drop, V)  # (B, h, L, d_k)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)

        out = self.out_proj(context)
        out = self.dropout(out) + residual
        return out, attn_weights


class AdaptiveGatedFusion(nn.Module):
    """
    Adaptive Learned Gated Fusion Layer.
    Dynamically balances continuous state-space drift (Mamba) vs.
    discrete pairwise relational jumps (Self-Attention):
      Gate G_t = sigmoid(W_g · [H_ssm ; H_attn] + b)
      H_fused = G_t ⊙ H_ssm + (1 - G_t) ⊙ H_attn
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.gate_linear = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_ssm: torch.Tensor, h_attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([h_ssm, h_attn], dim=-1)
        gate = self.gate_linear(combined)  # (B, L, d_model) in [0, 1]

        fused = gate * h_ssm + (1.0 - gate) * h_attn
        out = self.norm(fused)
        return out, gate


class AttentionTemporalPooling(nn.Module):
    """
    Learned Attention-Weighted Temporal Pooling.
    Calculates importance weight w_t for each flow in the 32-flow sequence,
    collapsing the window into a single representative threat vector while
    isolating the exact culprit flow that triggered the anomaly.
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.score_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (Batch, Seq_len, d_model)
            mask: (Batch, Seq_len) bool
        Returns:
            z: (Batch, d_model)
            weights: (Batch, Seq_len) attribution weights summing to 1.0
        """
        scores = self.score_proj(h).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)

        weights = F.softmax(scores, dim=-1)  # (B, L)
        # Weighted sum across sequence dimension
        z = torch.sum(h * weights.unsqueeze(-1), dim=1)  # (B, d_model)
        return z, weights


class BiMambaFormer(nn.Module):
    """
    Flagship Dual-Stream Hybrid Bi-MambaFormer for Network Intrusion Detection.
    Combines Bidirectional Mamba SSM + Multi-Head Temporal Self-Attention
    via Adaptive Gated Fusion.
    """
    def __init__(
        self,
        in_features: int = 133,
        d_model: int = 128,
        mamba_expand: int = 2,
        mamba_state: int = 16,
        attn_heads: int = 4,
        dropout: float = 0.15
    ):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model

        # 1. Flow Feature Tokenizer
        self.tokenizer = nn.Sequential(
            nn.Linear(in_features, d_model),
            RMSNorm(d_model),
            nn.Dropout(dropout)
        )

        # 2. Parallel Streams
        self.stream_mamba = BidirectionalMambaBlock(
            d_model=d_model,
            expand=mamba_expand,
            d_state=mamba_state
        )
        self.stream_attention = MultiHeadFlowAttention(
            d_model=d_model,
            n_heads=attn_heads,
            dropout=dropout
        )

        # 3. Adaptive Gated Fusion
        self.fusion = AdaptiveGatedFusion(d_model=d_model)

        # 4. Attention-Weighted Temporal Pooling
        self.pooling = AttentionTemporalPooling(d_model=d_model)

        # 5. Threat Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (Batch, Seq_len, 133)
            mask: (Batch, Seq_len) bool, optional
        Returns:
            Dict containing:
              - 'logits': (Batch, 1)
              - 'threat_score': (Batch, 1) in [0, 1]
              - 'flow_attribution': (Batch, Seq_len) weights indicating culprit flows
              - 'gate_weights': (Batch, Seq_len, d_model) Mamba vs Attention balance
              - 'attention_matrix': (Batch, h, Seq_len, Seq_len)
        """
        # Tokenize tabular flows
        tokens = self.tokenizer(x)  # (B, L, D)

        # Stream 1: Bidirectional Mamba SSM
        h_ssm = self.stream_mamba(tokens)  # (B, L, D)

        # Stream 2: Multi-Head Flow Self-Attention
        h_attn, attn_matrix = self.stream_attention(tokens, mask=mask)  # (B, L, D)

        # Step 3: Adaptive Gated Fusion
        h_fused, gate = self.fusion(h_ssm, h_attn)  # (B, L, D)

        # Step 4: Temporal Attention Pooling (isolates culprit flow)
        z, flow_attr = self.pooling(h_fused, mask=mask)  # (B, D), (B, L)

        # Step 5: Threat Classification
        logits = self.classifier(z)  # (B, 1)
        threat_score = torch.sigmoid(logits)

        return {
            "logits": logits,
            "threat_score": threat_score,
            "flow_attribution": flow_attr,
            "gate_weights": gate,
            "attention_matrix": attn_matrix,
            "latent": z
        }
