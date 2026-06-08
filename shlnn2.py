"""
╔══════════════════════════════════════════════════════════════════════════╗
║       Spiking Hypergraph Liquid Neural Network (SHLNN)                  ║
║       Single-file notebook edition  v2                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Combines:                                                               ║
║    - LTC ODE dynamics         (Hasani et al., AAAI 2021)                ║
║    - HGNN convolution         (Feng et al., AAAI 2019)                  ║
║    - Surrogate gradients      (Zenke & Ganguli, 2021)                   ║
║    - Continuous-Time RoPE     (content-dependent position encoding)     ║
║    - Multi-source streaming   (FineWeb-Edu / Wikipedia / StarCoder)     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  USAGE (Colab / Jupyter):                                                ║
║    Cell 1 — paste this whole file and run                               ║
║    Cell 2 — run_mock()    synthetic data, zero downloads                ║
║    Cell 3 — run_stream()  real web-scale data via HuggingFace streams   ║
║    Cell 4 — run_ptb()     classic PTB benchmark                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Install:  pip install tiktoken datasets                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

# 1. Pull the secret token securely from Kaggle's vault
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")

# 2. Login to the Hugging Face hub
login(token=hf_token)

print("Authenticated with Hugging Face successfully!")

# ── standard library ─────────────────────────────────────────────────────
import os, math, time, random, json, statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# ── third-party ──────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset

# ── tiktoken: BPE tokenizer (same vocab as GPT-2, 50257 tokens) ─────────
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    print("[WARN] tiktoken not found — streaming data sources disabled. "
          "Run: pip install tiktoken datasets")

# ── HuggingFace Datasets (streaming) ─────────────────────────────────────
try:
    import datasets as hf_datasets
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False
    print("[WARN] datasets not found — streaming sources disabled. "
          "Run: pip install datasets")


# ═══════════════════════════════════════════════════════════════════════════
# §1  REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════
# §2  CONTINUOUS-TIME RoPE  (CT-RoPE)
#
#     Standard RoPE assumes fixed integer positions [0, 1, 2, ...].
#     CT-RoPE replaces them with content-dependent cumulative timestamps:
#
#         dt_t  = softplus(W_dt - x_t)  > 0        [per-token time step]
#         tau_t = sum_{s <= t} dt_s                [cumulative position]
#         cos/sin applied at tau_t instead of t
#
#     Properties:
#       - Causal: tau_t depends only on x_0..x_t
#       - Identity-reducible: W_dt approx= 0 at init -> dt approx= 1 -> tau = [0,1,2,...]
#       - KV-cache compatible: tau_t is fixed once token t is processed
#
#     Source: adapted from LiQ-LM (2025); original RoPE: Su et al. (2021).
# ═══════════════════════════════════════════════════════════════════════════

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class CTRoPE(nn.Module):
    """
    Continuous-Time Rotary Position Embedding.

    Produces (cos, sin) tensors shaped (1, N, head_dim) for a sequence of
    N token nodes. Called once per SHLNNLayer forward; result modulates
    the x_hg representation before it feeds into the LTC parameter network.

    Parameters
    ----------
    embed_dim : dimension of the incoming feature vector (post-HGNN)
    head_dim  : dimension to apply RoPE over (= out_channels of HGNN layer)
    base      : frequency base (default 10 000, same as LLaMA/GPT-NeoX/RoPE)
    """
    def __init__(self, embed_dim: int, head_dim: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)        # (head_dim/2,)

        # Projects the token feature -> scalar dt > 0
        self.dt_proj = nn.Linear(embed_dim, 1, bias=True)
        self._init_dt()

    def _init_dt(self):
        """Init: softplus(output) approx= 1 -> tau approx= integer positions at t=0."""
        nn.init.zeros_(self.dt_proj.weight)
        self.dt_proj.bias.data.fill_(math.log(math.exp(1.0) - 1))  # approx= 0.541

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (N, embed_dim)

        Returns
        -------
        cos, sin : (1, N, head_dim) -- broadcast-ready rotation tensors
        """
        dt  = F.softplus(self.dt_proj(x)).clamp(max=4.0).squeeze(-1)  # (N,)
        tau = torch.cumsum(dt, dim=0) - dt[0]                          # (N,) first=0

        freqs = tau.unsqueeze(-1) * self.inv_freq                       # (N, head_dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)                       # (N, head_dim)

        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)           # (1, N, head_dim) each


def apply_ctrope(x: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply CT-RoPE rotation to a feature tensor.

    Parameters
    ----------
    x        : (N, D) -- feature tensor to rotate
    cos, sin : (1, N, D) -- from CTRoPE.forward()

    Returns (N, D) rotated tensor.
    """
    c = cos.squeeze(0)
    s = sin.squeeze(0)
    return (x * c) + (_rotate_half(x) * s)


# ═══════════════════════════════════════════════════════════════════════════
# §3  LTC SPIKING CELL
#     Closed-form ODE solution from Hasani et al. (AAAI 2021, arXiv:2006.04439)
#
#     LTC update:
#         v(t+dt) = v(t)-exp(-dt/tau) + A(x)-(1 - exp(-dt/tau))
#
#     Surrogate spike gradient (Zenke & Ganguli, 2021):
#         dH/dv approx= alpha / (2-(1 + alpha-|v - v_th|)-)
# ═══════════════════════════════════════════════════════════════════════════

class _SpikingLTCFunction(torch.autograd.Function):
    """
    Forward:  closed-form LTC ODE step + Heaviside spike.
    Backward: fast-sigmoid surrogate for spike path; exact grads for ODE.

    Tensors
    -------
    v_mem   (N, H) : membrane voltage at time t
    A       (N, H) : input-dependent attractor,  A = tanh(W_A x)
    inv_tau (N, H) : 1/tau(x), strictly positive
    """

    @staticmethod
    def forward(ctx, v_mem, A, inv_tau, dt, v_th, alpha):
        decay  = torch.exp(-dt * inv_tau)                 # e^{-dt/tau}
        v_pre  = v_mem * decay + A * (1.0 - decay)        # LTC update
        spikes = (v_pre >= v_th).to(v_mem.dtype)          # Heaviside
        v_next = torch.where(spikes.bool(), torch.zeros_like(v_pre), v_pre)  # hard reset

        ctx.save_for_backward(v_mem, A, inv_tau, v_pre, spikes, decay)
        ctx.dt, ctx.v_th, ctx.alpha = dt, v_th, alpha
        return spikes, v_next

    @staticmethod
    def backward(ctx, grad_spikes, grad_v_next):
        v_mem, A, inv_tau, v_pre, spikes, decay = ctx.saved_tensors
        alpha, v_th, dt = ctx.alpha, ctx.v_th, ctx.dt

        # Fast-sigmoid surrogate:  dH/dv approx= alpha / (2-(1 + alpha|v-v_th|)-)
        denom     = 1.0 + alpha * torch.abs(v_pre - v_th)
        surrogate = alpha / (2.0 * denom * denom)

        # Gradient routing: spike path uses surrogate; voltage path is zeroed at reset
        g_spike = grad_spikes * surrogate
        g_vnext = torch.where(spikes.bool(), torch.zeros_like(grad_v_next), grad_v_next)
        g_vpre  = g_spike + g_vnext                       # combined at v_pre

        # Back through LTC ODE
        grad_v_mem   = g_vpre * decay
        grad_A       = g_vpre * (1.0 - decay)
        grad_inv_tau = g_vpre * dt * decay * (A - v_mem)  # via d(decay)/d(inv_tau)

        return grad_v_mem, grad_A, grad_inv_tau, None, None, None


def spiking_ltc_step(
    v_mem: torch.Tensor, A: torch.Tensor, inv_tau: torch.Tensor,
    dt: float = 1.0, v_th: float = 1.0, alpha: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Public wrapper. Returns (spikes, v_next)."""
    return _SpikingLTCFunction.apply(v_mem, A, inv_tau, dt, v_th, alpha)


class LTCParameterNet(nn.Module):
    """
    Projects concatenated [x_hg || v_mem] into per-neuron LTC parameters:
        A(x)     = tanh(W_A x)             -- attractor in (-1, 1)
        tau(x)   = tau_min + softplus(W_tau x) -- time constant > tau_min
        inv_tau(x) = 1 / tau(x)            -- strictly positive
    """
    def __init__(self, in_features: int, out_features: int, tau_min: float = 0.1):
        super().__init__()
        self.tau_min = tau_min
        self.W_A   = nn.Linear(in_features, out_features)
        self.W_tau = nn.Linear(in_features, out_features)
        nn.init.zeros_(self.W_tau.bias)
        nn.init.xavier_uniform_(self.W_tau.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A       = torch.tanh(self.W_A(x))
        inv_tau = 1.0 / (self.tau_min + F.softplus(self.W_tau(x)))
        return A, inv_tau


# ═══════════════════════════════════════════════════════════════════════════
# §3  HYPERGRAPH CONVOLUTION
#     Spectral HGNN from Feng et al. (AAAI 2019, arXiv:1809.09401)
#
#     X' = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X Theta
#
#     Also implements AttentionHGNNConv (HyperGAT, Ding et al., EMNLP 2020)
# ═══════════════════════════════════════════════════════════════════════════

def _degree_norm(
    hyperedge_index: torch.Tensor,
    num_nodes: int, num_edges: int,
    edge_weight: Optional[torch.Tensor] = None,
    dtype=torch.float32,
    device=torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns D_v^{-1/2} and D_e^{-1} as 1-D tensors (sparse, no dense matrix)."""
    node_ids, edge_ids = hyperedge_index

    if edge_weight is None:
        edge_weight = torch.ones(num_edges, dtype=dtype, device=device)

    d_v = torch.zeros(num_nodes, dtype=dtype, device=device)
    d_v.scatter_add_(0, node_ids, edge_weight[edge_ids])
    d_v_invsqrt = d_v.clamp(min=1e-6).pow(-0.5)

    d_e = torch.zeros(num_edges, dtype=dtype, device=device)
    d_e.scatter_add_(0, edge_ids, torch.ones_like(node_ids, dtype=dtype))
    d_e_inv = d_e.clamp(min=1e-6).pow(-1.0)

    return d_v_invsqrt, d_e_inv


class HGNNConv(nn.Module):
    """
    Normalized spectral HGNN convolution with optional learnable edge weights.
    Learns log-weights per hyperedge (initialized 0 -> W=I at start).
    """
    def __init__(self, in_channels: int, out_channels: int,
                 use_bias: bool = True, learn_edge_w: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.learn_edge_w = learn_edge_w
        self.dropout      = dropout
        self.theta        = nn.Linear(in_channels, out_channels, bias=use_bias)
        nn.init.xavier_uniform_(self.theta.weight)
        self._log_edge_w: Optional[nn.Parameter] = None

    def _edge_weights(self, M: int, device, dtype) -> Optional[torch.Tensor]:
        if not self.learn_edge_w:
            return None
        if self._log_edge_w is None or self._log_edge_w.shape[0] != M:
            self._log_edge_w = nn.Parameter(torch.zeros(M, device=device, dtype=dtype))
            self.register_parameter("log_edge_w", self._log_edge_w)
        return self._log_edge_w.exp()

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None,
                ) -> torch.Tensor:
        device, dtype   = x.device, x.dtype
        node_ids, edge_ids = hyperedge_index
        N = num_nodes if num_nodes is not None else x.size(0)
        M = num_edges or int(edge_ids.max().item()) + 1

        ew = self._edge_weights(M, device, dtype)
        d_v_invsqrt, d_e_inv = _degree_norm(hyperedge_index, N, M, ew, dtype, device)

        x_n = x * d_v_invsqrt.unsqueeze(1)
        h = torch.zeros(M, x.size(1), dtype=dtype, device=device)
        h.index_add_(0, edge_ids, x_n[node_ids])
        h = h * d_e_inv.unsqueeze(1)
        if ew is not None:
            h = h * ew.unsqueeze(1)
        out = torch.zeros(N, x.size(1), dtype=dtype, device=device)
        out.index_add_(0, node_ids, h[edge_ids])
        out = out * d_v_invsqrt.unsqueeze(1)

        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout)
        return self.theta(out)


class AttentionHGNNConv(nn.Module):
    """
    HyperGAT-style multi-head attention over node-hyperedge membership.
    Replaces uniform aggregation with learned attention coefficients e_{i,e}.
    Reference: Ding et al., EMNLP 2020, arXiv:2011.00387
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_channels % num_heads == 0
        self.H, self.D = num_heads, out_channels // num_heads
        self.dropout   = dropout
        self.W_node    = nn.Linear(in_channels, out_channels, bias=False)
        self.att_src   = nn.Parameter(torch.empty(1, num_heads, self.D))
        self.att_dst   = nn.Parameter(torch.empty(1, num_heads, self.D))
        self.out_proj  = nn.Linear(out_channels, out_channels)
        nn.init.xavier_uniform_(self.W_node.weight)
        nn.init.xavier_normal_(self.att_src)
        nn.init.xavier_normal_(self.att_dst)

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None,
                ) -> torch.Tensor:
        device, dtype   = x.device, x.dtype
        node_ids, edge_ids = hyperedge_index
        N = num_nodes if num_nodes is not None else x.size(0)
        M = num_edges or int(edge_ids.max().item()) + 1
        H, D = self.H, self.D

        x_proj = self.W_node(x).view(N, H, D)

        e_feat = torch.zeros(M, H, D, dtype=dtype, device=device)
        e_feat.index_add_(0, edge_ids, x_proj[node_ids])
        counts = torch.bincount(edge_ids, minlength=M).clamp_min(1)
        e_feat = e_feat / counts.view(M, 1, 1).to(dtype)

        alpha = (x_proj[node_ids] * self.att_src).sum(-1) \
              + (e_feat[edge_ids]  * self.att_dst).sum(-1)
        alpha = F.leaky_relu(alpha, 0.2)

        a_exp = alpha.exp()
        a_sum = torch.zeros(M, H, dtype=dtype, device=device)
        a_sum.index_add_(0, edge_ids, a_exp)
        alpha_n = a_exp / (a_sum[edge_ids] + 1e-9)

        if self.dropout > 0 and self.training:
            alpha_n = F.dropout(alpha_n, p=self.dropout)

        msg   = x_proj[node_ids] * alpha_n.unsqueeze(-1)
        h_agg = torch.zeros(M, H, D, dtype=dtype, device=device)
        h_agg.index_add_(0, edge_ids, msg)
        out = torch.zeros(N, H, D, dtype=dtype, device=device)
        out.index_add_(0, node_ids, h_agg[edge_ids])

        return self.out_proj(out.view(N, H * D))


# ═══════════════════════════════════════════════════════════════════════════
# §4  SHLNN LAYER  (HGNN + LTC + Spiking, combined)
#
#     Information flow per timestep:
#       x --HGNNConv--> x_hg
#       x_hg + skip(x) --LayerNorm--> x_norm
#       [x_norm || v_{t-1}] --LTCParamNet--> A, 1/tau
#       v_{t-1}, A, 1/tau --LTCStep--> spikes, v_t
# ═══════════════════════════════════════════════════════════════════════════

class SHLNNLayer(nn.Module):
    """
    Single Spiking Hypergraph LNN layer.

    Parameters
    ----------
    in_channels  : input feature dim
    out_channels : hidden/output dim
    conv_type    : 'hgnn' | 'attention_hgnn'
    num_heads    : heads for attention variant
    v_th         : spike threshold
    alpha        : surrogate sharpness
    dt           : LTC ODE dt
    tau_min      : minimum time constant tau
    dropout      : dropout rate
    learn_edge_w : learnable per-hyperedge weight (hgnn only)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 conv_type: str = "hgnn", num_heads: int = 4,
                 v_th: float = 1.0, alpha: float = 2.0, dt: float = 1.0,
                 tau_min: float = 0.1, dropout: float = 0.1,
                 learn_edge_w: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.v_th, self.alpha, self.dt = v_th, alpha, dt

        if conv_type == "hgnn":
            self.hg_conv = HGNNConv(in_channels, out_channels,
                                    learn_edge_w=learn_edge_w, dropout=dropout)
        elif conv_type == "attention_hgnn":
            self.hg_conv = AttentionHGNNConv(in_channels, out_channels,
                                             num_heads=num_heads, dropout=dropout)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type!r}")

        self.ltc_params = LTCParameterNet(out_channels + out_channels,
                                          out_channels, tau_min=tau_min)
        self.skip = (nn.Linear(in_channels, out_channels, bias=False)
                     if in_channels != out_channels else nn.Identity())
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                v_mem: Optional[torch.Tensor] = None,
                num_nodes: Optional[int] = None,
                num_edges: Optional[int] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = x.size(0)
        if v_mem is None:
            v_mem = x.new_zeros(N, self.out_channels)

        x_hg = self.hg_conv(x, hyperedge_index, num_nodes=N, num_edges=num_edges)
        x_hg = self.norm(x_hg + self.skip(x))

        A, inv_tau = self.ltc_params(torch.cat([x_hg, v_mem], dim=-1))

        spikes, v_next = spiking_ltc_step(v_mem, A, inv_tau,
                                          dt=self.dt, v_th=self.v_th, alpha=self.alpha)
        return spikes, v_next


class SHLNNStack(nn.Module):
    """Stack of SHLNNLayer modules; carries per-layer v_mem across time."""
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int = 2, **kw):
        super().__init__()
        dims = [in_channels] + [hidden_dim] * num_layers
        self.layers     = nn.ModuleList([SHLNNLayer(dims[i], dims[i+1], **kw)
                                         for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                v_mems: Optional[List] = None,
                num_nodes: Optional[int] = None,
                num_edges: Optional[int] = None,
                ) -> Tuple[torch.Tensor, List]:
        if v_mems is None:
            v_mems = [None] * self.num_layers
        v_next_list, h = [], x
        for layer, v in zip(self.layers, v_mems):
            h, vn = layer(h, hyperedge_index, v_mem=v,
                          num_nodes=num_nodes, num_edges=num_edges)
            v_next_list.append(vn)
        return h, v_next_list


# ═══════════════════════════════════════════════════════════════════════════
# §5  FULL LANGUAGE MODEL
# ═══════════════════════════════════════════════════════════════════════════

class SpikingHypergraphLM(nn.Module):
    """
    Language model: Embedding -> SHLNNStack -> LM Head (weight-tied when dims match).

    forward()          -- single graph-snapshot step (one sentence)
    forward_sequence() -- temporal unrolling over a list of snapshots
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2,
                 conv_type: str = "hgnn", num_heads: int = 4,
                 v_th: float = 1.0, alpha: float = 2.0, dt: float = 1.0,
                 tau_min: float = 0.1, dropout: float = 0.1,
                 learn_edge_w: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding  = nn.Embedding(vocab_size, embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.shlnn      = SHLNNStack(embed_dim, hidden_dim, num_layers,
                                     conv_type=conv_type, num_heads=num_heads,
                                     v_th=v_th, alpha=alpha, dt=dt,
                                     tau_min=tau_min, dropout=dropout,
                                     learn_edge_w=learn_edge_w)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if embed_dim == hidden_dim:
            self.lm_head.weight = self.embedding.weight          # weight tying
        nn.init.normal_(self.embedding.weight, 0.0, 0.02)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, node_indices: torch.Tensor, hyperedge_index: torch.Tensor,
                v_mems: Optional[List] = None,
                num_nodes: Optional[int] = None,
                num_edges: Optional[int] = None,
                ) -> Tuple[torch.Tensor, List, torch.Tensor]:
        x = self.embed_drop(self.embedding(node_indices))
        spikes, v_mems_next = self.shlnn(x, hyperedge_index, v_mems=v_mems,
                                         num_nodes=num_nodes, num_edges=num_edges)
        return self.lm_head(spikes), v_mems_next, spikes

    def forward_sequence(self, token_ids_seq, hyperedge_index_seq,
                         v_mems=None):
        all_logits, all_spikes = [], []
        for tokens, hei in zip(token_ids_seq, hyperedge_index_seq):
            logits, v_mems, spikes = self.forward(tokens, hei, v_mems=v_mems)
            all_logits.append(logits)
            all_spikes.append(spikes)
            v_mems = [v.detach() for v in v_mems]
        return all_logits, v_mems, all_spikes


# ═══════════════════════════════════════════════════════════════════════════
# §6  HYPERGRAPH CONSTRUCTION
#     Strategy: n-gram sliding window + global sentence hyperedge
# ═══════════════════════════════════════════════════════════════════════════

def build_hypergraph(seq_len: int, window_size: int = 3,
                     add_global: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Build hyperedge_index (2, E_entries) for a token sequence of length seq_len.

    Hyperedge types:
        1. Sliding-window n-grams (width=window_size): local syntactic context
        2. Global sentence hyperedge:                  document-level context

    Returns (hyperedge_index, num_hyperedges).
    """
    nids, eids, eid = [], [], 0
    for start in range(seq_len - window_size + 1):
        for off in range(window_size):
            nids.append(start + off); eids.append(eid)
        eid += 1
    if add_global and seq_len > window_size:
        for i in range(seq_len):
            nids.append(i); eids.append(eid)
        eid += 1
    return torch.tensor([nids, eids], dtype=torch.long), eid


# ═══════════════════════════════════════════════════════════════════════════
# §7  DATA: Penn Treebank loader + mock generator
# ═══════════════════════════════════════════════════════════════════════════

class Vocabulary:
    PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"

    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str]      = []

    def build(self, tokens: List[str]) -> None:
        special  = [self.PAD, self.UNK, self.EOS]
        vocab    = special + sorted(set(tokens) - set(special))
        self.idx2word = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}

    def encode(self, t: str) -> int:
        return self.word2idx.get(t, self.word2idx[self.UNK])

    def __len__(self) -> int:
        return len(self.idx2word)


class PTBDataset(Dataset):
    """
    Penn Treebank sentence-level dataset.
    Each sample: (token_ids, hyperedge_index, num_edges).

    NOTE: the hypergraph is built over token_ids[:-1] (the model inputs),
    so that node count matches x = embedding(token_ids[:-1]) at train time.

    Download PTB from: https://github.com/wojzaremba/lstm/tree/master/data
    """
    def __init__(self, path: str, vocab: Optional[Vocabulary] = None,
                 max_seq_len: int = 70, window_size: int = 3,
                 add_global: bool = True):
        sentences = self._load(path)
        if vocab is None:
            vocab = Vocabulary()
            vocab.build([t for s in sentences for t in s])
        self.vocab   = vocab
        self.samples = []
        for sent in sentences:
            sent = sent[:max_seq_len]
            if len(sent) < 2:
                continue
            ids = torch.tensor([vocab.encode(t) for t in sent], dtype=torch.long)
            input_len = len(ids) - 1
            hg, ne = build_hypergraph(input_len, window_size, add_global)
            self.samples.append((ids, hg, ne))

    @staticmethod
    def _load(path: str) -> List[List[str]]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"PTB not found at {path}\n"
                "Get it from: https://github.com/wojzaremba/lstm/tree/master/data\n"
                "Files: ptb.train.txt  ptb.valid.txt  ptb.test.txt"
            )
        out = []
        with open(path) as f:
            for line in f:
                toks = line.strip().split()
                if toks:
                    out.append(toks + [Vocabulary.EOS])
        return out

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def _collate(batch):
    return batch


def get_ptb_loaders(data_dir: str, window_size: int = 3,
                    add_global: bool = True, max_seq_len: int = 70):
    """Returns (train_loader, valid_loader, test_loader, vocab)."""
    train_ds = PTBDataset(os.path.join(data_dir, "ptb.train.txt"),
                          vocab=None, max_seq_len=max_seq_len,
                          window_size=window_size, add_global=add_global)
    vocab    = train_ds.vocab
    valid_ds = PTBDataset(os.path.join(data_dir, "ptb.valid.txt"),
                          vocab=vocab, max_seq_len=max_seq_len,
                          window_size=window_size, add_global=add_global)
    test_ds  = PTBDataset(os.path.join(data_dir, "ptb.test.txt"),
                          vocab=vocab, max_seq_len=max_seq_len,
                          window_size=window_size, add_global=add_global)
    mk = lambda ds, sh: DataLoader(ds, batch_size=1, shuffle=sh, collate_fn=_collate)
    return mk(train_ds, True), mk(valid_ds, False), mk(test_ds, False), vocab


def get_mock_loader(vocab_size: int = 10_000, num_samples: int = 300,
                    seq_len: int = 20, window_size: int = 3):
    """Synthetic loader -- no PTB files required. Returns (loader, vocab_size)."""
    samples = []
    for _ in range(num_samples):
        L   = int(torch.randint(seq_len // 2, seq_len + 1, (1,)).item())
        ids = torch.randint(0, vocab_size, (L,))
        hg, ne = build_hypergraph(L - 1, window_size)
        samples.append((ids, hg, ne))
    return DataLoader(samples, batch_size=1, shuffle=True, collate_fn=_collate), vocab_size


# ═══════════════════════════════════════════════════════════════════════════
# §7b  STREAMING DATASET  (FineWeb-Edu + Wikipedia + Code-Python)
# ═══════════════════════════════════════════════════════════════════════════

TEXT_CHUNK_BYTES = 32_768

SOURCE_CONFIGS = [
    {
        "name": "fineweb-edu",
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_config": None,
        "data_dir": None,
        "text_field": "text",
        "shuffle_buffer": 200,
    },
    {
        "name": "wikipedia",
        "hf_path": "wikimedia/wikipedia",
        "hf_config": "20231101.en",
        "data_dir": None,
        "text_field": "text",
        "shuffle_buffer": 200,
    },
    {
        "name": "code-python",
        "hf_path": "bigcode/starcoderdata",
        "hf_config": None,
        "data_dir": "python",
        "text_field": "content",
        "shuffle_buffer": 200,
    },
]

SOURCE_MAP = {s["name"]: s for s in SOURCE_CONFIGS}


if _DATASETS_AVAILABLE and _TIKTOKEN_AVAILABLE:

    class StreamingLMDataset(IterableDataset):
        """
        Multi-source streaming from HuggingFace Datasets.
        Interleaves FineWeb-Edu, Wikipedia, and StarCoder (Python).
        Tokenizes on the fly with tiktoken BPE; builds hypergraphs per chunk.

        Sources (from SOURCE_CONFIGS):
            - fineweb-edu : HuggingFaceFW/fineweb-edu  (field: text)
            - wikipedia   : wikimedia/wikipedia/20231101.en  (field: text)
            - code-python : bigcode/starcoderdata/python  (field: content)

        Each source maintains a shuffle buffer of `shuffle_buffer` items.
        Text is truncated to TEXT_CHUNK_BYTES before tokenization.
        """

        def __init__(
            self,
            source_names: Optional[List[str]] = None,
            tokenizer_name: str = "gpt2",
            max_seq_len: int = 512,
            window_size: int = 3,
            add_global: bool = True,
            max_samples: Optional[int] = None,
        ):
            super().__init__()
            self.max_seq_len = max_seq_len
            self.window_size = window_size
            self.add_global = add_global
            self.max_samples = max_samples
            self.source_names = source_names or [s["name"] for s in SOURCE_CONFIGS]

            self.enc = tiktoken.get_encoding(tokenizer_name)
            self._vocab_size = self.enc.n_vocab

            ds_list = []
            for name in self.source_names:
                spec = SOURCE_MAP[name]
                load_kw = dict(
                    path=spec["hf_path"],
                    split="train",
                    streaming=True,
                )
                if spec["hf_config"] is not None:
                    load_kw["name"] = spec["hf_config"]
                if spec["data_dir"] is not None:
                    load_kw["data_dir"] = spec["data_dir"]
                ds = hf_datasets.load_dataset(**load_kw)
                ds = ds.shuffle(buffer_size=spec["shuffle_buffer"], seed=42)
                ds_list.append(ds)

            if len(ds_list) > 1:
                self.dataset = hf_datasets.interleave_datasets(
                    ds_list,
                    probabilities=[1.0 / len(ds_list)] * len(ds_list),
                    seed=42,
                    stopping_strategy="all_exhausted",
                )
            else:
                self.dataset = ds_list[0]

        @property
        def vocab_size(self) -> int:
            return self._vocab_size

        def _chunks(self, text: str):
            tokens = self.enc.encode(text[:TEXT_CHUNK_BYTES])
            for start in range(0, len(tokens), self.max_seq_len):
                chunk = tokens[start : start + self.max_seq_len]
                if len(chunk) >= 3:
                    hg, ne = build_hypergraph(
                        len(chunk) - 1, self.window_size, self.add_global
                    )
                    yield torch.tensor(chunk, dtype=torch.long), hg, ne

        def __iter__(self):
            count = 0
            for example in self.dataset:
                text = None
                for name in self.source_names:
                    text = example.get(SOURCE_MAP[name]["text_field"])
                    if text:
                        break
                if not text:
                    continue
                for item in self._chunks(text):
                    yield item
                    count += 1
                    if self.max_samples and count >= self.max_samples:
                        return

else:
    class StreamingLMDataset:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("pip install datasets tiktoken")


def get_streaming_loaders(
    source_names=None,
    max_seq_len=512,
    window_size=3,
    add_global=True,
    max_train_samples=None,
    max_val_samples=500,
):
    """Returns (train_loader, valid_loader, test_loader, vocab_size)."""
    train_ds = StreamingLMDataset(
        source_names=source_names,
        max_seq_len=max_seq_len,
        window_size=window_size,
        add_global=add_global,
        max_samples=max_train_samples,
    )
    val_ds = StreamingLMDataset(
        source_names=source_names,
        max_seq_len=max_seq_len,
        window_size=window_size,
        add_global=add_global,
        max_samples=max_val_samples,
    )
    mk = lambda ds: DataLoader(ds, batch_size=1, collate_fn=_collate)
    return mk(train_ds), mk(val_ds), mk(val_ds), train_ds.vocab_size


# ═══════════════════════════════════════════════════════════════════════════
# §8  TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class SOPCounter:
    """
    Synaptic Operation counter for energy-efficiency analysis.
    SOPs = sum_t sum_l spike_count_{t,l} - fan_out_l
    Reference: Yao et al. (NeurIPS 2023) "Spike-driven Transformer"
    """
    def __init__(self, fan_outs: List[int]):
        self.fan_outs  = fan_outs
        self.total_sop = 0
        self.total_ann = 0
        self.n_samples = 0

    def update(self, spikes: torch.Tensor, layer_idx: int, n_tokens: int):
        fan = self.fan_outs[min(layer_idx, len(self.fan_outs) - 1)]
        self.total_sop += int(spikes.sum().item()) * fan
        self.total_ann += n_tokens * spikes.shape[-1] * fan
        self.n_samples += n_tokens

    def report(self) -> Dict:
        if self.total_ann == 0:
            return {}
        return {
            "total_sops":    self.total_sop,
            "total_ann_ops": self.total_ann,
            "energy_ratio":  self.total_sop / self.total_ann,
            "avg_spike_rate": self.total_sop / max(self.n_samples, 1),
        }


def spike_rate_loss(spikes: torch.Tensor,
                    target_rate: float = 0.1, weight: float = 0.01) -> torch.Tensor:
    """
    L2 penalty on deviation from target spike rate.
    Prevents silent (rate->0) or saturated (rate->1) regimes.
    """
    return weight * (spikes.mean() - target_rate).pow(2)


@torch.no_grad()
def evaluate_ppl(model: SpikingHypergraphLM, loader,
                 device: torch.device, criterion: nn.CrossEntropyLoss) -> float:
    """Word-level perplexity: PPL = exp(mean NLL per token)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        for (token_ids, hei, ne) in batch:
            if len(token_ids) < 2:
                continue
            token_ids, hei = token_ids.to(device), hei.to(device)
            logits, _, _   = model(token_ids[:-1], hei, num_edges=ne)
            loss           = criterion(logits, token_ids[1:])
            n = token_ids[1:].size(0)
            total_loss   += loss.item() * n
            total_tokens += n
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


EVAL_PROMPTS = [
    "Photosynthesis is the process by which",
    "The Eiffel Tower is located in",
    "def fibonacci(n):",
    "Machine learning is a subset of",
    "The first law of thermodynamics states that",
]


@torch.no_grad()
def generate_text(
    model: SpikingHypergraphLM,
    prompt: str,
    enc,
    max_new_tokens: int = 15,
    window_size: int = 3,
    add_global: bool = True,
    device=None,
) -> str:
    """
    Greedy text generation for SHLNN.
    Rebuilds the hypergraph at each step (full-sequence recompute).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if len(ids) < 2:
            break
        hei, ne = build_hypergraph(len(ids) - 1, window_size, add_global)
        hei = hei.to(device)
        logits, _, _ = model(ids[:-1], hei, num_edges=ne)
        next_id = logits[-1].argmax().unsqueeze(0)
        ids = torch.cat([ids, next_id])

    return enc.decode(ids[len(enc.encode(prompt)):].tolist())


def evaluate_prompts(
    model: SpikingHypergraphLM,
    enc,
    prompts: List[str] = None,
    max_new_tokens: int = 15,
) -> List[Tuple[str, str]]:
    """Run generation on a list of prompts. Returns (prompt, completion) pairs."""
    if prompts is None:
        prompts = EVAL_PROMPTS
    results = []
    for prompt in prompts:
        completion = generate_text(model, prompt, enc, max_new_tokens=max_new_tokens)
        results.append((prompt, completion))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# §9  TRAINING CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # ── data ──
    data_dir:        str   = "./data/ptb"
    mock:            bool  = True
    mock_vocab:      int   = 10_000
    mock_samples:    int   = 300
    max_seq_len:     int   = 70
    window_size:     int   = 3
    add_global_edge: bool  = True
    # ── model ──
    embed_dim:       int   = 128
    hidden_dim:      int   = 256
    num_layers:      int   = 2
    conv_type:       str   = "hgnn"
    num_heads:       int   = 4
    v_th:            float = 0.5
    alpha:           float = 2.0
    dt:              float = 1.0
    tau_min:         float = 0.1
    dropout:         float = 0.1
    learn_edge_w:    bool  = False
    # ── training ──
    epochs:           int   = 20
    lr:               float = 1e-3
    weight_decay:     float = 1e-4
    grad_clip:        float = 1.0
    target_spike_rate: float = 0.1
    spike_loss_weight: float = 0.01
    eval_prompts:     bool  = True
    seed:             int   = 42
    save_dir:         str   = "./checkpoints"


# ═══════════════════════════════════════════════════════════════════════════
# §10  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train(cfg: TrainConfig,
          override_loaders: Optional[Tuple] = None,
          override_vocab_size: Optional[int] = None) -> Dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*64}")
    print(f"  Device : {device}  |  Seed : {cfg.seed}")

    # ── data ──
    if override_loaders is not None:
        train_loader, valid_loader, test_loader = override_loaders
        vocab_size = override_vocab_size
    elif cfg.mock:
        train_loader, vocab_size = get_mock_loader(
            cfg.mock_vocab, cfg.mock_samples, cfg.max_seq_len, cfg.window_size)
        valid_loader = test_loader = train_loader
    else:
        train_loader, valid_loader, test_loader, vocab = get_ptb_loaders(
            cfg.data_dir, cfg.window_size, cfg.add_global_edge, cfg.max_seq_len)
        vocab_size = len(vocab)

    print(f"  Vocab  : {vocab_size:,}  |  Mode : {'mock' if cfg.mock else 'PTB+' if override_loaders else 'PTB'}")

    # ── model ──
    model = SpikingHypergraphLM(
        vocab_size=vocab_size, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, conv_type=cfg.conv_type, num_heads=cfg.num_heads,
        v_th=cfg.v_th, alpha=cfg.alpha, dt=cfg.dt, tau_min=cfg.tau_min,
        dropout=cfg.dropout, learn_edge_w=cfg.learn_edge_w,
    ).to(device)
    print(f"  Params : {model.num_parameters:,}")
    print(f"{'='*64}")

    # ── optimiser & scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    sop = SOPCounter([cfg.hidden_dim] * cfg.num_layers)
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_val_ppl, history = float("inf"), []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss, ep_tokens, ep_spike_rates = 0.0, 0, []
        t0 = time.time()

        for batch in train_loader:
            for (token_ids, hei, ne) in batch:
                if len(token_ids) < 2:
                    continue
                token_ids, hei = token_ids.to(device), hei.to(device)
                optimizer.zero_grad(set_to_none=True)

                logits, v_mems, spikes = model(token_ids[:-1], hei, num_edges=ne)
                targets = token_ids[1:]

                lm_loss = criterion(logits, targets)
                sp_loss = spike_rate_loss(spikes, cfg.target_spike_rate,
                                          cfg.spike_loss_weight)
                (lm_loss + sp_loss).backward()

                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

                n = targets.size(0)
                ep_loss        += lm_loss.item() * n
                ep_tokens      += n
                ep_spike_rates.append(spikes.detach().mean().item())
                sop.update(spikes.detach(), 0, n)

        scheduler.step()

        train_ppl  = math.exp(ep_loss / max(ep_tokens, 1))
        val_ppl    = evaluate_ppl(model, valid_loader, device, criterion)
        spike_rate = float(np.mean(ep_spike_rates))
        elapsed    = time.time() - t0

        rec = dict(epoch=epoch, train_ppl=round(train_ppl, 3),
                   val_ppl=round(val_ppl, 3), spike_rate=round(spike_rate, 4),
                   lr=round(scheduler.get_last_lr()[0], 6), elapsed_s=round(elapsed, 1))
        history.append(rec)

        print(f"Ep {epoch:03d} | "
              f"Train PPL {train_ppl:8.2f} | "
              f"Val PPL {val_ppl:8.2f} | "
              f"Spike {spike_rate:.3f} | "
              f"LR {scheduler.get_last_lr()[0]:.5f} | "
              f"{elapsed:.1f}s")

        # ── prompt evaluation ──
        if cfg.eval_prompts:
            try:
                enc = tiktoken.get_encoding("gpt2")
                results = evaluate_prompts(model, enc, max_new_tokens=12)
                for prompt, comp in results:
                    suffix = comp.replace("\n", "\\n")[:60]
                    print(f"        | {prompt} -> {suffix}")
            except Exception:
                pass  # silent if tiktoken unavailable mid-run

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({"epoch": epoch, "val_ppl": val_ppl,
                        "model": model.state_dict(), "cfg": asdict(cfg)},
                       os.path.join(cfg.save_dir, "best_model.pt"))
            print(f"        v checkpoint  (val_ppl={val_ppl:.2f})")

    # ── final test eval ──
    ckpt = torch.load(os.path.join(cfg.save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    test_ppl   = evaluate_ppl(model, test_loader, device, criterion)
    sop_report = sop.report()

    print(f"\n{'='*64}")
    print(f"  Best Val PPL  : {best_val_ppl:.2f}")
    print(f"  Test PPL      : {test_ppl:.2f}")
    if sop_report:
        print(f"  Energy Ratio  : {sop_report['energy_ratio']:.4f}  (SNN vs ANN)")
        print(f"  Avg Spike Rate: {sop_report['avg_spike_rate']:.4f}")
    print(f"{'='*64}")

    summary = dict(test_ppl=test_ppl, best_val_ppl=best_val_ppl,
                   sop_report=sop_report, history=history, cfg=asdict(cfg))
    with open(os.path.join(cfg.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# §11  BASELINES  (LSTM + vanilla SNN LM)
# ═══════════════════════════════════════════════════════════════════════════

class LSTMLMBaseline(nn.Module):
    """
    Standard 2-layer LSTM LM (Zaremba et al., 2014, arXiv:1409.2329).
    Used as a classical recurrent baseline in the comparison table.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.drop      = nn.Dropout(dropout)
        self.lm_head   = nn.Linear(hidden_dim, vocab_size)
        self.lm_head.weight = self.embedding.weight

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, token_ids, hidden=None):
        x   = self.drop(self.embedding(token_ids.unsqueeze(0)))
        out, hidden = self.lstm(x, hidden)
        return self.lm_head(self.drop(out.squeeze(0))), hidden


class _LIFCell(nn.Module):
    """Leaky Integrate-and-Fire neuron (simple SNN baseline)."""
    def __init__(self, in_dim, out_dim, v_th=1.0, beta=0.9):
        super().__init__()
        self.proj, self.v_th, self.beta = nn.Linear(in_dim, out_dim), v_th, beta

    def forward(self, x, v=None):
        if v is None:
            v = x.new_zeros(x.size(0), self.proj.out_features)
        v_pre  = self.beta * v + self.proj(x)
        spikes = (v_pre >= self.v_th).float()
        v_next = torch.where(spikes.bool(), torch.zeros_like(v_pre), v_pre)
        return spikes, v_next


class SNNLMBaseline(nn.Module):
    """Spiking LM: LIF cells only, no hypergraph, no LTC."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        dims = [embed_dim] + [hidden_dim] * num_layers
        self.cells   = nn.ModuleList([_LIFCell(dims[i], dims[i+1])
                                      for i in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.lm_head.weight = self.embedding.weight

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, token_ids, v_mems=None):
        h = self.embedding(token_ids)
        if v_mems is None:
            v_mems = [None] * len(self.cells)
        vnext = []
        for cell, v in zip(self.cells, v_mems):
            h, vn = cell(h, v); vnext.append(vn)
        return self.lm_head(h), vnext


@torch.no_grad()
def _eval_baseline(model, loader, device, criterion, forward_fn):
    model.eval()
    tl, tt = 0.0, 0
    for batch in loader:
        for (ids, _, _) in batch:
            if len(ids) < 2: continue
            ids     = ids.to(device)
            logits, _ = forward_fn(model, ids[:-1])
            loss    = criterion(logits, ids[1:])
            n       = ids[1:].size(0)
            tl += loss.item() * n; tt += n
    return math.exp(tl / max(tt, 1))


def train_baseline(model: nn.Module, loader, device: torch.device,
                   epochs: int = 10, lr: float = 1e-3,
                   forward_fn=None) -> List[float]:
    """Generic training loop for LSTM / SNN baselines. Returns per-epoch PPL."""
    if forward_fn is None:
        forward_fn = lambda m, x: m(x)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ppls = []
    for epoch in range(1, epochs + 1):
        model.train()
        tl, tt = 0.0, 0
        for batch in loader:
            for (ids, _, _) in batch:
                if len(ids) < 2: continue
                ids = ids.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = forward_fn(model, ids[:-1])
                loss = criterion(logits, ids[1:])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                n = ids[1:].size(0); tl += loss.item() * n; tt += n
        ppl = math.exp(tl / max(tt, 1))
        ppls.append(ppl)
        print(f"  [{model.__class__.__name__}] Ep {epoch:02d} PPL {ppl:.2f}")
    return ppls


# ═══════════════════════════════════════════════════════════════════════════
# §12  QUICK-START FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_mock(epochs: int = 20, hidden_dim: int = 256, num_layers: int = 2,
             conv_type: str = "hgnn", seed: int = 42) -> Dict:
    """
    Fastest way to test the full pipeline -- no downloads needed.

    >>> summary = run_mock(epochs=20)
    """
    cfg = TrainConfig(
        mock=True, mock_vocab=10_000, mock_samples=300,
        embed_dim=128, hidden_dim=hidden_dim, num_layers=num_layers,
        conv_type=conv_type, epochs=epochs, lr=1e-3, seed=seed,
    )
    return train(cfg)


def run_ptb(data_dir: str = "./data/ptb", epochs: int = 40,
            embed_dim: int = 200, hidden_dim: int = 400,
            num_layers: int = 2, conv_type: str = "hgnn",
            seed: int = 42) -> Dict:
    """
    Full PTB training run.

    Download PTB first:
        wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt -P ./data/ptb/
        wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt -P ./data/ptb/
        wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt  -P ./data/ptb/

    >>> summary = run_ptb(data_dir="./data/ptb", epochs=40)
    """
    cfg = TrainConfig(
        mock=False, data_dir=data_dir, embed_dim=embed_dim,
        hidden_dim=hidden_dim, num_layers=num_layers,
        conv_type=conv_type, epochs=epochs, seed=seed,
    )
    return train(cfg)


def run_stream(
    source_names: Optional[List[str]] = None,
    max_seq_len: int = 512,
    embed_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 2,
    conv_type: str = "hgnn",
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 42,
    max_train_samples: Optional[int] = 5000,
) -> Dict:
    """
    Train SHLNN on real streaming web data using SOURCE_CONFIGS.

    Sources (default all three):
        - fineweb-edu : HuggingFaceFW/fineweb-edu
        - wikipedia   : wikimedia/wikipedia (20231101.en)
        - code-python : bigcode/starcoderdata (python)

    >>> summary = run_stream(epochs=5, max_train_samples=2000)
    """
    if not _TIKTOKEN_AVAILABLE:
        raise ImportError("tiktoken required -- run: pip install tiktoken")
    if not _DATASETS_AVAILABLE:
        raise ImportError("datasets required -- run: pip install datasets")

    source_names = source_names or [s["name"] for s in SOURCE_CONFIGS]

    loaders = get_streaming_loaders(
        source_names=source_names,
        max_seq_len=max_seq_len,
        max_train_samples=max_train_samples,
    )
    train_loader, valid_loader, test_loader, vocab_size = loaders

    cfg = TrainConfig(
        mock=True,
        mock_vocab=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        conv_type=conv_type,
        epochs=epochs,
        lr=lr,
        v_th=0.5,
        seed=seed,
        max_seq_len=max_seq_len,
    )
    return train(cfg, override_loaders=(train_loader, valid_loader, test_loader),
                 override_vocab_size=vocab_size)


def run_comparison(epochs: int = 10, seed: int = 42) -> None:
    """
    Train SHLNN + LSTM + SNN baselines on mock data and print a comparison table.

    >>> run_comparison(epochs=10)
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, vocab_size = get_mock_loader(vocab_size=5_000, num_samples=200)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    results = {}

    print("\n-- SHLNN ----")
    s = run_mock(epochs=epochs, seed=seed)
    results["SHLNN (ours)"] = s["test_ppl"]

    print("\n-- LSTM baseline ----")
    lstm = LSTMLMBaseline(vocab_size, 128, 256).to(device)
    train_baseline(lstm, loader, device, epochs=epochs,
                   forward_fn=lambda m, x: m(x))
    results["LSTM (Zaremba 2014)"] = _eval_baseline(
        lstm, loader, device, criterion, lambda m, x: m(x))

    print("\n-- SNN (LIF, no graph) ----")
    snn = SNNLMBaseline(vocab_size, 128, 256).to(device)
    train_baseline(snn, loader, device, epochs=epochs,
                   forward_fn=lambda m, x: m(x))
    results["SNN (LIF, no HG)"] = _eval_baseline(
        snn, loader, device, criterion, lambda m, x: m(x))

    print(f"\n{'='*50}")
    print(f"{'Model':<28} {'Test PPL':>10}")
    print(f"{'-'*50}")
    for name, ppl in results.items():
        marker = " <--" if name.startswith("SHLNN") else ""
        print(f"{name:<28} {ppl:>10.2f}{marker}")
    print(f"{'='*50}")


# ═══════════════════════════════════════════════════════════════════════════
# §13  SCRIPT ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Default: streaming from FineWeb-Edu + Wikipedia + Python code
    # For quick test without downloads:  run_mock(epochs=20)
    summary = run_stream(epochs=5, max_train_samples=2000)
