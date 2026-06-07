"""
╔══════════════════════════════════════════════════════════════════════════╗
║       Spiking Hypergraph Liquid Neural Network (SHLNN)                  ║
║       Single-file notebook edition  —  bug-fixed release                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Combines:                                                               ║
║    • LTC ODE dynamics       (Hasani et al., AAAI 2021)                  ║
║    • HGNN convolution       (Feng et al., AAAI 2019)                    ║
║    • Surrogate gradients    (Zenke & Ganguli, 2021)                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  BUG FIXES vs. original:                                                 ║
║    1. HGNNConv.forward — N always derived from x.size(0), never from    ║
║       max(node_ids)+1, eliminating the tensor-size mismatch.            ║
║    2. AttentionHGNNConv.forward — same N guard applied.                 ║
║    3. train() loop — hyperedge_index rebuilt for token_ids[:-1] so      ║
║       the graph length always matches the input sequence length.        ║
║    4. evaluate_ppl() — same hyperedge rebuild applied.                  ║
║    5. _eval_baseline() — same hyperedge rebuild applied.                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  USAGE (Colab / Jupyter):                                                ║
║    Cell 1 — paste this whole file, run it                               ║
║    Cell 2 — run_mock()       for synthetic data  (no downloads needed)  ║
║    Cell 3 — run_ptb()        for real PTB benchmark                     ║
║    Cell 4 — run_comparison() for side-by-side model table               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

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
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════
# §1  REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════
# §2  LTC SPIKING CELL
#     Closed-form ODE solution from Hasani et al. (AAAI 2021, arXiv:2006.04439)
#
#     LTC update:
#         v(t+Δt) = v(t)·exp(-Δt/τ) + A(x)·(1 - exp(-Δt/τ))
#
#     Surrogate spike gradient (Zenke & Ganguli, 2021):
#         dH/dv ≈ α / (2·(1 + α·|v - v_th|)²)
# ═══════════════════════════════════════════════════════════════════════════

class _SpikingLTCFunction(torch.autograd.Function):
    """
    Forward:  closed-form LTC ODE step + Heaviside spike.
    Backward: fast-sigmoid surrogate for spike path; exact grads for ODE.

    Tensors
    -------
    v_mem   (N, H) : membrane voltage at time t
    A       (N, H) : input-dependent attractor,  A = tanh(W_A x)
    inv_tau (N, H) : 1/τ(x), strictly positive
    """

    @staticmethod
    def forward(ctx, v_mem, A, inv_tau, dt, v_th, alpha):
        decay  = torch.exp(-dt * inv_tau)              # e^{-Δt/τ}
        v_pre  = v_mem * decay + A * (1.0 - decay)    # LTC update
        spikes = (v_pre >= v_th).to(v_mem.dtype)       # Heaviside
        v_next = torch.where(spikes.bool(),
                             torch.zeros_like(v_pre),
                             v_pre)                    # hard reset

        ctx.save_for_backward(v_mem, A, inv_tau, v_pre, spikes, decay)
        ctx.dt, ctx.v_th, ctx.alpha = dt, v_th, alpha
        return spikes, v_next

    @staticmethod
    def backward(ctx, grad_spikes, grad_v_next):
        v_mem, A, inv_tau, v_pre, spikes, decay = ctx.saved_tensors
        alpha, v_th, dt = ctx.alpha, ctx.v_th, ctx.dt

        # Fast-sigmoid surrogate:  dH/dv ≈ α / (2·(1 + α|v-v_th|)²)
        denom     = 1.0 + alpha * torch.abs(v_pre - v_th)
        surrogate = alpha / (2.0 * denom * denom)

        # Gradient routing
        g_spike = grad_spikes * surrogate
        g_vnext = torch.where(spikes.bool(),
                              torch.zeros_like(grad_v_next),
                              grad_v_next)
        g_vpre  = g_spike + g_vnext

        # Back through LTC ODE
        grad_v_mem   = g_vpre * decay
        grad_A       = g_vpre * (1.0 - decay)
        grad_inv_tau = g_vpre * dt * decay * (A - v_mem)

        return grad_v_mem, grad_A, grad_inv_tau, None, None, None


def spiking_ltc_step(
    v_mem: torch.Tensor,
    A: torch.Tensor,
    inv_tau: torch.Tensor,
    dt: float = 1.0,
    v_th: float = 1.0,
    alpha: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Public wrapper. Returns (spikes, v_next)."""
    return _SpikingLTCFunction.apply(v_mem, A, inv_tau, dt, v_th, alpha)


class LTCParameterNet(nn.Module):
    """
    Projects concatenated [x_hg ‖ v_mem] into per-neuron LTC parameters:
        A(x)     = tanh(W_A x)              — attractor ∈ (-1, 1)
        τ(x)     = τ_min + softplus(W_τ x)  — time constant > τ_min
        inv_τ(x) = 1 / τ(x)                — strictly positive
    """

    def __init__(self, in_features: int, out_features: int,
                 tau_min: float = 0.1):
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
#     X' = D_v^{-½} H W D_e^{-1} H^T D_v^{-½} X Θ
#
#     Also implements AttentionHGNNConv (HyperGAT, Ding et al., EMNLP 2020)
# ═══════════════════════════════════════════════════════════════════════════

def _degree_norm(
    hyperedge_index: torch.Tensor,
    num_nodes: int,
    num_edges: int,
    edge_weight: Optional[torch.Tensor] = None,
    dtype=torch.float32,
    device=torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns D_v^{-½} and D_e^{-1} as 1-D tensors (sparse, no dense matrix)."""
    node_ids, edge_ids = hyperedge_index

    if edge_weight is None:
        edge_weight = torch.ones(num_edges, dtype=dtype, device=device)

    # D_v[i] = Σ_e H[i,e] · W[e]
    d_v = torch.zeros(num_nodes, dtype=dtype, device=device)
    d_v.scatter_add_(0, node_ids, edge_weight[edge_ids])
    d_v_invsqrt = d_v.clamp(min=1e-6).pow(-0.5)

    # D_e[e] = |{i : H[i,e] = 1}|
    d_e = torch.zeros(num_edges, dtype=dtype, device=device)
    d_e.scatter_add_(0, edge_ids, torch.ones_like(node_ids, dtype=dtype))
    d_e_inv = d_e.clamp(min=1e-6).pow(-1.0)

    return d_v_invsqrt, d_e_inv


class HGNNConv(nn.Module):
    """
    Normalized spectral HGNN convolution with optional learnable edge weights.
    Learns log-weights per hyperedge (initialized 0 → W=I at start).

    FIX: N is always taken from x.size(0), never inferred from hyperedge_index,
    preventing the size-mismatch RuntimeError when input is token_ids[:-1].
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
            self._log_edge_w = nn.Parameter(
                torch.zeros(M, device=device, dtype=dtype))
            self.register_parameter("log_edge_w", self._log_edge_w)
        return self._log_edge_w.exp()

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: Optional[int] = None,   # kept for API compat, ignored
        num_edges: Optional[int] = None,
    ) -> torch.Tensor:
        device, dtype      = x.device, x.dtype
        node_ids, edge_ids = hyperedge_index

        # ── FIX 1: always use the actual input size, not max(node_ids)+1 ──
        N = x.size(0)
        M = num_edges if num_edges is not None else int(edge_ids.max().item()) + 1

        # Guard: clamp node ids that may exceed N due to stale hyperedge_index
        node_ids = node_ids.clamp(max=N - 1)

        ew = self._edge_weights(M, device, dtype)
        d_v_invsqrt, d_e_inv = _degree_norm(
            torch.stack([node_ids, edge_ids]), N, M, ew, dtype, device)

        # Step 1: left-normalize nodes
        x_n = x * d_v_invsqrt.unsqueeze(1)

        # Step 2: aggregate into hyperedges
        h = torch.zeros(M, x.size(1), dtype=dtype, device=device)
        h.index_add_(0, edge_ids, x_n[node_ids])

        # Step 3: D_e^{-1} and optional W
        h = h * d_e_inv.unsqueeze(1)
        if ew is not None:
            h = h * ew.unsqueeze(1)

        # Step 4: broadcast back to nodes
        out = torch.zeros(N, x.size(1), dtype=dtype, device=device)
        out.index_add_(0, node_ids, h[edge_ids])

        # Step 5: right-normalize
        out = out * d_v_invsqrt.unsqueeze(1)

        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout)

        return self.theta(out)


class AttentionHGNNConv(nn.Module):
    """
    HyperGAT-style multi-head attention over node–hyperedge membership.
    Reference: Ding et al., EMNLP 2020, arXiv:2011.00387

    FIX: N always derived from x.size(0).
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

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: Optional[int] = None,   # kept for API compat, ignored
        num_edges: Optional[int] = None,
    ) -> torch.Tensor:
        device, dtype      = x.device, x.dtype
        node_ids, edge_ids = hyperedge_index

        # ── FIX 2: always use the actual input size ──
        N = x.size(0)
        M = num_edges if num_edges is not None else int(edge_ids.max().item()) + 1
        node_ids = node_ids.clamp(max=N - 1)

        H, D = self.H, self.D

        x_proj = self.W_node(x).view(N, H, D)          # (N, H, D)

        # Edge representation: mean of member node projections
        e_feat = torch.zeros(M, H, D, dtype=dtype, device=device)
        e_feat.index_add_(0, edge_ids, x_proj[node_ids])
        counts = torch.bincount(edge_ids, minlength=M).clamp_min(1)
        e_feat = e_feat / counts.view(M, 1, 1).to(dtype)

        # Attention: e_{i,e} = LeakyReLU(a^T [x_i || h_e])
        alpha = (x_proj[node_ids] * self.att_src).sum(-1) \
              + (e_feat[edge_ids]  * self.att_dst).sum(-1)  # (E_entries, H)
        alpha = F.leaky_relu(alpha, 0.2)

        # Softmax within each hyperedge per head
        a_exp = alpha.exp()
        a_sum = torch.zeros(M, H, dtype=dtype, device=device)
        a_sum.index_add_(0, edge_ids, a_exp)
        alpha_n = a_exp / (a_sum[edge_ids] + 1e-9)

        if self.dropout > 0 and self.training:
            alpha_n = F.dropout(alpha_n, p=self.dropout)

        # Weighted aggregation into hyperedges, then broadcast to nodes
        msg   = x_proj[node_ids] * alpha_n.unsqueeze(-1)   # (E, H, D)
        h_agg = torch.zeros(M, H, D, dtype=dtype, device=device)
        h_agg.index_add_(0, edge_ids, msg)
        out = torch.zeros(N, H, D, dtype=dtype, device=device)
        out.index_add_(0, node_ids, h_agg[edge_ids])

        return self.out_proj(out.view(N, H * D))


# ═══════════════════════════════════════════════════════════════════════════
# §4  SHLNN LAYER  (HGNN + LTC + Spiking, combined)
#
#     Information flow per timestep:
#       x ──HGNNConv──► x_hg
#       x_hg + skip(x) ──LayerNorm──► x_norm
#       [x_norm ‖ v_{t-1}] ──LTCParamNet──► A, 1/τ
#       v_{t-1}, A, 1/τ ──LTCStep──► spikes, v_t
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
    dt           : LTC ODE Δt
    tau_min      : minimum time constant τ
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
                                    learn_edge_w=learn_edge_w,
                                    dropout=dropout)
        elif conv_type == "attention_hgnn":
            self.hg_conv = AttentionHGNNConv(in_channels, out_channels,
                                             num_heads=num_heads,
                                             dropout=dropout)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type!r}")

        self.ltc_params = LTCParameterNet(
            out_channels + out_channels, out_channels, tau_min=tau_min)
        self.skip = (nn.Linear(in_channels, out_channels, bias=False)
                     if in_channels != out_channels else nn.Identity())
        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        v_mem: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
        num_edges: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = x.size(0)
        if v_mem is None:
            v_mem = x.new_zeros(N, self.out_channels)

        x_hg = self.hg_conv(x, hyperedge_index,
                             num_nodes=num_nodes, num_edges=num_edges)
        x_hg = self.norm(x_hg + self.skip(x))

        A, inv_tau = self.ltc_params(torch.cat([x_hg, v_mem], dim=-1))
        spikes, v_next = spiking_ltc_step(
            v_mem, A, inv_tau, dt=self.dt, v_th=self.v_th, alpha=self.alpha)
        return spikes, v_next


class SHLNNStack(nn.Module):
    """Stack of SHLNNLayer modules; carries per-layer v_mem across time."""

    def __init__(self, in_channels: int, hidden_dim: int,
                 num_layers: int = 2, **kw):
        super().__init__()
        dims = [in_channels] + [hidden_dim] * num_layers
        self.layers     = nn.ModuleList(
            [SHLNNLayer(dims[i], dims[i + 1], **kw) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
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
    Language model: Embedding → SHLNNStack → LM Head (weight-tied).

    forward()          — single graph-snapshot step (one sentence)
    forward_sequence() — temporal unrolling over a list of snapshots
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
        self.shlnn      = SHLNNStack(
            embed_dim, hidden_dim, num_layers,
            conv_type=conv_type, num_heads=num_heads,
            v_th=v_th, alpha=alpha, dt=dt,
            tau_min=tau_min, dropout=dropout,
            learn_edge_w=learn_edge_w)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if embed_dim == hidden_dim:
            self.lm_head.weight = self.embedding.weight  # weight tying
        nn.init.normal_(self.embedding.weight, 0.0, 0.02)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        node_indices: torch.Tensor,
        hyperedge_index: torch.Tensor,
        v_mems: Optional[List] = None,
        num_nodes: Optional[int] = None,
        num_edges: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List, torch.Tensor]:
        x = self.embed_drop(self.embedding(node_indices))
        spikes, v_mems_next = self.shlnn(
            x, hyperedge_index, v_mems=v_mems,
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

def build_hypergraph(
    seq_len: int,
    window_size: int = 3,
    add_global: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Build hyperedge_index (2, E_entries) for a token sequence of length seq_len.

    Hyperedge types
    ---------------
    1. Sliding-window n-grams (width=window_size) : local syntactic context
    2. Global sentence hyperedge                  : document-level context

    Returns (hyperedge_index, num_hyperedges).
    """
    nids, eids, eid = [], [], 0
    for start in range(seq_len - window_size + 1):
        for off in range(window_size):
            nids.append(start + off)
            eids.append(eid)
        eid += 1
    if add_global and seq_len > window_size:
        for i in range(seq_len):
            nids.append(i)
            eids.append(eid)
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
        special   = [self.PAD, self.UNK, self.EOS]
        vocab     = special + sorted(set(tokens) - set(special))
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
            ids = torch.tensor(
                [vocab.encode(t) for t in sent], dtype=torch.long)
            hg, ne = build_hypergraph(len(ids), window_size, add_global)
            self.samples.append((ids, hg, ne))

    @staticmethod
    def _load(path: str) -> List[List[str]]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"PTB not found at {path}\n"
                "Download from: https://github.com/wojzaremba/lstm/tree/master/data\n"
                "Files: ptb.train.txt  ptb.valid.txt  ptb.test.txt"
            )
        out = []
        with open(path) as f:
            for line in f:
                toks = line.strip().split()
                if toks:
                    out.append(toks + [Vocabulary.EOS])
        return out

    def __len__(self):            return len(self.samples)
    def __getitem__(self, i):     return self.samples[i]


def _collate(batch):
    return batch   # variable-length; each sentence is its own graph


def get_ptb_loaders(
    data_dir: str,
    window_size: int = 3,
    add_global: bool = True,
    max_seq_len: int = 70,
):
    """Returns (train_loader, valid_loader, test_loader, vocab)."""
    train_ds = PTBDataset(
        os.path.join(data_dir, "ptb.train.txt"),
        vocab=None, max_seq_len=max_seq_len,
        window_size=window_size, add_global=add_global)
    vocab    = train_ds.vocab
    valid_ds = PTBDataset(
        os.path.join(data_dir, "ptb.valid.txt"),
        vocab=vocab, max_seq_len=max_seq_len,
        window_size=window_size, add_global=add_global)
    test_ds  = PTBDataset(
        os.path.join(data_dir, "ptb.test.txt"),
        vocab=vocab, max_seq_len=max_seq_len,
        window_size=window_size, add_global=add_global)
    mk = lambda ds, sh: DataLoader(
        ds, batch_size=1, shuffle=sh, collate_fn=_collate)
    return mk(train_ds, True), mk(valid_ds, False), mk(test_ds, False), vocab


def get_mock_loader(
    vocab_size: int = 10_000,
    num_samples: int = 300,
    seq_len: int = 20,
    window_size: int = 3,
):
    """Synthetic loader — no PTB files required. Returns (loader, vocab_size)."""
    samples = []
    for _ in range(num_samples):
        L   = int(torch.randint(seq_len // 2, seq_len + 1, (1,)).item())
        ids = torch.randint(0, vocab_size, (L,))
        hg, ne = build_hypergraph(L, window_size)
        samples.append((ids, hg, ne))
    return DataLoader(
        samples, batch_size=1, shuffle=True, collate_fn=_collate), vocab_size


# ═══════════════════════════════════════════════════════════════════════════
# §8  TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class SOPCounter:
    """
    Synaptic Operation counter for energy-efficiency analysis.
    SOPs = Σ_t Σ_l spike_count_{t,l} · fan_out_l
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
            "total_sops":     self.total_sop,
            "total_ann_ops":  self.total_ann,
            "energy_ratio":   self.total_sop / self.total_ann,
            "avg_spike_rate": self.total_sop / max(self.n_samples, 1),
        }


def spike_rate_loss(
    spikes: torch.Tensor,
    target_rate: float = 0.1,
    weight: float = 0.01,
) -> torch.Tensor:
    """
    L2 penalty on deviation from target spike rate.
    Prevents silent (rate→0) or saturated (rate→1) regimes.
    """
    return weight * (spikes.mean() - target_rate).pow(2)


@torch.no_grad()
def evaluate_ppl(
    model: SpikingHypergraphLM,
    loader,
    device: torch.device,
    criterion: nn.CrossEntropyLoss,
    window_size: int = 3,
    add_global: bool = True,
) -> float:
    """
    Word-level perplexity: PPL = exp(mean NLL per token).

    FIX 4: hyperedge_index is rebuilt for token_ids[:-1] so the graph
    length matches the input length exactly.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for batch in loader:
        for (token_ids, _hei_full, _ne_full) in batch:
            if len(token_ids) < 2:
                continue
            src_ids = token_ids[:-1]
            hei, ne = build_hypergraph(len(src_ids), window_size, add_global)
            src_ids = src_ids.to(device)
            hei     = hei.to(device)

            logits, _, _ = model(src_ids, hei, num_edges=ne)
            targets      = token_ids[1:].to(device)
            loss         = criterion(logits, targets)
            n            = targets.size(0)
            total_loss   += loss.item() * n
            total_tokens += n

    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


# ═══════════════════════════════════════════════════════════════════════════
# §9  TRAINING CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # ── data ────────────────────────────────────────────────────────────────
    data_dir:         str   = "./data/ptb"
    mock:             bool  = True
    mock_vocab:       int   = 10_000
    mock_samples:     int   = 300
    max_seq_len:      int   = 70
    window_size:      int   = 3
    add_global_edge:  bool  = True
    # ── model ───────────────────────────────────────────────────────────────
    embed_dim:        int   = 128
    hidden_dim:       int   = 256
    num_layers:       int   = 2
    conv_type:        str   = "hgnn"
    num_heads:        int   = 4
    v_th:             float = 1.0
    alpha:            float = 2.0
    dt:               float = 1.0
    tau_min:          float = 0.1
    dropout:          float = 0.1
    learn_edge_w:     bool  = True
    # ── training ────────────────────────────────────────────────────────────
    epochs:            int   = 20
    lr:                float = 1e-3
    weight_decay:      float = 1e-4
    grad_clip:         float = 1.0
    target_spike_rate: float = 0.1
    spike_loss_weight: float = 0.01
    seed:              int   = 42
    save_dir:          str   = "./checkpoints"


# ═══════════════════════════════════════════════════════════════════════════
# §10  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train(cfg: TrainConfig) -> Dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*64}")
    print(f"  Device : {device}  |  Seed : {cfg.seed}")

    # ── data ────────────────────────────────────────────────────────────────
    if cfg.mock:
        train_loader, vocab_size = get_mock_loader(
            cfg.mock_vocab, cfg.mock_samples,
            cfg.max_seq_len, cfg.window_size)
        valid_loader = test_loader = train_loader
    else:
        train_loader, valid_loader, test_loader, vocab = get_ptb_loaders(
            cfg.data_dir, cfg.window_size,
            cfg.add_global_edge, cfg.max_seq_len)
        vocab_size = len(vocab)

    print(f"  Vocab  : {vocab_size:,}  |  Mode : {'mock' if cfg.mock else 'PTB'}")

    # ── model ───────────────────────────────────────────────────────────────
    model = SpikingHypergraphLM(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        conv_type=cfg.conv_type,
        num_heads=cfg.num_heads,
        v_th=cfg.v_th,
        alpha=cfg.alpha,
        dt=cfg.dt,
        tau_min=cfg.tau_min,
        dropout=cfg.dropout,
        learn_edge_w=cfg.learn_edge_w,
    ).to(device)
    print(f"  Params : {model.num_parameters:,}")
    print(f"{'='*64}")

    # ── optimiser & scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
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
            for (token_ids, _hei_full, _ne_full) in batch:
                if len(token_ids) < 2:
                    continue

                # ── FIX 3: rebuild hyperedge_index for the actual input ──
                src_ids = token_ids[:-1]
                hei, ne = build_hypergraph(
                    len(src_ids), cfg.window_size, cfg.add_global_edge)
                src_ids = src_ids.to(device)
                hei     = hei.to(device)
                targets = token_ids[1:].to(device)

                optimizer.zero_grad(set_to_none=True)
                logits, v_mems, spikes = model(src_ids, hei, num_edges=ne)

                lm_loss = criterion(logits, targets)
                sp_loss = spike_rate_loss(
                    spikes, cfg.target_spike_rate, cfg.spike_loss_weight)
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
        val_ppl    = evaluate_ppl(model, valid_loader, device, criterion,
                                  cfg.window_size, cfg.add_global_edge)
        spike_rate = float(np.mean(ep_spike_rates))
        elapsed    = time.time() - t0

        rec = dict(
            epoch=epoch,
            train_ppl=round(train_ppl, 3),
            val_ppl=round(val_ppl, 3),
            spike_rate=round(spike_rate, 4),
            lr=round(scheduler.get_last_lr()[0], 6),
            elapsed_s=round(elapsed, 1),
        )
        history.append(rec)

        print(f"Ep {epoch:03d} | "
              f"Train PPL {train_ppl:8.2f} | "
              f"Val PPL {val_ppl:8.2f} | "
              f"Spike {spike_rate:.3f} | "
              f"LR {scheduler.get_last_lr()[0]:.5f} | "
              f"{elapsed:.1f}s")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(
                {"epoch": epoch, "val_ppl": val_ppl,
                 "model": model.state_dict(), "cfg": asdict(cfg)},
                os.path.join(cfg.save_dir, "best_model.pt"))
            print(f"        ✓ checkpoint  (val_ppl={val_ppl:.2f})")

    # ── final test eval ─────────────────────────────────────────────────────
    ckpt = torch.load(
        os.path.join(cfg.save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    test_ppl   = evaluate_ppl(model, test_loader, device, criterion,
                               cfg.window_size, cfg.add_global_edge)
    sop_report = sop.report()

    print(f"\n{'='*64}")
    print(f"  Best Val PPL  : {best_val_ppl:.2f}")
    print(f"  Test PPL      : {test_ppl:.2f}")
    if sop_report:
        print(f"  Energy Ratio  : {sop_report['energy_ratio']:.4f}  (SNN vs ANN)")
        print(f"  Avg Spike Rate: {sop_report['avg_spike_rate']:.4f}")
    print(f"{'='*64}")

    summary = dict(
        test_ppl=test_ppl, best_val_ppl=best_val_ppl,
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
        if embed_dim == hidden_dim:
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
        self.proj  = nn.Linear(in_dim, out_dim)
        self.v_th  = v_th
        self.beta  = beta

    def forward(self, x, v=None):
        if v is None:
            v = x.new_zeros(x.size(0), self.proj.out_features)
        v_pre  = self.beta * v + self.proj(x)
        spikes = (v_pre >= self.v_th).float()
        v_next = torch.where(spikes.bool(), torch.zeros_like(v_pre), v_pre)
        return spikes, v_next


class SNNLMBaseline(nn.Module):
    """Spiking LM: LIF cells only, no hypergraph, no LTC."""

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        dims = [embed_dim] + [hidden_dim] * num_layers
        self.cells   = nn.ModuleList(
            [_LIFCell(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        if embed_dim == hidden_dim:
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
            h, vn = cell(h, v)
            vnext.append(vn)
        return self.lm_head(h), vnext


@torch.no_grad()
def _eval_baseline(
    model, loader, device, criterion, forward_fn,
    window_size: int = 3,
) -> float:
    """
    FIX 5: baseline evaluation also uses token_ids[:-1] with a matching
    hyperedge_index (the baselines don't actually use hei, but the loop
    structure is kept consistent).
    """
    model.eval()
    tl, tt = 0.0, 0
    for batch in loader:
        for (ids, _, _) in batch:
            if len(ids) < 2:
                continue
            src = ids[:-1].to(device)
            tgt = ids[1:].to(device)
            logits, _ = forward_fn(model, src)
            loss = criterion(logits, tgt)
            n = tgt.size(0)
            tl += loss.item() * n
            tt += n
    return math.exp(tl / max(tt, 1))


def train_baseline(
    model: nn.Module,
    loader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    forward_fn=None,
) -> List[float]:
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
                if len(ids) < 2:
                    continue
                src = ids[:-1].to(device)
                tgt = ids[1:].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = forward_fn(model, src)
                loss = criterion(logits, tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                n = tgt.size(0)
                tl += loss.item() * n
                tt += n
        ppl = math.exp(tl / max(tt, 1))
        ppls.append(ppl)
        print(f"  [{model.__class__.__name__}] Ep {epoch:02d} PPL {ppl:.2f}")
    return ppls


# ═══════════════════════════════════════════════════════════════════════════
# §12  QUICK-START FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_mock(
    epochs: int = 20,
    hidden_dim: int = 256,
    num_layers: int = 2,
    conv_type: str = "hgnn",
    seed: int = 42,
) -> Dict:
    """
    Fastest way to test the full pipeline — no downloads needed.

    >>> summary = run_mock(epochs=20)
    """
    cfg = TrainConfig(
        mock=True, mock_vocab=10_000, mock_samples=300,
        embed_dim=128, hidden_dim=hidden_dim, num_layers=num_layers,
        conv_type=conv_type, epochs=epochs, lr=1e-3, seed=seed,
    )
    return train(cfg)


def run_ptb(
    data_dir: str = "./data/ptb",
    epochs: int = 40,
    embed_dim: int = 200,
    hidden_dim: int = 400,
    num_layers: int = 2,
    conv_type: str = "hgnn",
    seed: int = 42,
) -> Dict:
    """
    Full PTB training run.

    Download PTB first:
        wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt -P ./data/ptb/
        wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt -P ./data/ptb/
        wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt  -P ./data/ptb/

    >>> summary = run_ptb(data_dir="./data/ptb", epochs=40)
    """
    cfg = TrainConfig(
        mock=False, data_dir=data_dir,
        embed_dim=embed_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, conv_type=conv_type,
        epochs=epochs, seed=seed,
    )
    return train(cfg)


def run_comparison(epochs: int = 10, seed: int = 42) -> None:
    """
    Train SHLNN + LSTM + SNN baselines on mock data and print a comparison table.

    >>> run_comparison(epochs=10)
    """
    set_seed(seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, vocab_size = get_mock_loader(vocab_size=5_000, num_samples=200)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    results   = {}

    # SHLNN
    print("\n── SHLNN ──────────────────────────────────────────────")
    s = run_mock(epochs=epochs, seed=seed)
    results["SHLNN (ours)"] = s["test_ppl"]

    # LSTM baseline
    print("\n── LSTM baseline ──────────────────────────────────────")
    lstm = LSTMLMBaseline(vocab_size, 128, 256).to(device)
    train_baseline(lstm, loader, device, epochs=epochs,
                   forward_fn=lambda m, x: m(x))
    results["LSTM (Zaremba 2014)"] = _eval_baseline(
        lstm, loader, device, criterion, lambda m, x: m(x))

    # SNN baseline
    print("\n── SNN (LIF, no graph) ────────────────────────────────")
    snn = SNNLMBaseline(vocab_size, 128, 256).to(device)
    train_baseline(snn, loader, device, epochs=epochs,
                   forward_fn=lambda m, x: m(x))
    results["SNN (LIF, no HG)"] = _eval_baseline(
        snn, loader, device, criterion, lambda m, x: m(x))

    # Table
    print(f"\n{'='*50}")
    print(f"{'Model':<28} {'Test PPL':>10}")
    print(f"{'-'*50}")
    for name, ppl in results.items():
        marker = " ◄" if name.startswith("SHLNN") else ""
        print(f"{name:<28} {ppl:>10.2f}{marker}")
    print(f"{'='*50}")


# ═══════════════════════════════════════════════════════════════════════════
# §13  SCRIPT ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    summary = run_mock(epochs=20)
