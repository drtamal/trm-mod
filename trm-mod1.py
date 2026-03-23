import os, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

@dataclass
class TRMConfig:
    hidden_size: int = 1024
    num_layers: int = 12
    num_attention_heads: int = 16
    num_kv_heads: int = 8
    intermediate_size: int = 2816
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    n_latent: int = 8
    T_recurse: int = 4
    N_sup: int = 20
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    batch_size: int = 8 
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 512
    learning_rate: float = 3e-4
    max_steps: int = 50000
    save_interval: int = 1000
    output_dir: str = "./output_trm_l40s"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Components ──────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight

def precompute_rope(dim, max_len):
    freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    L = x.shape[2]
    cos_s, sin_s = cos[:L, None, :], sin[:L, None, :]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([x1 * cos_s - x2 * sin_s, x2 * cos_s + x1 * sin_s], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads, self.num_kv_heads = cfg.num_attention_heads, cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // self.num_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        cos, sin = precompute_rope(self.head_dim, cfg.max_position_embeddings)
        self.register_buffer("rope_cos", cos); self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, self.rope_cos, self.rope_sin), apply_rope(k, self.rope_cos, self.rope_sin)
        if self.num_heads // self.num_kv_heads > 1:
            k, v = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1), v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # SDPA (Flash Attention) for L40S
        return self.o_proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, D))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(cfg.hidden_size), CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        )
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))

class TRMForCausalLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.hidden_size)
        self.y_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        
        # FLATTENED: net.0, net.1, etc.
        self.net = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.output_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
        if cfg.tie_word_embeddings: self.output_head.weight = self.tok_emb.weight
        self.q_head = nn.Sequential(RMSNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, 1, bias=False))

    def forward(self, ids, labels=None):
        B, L = ids.shape
        x = self.tok_emb(ids)
        y, z = self.y_init.expand(B, L, -1), self.z_init.expand(B, L, -1)
        total_loss = 0
        
        # Recursive Training Loop
        for _ in range(self.cfg.N_sup):
            for _ in range(self.cfg.T_recurse):
                for _ in range(self.cfg.n_latent):
                    curr_z = x + y + z
                    for b in self.net: curr_z = b(curr_z)
                    z = self.final_norm(curr_z)
                curr_y = y + z
                for b in self.net: curr_y = b(curr_y)
                y = self.final_norm(curr_y)
            
            logits = self.output_head(y)
            if labels is not None:
                total_loss += F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
                
        return total_loss / self.cfg.N_sup, logits
