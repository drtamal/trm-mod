#!/usr/bin/env python3
import os
import math
import random
import time
import torch
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
    hidden_size: int = 256
    num_layers: int = 2
    num_attention_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 688
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    n_latent: int = 2
    T_recurse: int = 2
    N_sup: int = 2
    ema_decay: float = 0.999
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    dataset_subset: str = "sample-10BT"
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 10000
    save_interval: int = 1000
    output_dir: str = "./output_trm"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def get_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir): return None, 0
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts: return None, 0
    latest = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1]
    return os.path.join(output_dir, latest), int(latest.split("-")[-1])

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight

def precompute_rope(dim, max_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    L = x.shape[2]
    cos_s = cos[:L, None, :]
    sin_s = sin[:L, None, :]
    cos_s = cos_s.transpose(0, 1).unsqueeze(0)
    sin_s = sin_s.transpose(0, 1).unsqueeze(0)
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([x1 * cos_s - x2 * sin_s, x2 * cos_s + x1 * sin_s], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // self.num_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        cos, sin = precompute_rope(self.head_dim, cfg.max_position_embeddings)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        self.register_buffer("causal_mask", torch.triu(torch.full((cfg.max_position_embeddings, cfg.max_position_embeddings), float("-inf")), 1))

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        if self.num_heads // self.num_kv_heads > 1:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores + self.causal_mask[:L, :L]
        if mask is not None:
            scores = scores + (1.0 - mask.unsqueeze(1).unsqueeze(2)) * -1e9
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        )
    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class TRMForCausalLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.hidden_size)
        self.y_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.net = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.output_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.output_head.weight = self.tok_emb.weight
        self.q_head = nn.Sequential(RMSNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, 1, bias=False))

    def forward(self, ids, labels=None, mask=None):
        B, L = ids.shape
        x = self.tok_emb(ids)
        y, z = self.y_init.expand(B, L, -1), self.z_init.expand(B, L, -1)
        total_loss = 0
        for _ in range(self.cfg.N_sup):
            for _ in range(self.cfg.T_recurse):
                for _ in range(self.cfg.n_latent):
                    curr_z = x + y + z
                    for b in self.net: curr_z = b(curr_z, mask)
                    z = self.final_norm(curr_z)
                curr_y = y + z
                for b in self.net: curr_y = b(curr_y, mask)
                y = self.final_norm(curr_y)
            logits = self.output_head(y)
            q = self.q_head(y).mean(dim=1).squeeze(-1)
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
                shift_labels = labels[:, 1:].contiguous().view(-1)
                lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                with torch.no_grad():
                    acc = (logits[:, :-1, :].argmax(-1) == labels[:, 1:]).float().mean()
                    halt_target = (acc > 0.8).float().expand_as(q)
                total_loss += lm_loss + 0.1 * F.binary_cross_entropy_with_logits(q, halt_target)
        return total_loss / self.cfg.N_sup, logits

class FinewebStream(IterableDataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer, self.cfg = tokenizer, cfg
    def __iter__(self):
        while True:
            try:
                ds = load_dataset("HuggingFaceFW/fineweb-edu", name=self.cfg.dataset_subset, split="train", streaming=True)
                for x in ds.shuffle(buffer_size=1000, seed=self.cfg.seed):
                    e = self.tokenizer(x["text"]+self.tokenizer.eos_token, max_length=self.cfg.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
                    out = {k: v.squeeze(0) for k, v in e.items()}
                    out["labels"] = out["input_ids"].clone()
                    out["labels"][out["attention_mask"] == 0] =
