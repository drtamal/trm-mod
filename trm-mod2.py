#!/usr/bin/env python3
import os
import math
import random
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm


# ── Reproducibility ──────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Config ───────────────────────────────────────────────────────
@dataclass
class TRMConfig:
    hidden_size: int = 512
    num_layers: int = 2
    num_attention_heads: int = 8
    num_kv_heads: int = 4
    intermediate_size: int = 1376
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5

    n_latent: int = 6
    T_recurse: int = 3
    N_sup: int = 16
    inference_N_sup: int = 4

    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"

    batch_size: int = 2
    gradient_accumulation_steps: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000

    max_steps: int = 50000
    save_interval: int = 1000
    max_seq_length: int = 256

    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    output_dir: str = "./output_trm"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── RMSNorm ─────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ── RoPE ────────────────────────────────────────────────────────
def precompute_rope(dim, max_len, base=10000):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    B, H, L, D = x.shape
    cos = cos[:L].view(1, 1, L, D // 2)
    sin = sin[:L].view(1, 1, L, D // 2)

    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    return torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)


# ── Attention ───────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

        cos, sin = precompute_rope(self.head_dim, cfg.max_position_embeddings)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)

        if self.num_heads != self.num_kv_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn += (1 - mask[:, None, None, :]) * -1e9

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v

        return self.o_proj(out.transpose(1, 2).reshape(B, L, D))


# ── Transformer Block ───────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.SiLU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ── Core Network ────────────────────────────────────────────────
class TinyRecursiveNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size)

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x = blk(x, mask)
        return self.norm(x)


# ── Model ───────────────────────────────────────────────────────
class TRMForCausalLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(vocab_size, cfg.hidden_size)
        self.net = TinyRecursiveNet(cfg)
        self.head = nn.Linear(cfg.hidden_size, vocab_size)

    def forward(self, input_ids, mask=None):
        x = self.emb(input_ids)
        x = self.net(x, mask)
        logits = self.head(x)
        return logits

    def compute_loss(self, logits, labels):
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100
        )


# ── Dataset ─────────────────────────────────────────────────────
class FinewebEduStreamingDataset(IterableDataset):
    def __init__(self, tokenizer, max_len, subset):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=subset,
            split="train",
            streaming=True
        )

    def __iter__(self):
        for item in self.ds:
            text = item["text"]
            enc = self.tokenizer(
                text,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            out = {k: v.squeeze(0) for k, v in enc.items()}
            out["labels"] = out["input_ids"].clone()
            out["labels"][out["attention_mask"] == 0] = -100
            yield out


# ── Training ────────────────────────────────────────────────────
def train():
    cfg = TRMConfig()
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = TRMForCausalLM(cfg, len(tokenizer)).to(cfg.device)

    dataset = FinewebEduStreamingDataset(tokenizer, cfg.max_seq_length, cfg.dataset_subset)
    loader = DataLoader(dataset, batch_size=cfg.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.max_steps)

    step = 0
    pbar = tqdm(loader)

    for batch in pbar:
        input_ids = batch["input_ids"].to(cfg.device)
        labels = batch["labels"].to(cfg.device)
        mask = batch["attention_mask"].to(cfg.device)

        logits = model(input_ids, mask)
        loss = model.compute_loss(logits, labels)

        loss.backward()

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=loss.item())
        step += 1

        if step >= cfg.max_steps:
            break


# ── Entry ───────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
