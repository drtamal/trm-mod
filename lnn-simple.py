#!/usr/bin/env python3
import os, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

# ================= CONFIG =================
@dataclass
class LNNConfig:
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 8
    max_seq_length: int = 512
    vocab_size: int = 0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8        # 8 * 4 = 32 Effective Batch Size
    grad_accum: int = 4
    max_steps: int = 10000
    save_interval: int = 2000
    log_interval: int = 1
    eval_interval: int = 2000
    output_dir: str = "./output_lnn"
    label_smoothing: float = 0.05
    warmup_steps: int = 500
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1

# ================= TOKENIZER =================
class CharacterTokenizer:
    def __init__(self):
        chars = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\n\t"
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.pad_token_id = 0
        self.vocab_size = len(self.stoi) + 1
        self.unique_count = len(self.stoi)

    def encode(self, text):
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids):
        return ''.join([self.itos.get(i, '') for i in ids if i > 0])

# ================= CORE COMPONENTS =================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(1e-5).rsqrt()
        return x * norm * self.g

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos()[None, None, :, :])
        self.register_buffer('sin', emb.sin()[None, None, :, :])

    def forward(self, L):
        return self.cos[:, :, :L, :], self.sin[:, :, :L, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class CausalAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(L)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, L, D))

class LiquidCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.W = nn.Linear(dim, dim)
        self.U = nn.Linear(dim, dim)

    def forward(self, x, h):
        g = torch.sigmoid(self.alpha)
        dh = torch.tanh(self.W(x) + self.U(h))
        return h + g * dh * 0.1

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = CausalAttention(dim, heads)
        self.liquid = LiquidCell(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.ln1, self.ln2, self.ln3 = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)

    def forward(self, x, h):
        h = self.liquid(self.ln1(x), h)
        x = x + self.attn(self.ln2(x))
        x = x + self.mlp(self.ln3(h))
        return x, h

class LNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg.hidden_size, cfg.num_heads) for _ in range(cfg.num_layers)])
        self.ln = RMSNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids, labels=None):
        x = self.emb(ids)
        h = x
        for b in self.blocks: x, h = b(x, h)
        logits = self.head(self.ln(x))
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.cfg.vocab_size), 
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return loss, logits

# ================= DATA PACKING =================
class PackingDataset(IterableDataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        from datasets import load_dataset
        self.ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    def __iter__(self):
        buffer = []
        for example in self.ds:
            buffer.extend(self.tokenizer.encode(example["text"]))
            buffer.append(self.tokenizer.pad_token_id)
            while len(buffer) >= self.cfg.max_seq_length:
                chunk = buffer[:self.cfg.max_seq_length]
                buffer = buffer[self.cfg.max_seq_length:]
                yield {"input_ids": torch.tensor(chunk), "labels": torch.tensor(chunk)}

# ================= UTILS =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_lr(cfg, step):
    if step < cfg.warmup_steps:
        return cfg.max_lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

# ================= MAIN TRAIN =================
def train():
    cfg = LNNConfig()
    set_seed(cfg.seed)
    tokenizer = CharacterTokenizer()
    cfg.vocab_size = tokenizer.vocab_size
    
    model = LNN(cfg).to(cfg.device)
    num_params = sum(p.numel() for p in model.parameters())
    
    # EXACT OUTPUT HEADERS
    print(f"Vocabulary size: {cfg.vocab_size} ({tokenizer.unique_count} unique chars + pad)")
    print(f"Model parameters: {num_params:,}")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(PackingDataset(tokenizer, cfg), batch_size=cfg.batch_size)
    
    pbar = tqdm(enumerate(loader), total=cfg.max_steps, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    model.train()
    opt.zero_grad()
    
    for step, batch in pbar:
        # LR Scheduling
        current_lr = get_lr(cfg, step)
        for pg in opt.param_groups:
            pg['lr'] = current_lr

        loss, _ = model(batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device))
        scaled_loss = loss / cfg.grad_accum
        scaled_loss.backward()
        
        if (step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
        # Update TQDM postfix to match target exactly
        if step % cfg.log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")
            
        if step >= cfg.max_steps: 
            break

if __name__ == "__main__":
    train()
