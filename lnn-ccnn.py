#!/usr/bin/env python3
# LNN with Hybrid Character-CNN Front-end

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
    hidden_size: int = 192
    num_layers: int = 8
    num_heads: int = 8
    max_seq_length: int = 512 # Characters take more space, so we increased this
    char_vocab_size: int = 256 # Raw bytes
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    grad_accum: int = 4
    max_steps: int = 100000
    log_interval: int = 10
    eval_interval: int = 5000
    warmup_steps: int = 500
    max_lr: float = 4e-4 # CNNs can handle slightly higher initial LRs
    min_lr: float = 4e-5
    weight_decay: float = 0.1
    label_smoothing: float = 0.05
    output_model: str = "lnn_char_cnn_model.pt"

# ================= HYBRID FRONT-END =================
class CharCNNEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # We use a larger embedding for characters to give the CNN more to work with
        self.char_emb = nn.Embedding(cfg.char_vocab_size, 64) 
        self.convolutions = nn.ModuleList([
            nn.Conv1d(64, 128, kernel_size=k, padding=k//2) for k in [3, 5, 7]
        ])
        # Add Batchnorm to prevent the "0.4740" flat loss
        self.bn = nn.BatchNorm1d(384) 
        self.projection = nn.Linear(384, cfg.hidden_size)

    def forward(self, x):
        # Normalize input to 0-1 range to help the embedding layer
        x = self.char_emb(x).transpose(1, 2) # [B, 64, L]
        
        # Extract features
        x = torch.cat([F.silu(conv(x)) for conv in self.convolutions], dim=1)
        x = self.bn(x).transpose(1, 2)
        return self.projection(x)

# ================= CORE COMPONENTS =================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.g

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
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
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ln1, self.ln2, self.ln3 = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
    def forward(self, x, h):
        h = self.liquid(self.ln1(x), h)
        x = x + self.attn(self.ln2(x))
        x = x + self.mlp(self.ln3(h))
        return x, h

class LNN_Hybrid(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # THE HYBRID FRONT-END
        self.front_end = CharCNNEmbedding(cfg)
        
        self.blocks = nn.ModuleList([Block(cfg.hidden_size, cfg.num_heads) for _ in range(cfg.num_layers)])
        self.ln = RMSNorm(cfg.hidden_size)
        # Prediction head back to character space
        self.head = nn.Linear(cfg.hidden_size, cfg.char_vocab_size, bias=False)

    def forward(self, ids, labels=None):
        x = self.front_end(ids)
        h = x
        for b in self.blocks: x, h = b(x, h)
        logits = self.head(self.ln(x))
        
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, self.cfg.char_vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, label_smoothing=self.cfg.label_smoothing)
        return loss, logits

# ================= DATASET =================
class CharStream(IterableDataset):
    def __init__(self, cfg):
        self.cfg = cfg
    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        for x in ds.shuffle(buffer_size=1000, seed=self.cfg.seed):
            # Encode as raw bytes (UTF-8)
            bytes_data = x["text"].encode('utf-8')
            tokens = [min(b, 255) for b in bytes_data]
            
            if len(tokens) > self.cfg.max_seq_length: tokens = tokens[:self.cfg.max_seq_length]
            else: tokens += [0] * (self.cfg.max_seq_length - len(tokens))
            
            ids = torch.tensor(tokens, dtype=torch.long)
            yield {"input_ids": ids, "labels": ids.clone()}

# ================= GENERATION & EVAL =================
def generate(model, prompt, max_new_tokens=60, temperature=0.7):
    model.eval()
    # Convert string to list of bytes
    input_ids = list(prompt.encode('utf-8'))
    ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(model.cfg.device)
    
    generated_bytes = input_ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            _, logits = model(ids[:, -model.cfg.max_seq_length:])
            # Apply temperature sampling instead of argmax
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            ids = torch.cat([ids, next_id], dim=-1)
            generated_bytes.append(next_id.item())
            
    # Decode with 'ignore' to prevent crashes on half-learned UTF-8
    return bytes([b for b in generated_bytes if 0 < b < 256]).decode('utf-8', errors='ignore')

def train():
    cfg = LNNConfig()
    model = LNN_Hybrid(cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(CharStream(cfg), batch_size=cfg.batch_size)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    pbar = tqdm(enumerate(loader), total=cfg.max_steps)
    model.train()
    
    for step, batch in pbar:
        loss, _ = model(batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device))
        (loss / cfg.grad_accum).backward()
        
        if (step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        
        if step % cfg.log_interval == 0: pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        if step % cfg.eval_interval == 0 and step > 0:
            print(f"\nStep {step} Preview: {generate(model, 'The nature of')}"); model.train()
        
        if step >= cfg.max_steps: break

    torch.save(model.state_dict(), cfg.output_model)
    return model

if __name__ == "__main__":
    train()
