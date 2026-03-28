#!/usr/bin/env python3
import os, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import tiktoken  # Standard BPE Tokenizer

@dataclass
class LNNConfig:
    hidden_size: int = 256
    num_layers: int = 4
    max_seq_length: int = 256  # Increased for better context
    vocab_size: int = 50257    # GPT-2 Vocab Size
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8        # Adjusted for BPE
    max_steps: int = 10000
    save_interval: int = 5000
    log_interval: int = 100
    output_dir: str = "./output_lnn_bpe"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Components ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.g

class LiquidCell(nn.Module):
    """
    Implements a Liquid Neural Network cell where the state 
    update is gated by the input's 'fluidity'.
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.W = nn.Linear(dim, dim)
        self.U = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, h):
        # The 'Liquid' gating mechanism
        inter_gate = torch.sigmoid(self.gate(x) + self.alpha)
        dh = torch.tanh(self.W(x) + self.U(h))
        return h + inter_gate * dh * 0.1

class CausalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        return self.proj((attn @ v).transpose(1, 2).reshape(B, L, D))

class LiquidBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = CausalAttention(dim)
        self.liquid = LiquidCell(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
        self.ln3 = RMSNorm(dim)
    
    def forward(self, x, h):
        h = self.liquid(self.ln1(x), h)
        x = x + self.attn(self.ln2(x))
        x = x + self.mlp(self.ln3(h))
        return x, h

class LNNForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq_length, cfg.hidden_size))
        self.blocks = nn.ModuleList([LiquidBlock(cfg.hidden_size) for _ in range(cfg.num_layers)])
        self.ln_f = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight # Weight tying
        
    def forward(self, ids, labels=None):
        B, L = ids.shape
        x = self.tok_emb(ids) + self.pos_emb[:, :L, :]
        
        h = torch.zeros_like(x) # Initial hidden state
        for block in self.blocks:
            x, h = block(x, h)
        
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits

# --- Data ---

class TextStream(IterableDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.enc = tiktoken.get_encoding("gpt2")
    
    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        for x in ds:
            tokens = self.enc.encode_ordinary(x["text"])
            for i in range(0, len(tokens) - self.cfg.max_seq_length, self.cfg.max_seq_length):
                chunk = tokens[i:i + self.cfg.max_seq_length + 1]
                if len(chunk) < self.cfg.max_seq_length + 1: continue
                yield {
                    "input_ids": torch.tensor(chunk[:-1]),
                    "labels": torch.tensor(chunk[1:])
                }

# --- Generation ---

def generate(model, prompt, max_new_tokens=30, temperature=0.7):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor(enc.encode(prompt)).unsqueeze(0).to(model.cfg.device)
    
    for _ in range(max_new_tokens):
        input_ids = ids[:, -model.cfg.max_seq_length:]
        with torch.no_grad():
            _, logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            # Simple Top-K filtering
            v, _ = torch.topk(logits, 50)
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=-1)
            
    return enc.decode(ids[0].tolist())

# --- Training Loop ---

if __name__ == "__main__":
    cfg = LNNConfig()
    set_seed(cfg.seed)
    model = LNNForCausalLM(cfg).to(cfg.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    loader = DataLoader(TextStream(cfg), batch_size=cfg.batch_size)
    
    pbar = tqdm(enumerate(loader), total=cfg.max_steps)
    for step, batch in pbar:
        loss, _ = model(batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        
        if step % cfg.log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        if step >= cfg.max_steps: break

    print("\n--- Final Test ---")
    print(generate(model, "The scientific method is"))
