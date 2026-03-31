#!/usr/bin/env python3
# High-Performance LNN: 3 Epochs / 300k steps per epoch

import os, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

# Ensure tiktoken uses the local cache for offline/HPC environments
os.environ["TIKTOKEN_CACHE_DIR"] = "/data/aicoe_gpu/komal_paul_gtg/tiktoken_cache"
import tiktoken 

# ================= CONFIG =================
@dataclass
class LNNConfig:
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 8
    max_seq_length: int = 256
    vocab_size: int = 50257
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    grad_accum: int = 4
    
    # Epoch Configuration
    num_epochs: int = 3
    steps_per_epoch: int = 300000
    # Total steps for LR scheduler = 900,000
    total_steps: int = 3 * 300000 
    
    log_interval: int = 10
    eval_interval: int = 30000
    warmup_steps: int = 2000 # Increased warmup for longer run
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    label_smoothing: float = 0.05
    output_model: str = "lnn_final_model.pt"

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

class LNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg.hidden_size, cfg.num_heads) for _ in range(cfg.num_layers)])
        self.ln = RMSNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.emb.weight = self.head.weight
    def forward(self, ids, labels=None):
        x = self.emb(ids)
        h = x
        for b in self.blocks: x, h = b(x, h)
        logits = self.head(self.ln(x))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, label_smoothing=self.cfg.label_smoothing)
        return loss, logits

# ================= DATASET =================
class BPEStream(IterableDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.enc = tiktoken.get_encoding("gpt2")
        
    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        shuffled_ds = ds.shuffle(buffer_size=10000, seed=self.cfg.seed)
        
        for x in shuffled_ds:
            tokens = self.enc.encode_ordinary(x["text"])
            if len(tokens) > self.cfg.max_seq_length:
                start_idx = random.randint(0, len(tokens) - self.cfg.max_seq_length)
                tokens = tokens[start_idx : start_idx + self.cfg.max_seq_length]
            else:
                tokens += [0] * (self.cfg.max_seq_length - len(tokens))
            
            yield {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "labels": torch.tensor(tokens, dtype=torch.long)
            }

# ================= GENERATION & EVAL =================
def generate(model, prompt, max_new_tokens=40, temperature=0.8, top_k=40):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(model.cfg.device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            _, logits = model(ids[:, -model.cfg.max_seq_length:])
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat([ids, next_id], dim=-1)
    return enc.decode(ids[0].tolist())

def interactive_eval(model):
    print("\n" + "="*50)
    print("INTERACTIVE LNN EVALUATION MODE")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50)
    while True:
        prompt = input("\nPrompt > ")
        if prompt.lower() in ['quit', 'exit']: break
        result = generate(model, prompt)
        print(f"LNN Output: {result}")

# ================= TRAINING =================
def get_lr(cfg, global_step):
    if global_step < cfg.warmup_steps: 
        return cfg.max_lr * global_step / cfg.warmup_steps
    progress = (global_step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

def train():
    cfg = LNNConfig()
    model = LNN(cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    global_step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n--- Starting Epoch {epoch}/{cfg.num_epochs} ---")
        
        # Reset loader each epoch to restart the stream (with different shuffling via seed)
        loader = DataLoader(BPEStream(cfg), batch_size=cfg.batch_size)
        pbar = tqdm(enumerate(loader), total=cfg.steps_per_epoch, desc=f"Epoch {epoch}")
        model.train()
        
        for step, batch in pbar:
            # Learning Rate based on GLOBAL step
            lr = get_lr(cfg, global_step)
            for pg in opt.param_groups: pg['lr'] = lr
            
            loss, _ = model(batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device))
            (loss / cfg.grad_accum).backward()
            
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            
            if step % cfg.log_interval == 0: 
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}", global_step=global_step)
            
            # Internal epoch evaluations
            if step % cfg.eval_interval == 0 and step > 0:
                print(f"\nStep {step} (Global {global_step}) Preview: {generate(model, 'The nature of')}")
                model.train()
            
            global_step += 1
            if step >= cfg.steps_per_epoch: 
                break 

        # End of Epoch Evaluation
        print(f"\n--- Epoch {epoch} Complete ---")
        print(f"Final Epoch {epoch} Evaluation: {generate(model, 'Deep within the universe, there lies')}")
        
        # Intermediate checkpoint
        checkpoint_name = f"lnn_epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_name)
        print(f"Epoch checkpoint saved as {checkpoint_name}")

    torch.save(model.state_dict(), cfg.output_model)
    print(f"\nAll Epochs Complete. Final model saved as {cfg.output_model}")
    return model

if __name__ == "__main__":
    trained_model = train()
    interactive_eval(trained_model)
