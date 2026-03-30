#!/usr/bin/env python3
import os, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import tiktoken

# ================= CONFIG (MATCHES YOUR TRAINED MODEL) =================
@dataclass
class SFTConfig:
    hidden_size: int = 128     # Fixed to match the checkpoint!
    num_layers: int = 4        # Fixed to match the checkpoint!
    num_heads: int = 8         # Assuming 4 heads for a dim of 128
    max_seq_length: int = 256
    vocab_size: int = 50257    # Back to BPE vocab size
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SFT Specifics
    batch_size: int = 16       
    grad_accum: int = 2
    max_steps: int = 15000     
    log_interval: int = 10
    eval_interval: int = 500
    
    max_lr: float = 5e-5       
    min_lr: float = 5e-6
    weight_decay: float = 0.1
    label_smoothing: float = 0.0
    
    base_model: str = "lnn_final_model.pt"
    output_model: str = "lnn_chat_assistant.pt"

# ================= BPE MODEL ARCHITECTURE =================
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
        self.emb.weight = self.head.weight # Weight tying

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

# ================= BPE CHAT DATASET =================
class BPEChatStream(IterableDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.enc = tiktoken.get_encoding("gpt2")
        
    def __iter__(self):
        from datasets import load_dataset
        # Alpaca is a modern Parquet dataset, it will NEVER throw the script error
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        
        for x in ds.shuffle(buffer_size=1000, seed=self.cfg.seed):
            # 1. Get the user's prompt
            user_text = x["instruction"]
            if x["input"]: # Sometimes there is extra context
                user_text += f"\n{x['input']}"
            
            # 2. Get the AI's response
            bot_text = x["output"]
            
            # 3. Format into our Chat Template
            formatted_chat = f"<|user|>\n{user_text}\n<|bot|>\n{bot_text}\n<|end|>\n"
            
            tokens = self.enc.encode_ordinary(formatted_chat)
            
            # Truncate or Pad
            if len(tokens) > self.cfg.max_seq_length:
                tokens = tokens[-self.cfg.max_seq_length:]
            else:
                tokens += [0] * (self.cfg.max_seq_length - len(tokens))
                
            ids = torch.tensor(tokens, dtype=torch.long)
            yield {"input_ids": ids, "labels": ids.clone()}

# ================= BPE GENERATION =================
def chat_generate(model, prompt, max_new_tokens=100, temperature=0.7):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    formatted_prompt = f"<|user|>\n{prompt}\n<|bot|>\n"
    ids = torch.tensor(enc.encode(formatted_prompt), dtype=torch.long).unsqueeze(0).to(model.cfg.device)
    
    generated_tokens = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            _, logits = model(ids[:, -model.cfg.max_seq_length:])
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            ids = torch.cat([ids, next_id], dim=-1)
            generated_tokens.append(next_id.item())
            
            # Check for stop token
            current_text = enc.decode(generated_tokens)
            if "<|end|>" in current_text:
                return current_text.replace("<|end|>", "").strip()
                
    return enc.decode(generated_tokens).strip()

# ================= SFT TRAINING LOOP =================
def get_lr(cfg, step):
    progress = step / cfg.max_steps
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

def finetune():
    cfg = SFTConfig()
    model = LNN(cfg).to(cfg.device)
    
    print(f"Loading Base Knowledge from {cfg.base_model}...")
    model.load_state_dict(torch.load(cfg.base_model, map_location=cfg.device))
    print("Model Loaded successfully!")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(BPEChatStream(cfg), batch_size=cfg.batch_size)
    
    print("\nStarting Supervised Fine-Tuning (SFT) for Chat...")
    pbar = tqdm(enumerate(loader), total=cfg.max_steps)
    model.train()
    
    for step, batch in pbar:
        lr = get_lr(cfg, step)
        for pg in opt.param_groups: pg['lr'] = lr
            
        loss, _ = model(batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device))
        (loss / cfg.grad_accum).backward()
        
        if (step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        
        if step % cfg.log_interval == 0: 
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        if step % cfg.eval_interval == 0 and step > 0:
            print(f"\n[Test Chat] User: How is the weather today?")
            print(f"[Test Chat] LNN: {chat_generate(model, 'How is the weather today?')}")
            model.train()
        
        if step >= cfg.max_steps: break

    torch.save(model.state_dict(), cfg.output_model)
    print(f"\nSFT Complete! Assistant saved as {cfg.output_model}")

if __name__ == "__main__":
    finetune()
