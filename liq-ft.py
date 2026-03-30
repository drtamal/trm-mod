#!/usr/bin/env python3
import os, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm

# ================= CONFIG (FINE-TUNING) =================
@dataclass
class SFTConfig:
    hidden_size: int = 768     # Matches your 35M parameter model
    num_layers: int = 12
    num_heads: int = 12
    max_seq_length: int = 256
    char_vocab_size: int = 256
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SFT Specifics
    batch_size: int = 16       
    grad_accum: int = 2
    max_steps: int = 15000     
    log_interval: int = 10
    eval_interval: int = 1000
    
    max_lr: float = 5e-5       
    min_lr: float = 5e-6
    weight_decay: float = 0.1
    label_smoothing: float = 0.0
    
    base_model: str = "lnn_final_model.pt"
    output_model: str = "lnn_chat_assistant.pt"

# ================= MODEL ARCHITECTURE =================
class CharCNNEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.char_emb = nn.Embedding(cfg.char_vocab_size, 64)
        self.conv1 = nn.Conv1d(64, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.ln = nn.LayerNorm(256)
        self.projection = nn.Linear(256, cfg.hidden_size)

    def forward(self, x):
        x = self.char_emb(x).transpose(1, 2) 
        feat = F.gelu(self.conv1(x))
        feat = F.gelu(self.conv2(feat))
        x = feat.transpose(1, 2) 
        x = self.ln(x)
        return self.projection(x)

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
        self.front_end = CharCNNEmbedding(cfg)
        self.blocks = nn.ModuleList([Block(cfg.hidden_size, cfg.num_heads) for _ in range(cfg.num_layers)])
        self.ln = RMSNorm(cfg.hidden_size)
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
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=0, label_smoothing=self.cfg.label_smoothing)
        return loss, logits

# ================= CHAT DATASET =================
class ChatStream(IterableDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        
    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset("daily_dialog", split="train", streaming=True)
        
        for x in ds.shuffle(buffer_size=1000, seed=self.cfg.seed):
            dialogue = x["dialog"]
            formatted_chat = ""
            
            for i, turn in enumerate(dialogue):
                if i % 2 == 0:
                    formatted_chat += f"<|user|>\n{turn.strip()}\n"
                else:
                    formatted_chat += f"<|bot|>\n{turn.strip()}\n<|end|>\n"
            
            bytes_data = formatted_chat.encode('utf-8')
            tokens = [min(b, 255) for b in bytes_data]
            
            if len(tokens) > self.cfg.max_seq_length:
                tokens = tokens[-self.cfg.max_seq_length:]
            else:
                tokens += [0] * (self.cfg.max_seq_length - len(tokens))
                
            ids = torch.tensor(tokens, dtype=torch.long)
            yield {"input_ids": ids, "labels": ids.clone()}

# ================= GENERATION =================
def chat_generate(model, prompt, max_new_tokens=150, temperature=0.7):
    model.eval()
    formatted_prompt = f"<|user|>\n{prompt}\n<|bot|>\n"
    input_bytes = list(formatted_prompt.encode('utf-8'))
    ids = torch.tensor(input_bytes, dtype=torch.long).unsqueeze(0).to(model.cfg.device)
    
    generated = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            _, logits = model(ids[:, -model.cfg.max_seq_length:])
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            ids = torch.cat([ids, next_id], dim=-1)
            generated.append(next_id.item())
            
            current_text = bytes([b for b in generated if 0 < b < 256]).decode('utf-8', errors='ignore')
            if "<|end|>" in current_text:
                return current_text.replace("<|end|>", "").strip()
                
    return bytes([b for b in generated if 0 < b < 256]).decode('utf-8', errors='ignore').strip()

# ================= SFT TRAINING LOOP =================
def get_lr(cfg, step):
    progress = step / cfg.max_steps
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

def finetune():
    cfg = SFTConfig()
    model = LNN_Hybrid(cfg).to(cfg.device)
    
    if os.path.exists(cfg.base_model):
        print(f"Loading Base Knowledge from {cfg.base_model}...")
        model.load_state_dict(torch.load(cfg.base_model, map_location=cfg.device))
    else:
        print(f"ERROR: Could not find {cfg.base_model}. Cannot fine-tune.")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(ChatStream(cfg), batch_size=cfg.batch_size)
    
    print("\nStarting Supervised Fine-Tuning (SFT) for Chat...")
    pbar = tqdm(enumerate(loader), total=cfg.max_steps)
    model.train()
    
    for step, batch in pbar:
        lr = get_lr(cfg, step)
        for pg in opt.param_groups: 
            pg['lr'] = lr
            
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
        
        if step >= cfg.max_steps: 
            break

    torch.save(model.state_dict(), cfg.output_model)
    print(f"\nSFT Complete! Assistant saved as {cfg.output_model}")

if __name__ == "__main__":
    finetune()
