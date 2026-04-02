import os, math, random, torch, tiktoken, time, requests
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm

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
    num_epochs: int = 3
    steps_per_epoch: int = 30000
    val_steps: int = 500
    total_steps: int = 3 * 30000 
    log_interval: int = 10
    eval_interval: int = 5000 
    warmup_steps: int = 2000
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    label_smoothing: float = 0.05
    output_model: str = "lnn_final_model.pt"

# ================= COMPONENTS =================
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
    def forward(self, L, offset=0):
        return self.cos[:, :, offset:offset+L, :], self.sin[:, :, offset:offset+L, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class CausalAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)
        
    def forward(self, x, cache=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        
        offset = cache[0].shape[2] if cache is not None else 0
        cos, sin = self.rope(L, offset=offset)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k.detach(), v.detach())
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if L > 1:
            mask = torch.triu(torch.ones(L, k.shape[2], device=x.device), diagonal=k.shape[2]-L+1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, L, D)), new_cache

class LiquidCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.W, self.U = nn.Linear(dim, dim), nn.Linear(dim, dim)
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
        
    def forward(self, x, h, cache=None): # <-- FIXED: Now accepts cache
        h = self.liquid(self.ln1(x), h)
        attn_out, next_cache = self.attn(self.ln2(x), cache=cache)
        x = x + attn_out
        x = x + self.mlp(self.ln3(h))
        return x, h, next_cache

class LNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg.hidden_size, cfg.num_heads) for _ in range(cfg.num_layers)])
        self.ln = RMSNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.emb.weight = self.head.weight
        
    def forward(self, ids, h=None, caches=None, labels=None):
        x = self.emb(ids)
        if h is None: h = x
        new_caches = []
        for i, b in enumerate(self.blocks):
            layer_cache = caches[i] if caches is not None else None
            x, h, l_cache = b(x, h, cache=layer_cache)
            new_caches.append(l_cache)
        logits = self.head(self.ln(x))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=50256, label_smoothing=self.cfg.label_smoothing)
        return loss, logits, h, new_caches

# ================= DATA & TRAINING =================
def dynamic_collate_fn(batch):
    pad_id = 50256 
    ids = [item["input_ids"] for item in batch]
    lbls = [item["labels"] for item in batch]
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id),
        "labels": torch.nn.utils.rnn.pad_sequence(lbls, batch_first=True, padding_value=pad_id)
    }

class BPEStream(IterableDataset):
    def __init__(self, cfg, split="train"):
        self.cfg, self.enc, self.split = cfg, tiktoken.get_encoding("gpt2"), split
    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=self.split, streaming=True)
        for x in ds.shuffle(buffer_size=10000, seed=self.cfg.seed):
            tokens = self.enc.encode_ordinary(x["text"])
            if len(tokens) > self.cfg.max_seq_length:
                start = random.randint(0, len(tokens) - self.cfg.max_seq_length)
                tokens = tokens[start : start + self.cfg.max_seq_length]
            yield {"input_ids": torch.tensor(tokens, dtype=torch.long), "labels": torch.tensor(tokens, dtype=torch.long)}

@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()
    total_loss, steps = 0, 0
    for i, batch in enumerate(val_loader):
        if i >= cfg.val_steps: break
        loss, _, _, _ = model(batch["input_ids"].to(cfg.device), labels=batch["labels"].to(cfg.device))
        total_loss += loss.item()
        steps += 1
    model.train()
    return total_loss / steps, math.exp(total_loss / steps)

def train():
    cfg = LNNConfig()
    model = LNN(cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"🚀 SKAI-LNN Initialized | Params: {params:,} | Size: {(params*4)/(1024**2):.2f}MB")

    train_loader = DataLoader(BPEStream(cfg, split="train"), batch_size=cfg.batch_size, collate_fn=dynamic_collate_fn)
    val_loader = DataLoader(BPEStream(cfg, split="train"), batch_size=cfg.batch_size, collate_fn=dynamic_collate_fn)
    
    best_val_loss, global_step = float('inf'), 0
    for epoch in range(1, cfg.num_epochs + 1):
        pbar = tqdm(enumerate(train_loader), total=cfg.steps_per_epoch, desc=f"Epoch {epoch}")
        for step, batch in pbar:
            lr = (cfg.max_lr * min(1.0, global_step / cfg.warmup_steps))
            for pg in opt.param_groups: pg['lr'] = lr
            
            loss, _, _, _ = model(batch["input_ids"].to(cfg.device), labels=batch["labels"].to(cfg.device))
            (loss / cfg.grad_accum).backward()
            
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            
            if global_step % cfg.eval_interval == 0 and global_step > 0:
                v_loss, v_ppl = validate(model, val_loader, cfg)
                print(f"\n[VAL] Step {global_step}: Loss {v_loss:.4f} | PPL {v_ppl:.2f}")
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    torch.save(model.state_dict(), "lnn_best_model.pt")

            pbar.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{math.exp(loss.item()):.2f}")
            global_step += 1
            if step >= cfg.steps_per_epoch: break 

if __name__ == "__main__":
    train()
