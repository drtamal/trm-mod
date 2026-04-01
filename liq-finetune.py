#!/usr/bin/env python3
import os, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import tiktoken
# Force tiktoken to use your local cache and skip internet checks
os.environ["TIKTOKEN_CACHE_DIR"] = "/data/aicoe_gpu/komal_paul_gtg/tiktoken_cache"

# ================= CONFIG (4 EPOCHS / BATCH SIZE 2) =================
@dataclass
class SFTConfig:
    hidden_size: int = 384     
    num_layers: int = 8        
    num_heads: int = 8         
    max_seq_length: int = 256
    vocab_size: int = 50257    
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Updated Training Params
    num_epochs: int = 4
    batch_size: int = 2        # Batch size 2 as requested
    grad_accum: int = 8        # Increased to maintain stable gradients
    
    # Alpaca has ~52,000 samples. 52,000 / 2 = 26,000 steps per epoch.
    steps_per_epoch: int = 26000 
    total_steps: int = 4 * 26000 
    
    log_interval: int = 10
    max_lr: float = 3e-5       
    min_lr: float = 3e-6
    weight_decay: float = 0.1
    label_smoothing: float = 0.05
    
    base_model: str = "lnn_final_model.pt"
    output_model: str = "lnn_chat_assistant_final.pt"

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

# ================= BPE CHAT DATASET (LOOPABLE) =================
class BPEChatStream(IterableDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        try:
            self.enc = tiktoken.get_encoding("gpt2")
        except Exception:
            # If it fails, we try to force it to look at the cache directory
            import tiktoken
            from tiktoken.load import read_file, read_file_cached
            print("Offline Mode: Loading tokenizer from local cache...")
            self.enc = tiktoken.get_encoding("gpt2")
        
    def __iter__(self):
        from datasets import load_dataset
        # The training loop handles resetting the epoch, but we can stream endlessly here
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        # Unique shuffle per worker/epoch via seed
        shuffled_ds = ds.shuffle(buffer_size=2000, seed=self.cfg.seed + random.randint(0, 1000))
        
        for x in shuffled_ds:
            user_text = x["instruction"]
            if x["input"]: user_text += f"\n{x['input']}"
            bot_text = x["output"]
            
            formatted_chat = f"<|user|>\n{user_text}\n<|bot|>\n{bot_text}\n<|end|>\n"
            tokens = self.enc.encode_ordinary(formatted_chat)
            
            if len(tokens) > self.cfg.max_seq_length:
                tokens = tokens[-self.cfg.max_seq_length:]
            else:
                tokens += [0] * (self.cfg.max_seq_length - len(tokens))
                
            ids = torch.tensor(tokens, dtype=torch.long)
            yield {"input_ids": ids, "labels": ids.clone()}

# ================= CHAT EVALUATION =================
def chat_eval(model, prompt, temperature=0.7):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    formatted_prompt = f"<|user|>\n{prompt}\n<|bot|>\n"
    ids = torch.tensor(enc.encode(formatted_prompt), dtype=torch.long).unsqueeze(0).to(model.cfg.device)
    
    generated = []
    for _ in range(100):
        with torch.no_grad():
            _, logits = model(ids[:, -model.cfg.max_seq_length:])
            # Low temperature for stable evaluation
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=-1)
            generated.append(next_id.item())
            
            if "<|end|>" in enc.decode(generated):
                break
                
    return enc.decode(generated).replace("<|end|>", "").strip()

# ================= SFT TRAINING LOOP =================
def get_lr(cfg, global_step):
    progress = global_step / cfg.total_steps
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

def finetune():
    cfg = SFTConfig()
    model = LNN(cfg).to(cfg.device)
    
    print(f"Loading Base Knowledge from {cfg.base_model}...")
    model.load_state_dict(torch.load(cfg.base_model, map_location=cfg.device))
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    global_step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n--- Starting SFT Epoch {epoch}/{cfg.num_epochs} ---")
        loader = DataLoader(BPEChatStream(cfg), batch_size=cfg.batch_size)
        pbar = tqdm(enumerate(loader), total=cfg.steps_per_epoch)
        model.train()
        
        for step, batch in pbar:
            lr = get_lr(cfg, global_step)
            for pg in opt.param_groups: pg['lr'] = lr
                
            loss, _ = model(batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device))
            (loss / cfg.grad_accum).backward()
            
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            
            if step % cfg.log_interval == 0: 
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.7f}")
            
            global_step += 1
            if step >= cfg.steps_per_epoch: 
                break

        # --- END OF EPOCH EVALUATION ---
        print(f"\n--- Epoch {epoch} Evaluation ---")
        test_q = "Explain the importance of space exploration."
        print(f"User: {test_q}")
        print(f"LNN Assistant: {chat_eval(model, test_q)}")
        
        # Save per-epoch checkpoint
        checkpoint_path = f"lnn_assistant_epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    torch.save(model.state_dict(), cfg.output_model)
    print(f"\nSFT Complete! Final model saved as {cfg.output_model}")

if __name__ == "__main__":
    finetune()
