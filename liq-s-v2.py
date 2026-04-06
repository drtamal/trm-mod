#!/usr/bin/env python3
import os, math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm

# ================= PERFORMANCE & T4 STABILITY =================
torch.set_float32_matmul_precision('high') 
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0" 
import tiktoken
import torch
import gc
from torch.utils.checkpoint import checkpoint # Add this import


# Clear Python garbage collection
gc.collect()

# Clear the PyTorch CUDA cache
torch.cuda.empty_cache()

# Optional: if you have a zombie model variable
if 'model' in locals():
    del model
    
# This should show significantly more free memory now
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

@dataclass
class Config:
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 8
    max_seq_length: int = 256 
    vocab_size: int = 50257
    dropout: float = 0.1
    batch_size: int = 4      # Adjusted for T4 VRAM
    grad_accum: int = 4          
    num_epochs: int = 3
    steps_per_epoch: int = 10000 
    warmup_steps: int = 500
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    label_smoothing: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    num_workers: int = 2      
    log_interval: int = 10
    eval_interval: int = 1000
    output_model: str = "liq_parallel_final.pt"

    @property
    def total_steps(self) -> int: return self.num_epochs * self.steps_per_epoch

# ========================== THE RECURSIVE ENGINE ==========================
def parallel_associative_scan(a: torch.Tensor, b: torch.Tensor):
    """Functional Parallel Scan: O(log L) complexity without in-place errors."""
    L = a.shape[1]
    num_steps = int(math.log2(L))
    for i in range(num_steps):
        step = 2**i
        a_curr, b_curr = a[:, step:, :], b[:, step:, :]
        a_prev, b_prev = a[:, :-step, :], b[:, :-step, :]

        new_b = a_curr * b_prev + b_curr
        new_a = a_curr * a_prev
        
        a = torch.cat([a[:, :step, :], new_a], dim=1)
        b = torch.cat([b[:, :step, :], new_b], dim=1)
    return b

# ========================== MODEL COMPONENTS ==========================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    def forward(self, seq_len: int):
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(L)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.resid_drop(self.out_proj(out))

class ParallelLiquidCell(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.tau_proj = nn.Linear(dim, dim)
        self.input_proj = nn.Linear(dim, dim)
        self.log_dt = nn.Parameter(torch.zeros(dim))

    def forward(self, x, h0):
        tau = F.softplus(self.tau_proj(x)) + 1e-3
        dt = F.softplus(self.log_dt)
        alpha = torch.clamp(dt / tau, 0.0, 1.0)
        candidate = torch.tanh(self.input_proj(x))
        a = 1.0 - alpha
        b = alpha * candidate
        
        # Initial state integration
        h0_expanded = h0.unsqueeze(1) 
        b_init = a[:, :1, :] * h0_expanded + b[:, :1, :]
        b = torch.cat([b_init, b[:, 1:, :]], dim=1)
        
        return parallel_associative_scan(a, b)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.ln_attn = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, dropout)
        self.ln_liq = RMSNorm(dim)
        self.liquid = ParallelLiquidCell(dim)
        self.liq_gate = nn.Linear(dim, dim, bias=False)
        self.ln_ffn = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x, h0):
        x = x + self.attn(self.ln_attn(x))
        h_seq = self.liquid(self.ln_liq(x), h0)
        x = x + torch.sigmoid(self.liq_gate(h_seq)) * h_seq
        x = x + self.ffn(self.ln_ffn(x))
        return x, h_seq[:, -1]

class LiquidLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.use_checkpointing = True  # Toggle this for T4 memory savings
        
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.max_seq_length, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.h0 = nn.ParameterList([nn.Parameter(torch.zeros(cfg.hidden_size)) for _ in range(cfg.num_layers)])
        self.ln_out = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight 
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        x = self.emb_drop(self.token_emb(input_ids))
        
        for i, block in enumerate(self.blocks):
            h0_val = self.h0[i].expand(B, -1)
            
            # Manual Gradient Checkpointing logic
            if self.training and self.use_checkpointing:
                # use_reentrant=False is the modern PyTorch standard
                x, _ = checkpoint(block, x, h0_val, use_reentrant=False)
            else:
                x, _ = block(x, h0_val)
                
        logits = self.lm_head(self.ln_out(x))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, label_smoothing=self.cfg.label_smoothing, ignore_index=-100)
        return loss, logits

# ========================== RESILIENT DATASET ==========================
class BPEStream(IterableDataset):
    def __init__(self, cfg: Config, epoch: int = 0):
        self.cfg = cfg
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token
        self.epoch = epoch

    def __iter__(self):
        from datasets import load_dataset
        import datasets
        datasets.config.STREAMING_READ_MAX_RETRIES = 100
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info else 0
        num_w = worker_info.num_workers if worker_info else 1
        
        samples_processed = 0
        max_retries = 50
        while max_retries > 0:
            try:
                ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
                ds = ds.shuffle(buffer_size=10000, seed=self.cfg.seed + self.epoch * 1000 + wid)
                if num_w > 1: ds = ds.shard(num_shards=num_w, index=wid)
                if samples_processed > 0: ds = ds.skip(samples_processed)

                for sample in ds:
                    tokens = self.enc.encode_ordinary(sample["text"])
                    if not tokens: continue
                    sl = self.cfg.max_seq_length
                    if len(tokens) > sl:
                        s = random.randint(0, len(tokens) - sl)
                        ids = tokens[s : s+sl]
                    else:
                        pad = sl - len(tokens)
                        ids = tokens + [self.eot] * pad
                    yield {"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(ids, dtype=torch.long)}
                    samples_processed += 1
                break
            except Exception as e:
                max_retries -= 1
                wait = min(60, 5 * (2 ** (10 - min(10, max_retries))))
                print(f"\n[Network Error] Worker {wid}: {e}. Retrying in {wait}s...")
                time.sleep(wait)

# ========================== EVAL & UTILS ==========================
def get_lr(cfg, step):
    if step < cfg.warmup_steps: return cfg.max_lr * step / cfg.warmup_steps
    progress = min((step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps), 1.0)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def evaluate_prompts(model, prompts, max_new=50):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    print("\n" + "="*30 + "\nPROMPT EVALUATION\n" + "="*30)
    for p in prompts:
        ids = torch.tensor(enc.encode(p), dtype=torch.long, device=model.cfg.device).unsqueeze(0)
        for _ in range(max_new):
            _, logits = model(ids[:, -model.cfg.max_seq_length:])
            next_id = torch.multinomial(F.softmax(logits[:, -1, :] / 0.8, -1), 1)
            ids = torch.cat([ids, next_id], dim=-1)
            if next_id.item() == 50256: break
        print(f"Prompt: {p}\nOutput: {enc.decode(ids[0].tolist())}\n" + "-"*30)
    model.train()

# ========================== TRAINING ==========================
def train():
    cfg = Config()
    # Explicitly clear cache before starting
    torch.cuda.empty_cache()
    
    torch.manual_seed(cfg.seed)
    model = LiquidLM(cfg).to(cfg.device)
    
    # DO NOT call model.gradient_checkpointing_enable() 
    # It is now handled internally in the LiquidLM forward pass
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # --- PRINT MODEL SIZE ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*40}")
    print(f"MODEL INITIALIZED")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'='*40}\n")
    
    print("Compiling for T4 Stability...")
    model = torch.compile(model)
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    amp_dtype = torch.float16 # Fixed for T4 stability

    test_prompts = ["The future of liquid AI is", "In space, the stars look like"]

    global_step = 0
    for epoch in range(1, cfg.num_epochs + 1):
        loader = DataLoader(BPEStream(cfg, epoch-1), batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
        model.train()
        pbar = tqdm(loader, total=cfg.steps_per_epoch * cfg.grad_accum, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            if torch.cuda.is_available():
                torch.compiler.cudagraph_mark_step_begin()

            with torch.amp.autocast("cuda", enabled=cfg.use_amp, dtype=amp_dtype):
                loss, _ = model(batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device))
            
            scaler.scale(loss / cfg.grad_accum).backward()
            
            if (global_step * cfg.grad_accum + 1) % cfg.grad_accum == 0:
                global_step += 1
                for pg in opt.param_groups: pg["lr"] = get_lr(cfg, global_step)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                
                if global_step % cfg.log_interval == 0: 
                    pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
                if global_step >= epoch * cfg.steps_per_epoch: break
        
        evaluate_prompts(model, test_prompts)
        torch.save(model.state_dict(), f"liq_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()
