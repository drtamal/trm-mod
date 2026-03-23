#!/usr/bin/env python3
import os, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# ── Helper: Find Latest Checkpoint ───────────────────────────────
def get_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir): return None, 0
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts: return None, 0
    latest = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1]
    return os.path.join(output_dir, latest), int(latest.split("-")[-1])

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@dataclass
class TRMConfig:
    hidden_size: int = 1024
    num_layers: int = 12
    num_attention_heads: int = 16
    num_kv_heads: int = 8
    intermediate_size: int = 2816
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    n_latent: int = 8
    T_recurse: int = 4
    N_sup: int = 20
    ema_decay: float = 0.999
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    dataset_subset: str = "sample-10BT"
    # Hardware Optimized for your Node (16 CPU Cores / L40S)
    batch_size: int = 16 
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    learning_rate: float = 3e-4
    max_steps: int = 50000
    save_interval: int = 1000
    use_mixed_precision: bool = True
    output_dir: str = "./output_trm"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Architecture Components ──────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight

def precompute_rope(dim, max_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    # Fixed Shape Mismatch: (B, H, L, D) logic
    L = x.shape[2]
    cos_s = cos[:L, :].view(1, 1, L, -1)
    sin_s = sin[:L, :].view(1, 1, L, -1)
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([x1 * cos_s - x2 * sin_s, x2 * cos_s + x1 * sin_s], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads, self.num_kv_heads = cfg.num_attention_heads, cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // self.num_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        cos, sin = precompute_rope(self.head_dim, cfg.max_position_embeddings)
        self.register_buffer("rope_cos", cos); self.register_buffer("rope_sin", sin)
        self.register_buffer("causal_mask", torch.triu(torch.full((cfg.max_position_embeddings, cfg.max_position_embeddings), float("-inf")), 1))

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q, k, v = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2), \
                  self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2), \
                  self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, self.rope_cos, self.rope_sin), apply_rope(k, self.rope_cos, self.rope_sin)
        if self.num_heads // self.num_kv_heads > 1:
            k, v = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1), v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        # Optimized SDPA (Flash Attention)
        return self.o_proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, D))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(cfg.hidden_size), CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False), nn.SiLU(), nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False))
    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))

class TRMForCausalLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg, self.vocab_size = cfg, vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.hidden_size)
        self.y_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        # Flat structure to match your net.0, net.1 keys
        self.net = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.output_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
        if cfg.tie_word_embeddings: self.output_head.weight = self.tok_emb.weight
        self.q_head = nn.Sequential(RMSNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, 1, bias=False))

    def forward(self, ids, labels=None):
        B, L = ids.shape
        x = self.tok_emb(ids)
        y, z = self.y_init.expand(B, L, -1), self.z_init.expand(B, L, -1)
        total_loss = 0
        for _ in range(self.cfg.N_sup):
            for _ in range(self.cfg.T_recurse):
                for _ in range(self.cfg.n_latent):
                    curr_z = x + y + z
                    for b in self.net: curr_z = b(curr_z)
                    z = self.final_norm(curr_z)
                curr_y = y + z
                for b in self.net: curr_y = b(curr_y)
                y = self.final_norm(curr_y)
            logits = self.output_head(y)
            if labels is not None:
                total_loss += F.cross_entropy(logits[:, :-1, :].reshape(-1, self.vocab_size), labels[:, 1:].reshape(-1), ignore_index=-100)
        return total_loss / self.cfg.N_sup, logits

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay, self.shadow = decay, {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow: self.shadow[n].copy_(self.decay * self.shadow[n] + (1 - self.decay) * p.data)

def main():
    cfg = TRMConfig()
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name); tokenizer.pad_token = tokenizer.eos_token
    model = TRMForCausalLM(cfg, len(tokenizer)).to(cfg.device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.1)
    sched = get_cosine_schedule_with_warmup(opt, 2000, cfg.max_steps)

    ckpt_path, start_step = get_latest_checkpoint(cfg.output_dir)
    if ckpt_path:
        ckpt = torch.load(os.path.join(ckpt_path, "trm_model.pt"), map_location=cfg.device)
        model.load_state_dict(ckpt["model_state_dict"]); opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("ema_shadow"): ema.shadow = ckpt["ema_shadow"]

    # Dataset Optimized for L40S and 16 CPU Cores
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=cfg.dataset_subset, split="train", streaming=True)
    def gen():
        for x in ds.shuffle(buffer_size=1000, seed=cfg.seed):
            e = tokenizer(x["text"]+tokenizer.eos_token, max_length=cfg.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
            out = {k: v.squeeze(0) for k, v in e.items()}
            out["labels"] = out["input_ids"].clone()
            out["labels"][out["attention_mask"] == 0] = -100
            yield out
    
    loader = DataLoader(gen(), batch_size=cfg.batch_size, num_workers=4, pin_memory=True)
    pbar = tqdm(loader, total=cfg.max_steps, initial=start_step)
    optim_step, accum_loss = start_step, 0
    model.train()
    
    skip_batches = start_step * cfg.gradient_accumulation_steps
    for i, batch in enumerate(pbar):
        if i < skip_batches: continue
        input_ids, labels = batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(input_ids, labels)
            loss = loss / cfg.gradient_accumulation_steps
        loss.backward()
        accum_loss += loss.item()
        if (i + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); ema.update(model)
            optim_step += 1
            pbar.set_postfix(step=optim_step, loss=f"{accum_loss:.4f}")
            accum_loss = 0
            if optim_step % cfg.save_interval == 0:
                sd = os.path.join(cfg.output_dir, f"checkpoint-{optim_step}")
                os.makedirs(sd, exist_ok=True)
                torch.save({"step": optim_step, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sched.state_dict(), "ema_shadow": ema.shadow}, os.path.join(sd, "trm_model.pt"))
            if optim_step >= cfg.max_steps: break

if __name__ == "__main__": main()
