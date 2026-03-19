#!/usr/bin/env python3
"""
TRM-LM: Tiny Recursive Model for Generative Language Modeling
L40S Optimized - Fixed "Backward through graph a second time" Error.
"""
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# ── Checkpoint Helper ───────────────────────────────────────────
def get_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None, 0
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None, 0
    latest = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1]
    step = int(latest.split("-")[-1])
    return os.path.join(output_dir, latest), step

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class TRMConfig:
    hidden_size: int = 512
    num_layers: int = 2
    num_attention_heads: int = 8
    num_kv_heads: int = 4
    intermediate_size: int = 1376
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    
    n_latent: int = 6
    T_recurse: int = 3
    N_sup: int = 8  # Reduced slightly for stability, can be increased
    inference_N_sup: int = 4
    ema_decay: float = 0.999
    
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    dataset_subset: str = "sample-10BT"

    # --- L40S OPTIMIZED ---
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    # ----------------------

    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 50000
    save_interval: int = 1000
    use_mixed_precision: bool = True
    output_dir: str = "./output_trm"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Architecture ────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
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
    B, H, L, D = x.shape
    cos_sliced = cos[:L, :].view(1, 1, L, D // 2).contiguous()
    sin_sliced = sin[:L, :].view(1, 1, L, D // 2).contiguous()
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    return torch.cat([x1 * cos_sliced - x2 * sin_sliced, x2 * cos_sliced + x1 * sin_sliced], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.kv_group_size = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * self.head_dim, cfg.hidden_size, bias=False)
        cos, sin = precompute_rope(self.head_dim, cfg.max_position_embeddings)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("causal_mask", torch.triu(torch.full((cfg.max_position_embeddings, cfg.max_position_embeddings), float("-inf")), 1), persistent=False)

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, self.rope_cos, self.rope_sin), apply_rope(k, self.rope_cos, self.rope_sin)
        if self.kv_group_size > 1:
            k = k.repeat_interleave(self.kv_group_size, dim=1)
            v = v.repeat_interleave(self.kv_group_size, dim=1)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn + self.causal_mask[:L, :L].to(attn.dtype)
        if attention_mask is not None:
            attn = attn + (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)).to(attn.dtype) * -1e9
        return self.o_proj((F.softmax(attn, dim=-1) @ v).transpose(1, 2).reshape(B, L, -1))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(cfg.hidden_size), CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False), nn.SiLU(), nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False))
    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp[2](F.silu(self.mlp[0](self.mlp_norm(x))))

class TinyRecursiveNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size)
    def forward(self, h, mask=None):
        for block in self.blocks: h = block(h, mask)
        return self.final_norm(h)

class TRMForCausalLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg, self.vocab_size = cfg, vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.hidden_size)
        self.y_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.net = TinyRecursiveNet(cfg)
        self.output_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
        if cfg.tie_word_embeddings: self.output_head.weight = self.tok_emb.weight
        self.q_head = nn.Sequential(RMSNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, 1, bias=False))

    def _latent_recursion(self, x, y, z, mask, n):
        for _ in range(n): z = self.net(x + y + z, mask)
        return self.net(y + z, mask), z

    def forward(self, ids, labels=None, mask=None, n_sup=1):
        B, L = ids.shape
        x = self.tok_emb(ids)
        y, z = self.y_init.expand(B, L, -1), self.z_init.expand(B, L, -1)
        
        total_loss, last_logits, last_q = 0, None, None
        
        for _ in range(n_sup):
            # No detachment here to maintain graph for single backward
            for _ in range(self.cfg.T_recurse):
                y, z = self._latent_recursion(x, y, z, mask, self.cfg.n_latent)
            
            logits = self.output_head(y)
            q = self.q_head(y).mean(dim=1).squeeze(-1)
            
            if labels is not None:
                loss_step, _ = self.compute_loss(logits, q, labels)
                total_loss += loss_step
            
            last_logits, last_q = logits, q
            
        return total_loss / n_sup, last_logits, last_q

    def compute_loss(self, logits, q, labels):
        shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        with torch.no_grad():
            correct = ((logits[:, :-1, :].argmax(-1) == labels[:, 1:]).float() * (labels[:, 1:] != -100).float()).sum() / (labels[:, 1:] != -100).float().sum().clamp(min=1)
            halt_target = (correct > 0.8).float().expand_as(q)
        return lm_loss + 0.1 * F.binary_cross_entropy_with_logits(q, halt_target), lm_loss.item()

# ── EMA & Training Loop ──────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay, self.shadow = decay, {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow: self.shadow[n].copy_(self.decay * self.shadow[n] + (1 - self.decay) * p.data)
    def apply(self, model):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow: p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n])

def train_stream(model, ema, tokenizer, loader, optimizer, scheduler, cfg, start_step=0):
    model.train()
    skip_batches = start_step * cfg.gradient_accumulation_steps
    pbar = tqdm(loader, total=cfg.max_steps * cfg.gradient_accumulation_steps, desc="Training")
    optim_step, accum_loss = start_step, 0.0

    for i, batch in enumerate(pbar):
        if i < skip_batches: continue
        
        ids, labels, mask = batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device), batch["attention_mask"].to(cfg.device)
        
        with torch.autocast("cuda", enabled=cfg.use_mixed_precision, dtype=torch.bfloat16):
            loss, logits, q = model(ids, labels, mask, n_sup=cfg.N_sup)
            scaled_loss = loss / cfg.gradient_accumulation_steps

        scaled_loss.backward()
        accum_loss += loss.item()

        if (i + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ema.update(model)
            optim_step += 1
            
            pbar.set_postfix(step=optim_step, loss=f"{accum_loss/cfg.gradient_accumulation_steps:.4f}")
            accum_loss = 0.0
            
            if optim_step % cfg.save_interval == 0:
                save_checkpoint(model, ema, optimizer, scheduler, optim_step, tokenizer, cfg)
            if optim_step >= cfg.max_steps: break

def save_checkpoint(model, ema, optimizer, scheduler, step, tokenizer, cfg):
    ema.apply(model)
    save_dir = os.path.join(cfg.output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "ema_shadow": ema.shadow}, os.path.join(save_dir, "trm_model.pt"))
    tokenizer.save_pretrained(save_dir)
    ema.restore(model)

# ── Main ─────────────────────────────────────────────────────────
class FinewebEduStreamingDataset(IterableDataset):
    def __init__(self, tokenizer, max_len, subset):
        super().__init__()
        self.tokenizer, self.max_len = tokenizer, max_len
        self.ds = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True).shuffle(buffer_size=10000, seed=42)
    def __iter__(self):
        for item in self.ds:
            enc = self.tokenizer(item["text"] + self.tokenizer.eos_token, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
            out = {k: v.squeeze(0) for k, v in enc.items()}
            out["labels"] = out["input_ids"].clone()
            out["labels"][out["attention_mask"] == 0] = -100
            yield out

def main():
    cfg = TRMConfig()
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = TRMForCausalLM(cfg, len(tokenizer)).to(cfg.device)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, cfg.warmup_steps, cfg.max_steps)

    ckpt_path, start_step = get_latest_checkpoint(cfg.output_dir)
    if ckpt_path:
        ckpt = torch.load(os.path.join(ckpt_path, "trm_model.pt"), map_location=cfg.device)
        model.load_state_dict(ckpt["model_state_dict"]); opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"]); ema.shadow = ckpt["ema_shadow"]
        start_step = ckpt["step"]

    loader = DataLoader(FinewebEduStreamingDataset(tokenizer, cfg.max_seq_length, cfg.dataset_subset), batch_size=cfg.batch_size)
    train_stream(model, ema, tokenizer, loader, opt, sched, cfg, start_step=start_step)

if __name__ == "__main__": main()
