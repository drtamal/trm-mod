#!/usr/bin/env python3
"""
TRM-LM: Tiny Recursive Model for Generative Language Modeling
OPTIMIZED FOR NVIDIA L40S - NO CHECKPOINTING (MAX THROUGHPUT)
"""
import os
import sys
import json
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

# ── Reproducibility ──────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ── L40S Optimized Config ───────────────────────────────────────
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

    # Recursive parameters
    n_latent: int = 8
    T_recurse: int = 4
    N_sup: int = 20
    inference_N_sup: int = 6
    ema_decay: float = 0.999

    # Data
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"

    # Training
    batch_size: int = 8 
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 50000
    save_interval: int = 1000
    max_seq_length: int = 512
    
    # Checkpointing DISABLED for L40S speed
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16"
    compile_model: bool = True
    output_dir: str = "./output_trm_l40s"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Architecture Components ──────────────────────────────────────

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

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.kv_group_size > 1:
            k = k.repeat_interleave(self.kv_group_size, dim=1)
            v = v.repeat_interleave(self.kv_group_size, dim=1)

        return self.o_proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, -1))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        )

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp[2](F.silu(self.mlp[0](self.mlp_norm(x))))
        return x

# ── Main Model ───────────────────────────────────────────────────

class TRMForCausalLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.hidden_size)
        self.y_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        
        # Flattened list for efficient sequential processing
        self.net = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        
        self.output_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.output_head.weight = self.tok_emb.weight
            
        self.q_head = nn.Sequential(
            RMSNorm(cfg.hidden_size), 
            nn.Linear(cfg.hidden_size, 1, bias=False)
        )
        self._init_weights()

    def _init_weights(self):
        eff_depth = self.cfg.T_recurse * (self.cfg.n_latent + 1) * self.cfg.num_layers
        std = 0.02 / math.sqrt(eff_depth)
        for name, p in self.named_parameters():
            if p.dim() < 2: continue
            nn.init.normal_(p, mean=0.0, std=std if "tok_emb" not in name else 0.02)

    def _apply_net(self, h):
        for block in self.net:
            h = block(h)
        return self.final_norm(h)

    def _latent_recursion(self, x, y, z, n):
        for _ in range(n):
            z = self._apply_net(x + y + z)
        y = self._apply_net(y + z)
        return y, z

    def forward(self, ids, labels=None):
        B, L = ids.shape
        x = self.tok_emb(ids)
        y, z = self.y_init.expand(B, L, -1).contiguous(), self.z_init.expand(B, L, -1).contiguous()
        
        total_loss = 0
        lm_loss_val = 0
        
        for s in range(self.cfg.N_sup):
            # Recurse
            for _ in range(self.cfg.T_recurse):
                y, z = self._latent_recursion(x, y, z, self.cfg.n_latent)
            
            logits = self.output_head(y)
            q = self.q_head(y).mean(dim=1).squeeze(-1)
            
            if labels is not None:
                # Loss Calculation
                shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
                shift_labels = labels[:, 1:].contiguous().view(-1)
                lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                
                with torch.no_grad():
                    preds = logits[:, :-1, :].argmax(dim=-1)
                    targets = labels[:, 1:]
                    valid_mask = (targets != -100).float()
                    correct = ((preds == targets).float() * valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1).clamp(min=1)
                    halt_target = (correct > 0.8).float()
                
                halt_loss = F.binary_cross_entropy_with_logits(q, halt_target)
                total_loss += (lm_loss + 0.1 * halt_loss)
                lm_loss_val += lm_loss.item()
                
                # Detach for next supervision stage to save memory if needed
                y, z = y.detach(), z.detach()

            if torch.sigmoid(q).mean() > 0.9: break

        return (total_loss / self.cfg.N_sup if labels is not None else None), logits

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=50, temperature=0.8, eos_token_id=0):
        self.eval()
        for _ in range(max_new_tokens):
            _, logits = self.forward(ids[:, -self.cfg.max_position_embeddings:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_token], dim=1)
            if next_token.item() == eos_token_id: break
        return ids

# ── EMA & Data ───────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
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

class FinewebEduStreamingDataset(IterableDataset):
    def __init__(self, tokenizer, max_len, subset="sample-10BT"):
        super().__init__()
        self.tokenizer, self.max_len = tokenizer, max_len
        self.ds = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True)
    def __iter__(self):
        for item in self.ds:
            enc = self.tokenizer(item["text"] + self.tokenizer.eos_token, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
            out = {k: v.squeeze(0) for k, v in enc.items()}
            out["labels"] = out["input_ids"].clone()
            out["labels"][out["attention_mask"] == 0] = -100
            yield out

# ── Training ─────────────────────────────────────────────────────

def main():
    cfg = TRMConfig()
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    loader = DataLoader(FinewebEduStreamingDataset(tokenizer, cfg.max_seq_length, cfg.dataset_subset), batch_size=cfg.batch_size, num_workers=4, pin_memory=True)
    model = TRMForCausalLM(cfg, len(tokenizer)).to(cfg.device)
    
    if cfg.compile_model: model = torch.compile(model)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, cfg.warmup_steps, cfg.max_steps)
    
    print(f"Starting Training: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M Params")
    model.train()
    for i, batch in enumerate(tqdm(loader, total=cfg.max_steps * cfg.gradient_accumulation_steps)):
        ids, labels = batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device)
        
        with torch.autocast(cfg.device, enabled=cfg.use_mixed_precision, dtype=torch.bfloat16):
            loss, _ = model(ids, labels)
            loss = loss / cfg.gradient_accumulation_steps
            
        loss.backward()
        
        if (i + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad()
            ema.update(model)
            
            step = (i + 1) // cfg.gradient_accumulation_steps
            if step % cfg.save_interval == 0:
                ema.apply(model)
                test_ids = tokenizer("The universe is", return_tensors="pt")["input_ids"].to(cfg.device)
                print(f"\nSample: {tokenizer.decode(model.generate(test_ids)[0], skip_special_tokens=True)}")
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"trm_step_{step}.pt"))
                ema.restore(model)
        
        if (i + 1) // cfg.gradient_accumulation_steps >= cfg.max_steps: break

if __name__ == "__main__":
    main()
