#!/usr/bin/env python3
import os
import sys
import json
import math
import random
import re
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

# ── Config ───────────────────────────────────────────────────────
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
    N_sup: int = 16                  
    inference_N_sup: int = 4         
    ema_decay: float = 0.999         

    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"

    batch_size: int = 2
    gradient_accumulation_steps: int = 32
    learning_rate: float = 5e-4      
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    
    max_steps: int = 50000         
    save_interval: int = 1000      
    max_seq_length: int = 256

    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    output_dir: str = "./output_trm"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Architecture Components ──────────────────────────────────────

class RMSNorm(nn.Module):
    def _init_(self, dim: int, eps: float = 1e-5):
        super()._init_()
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
    def _init_(self, cfg):
        super()._init_()
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
        causal_mask = torch.triu(torch.full((cfg.max_position_embeddings, cfg.max_position_embeddings), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

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
            pad_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)).to(attn.dtype) * -1e9
            attn = attn + pad_mask
        attn = F.softmax(attn, dim=-1)
        return self.o_proj((attn @ v).transpose(1, 2).reshape(B, L, -1))

class TransformerBlock(nn.Module):
    def _init_(self, cfg):
        super()._init_()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        )
    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        gate_out = self.mlp[0](self.mlp_norm(x))
        x = x + self.mlp[2](F.silu(gate_out))
        return x

class TinyRecursiveNet(nn.Module):
    def _init_(self, cfg):
        super()._init_()
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.use_checkpoint = cfg.use_gradient_checkpointing
    def forward(self, h, mask=None):
        for block in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(block, h, mask, use_reentrant=False)
            else:
                h = block(h, mask)
        return self.final_norm(h)

class TRMForCausalLM(nn.Module):
    def _init_(self, cfg, vocab_size):
        super()._init_()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.hidden_size)
        self.y_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.net = TinyRecursiveNet(cfg)
        self.output_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.output_head.weight = self.tok_emb.weight
        self.q_head = nn.Sequential(RMSNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, 1, bias=False))
        self._init_weights()

    def _init_weights(self):
        eff_depth = self.cfg.T_recurse * (self.cfg.n_latent + 1) * self.cfg.num_layers
        std = 0.02
        for name, p in self.named_parameters():
            if p.dim() < 2: continue
            if "o_proj" in name or "mlp.2" in name:
                nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * eff_depth))
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    def embed(self, input_ids):
        B, L = input_ids.shape
        x = self.tok_emb(input_ids).detach()
        y = self.y_init.expand(B, L, -1).contiguous()
        z = self.z_init.expand(B, L, -1).contiguous()
        return x, y, z

    def _latent_recursion(self, x, y, z, mask, n):
        for _ in range(n):
            z = self.net(x + y + z, mask)
        y = self.net(y + z, mask)
        return y, z

    def _deep_recursion(self, x, y, z, mask):
        T, n = self.cfg.T_recurse, self.cfg.n_latent
        with torch.no_grad():
            for _ in range(T - 1):
                y, z = self._latent_recursion(x, y, z, mask, n)
        if self.training:
            torch.clear_autocast_cache()
        y, z = self._latent_recursion(x, y, z, mask, n)
        logits = self.output_head(y)
        q = self.q_head(y).mean(dim=1).squeeze(-1)
        return y, z, logits, q

    def compute_loss(self, logits, q, labels):
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
        return lm_loss + 0.1 * halt_loss, lm_loss.item()

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_p=0.9, eos_token_id=0):
        self.eval()
        for _ in range(max_new_tokens):
            x, y, z = self.embed(input_ids[:, -self.cfg.max_position_embeddings:])
            for _ in range(self.cfg.inference_N_sup):
                y, z, logits, q = self._deep_recursion(x, y, z, None)
                if torch.sigmoid(q).mean() > 0.9: break
            next_logits = logits[:, -1, :] / temperature
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask_remove = cumulative - F.softmax(sorted_logits, dim=-1) >= top_p
                mask_remove[..., 1:] = mask_remove[..., :-1].clone()
                mask_remove[..., 0] = 0
                sorted_logits[mask_remove] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == eos_token_id: break
        return input_ids

# ── EMA & Helper Logic ───────────────────────────────────────────

class EMA:
    def _init_(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].copy_(self.decay * self.shadow[n] + (1 - self.decay) * p.data)
    def apply(self, model):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow: p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n])

def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    # Sort by the integer step number
    checkpoints.sort(key=lambda x: int(re.sub(r'\D', '', x)))
    latest = checkpoints[-1]
    return os.path.join(output_dir, latest)

# ── Training Loop ────────────────────────────────────────────────

def train_stream(model, ema, tokenizer, loader, optimizer, scheduler, cfg, start_step=0):
    model.train()
    optimizer.zero_grad()
    
    optim_step = start_step
    total_samples_seen = start_step * cfg.gradient_accumulation_steps
    
    pbar = tqdm(loader, desc="Training")
    batch_lm_accum = 0.0
    
    for i, batch in enumerate(pbar):
        # FAST SKIP: Skip data points already processed
        if i < total_samples_seen:
            if i % 500 == 0:
                pbar.set_description(f"Skipping to step {start_step} (idx {i})")
            continue

        ids, labels, mask = batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device), batch["attention_mask"].to(cfg.device)
        
        with torch.autocast("cuda", enabled=cfg.use_mixed_precision, dtype=torch.bfloat16):
            x, y, z = model.embed(ids)
        
        batch_lm, steps_taken = 0, 0
        for s in range(cfg.N_sup):
            y_in, z_in = (y, z) if s == 0 else (y.detach(), z.detach())
            with torch.autocast("cuda", enabled=cfg.use_mixed_precision, dtype=torch.bfloat16):
                y_out, z_out, logits, q = model._deep_recursion(x, y_in, z_in, mask)
                loss, lm_val = model.compute_loss(logits, q, labels)
                scaled_loss = loss / cfg.gradient_accumulation_steps
            scaled_loss.backward()
            y, z, batch_lm, steps_taken = y_out.detach(), z_out.detach(), batch_lm + lm_val, steps_taken + 1
            if torch.sigmoid(q).mean() > 0.9: break
        
        batch_lm_accum += (batch_lm / steps_taken)
        
        if (i + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ema.update(model)
            optim_step += 1
            
            avg_step_loss = batch_lm_accum / cfg.gradient_accumulation_steps
            batch_lm_accum = 0.0
            ppl = math.exp(min(avg_step_loss, 20.0))
            pbar.set_postfix(step=optim_step, loss=f"{avg_step_loss:.4f}", ppl=f"{ppl:.1f}", sup=steps_taken)
            
            if optim_step % cfg.save_interval == 0:
                save_checkpoint(model, ema, optimizer, scheduler, tokenizer, cfg, optim_step, avg_step_loss)

            if optim_step >= cfg.max_steps:
                break

def save_checkpoint(model, ema, optimizer, scheduler, tokenizer, cfg, step, loss):
    print(f"\n--- Saving Checkpoint @ Step {step} ---")
    ema.apply(model)
    save_dir = os.path.join(cfg.output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    
    ckpt_path = os.path.join(save_dir, "trm_model.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "config": vars(cfg)
    }, ckpt_path)
    tokenizer.save_pretrained(save_dir)
    ema.restore(model)
    print(f"✅ Saved to {ckpt_path}\n")

# ── Data & Runner ────────────────────────────────────────────────

class FinewebEduStreamingDataset(IterableDataset):
    def _init_(self, tokenizer, max_len, subset="sample-10BT", split="train"):
        self.tokenizer, self.max_len = tokenizer, max_len
        self.ds = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split=split, streaming=True)
        self.ds = self.ds.shuffle(buffer_size=10000, seed=42)
    def _iter_(self):
        for item in self.ds:
            txt = item["text"] + self.tokenizer.eos_token
            enc = self.tokenizer(txt, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
            out = {k: v.squeeze(0) for k, v in enc.items()}
            out["labels"] = out["input_ids"].clone()
            out["labels"][out["attention_mask"] == 0] = -100
            yield out

def main():
    cfg = TRMConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = TRMForCausalLM(cfg, len(tokenizer)).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, cfg.warmup_steps, cfg.max_steps)
    
    # --- Checkpoint Logic ---
    start_step = 0
    latest_ckpt_dir = find_latest_checkpoint(cfg.output_dir)
    
    if latest_ckpt_dir:
        ckpt_path = os.path.join(latest_ckpt_dir, "trm_model.pt")
        print(f"🔄 Found checkpoint: {latest_ckpt_dir}. Loading...")
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        sched.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"]
        print(f"✅ Resumed from step {start_step}")
    
    ema = EMA(model)
    train_ds = FinewebEduStreamingDataset(tokenizer, cfg.max_seq_length, subset=cfg.dataset_subset)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size)
    
    print(f"🚀 Starting training from step {start_step}...")
    train_stream(model, ema, tokenizer, loader, opt, sched, cfg, start_step=start_step)

if _name_ == "_main_":
    main()
