#!/usr/bin/env python3
"""
TRM-LM: Tiny Recursive Model for Generative Language Modeling
OPTIMIZED FOR NVIDIA L40S (46GB VRAM) with SDPA Attention
"""
import os
import sys
import json
import math
import random
import argparse
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
    # Model Architecture - Scaled up for L40S
    hidden_size: int = 1024  # Increased from 512 (2x)
    num_layers: int = 12      # Increased from 2 (6x)
    num_attention_heads: int = 16  # Increased from 8
    num_kv_heads: int = 8          # GQA for efficiency
    intermediate_size: int = 2816  # Increased proportionally (2.75x hidden)
    max_position_embeddings: int = 4096  # Increased from 2048
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    # Recursive parameters
    n_latent: int = 8          # Increased from 6
    T_recurse: int = 4          # Increased from 3
    N_sup: int = 20             # Increased from 16
    inference_N_sup: int = 6    # Increased from 4
    ema_decay: float = 0.999

    # Data
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"

    # Training - Optimized for L40S throughput
    batch_size: int = 8          # Increased from 2 (4x)
    gradient_accumulation_steps: int = 16  # Reduced from 32
    # Effective batch size: 8 * 16 = 128 sequences
    # Tokens per step: 128 * 512 = 65,536 tokens (much higher throughput)
    
    learning_rate: float = 3e-4   # Slightly lower for larger model
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000      # More warmup for larger model
    
    # Scaled for 50k steps but more tokens per step
    max_steps: int = 50000
    save_interval: int = 1000
    max_seq_length: int = 512      # Increased from 256 (2x)
    
    # L40S optimizations
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16"  # L40S loves bfloat16
    compile_model: bool = True     # Enable torch.compile on L40S
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
        self.hidden_size = cfg.hidden_size

        # Projections
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * self.head_dim, cfg.hidden_size, bias=False)

        # RoPE embeddings
        cos, sin = precompute_rope(self.head_dim, cfg.max_position_embeddings)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        
        # Project to q, k, v
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Handle GQA by repeating k/v if needed
        if self.kv_group_size > 1:
            k = k.repeat_interleave(self.kv_group_size, dim=1)
            v = v.repeat_interleave(self.kv_group_size, dim=1)

        # Prepare mask for SDPA
        attn_mask = None
        is_causal = True
        
        if attention_mask is not None:
            # Convert padding mask to additive mask for SDPA
            # attention_mask: (B, L) with 1 for valid tokens, 0 for padding
            expanded_mask = attention_mask[:, None, None, :]  # (B, 1, 1, L)
            expanded_mask = expanded_mask.expand(-1, self.num_heads, L, -1)  # (B, H, L, L)
            
            # Create causal mask (upper triangular -inf)
            causal_part = torch.triu(
                torch.ones(L, L, device=x.device, dtype=x.dtype) * float('-inf'), 
                diagonal=1
            )
            
            # Combine: where padding mask is 0, set to -inf regardless of causal position
            attn_mask = torch.where(
                expanded_mask.bool(),
                causal_part.unsqueeze(0).unsqueeze(0),  # Use causal mask where valid
                float('-inf')  # Mask out padding tokens entirely
            )
            is_causal = False

        # Use PyTorch's scaled_dot_product_attention (FlashAttention v2)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,  # No dropout in attention
            is_causal=is_causal,
            scale=None  # Default 1/sqrt(head_dim)
        )

        # Reshape and project output
        return self.o_proj(attn_output.transpose(1, 2).reshape(B, L, -1))

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

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        gate_out = self.mlp[0](self.mlp_norm(x))
        x = x + self.mlp[2](F.silu(gate_out))
        return x

class TinyRecursiveNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
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

# ── Main Model ───────────────────────────────────────────────────

class TRMForCausalLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.hidden_size)
        
        # Initialize with smaller std for stability
        self.y_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        
        self.net = TinyRecursiveNet(cfg)
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
        std = 0.02 / math.sqrt(eff_depth)  # Scaled initialization for deeper model
        
        for name, p in self.named_parameters():
            if p.dim() < 2: 
                continue
            if "o_proj" in name or "mlp.2" in name:
                nn.init.normal_(p, mean=0.0, std=std)
            elif "tok_emb" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
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
        
        # First T-1 recursions without gradients (memory efficient)
        with torch.no_grad():
            for _ in range(T - 1):
                y, z = self._latent_recursion(x, y, z, mask, n)
        
        # Final recursion with gradients
        y, z = self._latent_recursion(x, y, z, mask, n)
        
        logits = self.output_head(y)
        q = self.q_head(y).mean(dim=1).squeeze(-1)
        return y, z, logits, q

    def compute_loss(self, logits, q, labels):
        shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        
        # Dynamic halt target based on accuracy
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

# ── EMA for stability ────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].copy_(self.decay * self.shadow[n] + (1 - self.decay) * p.data)
    
    def apply(self, model):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow: 
                p.data.copy_(self.shadow[n])
    
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup: 
                p.data.copy_(self.backup[n])

# ── Streaming Dataset ────────────────────────────────────────────

class FinewebEduStreamingDataset(IterableDataset):
    def __init__(self, tokenizer, max_len, subset="sample-10BT", split="train"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Loading HuggingFaceFW/fineweb-edu ({subset}) in STREAMING mode...")
        
        self.ds = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split=split, streaming=True)
        self.ds = self.ds.shuffle(buffer_size=10000, seed=42)

    def __iter__(self):
        for item in self.ds:
            txt = item["text"] + self.tokenizer.eos_token
            enc = self.tokenizer(
                txt, 
                max_length=self.max_len, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            out = {k: v.squeeze(0) for k, v in enc.items()}
            out["labels"] = out["input_ids"].clone()
            out["labels"][out["attention_mask"] == 0] = -100
            yield out

# ── L40S Optimized Training Loop ─────────────────────────────────

def train_stream(model, ema, tokenizer, loader, optimizer, scheduler, cfg, scaler=None):
    model.train()
    optimizer.zero_grad()
    
    # Calculate total steps for progress bar
    total_batches = cfg.max_steps * cfg.gradient_accumulation_steps
    pbar = tqdm(loader, total=total_batches, desc="Training L40S")
    
    batch_lm_accum = 0.0
    optim_step = 0
    
    # Enable cuDNN auto-tuner for L40S
    torch.backends.cudnn.benchmark = True
    
    for i, batch in enumerate(pbar):
        # Move data to GPU efficiently
        ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        labels = batch["labels"].to(cfg.device, non_blocking=True)
        mask = batch["attention_mask"].to(cfg.device, non_blocking=True)
        
        # Embed tokens
        with torch.autocast(cfg.device, enabled=cfg.use_mixed_precision, dtype=torch.bfloat16):
            x, y, z = model.embed(ids)
        
        batch_lm = 0
        steps_taken = 0

        # Deep supervision loop
        for s in range(cfg.N_sup):
            if s == 0:
                y_in, z_in = y, z
            else:
                y_in = y.detach()
                z_in = z.detach()
            
            with torch.autocast(cfg.device, enabled=cfg.use_mixed_precision, dtype=torch.bfloat16):
                y_out, z_out, logits, q = model._deep_recursion(x, y_in, z_in, mask)
                loss, lm_val = model.compute_loss(logits, q, labels)
                scaled_loss = loss / cfg.gradient_accumulation_steps
            
            # Use GradScaler for mixed precision if needed
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            y, z = y_out.detach(), z_out.detach()
            batch_lm += lm_val
            steps_taken += 1

            # Early halt if q > 0.9 (model is confident)
            if torch.sigmoid(q).mean() > 0.9: 
                break
        
        batch_lm_accum += (batch_lm / steps_taken)
        
        # Optimizer step with gradient accumulation
        if (i + 1) % cfg.gradient_accumulation_steps == 0:
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            
            # Optimizer step with or without scaler
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            ema.update(model)
            optim_step += 1
            
            # Calculate metrics
            avg_step_loss = batch_lm_accum / cfg.gradient_accumulation_steps
            batch_lm_accum = 0.0
            
            try:
                ppl = math.exp(min(avg_step_loss, 20.0))
            except OverflowError:
                ppl = float('inf')
                
            # Update progress bar
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                step=optim_step, 
                loss=f"{avg_step_loss:.4f}", 
                ppl=f"{ppl:.1f}", 
                lr=f"{lr:.2e}",
                sup=steps_taken
            )
            
            # Save checkpoint
            if optim_step % cfg.save_interval == 0:
                print(f"\n\n--- Checkpoint @ Step {optim_step} ---")
                
                # Use EMA for generation
                ema.apply(model)
                
                # Generate sample
                test_prompt = "The basic principles of physics include"
                test_ids = tokenizer(test_prompt, return_tensors="pt")["input_ids"].to(cfg.device)
                with torch.no_grad():
                    gen = model.generate(test_ids, max_new_tokens=40, eos_token_id=tokenizer.eos_token_id)
                print(f"Sample: {tokenizer.decode(gen[0], skip_special_tokens=True)}")
                
                # Save checkpoint
                save_dir = os.path.join(cfg.output_dir, f"checkpoint-{optim_step}")
                os.makedirs(save_dir, exist_ok=True)
                
                ckpt_path = os.path.join(save_dir, "trm_model.pt")
                torch.save({
                    "step": optim_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_step_loss,
                    "config": vars(cfg)
                }, ckpt_path)
                
                # Save tokenizer
                tokenizer.save_pretrained(save_dir)
                
                print(f"✅ Checkpoint saved to: {ckpt_path}")
                print("-" * 30 + "\n")
                
                # Restore from EMA
                ema.restore(model)
                
            if optim_step >= cfg.max_steps:
                print("\n🎯 Reached max_steps. Training complete!")
                break

def main():
    cfg = TRMConfig()
    set_seed(cfg.seed)
    
    print(f"\n{'='*60}")
    print(f"🚀 TRM-LM Training on NVIDIA L40S (46GB VRAM)")
    print(f"{'='*60}")
    print(f"Model size: ~{(cfg.hidden_size * cfg.num_layers * 12) / 1e6:.1f}M parameters")
    print(f"Batch size: {cfg.batch_size} | Grad accum: {cfg.gradient_accumulation_steps}")
    print(f"Effective batch size: {cfg.batch_size * cfg.gradient_accumulation_steps} sequences")
    print(f"Tokens per step: {cfg.batch_size * cfg.gradient_accumulation_steps * cfg.max_seq_length:,}")
    print(f"Max steps: {cfg.max_steps}")
    print(f"{'='*60}\n")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    train_ds = FinewebEduStreamingDataset(
        tokenizer, 
        cfg.max_seq_length, 
        subset=cfg.dataset_subset, 
        split="train"
    )
    
    # Optimized dataloader for L40S
    loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2  # Prefetch batches
    )
    
    # Create model
    model = TRMForCausalLM(cfg, len(tokenizer)).to(cfg.device)
    
    # Print model size
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count / 1e6:.2f}M")
    print(f"Estimated VRAM usage: ~{(param_count * 4 * 2.5) / 1e9:.2f}GB\n")  # Rough estimate
    
    # Compile model if enabled (PyTorch 2.0+)
    if cfg.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Initialize EMA
    ema = EMA(model)
    
    # Optimizer with AdamW
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Cosine schedule with warmup
    sched = get_cosine_schedule_with_warmup(
        opt, 
        cfg.warmup_steps, 
        cfg.max_steps
    )
    
    # GradScaler for mixed precision (though bfloat16 doesn't need it)
    scaler = None
    if cfg.use_mixed_precision and cfg.mixed_precision_dtype == "float16":
        scaler = torch.cuda.amp.GradScaler()
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)
    
    # Start training
    print(f"\n🔥 Starting training on {cfg.device.upper()}...\n")
    train_stream(model, ema, tokenizer, loader, opt, sched, cfg, scaler)
    
    # Save final model
    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    # Use EMA for final save
    ema.apply(model)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(cfg)
    }, os.path.join(final_dir, "trm_model_final.pt"))
    tokenizer.save_pretrained(final_dir)
    
    print(f"\n✅ Training complete! Final model saved to {final_dir}")

if __name__ == "__main__":
    main()
