import os, math, random, time, torch, re
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# --- Configuration ---
@dataclass
class HybridConfig:
    hidden_size: int = 512  # Optimized for 8GB VRAM
    num_layers: int = 1     # We rely on recursion for "depth"
    num_heads: int = 8
    intermediate_size: int = 1024
    max_seq_length: int = 256
    
    # Recurrence & Liquid Params
    n_latent: int = 2       # Inner loops
    T_recurse: int = 2      # Outer loops
    N_sup: int = 1          # Supervision steps (Set to 1 for 8GB VRAM stability)
    
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    dataset_subset: str = "sample-10BT"
    batch_size: int = 2     # Keep low for 8GB
    grad_accum: int = 16    # High accum to compensate for small batch
    lr: float = 4e-4
    
    # Epoch settings
    num_epochs: int = 3
    steps_per_epoch: int = 100000
    eval_interval: int = 50000
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./hybrid_model"

# --- Components ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight

class ClosedFormLiquidBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = dim // cfg.num_heads
        
        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Liquid Gates (The CfC Magic)
        self.f_gate = nn.Linear(dim, dim) # Time/Sensitivity
        self.g_target = nn.Linear(dim, dim) # Goal state
        
        self.norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, cfg.intermediate_size),
            nn.SiLU(),
            nn.Linear(cfg.intermediate_size, dim)
        )

    def forward(self, x, mask=None):
        B, L, D = x.shape
        # 1. Attention as context generator
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None: attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        context = (attn @ v).transpose(1, 2).reshape(B, L, D)
        
        # 2. Liquid Update Formula
        ff = torch.sigmoid(self.f_gate(context))
        gg = torch.tanh(self.g_target(context))
        h = x * (1 - ff) + gg * ff
        
        # 3. Residual MLP
        return h + self.mlp(self.norm(h))

class HybridRecursiveNet(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(vocab_size, cfg.hidden_size)
        self.block = ClosedFormLiquidBlock(cfg)
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.output_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
        self.output_head.weight = self.emb.weight # Weight tying
        
        # Init Latents
        self.y_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)

    def forward(self, ids, labels=None):
        B, L = ids.shape
        x = self.emb(ids)
        y = self.y_init.expand(B, L, -1)
        z = self.z_init.expand(B, L, -1)
        
        # Create Causal Mask
        mask = torch.triu(torch.full((L, L), float("-inf"), device=ids.device), 1)

        # Recurrent Latent Logic
        for _ in range(self.cfg.T_recurse):
            for _ in range(self.cfg.n_latent):
                # Gradient Checkpointing here saves massive VRAM
                z = torch.utils.checkpoint.checkpoint(self.block, (x + y + z), mask, use_reentrant=False)
            y = self.block(y + z, mask)
        
        logits = self.output_head(self.final_norm(y))
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generates new tokens iteratively."""
        self.eval() 
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        self.train() 
        return idx

# --- Data & Utilities ---
class FastFineweb(IterableDataset):
    def __init__(self, tokenizer, cfg, skip_steps=0):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.skip_steps = skip_steps
        
    def __iter__(self):
        local_yielded = 0
        retry_count = 0
        
        while True:
            try:
                # 1. Attempt to connect to the dataset
                ds = load_dataset(
                    "HuggingFaceFW/fineweb-edu", 
                    name=self.cfg.dataset_subset, 
                    split="train", 
                    streaming=True
                )
                
                # 2. Fast-forward past globally skipped steps AND locally yielded steps
                total_skip = self.skip_steps + local_yielded
                if total_skip > 0:
                    ds = ds.skip(total_skip)
                    
                # 3. Stream the data
                for x in ds:
                    enc = self.tokenizer(
                        x["text"] + self.tokenizer.eos_token, 
                        max_length=self.cfg.max_seq_length, 
                        padding="max_length", 
                        truncation=True, 
                        return_tensors="pt"
                    )
                    
                    item = {k: v.squeeze(0) for k, v in enc.items()}
                    item["labels"] = item["input_ids"].clone()
                    item["labels"][item["attention_mask"] == 0] = -100
                    
                    yield item
                    
                    # Successfully yielded, update trackers
                    local_yielded += 1
                    retry_count = 0 # Reset retry timer on a successful stream
                    
                # If the loop finishes naturally (end of dataset), break the infinite loop
                break
                
            except KeyboardInterrupt:
                # Allow you to manually stop the script without triggering a "retry"
                print("\n[!] Training interrupted by user.")
                raise
                
            except Exception as e:
                # Catch connection drops, HTTP errors, read timeouts, etc.
                retry_count += 1
                
                # Exponential backoff: wait 2s, 4s, 8s, up to a max of 60 seconds
                wait_time = min(2 ** retry_count, 60) 
                
                total_skip = self.skip_steps + local_yielded
                print(f"\n[!] Dataset connection error: {e}")
                print(f"[*] Reconnecting in {wait_time} seconds... (Will resume stream at offset {total_skip})")
                
                time.sleep(wait_time)

def main():
    cfg = HybridConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = HybridRecursiveNet(cfg, len(tokenizer)).to(cfg.device)
    
    # --- Model Size Calculation ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*40}")
    print(f"[*] Initializing Hybrid Recursive Net")
    print(f"[*] Total Trainable Parameters: {total_params / 1e6:.2f} M")
    print(f"{'='*40}\n")
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    
    # Scheduler needs to know the TOTAL steps across all epochs
    total_training_steps = cfg.num_epochs * cfg.steps_per_epoch
    sched = get_cosine_schedule_with_warmup(opt, 500, total_training_steps)
    
    eval_prompt_text = "The most important discovery in modern physics is"
    global_step = 0
    
    # --- Outer Epoch Loop ---
    for epoch in range(cfg.num_epochs):
        print(f"\n{'-'*40}")
        print(f"[*] Starting Epoch {epoch + 1} / {cfg.num_epochs}")
        print(f"{'-'*40}\n")
        
        # Re-initialize the DataLoader to start the dataset stream from the beginning for the new epoch
        loader = DataLoader(FastFineweb(tokenizer, cfg), batch_size=cfg.batch_size)
        pbar = tqdm(enumerate(loader), total=cfg.steps_per_epoch, desc=f"Epoch {epoch + 1}")
        
        model.train()
        
        # --- Inner Step Loop ---
        for i, batch in pbar:
            input_ids = batch["input_ids"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)
            
            with torch.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", dtype=torch.bfloat16):
                _, loss = model(input_ids, labels)
                loss = loss / cfg.grad_accum
                
            loss.backward()
            
            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                
                # Update progress bar with loss and current learning rate
                current_lr = sched.get_last_lr()[0]
                pbar.set_postfix(loss=loss.item() * cfg.grad_accum, lr=f"{current_lr:.2e}")
            
            global_step += 1
                
            # --- Evaluation Generation Hook ---
            # Triggered based on global_step so it doesn't break across epoch boundaries
            if global_step % cfg.eval_interval == 0:
                print(f"\n\n--- Global Step {global_step} Evaluation ---")
                eval_inputs = tokenizer.encode(eval_prompt_text, return_tensors="pt").to(cfg.device)
                
                generated_ids = model.generate(eval_inputs, max_new_tokens=50, top_k=10)
                output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                print(f"Prompt:  {eval_prompt_text}")
                print(f"Output:  {output_text}")
                print(f"--------------------------------------\n")
                pbar.refresh()
                
            # Stop the current epoch once we hit the specified steps
            if (i + 1) >= cfg.steps_per_epoch:
                break

if __name__ == "__main__":
    main()
