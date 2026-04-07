#!/usr/bin/env python3
import os, math, random, time, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from tqdm import tqdm

# ================= PERFORMANCE & T4 STABILITY =================
# Optimizes matrix multiplications for newer kernels on T4/L40S
torch.set_float32_matmul_precision('high') 

# Hugging Face & Tokenizer setup
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0" 
import tiktoken

# CRITICAL: Manually clear the GPU before starting to prevent "Zombie" OOMs
gc.collect()
torch.cuda.empty_cache()

@dataclass
class Config:
    """Hyperparameters for the LiQ-LM model."""
    hidden_size: int = 384     # Width of the model layers
    num_layers: int = 8        # Number of Transformer + Liquid blocks
    num_heads: int = 8         # Attention heads (hidden_size must be divisible by this)
    max_seq_length: int = 256  # Maximum tokens the model can see at once
    vocab_size: int = 50257    # GPT-2 standard vocabulary size
    dropout: float = 0.1       # Regularization to prevent overfitting
    batch_size: int = 4        # Number of samples per GPU micro-step
    grad_accum: int = 4        # Updates weights every (batch_size * grad_accum) samples
    num_epochs: int = 3        # Total passes through the data limit
    steps_per_epoch: int = 1000 # Optimizer updates per epoch
    warmup_steps: int = 500    # LR slowly increases for stability at start
    max_lr: float = 3e-4       # Peak learning rate
    min_lr: float = 3e-5       # Final learning rate after decay
    weight_decay: float = 0.1  # L2 regularization
    label_smoothing: float = 0.05 # Softens target labels for better generalization
    max_grad_norm: float = 1.0 # Clips gradients to prevent "exploding gradients"
    seed: int = 42             # Ensures reproducibility
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True       # Uses Float16 for faster training on T4
    num_workers: int = 2       # CPU threads for data loading
    log_interval: int = 10     # How often to update the tqdm bar
    eval_interval: int = 500   # How often to run text generation tests
    output_model: str = "liq_parallel_final.pt"

    @property
    def total_steps(self) -> int: return self.num_epochs * self.steps_per_epoch

# ========================== UTILITIES ==========================

def get_alibi_slopes(num_heads):
    """
    Generates slopes for ALiBi (Attention with Linear Biases).
    Each head gets a different 'penalty' for distance between tokens.
    """
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start * ratio**i for i in range(n)]
    
    # If head count is a power of 2, use standard geometric sequence
    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads))
    
    # Interpolate slopes if head count isn't a power of 2
    closest_power_of_2 = 2**math.floor(math.log2(num_heads))
    slopes_base = get_slopes_power_of_2(closest_power_of_2)
    slopes_extra = get_slopes_power_of_2(2 * closest_power_of_2)[1::2]
    return torch.tensor((slopes_base + slopes_extra)[:num_heads])

def parallel_associative_scan(a: torch.Tensor, b: torch.Tensor):
    """
    Core math for Parallel Liquid Recurrence.
    Converts a sequential ODE (h_t = a*h_{t-1} + b) into a parallel tree.
    Complexity: O(log L) instead of O(L).
    """
    L = a.shape[1]
    num_steps = int(math.log2(L))
    for i in range(num_steps):
        step = 2**i
        # Slices for 'current' and 'previous' elements in the prefix sum
        a_curr, b_curr = a[:, step:, :], b[:, step:, :]
        a_prev, b_prev = a[:, :-step, :], b[:, :-step, :]
        
        # Associative operator for linear recurrence
        new_b = a_curr * b_prev + b_curr
        new_a = a_curr * a_prev
        
        # Combine the calculated steps back into the sequence
        a = torch.cat([a[:, :step, :], new_a], dim=1)
        b = torch.cat([b[:, :step, :], new_b], dim=1)
    return b

# ========================== MODEL COMPONENTS ==========================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama style). Faster than standard LayerNorm."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class AlibiAttention(nn.Module):
    """
    Attention that uses distance-based biases instead of positional embeddings.
    Allows for faster training (it/s) and better length extrapolation.
    """
    def __init__(self, dim, num_heads, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout 
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.resid_drop = nn.Dropout(dropout)

        # Precompute ALiBi distances and biases
        slopes = get_alibi_slopes(num_heads)
        context_pos = torch.arange(max_seq_len).view(1, 1, max_seq_len)
        memory_pos = torch.arange(max_seq_len).view(1, max_seq_len, 1)
        relative_dist = memory_pos - context_pos
        alibi_bias = slopes.view(1, num_heads, 1, 1) * relative_dist.view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("alibi_bias", alibi_bias)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        
        # Multi-head split: [Batch, Head, Length, Head_Dim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Build causal mask + ALiBi bias
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        combined_mask = self.alibi_bias[:, :, :L, :L].masked_fill(mask, float("-inf"))

        # Use Scaled Dot Product Attention (SDPA) - Optimized for T4
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=combined_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False 
        )
        
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.resid_drop(self.out_proj(out))

class ParallelLiquidCell(nn.Module):
    """
    A Liquid Time-Constant (LTC) cell that can be processed in parallel.
    Simulates an ODE where the time-constant 'tau' is input-dependent.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.tau_proj = nn.Linear(dim, dim)
        self.input_proj = nn.Linear(dim, dim)
        self.log_dt = nn.Parameter(torch.zeros(dim)) # Learnable base time-step

    def forward(self, x, h0):
        # Calculate dynamic time-constant (tau)
        tau = F.softplus(self.tau_proj(x)) + 1e-3
        dt = F.softplus(self.log_dt)
        alpha = torch.clamp(dt / tau, 0.0, 1.0) # Mixing ratio
        
        candidate = torch.tanh(self.input_proj(x))
        a, b = 1.0 - alpha, alpha * candidate
        
        # Inject the initial hidden state into the first position
        b_init = a[:, :1, :] * h0.unsqueeze(1) + b[:, :1, :]
        b = torch.cat([b_init, b[:, 1:, :]], dim=1)
        
        # Run the parallel recurrence tree
        return parallel_associative_scan(a, b)

class TransformerBlock(nn.Module):
    """A hybrid layer containing both Attention (global) and Liquid (local/recurrent) cells."""
    def __init__(self, dim, num_heads, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.ln_attn = RMSNorm(dim)
        self.attn = AlibiAttention(dim, num_heads, max_seq_len, dropout)
        
        self.ln_liq = RMSNorm(dim)
        self.liquid = ParallelLiquidCell(dim)
        self.liq_gate = nn.Linear(dim, dim, bias=False) # Gating for Liquid output
        
        self.ln_ffn = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x, h0):
        # Attention sub-layer
        x = x + self.attn(self.ln_attn(x))
        
        # Liquid Recurrence sub-layer
        h_seq = self.liquid(self.ln_liq(x), h0)
        x = x + torch.sigmoid(self.liq_gate(h_seq)) * h_seq
        
        # Feed-forward sub-layer
        x = x + self.ffn(self.ln_ffn(x))
        return x, h_seq[:, -1] # Returns sequence and the final hidden state

class LiquidLM(nn.Module):
    """The full Language Model architecture."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.use_checkpointing = False # Can be toggled to True to save VRAM at cost of speed
        
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.emb_drop = nn.Dropout(cfg.dropout)
        
        # Stack of hybrid blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.max_seq_length, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        
        # Initial states for each layer's recurrence
        self.h0 = nn.ParameterList([nn.Parameter(torch.zeros(cfg.hidden_size)) for _ in range(cfg.num_layers)])
        
        self.ln_out = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        
        # Weight Tying: The embedding and output head share the same weights
        self.token_emb.weight = self.lm_head.weight 
        self._init_weights()

    def _init_weights(self):
        """Standard small-constant initialization for LLMs."""
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
            # Support for Gradient Checkpointing on T4
            if self.training and self.use_checkpointing:
                x, _ = checkpoint(block, x, h0_val, use_reentrant=False)
            else:
                x, _ = block(x, h0_val)
                
        logits = self.lm_head(self.ln_out(x))
        loss = None
        if labels is not None:
            # Shift logits/labels for Causal LM training (predict next token)
            shift_logits = logits[:, :-1, :].reshape(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, label_smoothing=self.cfg.label_smoothing, ignore_index=-100)
        return loss, logits

# ========================== DATA LOADING ==========================

class BPEStream(IterableDataset):
    """Streams tokenized text from HuggingFace FineWeb-Edu."""
    def __init__(self, cfg: Config, epoch: int = 0):
        self.cfg = cfg
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token
        self.epoch = epoch

    def __iter__(self):
        from datasets import load_dataset
        import datasets
        # Resilient network settings for long training runs
        datasets.config.STREAMING_READ_MAX_RETRIES = 100
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info else 0
        num_w = worker_info.num_workers if worker_info else 1
        
        samples_processed, max_retries = 0, 50
        while max_retries > 0:
            try:
                ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
                ds = ds.shuffle(buffer_size=1000, seed=self.cfg.seed + self.epoch * 1000 + wid)
                if num_w > 1: ds = ds.shard(num_shards=num_w, index=wid)
                
                for sample in ds:
                    tokens = self.enc.encode_ordinary(sample["text"])
                    if not tokens: continue
                    sl = self.cfg.max_seq_length
                    # Random slicing for variety; padding if too short
                    if len(tokens) > sl:
                        s = random.randint(0, len(tokens) - sl)
                        ids = tokens[s : s+sl]; lbl = ids
                    else:
                        pad = sl - len(tokens)
                        ids = tokens + [self.eot] * pad; lbl = tokens + [-100] * pad
                    yield {"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(lbl, dtype=torch.long)}
                    samples_processed += 1
                break
            except Exception:
                max_retries -= 1
                time.sleep(5)

# ========================== TRAINING HELPERS ==========================

def get_lr(cfg, step):
    """Cosine Learning Rate Schedule with Warmup."""
    if step < cfg.warmup_steps: return cfg.max_lr * step / cfg.warmup_steps
    progress = min((step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps), 1.0)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
@torch.compiler.disable # Prevent re-compilation freezes during generation
def evaluate_prompts(model, prompts, max_new=40):
    """Tests the model by generating text from fixed prompts."""
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    print(f"\n{'='*30}\nPROMPT EVALUATION\n{'='*30}")
    for p in prompts:
        ids = torch.tensor(enc.encode(p), dtype=torch.long, device=model.cfg.device).unsqueeze(0)
        for _ in range(max_new):
            _, logits = model(ids[:, -model.cfg.max_seq_length:])
            # Top-p/Temperature sampling for better variety
            next_id = torch.multinomial(F.softmax(logits[:, -1, :] / 0.8, -1), 1)
            ids = torch.cat([ids, next_id], dim=-1)
            if next_id.item() == 50256: break
        print(f"Prompt: {p}\nOutput: {enc.decode(ids[0].tolist())}\n{'-'*30}")
    model.train()

def train():
    cfg = Config()
    model = LiquidLM(cfg).to(cfg.device)
    
    # Print model statistics
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*40}\nMODEL INITIALIZED\nTotal Params: {total_p:,}\n{'='*40}\n")
    
    # Use torch.compile to fuse kernels for T4 speedup
    compiled_model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")
    
    global_step, micro_step = 0, 0
    test_prompts = ["The future of liquid AI is", "In space, the stars look like"]

    for epoch in range(1, cfg.num_epochs + 1):
        # We set num_workers to 1 or 2 to avoid deadlocks on T4 instances
        loader = DataLoader(BPEStream(cfg, epoch-1), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        pbar = tqdm(loader, total=cfg.steps_per_epoch * cfg.grad_accum, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            input_ids, labels = batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device)
            
            # Start of the optimization step for the compiler
            if torch.cuda.is_available(): torch.compiler.cudagraph_mark_step_begin()

            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, _ = compiled_model(input_ids, labels)
            
            # Scale loss for gradient accumulation
            scaler.scale(loss / cfg.grad_accum).backward()
            micro_step += 1
            
            # --- THE GLOBAL STEP (Updates weights every 'grad_accum' micro-steps) ---
            if micro_step % cfg.grad_accum == 0:
                global_step += 1
                
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                
                # Dynamic learning rate update
                for pg in opt.param_groups: pg["lr"] = get_lr(cfg, global_step)
                
                # Check for Periodic Evaluation
                if global_step % cfg.eval_interval == 0:
                    evaluate_prompts(model, test_prompts)
                
                if global_step % cfg.log_interval == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
                    
                # End epoch if we reached the step limit
                if global_step >= epoch * cfg.steps_per_epoch: break

    # Final save
    torch.save(model.state_dict(), cfg.output_model)

if __name__ == "__main__":
    train()
