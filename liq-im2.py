#!/usr/bin/env python3
"""
LiQ-LM v2: Liquid Time-Constant Language Model
================================================
Hybrid Transformer + LTC architecture with TRUE temporal recurrence.

Key improvements over liq-im1.py:
  1. LTC cells recur across sequence positions (not just across layers)
  2. Proper padding: uses -100 ignore_index so loss ignores pad tokens
  3. global_step counts optimizer steps, not micro-batches
  4. Gradient norm is logged
  5. num_workers > 0 for data loading
  6. cudnn.benchmark enabled for speed
  7. Narrower exception handling (no bare Exception for network errors)
  8. Per-layer learned initial hidden states
  9. Gated liquid injection into the residual stream
"""

import os, math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm

os.environ["TIKTOKEN_CACHE_DIR"] = "/data/aicoe_gpu/komal_paul_gtg/tiktoken_cache"
import tiktoken


# ========================== CONFIG ==========================
@dataclass
class Config:
    # Model
    hidden_size: int = 1024
    num_layers: int = 8
    num_heads: int = 8
    max_seq_length: int = 256
    vocab_size: int = 50257
    dropout: float = 0.1

    # Training
    batch_size: int = 4
    grad_accum: int = 4
    num_epochs: int = 3
    steps_per_epoch: int = 300_000   # optimizer steps per epoch
    warmup_steps: int = 2000
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    label_smoothing: float = 0.05
    max_grad_norm: float = 1.0

    # System
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    num_workers: int = 2
    log_interval: int = 10
    eval_interval: int = 20_000
    output_model: str = "liq_final.pt"

    @property
    def total_steps(self) -> int:
        return self.num_epochs * self.steps_per_epoch

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


# ========================== UTILITIES ==========================
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


@torch.jit.script
def sequential_scan(gates: torch.Tensor, inputs: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Linear recurrence via sequential scan:  h_t = gates_t * h_{t-1} + inputs_t
    Args: gates (B,L,D), inputs (B,L,D), h0 (B,D)  →  returns (B,L,D)
    """
    B, L, D = gates.shape
    outputs = torch.empty_like(inputs)
    h = h0
    for t in range(L):
        h = gates[:, t] * h + inputs[:, t]
        outputs[:, t] = h
    return outputs


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
        assert dim % num_heads == 0
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
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = self.attn_drop(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.resid_drop(self.out_proj(out))


class LiquidTimeConstantCell(nn.Module):
    """
    LTC cell with TRUE temporal recurrence across sequence positions.

    Discretized ODE:
        h_t = (1 - α_t) · h_{t-1} + α_t · f(x_t)
    where:
        α_t = clamp(softplus(Δt) / (softplus(τ(x_t)) + ε), 0, 1)
        f(x_t) = tanh(W_f · x_t)        — candidate state
        τ(x_t) = W_τ · x_t              — data-dependent time constant
        Δt     = per-dim learnable step

    Large τ → slow decay → long memory.  Small τ → fast adaptation.
    Recurrence computed via sequential_scan (JIT-compiled).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.tau_proj = nn.Linear(dim, dim)
        self.input_proj = nn.Linear(dim, dim)
        self.log_dt = nn.Parameter(torch.zeros(dim))  # → softplus ≈ 0.69

    def forward(self, x, h0):
        """x: (B,L,D), h0: (B,D) → (B,L,D) hidden states at every position."""
        tau = F.softplus(self.tau_proj(x)) + 1e-3
        dt = F.softplus(self.log_dt)
        alpha = torch.clamp(dt / tau, 0.0, 1.0)
        candidate = torch.tanh(self.input_proj(x))
        gates = 1.0 - alpha
        inputs = alpha * candidate
        return sequential_scan(gates, inputs, h0)


class TransformerBlock(nn.Module):
    """
    Pre-norm block:  Attention → LTC recurrence → FFN
    The liquid state provides a smooth recurrent "memory lane" that
    complements attention's direct-access pattern.
    """

    def __init__(self, dim, num_heads, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.ln_attn = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, dropout)

        self.ln_liq = RMSNorm(dim)
        self.liquid = LiquidTimeConstantCell(dim)
        self.liq_gate = nn.Linear(dim, dim, bias=False)  # gated injection

        self.ln_ffn = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x, h0):
        # 1. Self-attention
        x = x + self.attn(self.ln_attn(x))
        # 2. Liquid temporal recurrence
        h_seq = self.liquid(self.ln_liq(x), h0)
        x = x + torch.sigmoid(self.liq_gate(h_seq)) * h_seq
        # 3. FFN
        x = x + self.ffn(self.ln_ffn(x))
        return x, h_seq[:, -1]


# ========================== FULL MODEL ==========================
class LiquidLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.emb_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.max_seq_length, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        # Per-layer initial hidden state
        self.h0 = nn.ParameterList([
            nn.Parameter(torch.zeros(1, cfg.hidden_size))
            for _ in range(cfg.num_layers)
        ])
        self.ln_out = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        x = self.emb_drop(self.token_emb(input_ids))
        for i, block in enumerate(self.blocks):
            x, _ = block(x, self.h0[i].expand(B, -1))
        logits = self.lm_head(self.ln_out(x))

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous().view(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(
                shift_logits, shift_labels,
                label_smoothing=self.cfg.label_smoothing,
                ignore_index=-100,
            )
        return loss, logits


# ========================== DATASET ==========================
class BPEStream(IterableDataset):
    """Streaming tokenizer with proper padding (labels=-100 on pad positions)."""

    def __init__(self, cfg: Config, epoch: int = 0):
        self.cfg = cfg
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token
        self.epoch = epoch

    def __iter__(self):
        from datasets import load_dataset

        # Shard across DataLoader workers to avoid duplicate data
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        attempt, max_retries, base_delay = 0, 10, 5
        while attempt < max_retries:
            try:
                ds = load_dataset(
                    "HuggingFaceFW/fineweb-edu", name="sample-10BT",
                    split="train", streaming=True,
                )
                # Unique seed per worker so each shuffles differently
                seed = self.cfg.seed + self.epoch * 1000 + attempt + worker_id
                ds = ds.shuffle(buffer_size=10_000, seed=seed)

                # Each worker takes a disjoint slice of the stream
                if num_workers > 1:
                    ds = ds.shard(num_shards=num_workers, index=worker_id)

                print(f"Stream ready (epoch {self.epoch + 1}, attempt {attempt + 1}, worker {worker_id}/{num_workers})")

                for sample in ds:
                    try:
                        tokens = self.enc.encode_ordinary(sample["text"])
                        if not tokens:
                            continue
                        sl = self.cfg.max_seq_length
                        if len(tokens) > sl:
                            s = random.randint(0, len(tokens) - sl)
                            tokens = tokens[s : s + sl]
                            ids, lbl = tokens, tokens
                        else:
                            pad = sl - len(tokens)
                            ids = tokens + [self.eot] * pad
                            lbl = tokens + [-100] * pad

                        yield {
                            "input_ids": torch.tensor(ids, dtype=torch.long),
                            "labels": torch.tensor(lbl, dtype=torch.long),
                        }
                    except Exception as e:
                        print(f"\nSkipping bad sample: {e}")
                        continue
                break
            except (ConnectionError, TimeoutError, OSError) as e:
                attempt += 1
                wait = base_delay * (2 ** (attempt - 1))
                print(f"\n[NET ERROR] {e} — retry {attempt}/{max_retries} in {wait}s")
                time.sleep(wait)

        if attempt >= max_retries:
            raise ConnectionError("Stream failed after max retries.")


# ========================== GENERATION ==========================
@torch.no_grad()
def generate(model, prompt, max_new=50, temperature=0.8, top_k=40):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=model.cfg.device).unsqueeze(0)
    for _ in range(max_new):
        ctx = ids[:, -model.cfg.max_seq_length :]
        _, logits = model(ctx)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        ids = torch.cat([ids, torch.multinomial(F.softmax(logits, -1), 1)], dim=-1)
    return enc.decode(ids[0].tolist())


def interactive_eval(model):
    print("\n" + "=" * 50 + "\nINTERACTIVE EVAL — type 'quit' to exit\n" + "=" * 50)
    while True:
        try:
            p = input("\nPrompt > ")
        except (EOFError, KeyboardInterrupt):
            break
        if p.strip().lower() in ("quit", "exit", "q"):
            break
        print(f"Output: {generate(model, p)}")


# ========================== LR SCHEDULE ==========================
def get_lr(cfg: Config, step: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * step / cfg.warmup_steps
    progress = min((step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1), 1.0)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


# ========================== TRAINING ==========================
def train():
    cfg = Config()
    seed_everything(cfg.seed)
    model = LiquidLM(cfg).to(cfg.device)

    # Optimizer — exclude norms, biases, liquid scalars from weight decay
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(k in name for k in ("ln", "bias", "h0", "log_dt")):
            no_decay.append(p)
        else:
            decay.append(p)

    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.max_lr, betas=(0.9, 0.95), eps=1e-8)

    use_amp = cfg.use_amp and cfg.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}\n  LiQ-LM v2 Training\n{'='*60}")
    print(f"  Params:          {total_p:,} ({train_p:,} trainable)")
    print(f"  AMP:             {use_amp} ({amp_dtype})")
    print(f"  Eff. batch:      {cfg.effective_batch_size}")
    print(f"  Steps/epoch:     {cfg.steps_per_epoch:,} (optimizer steps)")
    print(f"  Total steps:     {cfg.total_steps:,}")
    print(f"{'='*60}\n")

    global_step = 0
    best_loss = float("inf")
    running_loss = 0.0
    loss_count = 0

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n{'='*60}\n  EPOCH {epoch}/{cfg.num_epochs}\n{'='*60}")
        loader = DataLoader(
            BPEStream(cfg, epoch - 1), batch_size=cfg.batch_size,
            num_workers=cfg.num_workers, pin_memory=True,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )
        model.train()
        opt.zero_grad(set_to_none=True)
        micro = 0
        epoch_loss, epoch_n = 0.0, 0
        pbar = tqdm(loader, total=cfg.steps_per_epoch * cfg.grad_accum, desc=f"Epoch {epoch}")

        for batch in pbar:
            ids = batch["input_ids"].to(cfg.device, non_blocking=True)
            lbl = batch["labels"].to(cfg.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss, _ = model(ids, lbl)
                scaled = loss / cfg.grad_accum

            scaler.scale(scaled).backward()
            micro += 1
            lv = loss.item()
            running_loss += lv
            loss_count += 1
            epoch_loss += lv
            epoch_n += 1

            if micro % cfg.grad_accum == 0:
                global_step += 1
                lr = get_lr(cfg, global_step)
                for pg in opt.param_groups:
                    pg["lr"] = lr

                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                if global_step % cfg.log_interval == 0:
                    avg = running_loss / max(loss_count, 1)
                    pbar.set_postfix(loss=f"{lv:.4f}", avg=f"{avg:.4f}",
                                     gnorm=f"{gn:.2f}", lr=f"{lr:.2e}", step=global_step)

                if global_step > 0 and global_step % cfg.eval_interval == 0:
                    avg = running_loss / max(loss_count, 1)
                    print(f"\n[Eval @ step {global_step}] Avg Loss: {avg:.4f} | ‖∇‖: {gn:.2f}")
                    print(f"  → {generate(model, 'The nature of')}")
                    print(f"  → {generate(model, 'In recent years')}")
                    if avg < best_loss:
                        best_loss = avg
                        torch.save(model.state_dict(), "liq_best.pt")
                        print(f"  ★ Best model saved (loss: {best_loss:.4f})")
                    running_loss, loss_count = 0.0, 0
                    model.train()

                if global_step >= epoch * cfg.steps_per_epoch:
                    break

        avg_ep = epoch_loss / max(epoch_n, 1)
        print(f"\n--- Epoch {epoch} done — Avg Loss: {avg_ep:.4f} ---")
        print(f"  → {generate(model, 'Deep within the universe')}")
        torch.save({
            "epoch": epoch, "global_step": global_step,
            "model": model.state_dict(), "opt": opt.state_dict(),
            "scaler": scaler.state_dict(), "best_loss": best_loss, "cfg": cfg,
        }, f"liq_epoch_{epoch}.pt")
        print(f"  Checkpoint: liq_epoch_{epoch}.pt")

    torch.save(model.state_dict(), cfg.output_model)
    print(f"\n{'='*60}\n  Training complete!\n  Final: {cfg.output_model}\n"
          f"  Best:  liq_best.pt (loss: {best_loss:.4f})\n  Steps: {global_step:,}\n{'='*60}")
    return model


if __name__ == "__main__":
    trained = train()
    interactive_eval(trained)
