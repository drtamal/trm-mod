#!/usr/bin/env python3
import os, math, random, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from torchdiffeq import odeint

@dataclass
class LNNConfig:
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 688
    max_seq_length: int = 256
    time_steps: int = 4
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    n_latent: int = 2
    T_recurse: int = 2
    N_sup: int = 2
    ema_decay: float = 0.999
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    dataset_subset: str = "sample-10BT"
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: int = 100000
    save_interval: int = 10000
    log_interval: int = 100
    output_dir: str = "./output_lnn"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def get_latest_checkpoint(output_dir):
    return None, 0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight

class LiquidGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_constant = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
    def forward(self, x):
        tc = torch.sigmoid(self.time_constant)
        return tc * self.gate(x)

class LiquidCell(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.U = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(dim))
        self.liquid_gate = LiquidGate(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, dim)
    def ode_step(self, t, x, input_features):
        dxdt = -F.relu(x @ self.W.T) + input_features @ self.U.T + self.b
        return self.liquid_gate(x) * dxdt
    def forward(self, x, input_features):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(input_features).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(input_features).view(B, L, self.num_heads, self.head_dim)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        attn_features = (attn @ v).reshape(B, L, -1)
        ode_features = x + attn_features
        t = torch.linspace(0, 1, 8, device=x.device)
        x_out = odeint(lambda t, s: self.ode_step(t, s, ode_features), x, t, method="euler")[-1]
        return self.output_proj(self.norm(x_out))

class LiquidBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.liquid_cell = LiquidCell(cfg.hidden_size, cfg.num_heads)
        self.mlp = nn.Sequential(nn.Linear(cfg.hidden_size, cfg.intermediate_size), nn.GELU(), nn.Linear(cfg.intermediate_size, cfg.hidden_size))
        self.mlp_norm = RMSNorm(cfg.hidden_size)
    def forward(self, x):
        return x + self.mlp(self.mlp_norm(x + self.liquid_cell(x, x)))

class LNNForCausalLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_seq_length, cfg.hidden_size) * 0.02)
        self.y_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.layers = nn.ModuleList([LiquidBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.output_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.output_head.weight = self.tok_emb.weight
        self.q_head = nn.Sequential(RMSNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, 1))
    def forward(self, ids, labels=None, mask=None):
        B, L = ids.shape
        L = min(L, self.cfg.max_seq_length)
        x = self.tok_emb(ids[:, :L]) + self.pos_emb[:, :L, :]
        y = self.y_init.expand(B, L, -1)
        z = self.z_init.expand(B, L, -1)
        total_loss = 0
        for _ in range(self.cfg.N_sup):
            for _ in range(self.cfg.T_recurse):
                for _ in range(self.cfg.n_latent):
                    curr_z = x + y + z
                    for layer in self.layers: curr_z = layer(curr_z)
                    z = z + self.final_norm(curr_z)  # Residual update instead of overwrite
                y = self.final_norm(sum(layer(y) for layer in self.layers))
            logits = self.output_head(y)
            q = self.q_head(y).mean(dim=1).squeeze(-1)
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
                shift_labels = labels[:, 1:L].contiguous().view(-1)
                lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                with torch.no_grad():
                    acc = (logits[:, :-1, :].argmax(-1) == labels[:, 1:L]).float().mean()
                    halt_target = (acc > 0.8).float().expand_as(q)
                total_loss += lm_loss + 0.1 * F.binary_cross_entropy_with_logits(q, halt_target)
        return total_loss / self.cfg.N_sup, logits

class FinewebStream(IterableDataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer, self.cfg = tokenizer, cfg
    def __iter__(self):
        while True:
            try:
                ds = load_dataset("HuggingFaceFW/fineweb-edu", name=self.cfg.dataset_subset, split="train", streaming=True)
                for x in ds.shuffle(buffer_size=1000, seed=self.cfg.seed):
                    e = self.tokenizer(x["text"]+self.tokenizer.eos_token, max_length=self.cfg.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
                    out = {k: v.squeeze(0) for k, v in e.items()}
                    out["labels"] = out["input_ids"].clone()
                    out["labels"][out["attention_mask"] == 0] = -100
                    yield out
                break
            except Exception as e:
                print(f"Stream error: {e}. Retrying...")
                time.sleep(5)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].copy_(self.decay * self.shadow[n] + (1 - self.decay) * p.data)

def main():
    cfg = LNNConfig()
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = LNNForCausalLM(cfg, len(tokenizer)).to(cfg.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    sched = get_cosine_schedule_with_warmup(opt, 100, cfg.max_steps)
    loader = DataLoader(FinewebStream(tokenizer, cfg), batch_size=cfg.batch_size, num_workers=0)
    pbar = tqdm(loader, total=cfg.max_steps)
    optim_step, accum_loss = 0, 0
    model.train()
    for batch in pbar:
        input_ids = batch["input_ids"].to(cfg.device)
        labels = batch["labels"].to(cfg.device)
        if cfg.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = model(input_ids, labels)
        else:
            loss, _ = model(input_ids, labels)
        loss.backward()
        
        # Check for NaN/Inf in loss and gradients
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[Step {optim_step}] Loss is NaN/Inf - skipping")
            opt.zero_grad()
            continue
        
        has_nan_grad = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        has_inf_grad = any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)
        if has_nan_grad or has_inf_grad:
            print(f"\n[Step {optim_step}] NaN/Inf gradients detected - skipping")
            opt.zero_grad()
            continue
        
        # Log gradient stats periodically for debugging
        if optim_step % cfg.log_interval == 0:
            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            avg_grad = np.mean(grad_norms)
            max_grad = np.max(grad_norms)
            if optim_step % (cfg.log_interval * 10) == 0:
                print(f"\n[Step {optim_step}] Grad stats - avg: {avg_grad:.4f}, max: {max_grad:.4f}, loss: {loss.item():.4f}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Fixed accumulation - only reset after accumulation steps
        accum_loss += loss.item()
        
        if (optim_step + 1) % cfg.gradient_accumulation_steps == 0:
            pbar.set_postfix(step=optim_step + 1, loss=f"{accum_loss/cfg.gradient_accumulation_steps:.4f}")
            accum_loss = 0
        
        opt.step(); sched.step(); opt.zero_grad(); ema.update(model)
        optim_step += 1
        if optim_step % cfg.save_interval == 0:
            sd = os.path.join(cfg.output_dir, f"checkpoint-{optim_step}")
            os.makedirs(sd, exist_ok=True)
            torch.save({"step": optim_step, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sched.state_dict(), "ema_shadow": ema.shadow}, os.path.join(sd, "lnn_model.pt"))
            print(f"\nSaved checkpoint-{optim_step}")
        if optim_step >= cfg.max_steps: break

if __name__ == "__main__":
    main()
