import os, json, gc, torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import tiktoken
import math, random, time, gc
from torch.utils.data import IterableDataset, DataLoader, Dataset, Subset
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
import matplotlib.pyplot as plt
# --- LOAD YOUR EXISTING CONFIG AND MODEL CLASSES ---
# (Ensure LiquidLM, RMSNorm, etc., are defined in your environment)

@dataclass
class Config:
    """Hyperparameters for the LiQ-LM model."""
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    max_seq_length: int = 128
    vocab_size: int = 50257
    dropout: float = 0.1
    batch_size: int = 4
    grad_accum: int = 4
    num_epochs: int = 3
    steps_per_epoch: int = 10000 
    warmup_steps: int = 2000
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
    eval_interval: int = 5000
    val_split: float = 0.1  # <--- THIS WAS MISSING
    output_model: str = "liq_parallel_final.pt"

    @property
    def total_steps(self) -> int: return self.num_epochs * self.steps_per_epoch

# ... [Utilities: get_alibi_slopes, parallel_associative_scan remain unchanged] ...

def get_alibi_slopes(num_heads):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start * ratio**i for i in range(n)]
    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads))
    closest_power_of_2 = 2**math.floor(math.log2(num_heads))
    slopes_base = get_slopes_power_of_2(closest_power_of_2)
    slopes_extra = get_slopes_power_of_2(2 * closest_power_of_2)[1::2]
    return torch.tensor((slopes_base + slopes_extra)[:num_heads])

def parallel_associative_scan(a: torch.Tensor, b: torch.Tensor):
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

# ... [Model Components: RMSNorm, AlibiAttention, ParallelLiquidCell, TransformerBlock, LiquidLM remain unchanged] ...

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class AlibiAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout 
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.resid_drop = nn.Dropout(dropout)
        slopes = get_alibi_slopes(num_heads)
        context_pos = torch.arange(max_seq_len).view(1, 1, max_seq_len)
        memory_pos = torch.arange(max_seq_len).view(1, max_seq_len, 1)
        relative_dist = memory_pos - context_pos
        alibi_bias = slopes.view(1, num_heads, 1, 1) * relative_dist.view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("alibi_bias", alibi_bias)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        combined_mask = self.alibi_bias[:, :, :L, :L].masked_fill(mask, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=combined_mask, 
                                            dropout_p=self.dropout_p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, D)
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
        a, b = 1.0 - alpha, alpha * candidate
        b_init = a[:, :1, :] * h0.unsqueeze(1) + b[:, :1, :]
        b = torch.cat([b_init, b[:, 1:, :]], dim=1)
        return parallel_associative_scan(a, b)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.ln_attn = RMSNorm(dim)
        self.attn = AlibiAttention(dim, num_heads, max_seq_len, dropout)
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
        self.use_checkpointing = False
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
            x, _ = block(x, h0_val)
        logits = self.lm_head(self.ln_out(x))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, label_smoothing=self.cfg.label_smoothing, ignore_index=-100)
        return loss, logits

def count_parameters(model):
    """Calculates model size for the Grant Report."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#============================================================
# ================= DATASET =================
class SyntheticARCDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        print(f"📖 Loading synthetic data from {file_path}...")
        self.data = [json.loads(line) for line in open(file_path, "r")]
        self.tokenizer, self.max_len = tokenizer, max_len
    def grid_to_str(self, grid): return " ".join([" ".join([str(v) for v in row]) for row in grid])
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = self.tokenizer.encode(f"In: {self.grid_to_str(item['input'])} Out: {self.grid_to_str(item['output'])}")
        if len(tokens) < self.max_len: tokens += [self.tokenizer.eot_token] * (self.max_len - len(tokens))
        return torch.tensor(tokens[:self.max_len])

# ================= RESEARCH PIPELINE =================
def run_research_pipeline(model_path, data_path, cfg):
    tokenizer = tiktoken.get_encoding("gpt2")
    model = LiquidLM(cfg).to(cfg.device)
    
    # 1. MODEL SIZE PRINTING
    total_params = count_parameters(model)
    print("="*40)
    print(f"📊 SKAI-LNN ARCHITECTURE REPORT")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size: {total_params / 1e6:.2f}M")
    #print(f"Target Hardware: Asus Dual RX 9060 XT (16GB)")
    print("="*40)
    
    if os.path.exists(model_path):
        print(f"🔄 Resuming from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=cfg.device), strict=False)
    
    # 2. DATA SPLITTING
    full_dataset = SyntheticARCDataset(data_path, tokenizer, max_len=cfg.max_seq_length)
    train_size = int((1 - cfg.val_split) * len(full_dataset))
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=cfg.batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    
    train_losses, val_losses = [], []
    
    # 3. EPOCH LOOP
    for epoch in range(cfg.num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        epoch_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Train]")
        
        for batch in pbar:
            batch = batch.to(cfg.device)
            optimizer.zero_grad()
            loss, _ = model(batch, labels=batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            epoch_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # --- EVALUATION PHASE (After each epoch) ---
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(cfg.device)
                loss, _ = model(batch, labels=batch)
                epoch_val_loss += loss.item()
        
        avg_val = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val)
        
        print(f"📈 Result - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        scheduler.step()
        torch.save(model.state_dict(), model_path)

    # 4. LOSS PLOTTING
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, cfg.num_epochs + 1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, cfg.num_epochs + 1), val_losses, 'r--', label='Val Loss')
    plt.title(f"SKAI-LNN 7.4M Phase 4 Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve.png")
    print("💾 Research artifacts (plot/model) saved.")

if __name__ == "__main__":
    config = Config()
    run_research_pipeline(config.output_model, "synthetic_arc.jsonl", config)
