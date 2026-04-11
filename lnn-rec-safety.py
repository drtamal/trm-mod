#!/usr/bin/env python3
import os, math, random, time, gc, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from tqdm import tqdm

# ================= COMMERCIAL SAFETY LAYER =================
class SafetyGuard:
    """Deterministically blocks high-risk prompts for commercial safety."""
    def __init__(self):
        # Add industry-standard forbidden topics here
        self.forbidden_keywords = [
            r"how to (make|build|create) a (bomb|weapon|explosive)",
            r"hack into", r"bypass (security|password)",
            r"generate (hate speech|racist|sexist)",
            r"private (address|phone|email) of"
        ]
        self.redaction_patterns = {
            "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        }

    def is_safe(self, text):
        for pattern in self.forbidden_keywords:
            if re.search(pattern, text.lower()):
                return False
        return True

    def redact_pii(self, text):
        """Replaces sensitive data with [REDACTED] for privacy compliance."""
        for label, pattern in self.redaction_patterns.items():
            text = re.sub(pattern, f"[{label}_REDACTED]", text)
        return text

# ================= PERFORMANCE & STABILITY =================
torch.set_float32_matmul_precision('high') 
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0" 
import tiktoken

gc.collect()
torch.cuda.empty_cache()

@dataclass
class Config:
    hidden_size: int = 384 # Your current 177M setup
    num_layers: int = 8
    num_heads: int = 8
    max_seq_length: int = 256
    vocab_size: int = 50257
    dropout: float = 0.1
    batch_size: int = 32 # Your current L40S setup
    grad_accum: int = 4
    num_epochs: int = 3
    steps_per_epoch: int = 50000 
    warmup_steps: int = 500
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
    eval_interval: int = 500
    output_model: str = "liq_commercial_v1.pt"
    # COMMERCIAL SYSTEM PROMPT
    system_prompt: str = "You are a professional, helpful, and safe SKAI-LNN Assistant. "

    @property
    def total_steps(self) -> int: return self.num_epochs * self.steps_per_epoch

# ... [Keep your RMSNorm, AlibiAttention, ParallelLiquidCell, TransformerBlock classes here] ...

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

# ... [Keep BPEStream and get_lr as they were] ...

@torch.no_grad()
def evaluate_prompts(model, prompts, max_new=60):
    """COMMERCIAL VERSION: Includes safety guard and system anchoring."""
    model.eval()
    guard = SafetyGuard()
    enc = tiktoken.get_encoding("gpt2")
    
    print(f"\n{'='*30}\nCOMMERCIAL PROMPT EVALUATION\n{'='*30}")
    
    for user_p in prompts:
        # 1. Deterministic Input Safety Check
        if not guard.is_safe(user_p):
            print(f"Prompt: {user_p}\nOutput: [REJECTED] Violation of Safety Policy.\n{'-'*30}")
            continue

        # 2. System Prompt Anchoring
        full_p = model.cfg.system_prompt + user_p
        ids = torch.tensor(enc.encode(full_p), dtype=torch.long, device=model.cfg.device).unsqueeze(0)
        
        # 3. Generation Loop
        for _ in range(max_new):
            _, logits = model(ids[:, -model.cfg.max_seq_length:])
            # Temperature 0.7 for commercial stability
            next_id = torch.multinomial(F.softmax(logits[:, -1, :] / 0.7, -1), 1)
            ids = torch.cat([ids, next_id], dim=-1)
            if next_id.item() == 50256: break
            
        output_text = enc.decode(ids[0].tolist())
        
        # 4. PII Redaction
        final_output = guard.redact_pii(output_text)
        print(f"Prompt: {user_p}\nOutput: {final_output}\n{'-'*30}")
    
    model.train()

# ... [Keep train() function as it was, just ensure it uses Config(hidden_size=1024)] ...
