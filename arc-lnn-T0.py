import os, math, json, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm
import tiktoken
import arckit

# ================= PERFORMANCE & CONFIG =================
torch.set_float32_matmul_precision('high')

@dataclass
class Config:
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    max_seq_length: int = 128
    vocab_size: int = 50257
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ================= CORE LNN COMPONENTS =================

def get_alibi_slopes(num_heads):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start * ratio**i for i in range(n)]
    if math.log2(num_heads).is_integer(): return torch.tensor(get_slopes_power_of_2(num_heads))
    closest = 2**math.floor(math.log2(num_heads))
    return torch.tensor((get_slopes_power_of_2(closest) + get_slopes_power_of_2(2*closest)[1::2])[:num_heads])

def parallel_associative_scan(a, b):
    L = a.shape[1]
    for i in range(int(math.log2(L))):
        step = 2**i
        a_curr, b_curr = a[:, step:, :], b[:, step:, :]
        a_prev, b_prev = a[:, :-step, :], b[:, :-step, :]
        new_b = a_curr * b_prev + b_curr
        new_a = a_curr * a_prev
        a = torch.cat([a[:, :step, :], new_a], dim=1)
        b = torch.cat([b[:, :step, :], new_b], dim=1)
    return b

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class AlibiAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv, self.out_proj = nn.Linear(dim, 3*dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.register_buffer("slopes", get_alibi_slopes(num_heads))
    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = [t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)]
        dist = torch.arange(L, device=x.device).view(1, L, 1) - torch.arange(L, device=x.device).view(1, 1, L)
        bias = self.slopes.view(1, self.num_heads, 1, 1) * dist.view(1, 1, L, L)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.masked_fill(mask, float("-inf")))
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))

class ParallelLiquidCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tau_proj, self.input_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.log_dt = nn.Parameter(torch.zeros(dim))
    def forward(self, x, h0):
        alpha = torch.clamp(F.softplus(self.log_dt) / (F.softplus(self.tau_proj(x)) + 1e-3), 0.0, 1.0)
        a, b = 1.0 - alpha, alpha * torch.tanh(self.input_proj(x))
        b[:, :1, :] = a[:, :1, :] * h0.unsqueeze(1) + b[:, :1, :]
        return parallel_associative_scan(a, b)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=256, dropout=0.0):
        super().__init__()
        self.attn, self.liquid = AlibiAttention(dim, num_heads), ParallelLiquidCell(dim)
        self.ln1, self.ln2, self.ln3 = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
        self.liq_gate = nn.Linear(dim, dim, bias=False)
    def forward(self, x, h0):
        x = x + self.attn(self.ln1(x))
        h = self.liquid(self.ln2(x), h0)
        x = x + torch.sigmoid(self.liq_gate(h)) * h + self.ffn(self.ln3(x))
        return x, h[:, -1]

class LiquidLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.token_emb = cfg, nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.max_seq_length, cfg.dropout) for _ in range(cfg.num_layers)])
        self.h0 = nn.ParameterList([nn.Parameter(torch.zeros(cfg.hidden_size)) for _ in range(cfg.num_layers)])
        self.ln_out, self.lm_head = RMSNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight
    def forward(self, ids, labels=None):
        x = self.token_emb(ids)
        for i, block in enumerate(self.blocks): x, _ = block(x, self.h0[i].expand(ids.size(0), -1))
        logits = self.lm_head(self.ln_out(x))
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, self.cfg.vocab_size), labels[:, 1:].reshape(-1)) if labels is not None else None
        return loss, logits

# ========================== ARC LOGIC ==========================

class ARCEncoder:
    def grid_to_tokens(self, grid):
        return " ".join([" ".join([f"r{r}c{c}v{v}" for c, v in enumerate(row)]) for r, row in enumerate(grid)])

    def task_to_prompt(self, task):
        p = "ARC:\n"
        for pair in task['train']:
            p += f"In: {self.grid_to_tokens(pair['input'])} => Out: {self.grid_to_tokens(pair['output'])}\n"
        p += f"Test: {self.grid_to_tokens(task['test'][0]['input'])} => Out:"
        return p

class ARCEvaluator:
    def __init__(self, model_path, cfg):
        self.cfg, self.device, self.tokenizer = cfg, cfg.device, tiktoken.get_encoding("gpt2")
        self.encoder = ARCEncoder()
        self.model = LiquidLM(cfg).to(self.device)
        sd = torch.load(model_path, map_location=self.device)
        for i in range(cfg.num_layers):
            bk, sk = f"blocks.{i}.attn.alibi_bias", f"blocks.{i}.attn.slopes"
            if bk in sd: sd[sk] = sd[bk][0, :, 1, 0]; del sd[bk]
        self.model.load_state_dict(sd, strict=False)
        self.valid_tokens = [self.tokenizer.encode(c)[0] for c in "rcv0123456789| \n"]

    def test_time_train(self, prompt, steps=20):
        self.model.train()
        with torch.enable_grad():
            opt = torch.optim.AdamW(self.model.parameters(), lr=5e-6)
            ids = torch.tensor(self.tokenizer.encode(prompt), device=self.device).unsqueeze(0)
            for _ in range(steps):
                opt.zero_grad(); loss, _ = self.model(ids, labels=ids); loss.backward(); opt.step()
        self.model.eval()

    def solve_task(self, task_json):
        ttt_prompt = self.encoder.task_to_prompt(task_json)
        self.test_time_train(ttt_prompt)

        nudge = " r0c0v"
        input_ids = torch.tensor(self.tokenizer.encode(ttt_prompt + nudge), device=self.device).unsqueeze(0)
        
        t_r, t_c, t_v, t_space = [self.tokenizer.encode(c)[0] for c in "rcv "]
        t_digits = [self.tokenizer.encode(str(d))[0] for d in range(10)]
        
        generated, state = [], "VALUE"
        visited_coords = set()
        curr_r, curr_c = 0, 0 # Initialize with nudge values

        with torch.no_grad():
            for _ in range(400):
                _, logits = self.model(input_ids[:, -self.cfg.max_seq_length:])
                mask = torch.full_like(logits[0, -1, :], float("-inf"))
                
                # STATE MACHINE: Forces the rXcXvX structure
                if state == "VALUE": 
                    mask[t_digits] = 0
                    
                    # GLOBAL DIVERSITY BIAS
                    # Count how many times each color digit (0-9) has appeared
                    color_counts = torch.zeros(10, device=self.device)
                    for tok in generated:
                        if tok in t_digits:
                            color_counts[t_digits.index(tok)] += 1
                    
                    # Apply a penalty proportional to how many times a color has appeared
                    # This prevents the "v2 v2 v2" sinkhole
                    for idx, count in enumerate(color_counts):
                        logits[0, -1, t_digits[idx]] -= (count * 0.5)
                    
                    ns = "SPACE"
                elif state == "SPACE": 
                    mask[t_space] = 0; ns = "ROW"
                elif state == "ROW": 
                    mask[t_r] = 0; ns = "RIDX"
                elif state == "RIDX": 
                    # Filter digits to ensure we pick a NEW coordinate
                    valid_r = []
                    for d in range(10):
                        # Simple check: can we find ANY column for this row we haven't used?
                        if any((d, c) not in visited_coords for c in range(10)):
                            valid_r.append(t_digits[d])
                    mask[valid_r if valid_r else t_digits] = 0
                    ns = "COL"
                elif state == "COL": 
                    mask[t_c] = 0; ns = "CIDX"
                elif state == "CIDX": 
                    valid_c = [t_digits[d] for d in range(10) if (curr_r, d) not in visited_coords]
                    mask[valid_c if valid_c else t_digits] = 0
                    ns = "VTAG"
                elif state == "VTAG": 
                    mask[t_v] = 0; ns = "VALUE"

                next_id = torch.argmax(logits[0, -1, :] + mask).unsqueeze(0).unsqueeze(0)
                
                # Update our coordinate tracker safely
                token_str = self.tokenizer.decode([next_id.item()]).strip()
                if state == "RIDX" and token_str.isdigit():
                    curr_r = int(token_str)
                elif state == "CIDX" and token_str.isdigit():
                    curr_c = int(token_str)
                    visited_coords.add((curr_r, curr_c))

                input_ids = torch.cat([input_ids, next_id], dim=-1)
                generated.append(next_id.item()); state = ns
                
                if len(visited_coords) >= 40: break # Standard ARC max
        
        return nudge + self.tokenizer.decode(generated)

if __name__ == "__main__":
    eval_cfg = Config()
    _, eval_set = arckit.load_data("arcagi")
    ev = ARCEvaluator("liq_parallel_final.pt", eval_cfg)
    print("\n🔬 TASK 0 PREDICTION:\n", ev.solve_task(eval_set[0].to_dict()))
