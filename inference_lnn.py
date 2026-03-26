#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./output_lnn"
CHECKPOINT = "checkpoint-10000"

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
        t = torch.linspace(0, 1, 2, device=x.device)
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
        for _ in range(self.cfg.N_sup):
            for _ in range(self.cfg.T_recurse):
                for _ in range(self.cfg.n_latent):
                    curr_z = x + y + z
                    for layer in self.layers: curr_z = layer(curr_z)
                    z = self.final_norm(curr_z)
                y = self.final_norm(sum(layer(y) for layer in self.layers))
            logits = self.output_head(y)
        return None, logits

class LNNConfig:
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 688
    max_seq_length: int = 256
    tie_word_embeddings: bool = True
    n_latent: int = 2
    T_recurse: int = 2
    N_sup: int = 2
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"

def load_model():
    checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT, "lnn_model.pt")
    print(f"Loading checkpoint from {checkpoint_path}")
    
    cfg = LNNConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = LNNForCausalLM(cfg, len(tokenizer)).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded on {DEVICE}")
    return model, tokenizer

@torch.no_grad()
def generate(prompt, max_new_tokens=50, temperature=0.7, top_k=50):
    model, tokenizer = load_model()
    
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    
    for _ in range(max_new_tokens):
        _, logits = model(inputs)
        next_token_logits = logits[0, -1, :] / temperature
        
        if top_k > 0:
            indices = torch.topk(next_token_logits, top_k).indices
            next_token_logits[~torch.isin(torch.arange(len(next_token_logits), device=DEVICE), indices)] = float('-inf')
        
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(inputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In a distant future",
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt}")
        print(f"{'='*50}")
        output = generate(prompt, max_new_tokens=50)
        print(f"Output: {output}")
