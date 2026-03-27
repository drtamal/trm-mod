#!/usr/bin/env python3
import os, math, random, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False
from torchdiffeq import odeint

OUTPUT_ROOT = "./output_lnn"

@dataclass
class LNNConfig:
    vocab_size: int = 128
    hidden_size: int = 96
    num_layers: int = 1
    num_heads: int = 2
    intermediate_size: int = 192
    max_seq_length: int = 32
    time_steps: int = 2
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    n_latent: int = 1
    T_recurse: int = 1
    N_sup: int = 1
    ema_decay: float = 0.99
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 50000
    learning_rate: float = 1e-3
    save_interval: int = 1000
    output_dir: str = OUTPUT_ROOT
    seed: int = 42
    device: str = "cpu"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        return torch.sigmoid(self.time_constant) * self.gate(x)

class LiquidCell(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
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
        ode_features = x + (attn @ v).reshape(B, L, -1)
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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_seq_length, cfg.hidden_size) * 0.02)
        self.y_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        self.layers = nn.ModuleList([LiquidBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.output_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings: self.output_head.weight = self.tok_emb.weight

    def forward(self, ids, labels=None):
        B, L = ids.shape
        L = min(L, self.cfg.max_seq_length)
        x = self.tok_emb(ids[:, :L]) + self.pos_emb[:, :L, :]
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        logits = self.output_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:L].contiguous().view(-1)
            shift_labels[shift_labels == 0] = -100
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return loss, logits

class CharDataset(IterableDataset):
    def __init__(self, data, cfg):
        self.cfg = cfg
        self.vocab = sorted(set(ord(c) for c in data))
        chars = [c for c in map(ord, data) if c in self.vocab]
        print(f"Vocab size: {len(self.vocab)}, chars: {[chr(v) for v in self.vocab[:20]]}")
        self.chunks = []
        for i in range(0, len(chars) - cfg.max_seq_length, cfg.max_seq_length // 2):
            self.chunks.append(chars[i:i+cfg.max_seq_length])
        random.shuffle(self.chunks)
        print(f"Total chunks: {len(self.chunks)}")
    
    def __iter__(self):
        idx = 0
        while True:
            if idx >= len(self.chunks):
                random.shuffle(self.chunks)
                idx = 0
            chunk = self.chunks[idx]
            if len(chunk) < self.cfg.max_seq_length:
                chunk = chunk + [0] * (self.cfg.max_seq_length - len(chunk))
            yield {"input_ids": torch.tensor(chunk, dtype=torch.long), "labels": torch.tensor(chunk, dtype=torch.long)}
            idx += 1

def encode_text(text, cfg):
    vocab = set(ord(c) for c in "".join(chr(i) for i in range(128)))
    ids = [ord(c) if ord(c) in vocab else 0 for c in text]
    return ids

def decode_ids(ids, cfg):
    return ''.join(chr(c) if c > 32 and c < 127 else '' for c in ids)

def generate(model, cfg, prompt, max_new_tokens=50, temperature=0.8):
    model.eval()
    input_ids = encode_text(prompt, cfg)
    while len(input_ids) < cfg.max_seq_length:
        input_ids.append(0)
    
    generated = input_ids.copy()
    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([generated[-cfg.max_seq_length:]], dtype=torch.long).to(cfg.device)
        with torch.no_grad():
            _, logits = model(input_tensor)
        logits = logits[0, -1] / temperature
        probs = torch.softmax(logits, dim=0)
        probs[0] = 0
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        if next_token == 0:
            break
    
    result = decode_ids(generated[len(prompt):], cfg)
    return result

def evaluate_model(model, cfg):
    test_prompts = [
        "The quick brown",
        "Hello world",
        "Liquid neural",
    ]
    print("\n--- Evaluation ---")
    model.eval()
    for prompt in test_prompts:
        input_ids = encode_text(prompt, cfg)
        while len(input_ids) < cfg.max_seq_length:
            input_ids.append(0)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(cfg.device)
        with torch.no_grad():
            _, logits = model(input_tensor)
        probs = torch.softmax(logits[0, -1], dim=0)
        probs[0] = 0
        top_probs, top_indices = probs.topk(5)
        print(f"Prompt: '{prompt}'")
        print(f"Top tokens: {[(decode_ids([idx.item()], cfg), idx.item(), p.item()) for idx, p in zip(top_indices, top_probs)]}")
        
        generated = generate(model, cfg, prompt, max_new_tokens=30)
        print(f"Generated: '{generated}'")
        print()

def main():
    sample_text = (
        "The quick brown fox jumps over the lazy dog. Hello world! Liquid neural networks are amazing. " * 1000 +
        "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 500 +
        "0123456789 .!?," * 300 +
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " * 300 +
        "apple banana cherry date grape lemon orange peach raspberry strawberry watermelon " * 300 +
        "Monday Tuesday Wednesday Thursday Friday Saturday Sunday " * 300 +
        "red blue green yellow orange purple pink black white brown gray cyan magenta " * 300 +
        "cat dog bird fish horse rabbit turtle snake lizard frog mouse elephant lion tiger " * 300 +
        "sun moon star planet galaxy universe asteroid comet meteor nebula blackhole " * 300 +
        "water fire earth air lightning ice rock paper scissors " * 300
    )
    
    actual_vocab = len(set(c for c in sample_text))
    print(f"Auto-detected vocab size: {actual_vocab}")
    
    cfg = LNNConfig()
    cfg.vocab_size = 128
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.set_num_threads(2)

    model = LNNForCausalLM(cfg).to(cfg.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    if num_params > 100000:
        print(f"WARNING: Parameters exceed 100k limit!")

    dataset = CharDataset(sample_text, cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    sched = get_cosine_schedule_with_warmup(opt, 500, cfg.max_steps)

    pbar = tqdm(loader, total=cfg.max_steps)
    model.train()
    
    loss_history = []

    for i, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(cfg.device)
        labels = batch["labels"].to(cfg.device)

        loss, _ = model(input_ids, labels)
        if loss is None:
            continue
        loss = loss / cfg.gradient_accumulation_steps
        loss.backward()

        if (i + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            loss_history.append(loss.item() * cfg.gradient_accumulation_steps)
        pbar.set_postfix(loss=f"{loss.item() if loss else 0:.4f}")

        step = i // cfg.gradient_accumulation_steps
        if step > 0 and i % (cfg.save_interval * cfg.gradient_accumulation_steps) == 0:
            path = os.path.join(cfg.output_dir, f"checkpoint-{step}.pt")
            torch.save(model.state_dict(), path)

        if step >= cfg.max_steps: break

    if HAS_MATPLOTLIB and loss_history:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.savefig(os.path.join(cfg.output_dir, 'loss_curve.png'), dpi=150)
        plt.close()
        print(f"\nLoss curve saved to {cfg.output_dir}/loss_curve.png")
        
        if len(loss_history) > 100:
            smoothed = np.convolve(loss_history, np.ones(100)/100, mode='valid')
            plt.figure(figsize=(10, 5))
            plt.plot(smoothed)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve (Smoothed)')
            plt.grid(True)
            plt.savefig(os.path.join(cfg.output_dir, 'loss_curve_smoothed.png'), dpi=150)
            plt.close()
            print(f"Smoothed loss curve saved to {cfg.output_dir}/loss_curve_smoothed.png")

    evaluate_model(model, cfg)

    print("\n--- Interactive Mode ---")
    while True:
        prompt = input("\nEnter prompt (or 'quit'): ")
        if prompt.lower() == 'quit':
            break
        generated = generate(model, cfg, prompt, max_new_tokens=50)
        print(f"Generated: '{generated}'")

if __name__ == "__main__":
    main()
