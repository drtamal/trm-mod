#!/usr/bin/env python3
import os, math, random, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np


@dataclass
class LNNConfig:
    hidden_size: int = 64
    num_layers: int = 4
    max_seq_length: int = 128
    vocab_size: int = 256
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    max_steps: int = 50000
    save_interval: int = 25000
    log_interval: int = 100
    output_dir: str = "./output_lnn"


class CharacterTokenizer:
    def __init__(self):
        chars = " abcdefghijklmnopqrstuvwxyz.,!?-():;"
        self.chars = chars
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi) + 1
        self.pad_token_id = 0
    
    def encode(self, text, max_length=128, padding=False, truncation=False):
        ids = [self.stoi.get(ch, 0) for ch in text]
        ids.append(0)
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        if padding and len(ids) < max_length:
            ids.extend([self.pad_token_id] * (max_length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids):
        return ''.join([self.itos.get(i, '') for i in ids if i > 0])
    
    def __call__(self, text, max_length=128, padding=False, truncation=False, return_tensors=None):
        e = self.encode(text, max_length, padding, truncation)
        return {"input_ids": e.unsqueeze(0)}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm * self.g).to(x.dtype)


class LiquidGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gate = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x):
        a = torch.sigmoid(self.alpha)
        g = torch.sigmoid(self.gate(x))
        return a * g


class LiquidCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.liquid_gate = LiquidGate(dim)
        self.W = nn.Linear(dim, dim, bias=True)
        self.U = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.W.weight)
        nn.init.zeros_(self.U.weight)
    
    def forward(self, x, h):
        g = self.liquid_gate(x)
        dh = torch.tanh(self.W(x) + self.U(h))
        return h + g * dh * 0.1


class CausalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        return self.proj((attn @ v).transpose(1, 2).reshape(B, L, D))


class LiquidBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = CausalAttention(dim, num_heads)
        self.liquid = LiquidCell(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
        self.ln3 = RMSNorm(dim)
    
    def forward(self, x, h):
        h = self.liquid(self.ln1(x), h)
        x = x + self.attn(self.ln2(x))
        x = x + self.mlp(self.ln3(h))
        return x, h


class LNNForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq_length, cfg.hidden_size))
        self.blocks = nn.ModuleList([LiquidBlock(cfg.hidden_size, 4) for _ in range(cfg.num_layers)])
        self.ln_f = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        
        self.apply(self._init)
    
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, ids, labels=None):
        B, L = ids.shape
        L = min(L, self.cfg.max_seq_length)
        
        x = self.tok_emb(ids[:, :L])
        x = x + self.pos_emb[:, :L, :]
        
        h = x
        for block in self.blocks:
            x, h = block(x, h)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        total_loss = torch.tensor(0.0, device=x.device)
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:L].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            total_loss = loss
        
        return total_loss, logits


class TextStream(IterableDataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be that is the question.",
            "All that glitters is not gold.",
            "Actions speak louder than words.",
            "Better late than never.",
            "Birds of a feather flock together.",
            "Curiosity killed the cat.",
            "Every cloud has a silver lining.",
            "Fortune favors the bold.",
            "Knowledge is power.",
            "Laughter is the best medicine.",
            "Nothing ventured nothing gained.",
            "Practice makes perfect.",
            "Quality matters more than quantity.",
            "Rome was not built in a day.",
            "Slow and steady wins the race.",
            "The early bird catches the worm.",
            "Variety is the spice of life.",
            "When in Rome do as the Romans do.",
            "You cannot judge a book by its cover.",
            "An apple a day keeps the doctor away.",
            "Calm waters run deep.",
            "Easy come easy go.",
            "Time and tide wait for no one.",
            "Where there is a will there is a way.",
            "No pain no gain.",
            "Think before you speak.",
            "Read before you write.",
            "Learn before you teach.",
            "Pack light travel far.",
            "Think clearly act boldly.",
            "Write well read more.",
            "Learn fast fail less.",
            "Build strong test early.",
            "Ship fast iterate slow.",
            "The cat sat on the mat.",
            "The dog ran in the park.",
            "Birds fly in the sky.",
            "Fish swim in the sea.",
            "Stars shine at night.",
            "The sun rises in the east.",
            "Rain falls from clouds.",
            "Snow covers the ground.",
            "Winds blow through trees.",
            "Rivers flow to the sea.",
            "Mountains reach the sky.",
            "Deserts are hot and dry.",
            "Forests are green and dense.",
            "Oceans are vast and deep.",
            "Islands dot the Pacific.",
            "Cities grow and change.",
            "People work and play.",
            "Children learn and grow.",
            "Music fills the air.",
            "Art expresses the soul.",
            "Science探索s the unknown.",
            "History repeats itself.",
            "Math describes patterns.",
            "Language connects minds.",
            "Culture shapes society.",
            "Technology changes fast.",
            "Ideas power progress.",
            "Dreams inspire action.",
            "Hope lights the way.",
            "Love conquers all.",
        ]
    
    def __iter__(self):
        rng = np.random.default_rng(self.cfg.seed)
        while True:
            for _ in range(self.cfg.batch_size):
                texts = rng.choice(self.corpus, size=3, replace=False)
                text = " ".join(texts)
                e = self.tokenizer(text, max_length=self.cfg.max_seq_length, padding=True, truncation=True)
                out = {k: v.squeeze(0) for k, v in e.items()}
                out["labels"] = out["input_ids"].clone()
                out["labels"][out["input_ids"] == 0] = -100
                yield out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    cfg = LNNConfig()
    set_seed(cfg.seed)
    tokenizer = CharacterTokenizer()
    cfg.vocab_size = tokenizer.vocab_size
    
    model = LNNForCausalLM(cfg).to(cfg.device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    if num_params > 100000:
        print("WARNING: Parameters exceed 100k limit!")
    else:
        print(f"Model size OK ({num_params} <= 100000)")
    
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=cfg.max_steps + 1, pct_start=0.1)
    
    loader = DataLoader(TextStream(tokenizer, cfg), batch_size=cfg.batch_size, num_workers=0)
    pbar = tqdm(loader, total=cfg.max_steps)
    
    model.train()
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(cfg.device)
        labels = batch["labels"].to(cfg.device)
        
        opt.zero_grad()
        
        if cfg.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = model(input_ids, labels)
        else:
            loss, _ = model(input_ids, labels)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[Step {step}] Loss NaN/Inf - skipping")
            continue
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if step % cfg.log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.6f}")
        
        if step % cfg.save_interval == 0 and step > 0:
            sd = os.path.join(cfg.output_dir, f"checkpoint-{step}")
            os.makedirs(sd, exist_ok=True)
            torch.save({"step": step, "model_state_dict": model.state_dict(), "config": cfg}, os.path.join(sd, "lnn_model.pt"))
            print(f"\nSaved checkpoint-{step}")
        
        if step >= cfg.max_steps:
            break
    
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=40):
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = tokenizer(prompt, max_length=model.cfg.max_seq_length, truncation=True)["input_ids"].to(device)
    prompt_len = input_ids.shape[-1]
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.shape[-1] >= model.cfg.max_seq_length:
                break
            
            _, logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == 0:
                break
    
    full_text = tokenizer.decode(input_ids[0].tolist())
    if len(full_text) > prompt_len:
        return full_text[prompt_len:]
    return full_text


if __name__ == "__main__":
    model, tokenizer = train()
    
    print("\n--- Testing Generation ---")
    test_prompts = [
        "The quick brown",
        "The Liquid Neural",
        "Knowledge is",
        "Practice makes",
        "Rome was not",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = generate(model, tokenizer, prompt, max_new_tokens=40)
        print(f"Generated: '{generated}'")
