#!/usr/bin/env python3
import os, math, random, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np


@dataclass
class LNNConfig:
    hidden_size: int = 64
    num_layers: int = 2
    intermediate_size: int = 128
    max_seq_length: int = 128
    vocab_size: int = 256
    n_latent: int = 2
    T_recurse: int = 1
    tie_word_embeddings: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    max_steps: int = 50000
    save_interval: int = 10000
    log_interval: int = 100
    output_dir: str = "./output_lnn"


class CharacterTokenizer:
    def __init__(self):
        chars = list(" \nABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?-'\"()[]{}:;@#$%^&*+=<>|/\\~`")
        self.stoi = {ch: i + 3 for i, ch in enumerate(chars)}
        self.stoi['<pad>'] = 0
        self.stoi['<bos>'] = 1
        self.stoi['<eos>'] = 2
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.eos_token = '<eos>'
    
    def encode(self, text, max_length=128, padding=False, truncation=False):
        ids = [self.bos_token_id]
        for ch in text:
            ids.append(self.stoi.get(ch, 2))
        ids.append(self.eos_token_id)
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        if padding and len(ids) < max_length:
            ids.extend([self.pad_token_id] * (max_length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids):
        return ''.join([self.itos.get(i, '<unk>') for i in ids if i not in [0, 1, 2]])
    
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
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        norm = torch.norm(x.float(), p=2, dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5 + self.eps)
        return (x.float() / (rms + self.eps)).to(x.dtype) * self.weight


class LiquidGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_alpha = nn.Parameter(torch.tensor(0.5))
        self.gate_proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        alpha = torch.sigmoid(self.time_alpha)
        gate = torch.sigmoid(self.gate_proj(x))
        return alpha * gate


class LiquidCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)
        self.U = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(dim))
        self.liquid_gate = LiquidGate(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)
    
    def forward(self, x, input_features):
        B, L, D = x.shape
        head_dim = max(1, D // 4)
        
        q = self.q_proj(x)[:, :, :head_dim*4]
        k = self.k_proj(input_features)[:, :, :head_dim*4]
        v = self.v_proj(input_features)[:, :, :head_dim*4]
        
        q = q.view(B, L, 4, head_dim)
        k = k.view(B, L, 4, head_dim)
        v = v.view(B, L, 4, head_dim)
        
        scale = head_dim ** -0.5
        attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        attn_out = (attn @ v).reshape(B, L, -1)
        
        dx = F.silu(x @ self.W.T) + attn_out + input_features @ self.U.T + self.b
        return x + self.liquid_gate(x) * dx


class LiquidBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.liquid_cell = LiquidCell(cfg.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        )
        self.mlp_norm = RMSNorm(cfg.hidden_size)
        self.cell_norm = RMSNorm(cfg.hidden_size)
    
    def forward(self, x):
        cell_out = self.cell_norm(x + self.liquid_cell(x, x))
        return x + self.mlp(self.mlp_norm(cell_out))


class LNNForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_seq_length, cfg.hidden_size) * 0.02)
        self.y_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.01)
        self.z_init = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.01)
        self.layers = nn.ModuleList([LiquidBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.output_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        
        if cfg.tie_word_embeddings:
            self.output_head.weight = self.tok_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, ids, labels=None):
        B, L = ids.shape
        L = min(L, self.cfg.max_seq_length)
        
        x = self.tok_emb(ids[:, :L]) + self.pos_emb[:, :L, :]
        y = self.y_init.expand(B, L, -1)
        z = self.z_init.expand(B, L, -1)
        
        for _ in range(self.cfg.T_recurse):
            for _ in range(self.cfg.n_latent):
                curr_z = x + y + z
                for layer in self.layers:
                    curr_z = layer(curr_z)
                z = z + self.final_norm(curr_z)
            
            for layer in self.layers:
                y = layer(y)
            y = self.final_norm(y)
        
        logits = self.output_head(y)
        
        total_loss = 0
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:L].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            total_loss = loss
        
        return total_loss, logits


class TextStream(IterableDataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.corpus = [
            "The quick brown fox jumps over the lazy dog. ",
            "A journey of a thousand miles begins with a single step. ",
            "To be or not to be that is the question. ",
            "All that glitters is not gold. ",
            "Actions speak louder than words. ",
            "Better late than never. ",
            "Birds of a feather flock together. ",
            "Curiosity killed the cat but satisfaction brought it back. ",
            "Don t count your chickens before they hatch. ",
            "Every cloud has a silver lining. ",
            "Fortune favors the bold. ",
            "Knowledge is power. ",
            "Laughter is the best medicine. ",
            "Make virtue of necessity. ",
            "Nothing ventured nothing gained. ",
            "Opportunity knocks but once. ",
            "Practice makes perfect. ",
            "Quality matters more than quantity. ",
            "Rome was not built in a day. ",
            "Slow and steady wins the race. ",
            "The early bird catches the worm. ",
            "Understanding is the key to success. ",
            "Variety is the spice of life. ",
            "When in Rome do as the Romans do. ",
            "You cannot judge a book by its cover. ",
            "An apple a day keeps the doctor away. ",
            "Biting the bullet and moving forward. ",
            "Calm waters run deep. ",
            "Dawn waits for no one. ",
            "Easy come easy go. ",
        ]
        self.base_text = "The Liquid Neural Network processes sequential data through continuous transformations. "
        "Each unit maintains an internal state that evolves over time. "
        "The network learns to gate information flow adaptively. "
        "Gradient descent optimizes the model parameters. "
        "Attention mechanisms help focus on relevant features. "
        "Recurrent connections enable memory of past inputs. "
        "The model architecture balances expressiveness and efficiency. "
        "Training involves minimizing prediction errors across sequences. "
        "Token embeddings capture semantic relationships between symbols. "
        "Position encodings provide order information to the model. "
        "Normalization stabilizes training dynamics. "
        "Skip connections facilitate gradient flow through deep networks. "
        "Nonlinearities enable modeling of complex patterns. "
        "The loss function measures the difference between predictions and targets. "
        "Backpropagation computes gradients efficiently through the computational graph. "
    
    def __iter__(self):
        rng = np.random.default_rng(self.cfg.seed)
        while True:
            texts = []
            for _ in range(self.cfg.batch_size):
                text = self.base_text + rng.choice(self.corpus) * 3
                texts.append(text)
            
            for text in texts:
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
        
        if step % cfg.log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm.item():.4f}")
        
        if step % cfg.save_interval == 0 and step > 0:
            sd = os.path.join(cfg.output_dir, f"checkpoint-{step}")
            os.makedirs(sd, exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": cfg,
            }, os.path.join(sd, "lnn_model.pt"))
            print(f"\nSaved checkpoint-{step}")
        
        if step >= cfg.max_steps:
            break
    
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=40):
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = tokenizer(prompt, max_length=model.cfg.max_seq_length, truncation=True)["input_ids"].to(device)
    
    if input_ids.shape[-1] >= model.cfg.max_seq_length:
        input_ids = input_ids[:, -model.cfg.max_seq_length+1:]
    
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if generated.shape[-1] >= model.cfg.max_seq_length:
                break
            
            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated[0].tolist())


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
