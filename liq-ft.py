#!/usr/bin/env python3
import os, math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm

# ================= CONFIG (FINE-TUNING) =================
@dataclass
class SFTConfig:
    hidden_size: int = 768     # MUST match your pre-trained model
    num_layers: int = 12       # MUST match your pre-trained model
    num_heads: int = 12        # MUST match your pre-trained model
    max_seq_length: int = 256
    char_vocab_size: int = 256
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SFT Specifics
    batch_size: int = 16       # Can be larger since sequences are shorter dialogues
    grad_accum: int = 2
    max_steps: int = 15000     # Fine-tuning requires far fewer steps than pre-training
    log_interval: int = 10
    eval_interval: int = 1000
    
    # Lower Learning Rate to preserve base knowledge!
    max_lr: float = 5e-5       
    min_lr: float = 5e-6
    weight_decay: float = 0.1
    label_smoothing: float = 0.0
    
    base_model: str = "lnn_final_model.pt"
    output_model: str = "lnn_chat_assistant.pt"

# ---> [PASTE YOUR EXISTING MODEL ARCHITECTURE HERE] <---
# (CharCNNEmbedding, RMSNorm, RotaryEmbedding, CausalAttention, LiquidCell, Block, LNN_Hybrid)

# ================= CHAT DATASET =================
class ChatStream(IterableDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        
    def __iter__(self):
        from datasets import load_dataset
        # Daily Dialog is a great, clean chit-chat dataset
        ds = load_dataset("daily_dialog", split="train", streaming=True)
        
        for x in ds.shuffle(buffer_size=1000, seed=self.cfg.seed):
            dialogue = x["dialog"]
            formatted_chat = ""
            
            # Build the Chat Template: <|user|> ... <|bot|> ... <|end|>
            for i, turn in enumerate(dialogue):
                if i % 2 == 0:
                    formatted_chat += f"<|user|>\n{turn.strip()}\n"
                else:
                    formatted_chat += f"<|bot|>\n{turn.strip()}\n<|end|>\n"
            
            # Convert to raw bytes
            bytes_data = formatted_chat.encode('utf-8')
            tokens = [min(b, 255) for b in bytes_data]
            
            # Chunk it to fit the max_seq_length
            if len(tokens) > self.cfg.max_seq_length:
                # We take the end of the conversation if it's too long
                tokens = tokens[-self.cfg.max_seq_length:]
            else:
                tokens += [0] * (self.cfg.max_seq_length - len(tokens))
                
            ids = torch.tensor(tokens, dtype=torch.long)
            yield {"input_ids": ids, "labels": ids.clone()}

# ================= GENERATION WITH STOP TOKEN =================
def chat_generate(model, prompt, max_new_tokens=150, temperature=0.7):
    model.eval()
    
    # Format the prompt using our exact template
    formatted_prompt = f"<|user|>\n{prompt}\n<|bot|>\n"
    input_bytes = list(formatted_prompt.encode('utf-8'))
    ids = torch.tensor(input_bytes, dtype=torch.long).unsqueeze(0).to(model.cfg.device)
    
    generated = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            _, logits = model(ids[:, -model.cfg.max_seq_length:])
            
            # Temperature sampling
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            ids = torch.cat([ids, next_id], dim=-1)
            generated.append(next_id.item())
            
            # Look for the <|end|> trigger to stop generating automatically
            current_text = bytes([b for b in generated if 0 < b < 256]).decode('utf-8', errors='ignore')
            if "<|end|>" in current_text:
                return current_text.replace("<|end|>", "").strip()
                
    return bytes([b for b in generated if 0 < b < 256]).decode('utf-8', errors='ignore').strip()

# ================= SFT TRAINING LOOP =================
def get_lr(cfg, step):
    progress = step / cfg.max_steps
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

def finetune():
    cfg = SFTConfig()
    model = LNN_Hybrid(cfg).to(cfg.device)
    
    # 1. LOAD THE PRE-TRAINED BASE MODEL
    if os.path.exists(cfg.base_model):
        print(f"Loading Base Knowledge from {cfg.base_model}...")
        model.load_state_dict(torch.load(cfg.base_model, map_location=cfg.device))
    else:
        print(f"ERROR: Could not find {cfg.base_model}. Cannot fine-tune.")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(ChatStream(cfg), batch_size=cfg.batch_size)
    
    print("\nStarting Supervised Fine-Tuning (SFT) for Chat...")
    pbar = tqdm(enumerate(loader), total=cfg.max_steps)
    model.train()
    
    for step, batch in pbar:
        lr = get_lr(cfg, step)
        for pg in opt.param_groups: 
            pg['lr'] = lr
            
        loss, _ = model(batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device))
        (loss / cfg.grad_accum).backward()
        
        if (step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        
        if step % cfg.log_interval == 0: 
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        if step % cfg.eval_interval == 0 and step > 0:
            print(f"\n[Test Chat] User: How is the weather today?")
            print(f"[Test Chat] LNN: {chat_generate(model, 'How is the weather today?')}")
            model.train()
        
        if step >= cfg.max_steps: 
            break

    # Save the Assistant Model
    torch.save(model.state_dict(), cfg.output_model)
    print(f"\nSFT Complete! Assistant saved as {cfg.output_model}")

if __name__ == "__main__":
    finetune()
