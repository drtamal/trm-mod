#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken
import os

# ================= CONFIG (MATCHES YOUR 35M / 8-LAYER MODEL) =================
@dataclass
class EvalConfig:
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 8
    max_seq_length: int = 256
    vocab_size: int = 50257
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Change this to "lnn_grammar_epoch_1.pt" if epoch 2 was overfitted
    model_path: str = "lnn_grammar_expert.pt" 

# ================= CORE ARCHITECTURE =================
# [Insert your RMSNorm, RotaryEmbedding, CausalAttention, LiquidCell, Block, and LNN classes here]
# ... (Ensure these are the same as your training script) ...

# ================= ADVANCED GENERATION ENGINE =================
def generate_response(model, prompt, max_tokens=128, temp=0.7, repetition_penalty=1.2, top_p=0.9):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    
    # Wrap prompt in the Chat Template
    full_prompt = f"<|user|>\n{prompt}\n<|bot|>\n"
    ids = torch.tensor(enc.encode(full_prompt), dtype=torch.long).unsqueeze(0).to(model.cfg.device)
    
    generated_tokens = []
    
    for _ in range(max_tokens):
        with torch.no_grad():
            _, logits = model(ids[:, -model.cfg.max_seq_length:])
            next_token_logits = logits[0, -1, :]
            
            # 1. Repetition Penalty
            for token_id in set(generated_tokens):
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= repetition_penalty
                else:
                    next_token_logits[token_id] /= repetition_penalty

            # 2. Nucleus (Top-P) Filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits / temp, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            next_token_logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

            # 3. Sample
            probs = F.softmax(next_token_logits / temp, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=-1)
            generated_tokens.append(next_id.item())
            
            # Stream output to terminal
            token_str = enc.decode([next_id.item()])
            print(token_str, end="", flush=True)
            
            if "<|end|>" in enc.decode(generated_tokens):
                break
    print() # Newline at end

# ================= INTERACTIVE TERMINAL =================
def run_terminal():
    cfg = EvalConfig()
    print(f"Loading Expert Model from {cfg.model_path}...")
    
    model = LNN(cfg).to(cfg.device)
    try:
        model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: {cfg.model_path} not found. Check your file path.")
        return

    print("\n" + "="*60)
    print("LNN HYBRID EXPERT: CHAT & GRAMMAR TERMINAL")
    print("Commands: 'quit' to exit | 'temp [val]' to change creativity")
    print("Tip: Use 'Correct the grammar: [text]' for best editing results.")
    print("="*60)

    current_temp = 0.7
    
    while True:
        user_input = input("\nUser > ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if user_input.lower().startswith('temp '):
            try:
                current_temp = float(user_input.split()[1])
                print(f"Temperature set to {current_temp}")
                continue
            except:
                print("Invalid temperature format.")
                continue

        print("Bot   > ", end="")
        generate_response(model, user_input, temp=current_temp)

if __name__ == "__main__":
    run_terminal()
