import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import arckit

#======================LNN core==========================================

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
    def __init__(self, dim, num_heads, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout 
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.resid_drop = nn.Dropout(dropout)
        
        # We still store slopes, but we will compute the bias dynamically
        self.register_buffer("slopes", get_alibi_slopes(num_heads))

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # DYNAMIC ALIBI BIAS GENERATION
        # Create distance matrix for current sequence length L
        context_pos = torch.arange(L, device=x.device).view(1, 1, L)
        memory_pos = torch.arange(L, device=x.device).view(1, L, 1)
        relative_dist = memory_pos - context_pos
        
        # Apply slopes to the relative distances
        current_bias = self.slopes.view(1, self.num_heads, 1, 1) * relative_dist.view(1, 1, L, L)
        
        # Causal Mask
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        combined_mask = current_bias.masked_fill(mask, float("-inf"))

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


# ========================== GRID SERIALIZATION ==========================

class ARCEncoder:
    """Converts 2D ARC grids into LLM-compatible token strings."""
    def __init__(self):
        # We map colors 0-9 to tokens that are visually/conceptually distinct
        self.color_map = {i: str(i) for i in range(10)}
        self.row_sep = "|" 
        self.pair_sep = "=>"

    def grid_to_tokens(self, grid):
        """Flatten a 2D list into a string: '1 2 | 3 4'"""
        rows = [" ".join(self.color_map[c] for c in row) for row in grid]
        return f" {self.row_sep} ".join(rows)

    def task_to_prompt(self, task, include_test_output=False):
        """Converts an entire ARC JSON task into a sequence of demonstrations."""
        prompt = "ARC Reasoning Task:\n"
        for i, pair in enumerate(task['train']):
            inp = self.grid_to_tokens(pair['input'])
            out = self.grid_to_tokens(pair['output'])
            prompt += f"Example {i+1}: Input: {inp} {self.pair_sep} Output: {out}\n"
        
        test_inp = self.grid_to_tokens(task['test'][0]['input'])
        prompt += f"Test: Input: {test_inp} {self.pair_sep} Output:"
        
        if include_test_output:
            prompt += f" {self.grid_to_tokens(task['test'][0]['output'])}"
        return prompt

# ========================== EVALUATOR ENGINE ==========================

class ARCEvaluator:
    def __init__(self, model_path, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.encoder = ARCEncoder()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # 1. Initialize the model with the NEW dynamic architecture
        self.model = LiquidLM(cfg).to(self.device)
        
        # 2. Load the state_dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 3. FIX: Convert 'alibi_bias' back to 'slopes' for the new architecture
        # The first value of the bias at distance 1 is the slope.
        new_state_dict = self.model.state_dict()
        for i in range(cfg.num_layers):
            bias_key = f"blocks.{i}.attn.alibi_bias"
            slope_key = f"blocks.{i}.attn.slopes"
            
            if bias_key in state_dict:
                # Extract slopes from the stored bias: bias[0, head, 1, 0] is the slope for that head
                # Since relative_dist was (memory_pos - context_pos), dist 1 is at index [1,0]
                extracted_slopes = state_dict[bias_key][0, :, 1, 0] 
                state_dict[slope_key] = extracted_slopes
                del state_dict[bias_key] # Remove the old key
        
        # 4. Load with strict=False to ignore the change in bias/slope structure
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print("✅ Weights loaded and slopes reconstructed for Dynamic Attention.")

    def test_time_train(self, prompt, steps=5, lr=5e-5):
        """
        FIXED: Ensures the TTT loop can actually calculate gradients.
        """
        self.model.train() # Must be in train mode for backward()
        
        # Explicitly enable grad just for this block to override any global no_grad
        with torch.enable_grad():
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            
            # Encode tokens and ensure we don't accidentally freeze them
            token_list = self.tokenizer.encode(prompt)
            if not token_list: return # Safety check for empty prompts
            
            tokens = torch.tensor(token_list, device=self.device).unsqueeze(0)
            
            for _ in range(steps):
                optimizer.zero_grad()
                
                # Forward pass: labels=tokens ensures CrossEntropy is calculated
                loss, _ = self.model(tokens, labels=tokens)
                
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                else:
                    print("⚠️ Warning: Loss does not require grad. Check if parameters are frozen.")
                    break
        
        self.model.eval() # Return to eval mode for the actual test guess

    @torch.no_grad()
    def solve_task(self, task_json, use_ttt=True):
        """Attempts to solve a single ARC task."""
        prompt = self.encoder.task_to_prompt(task_json)
        
        if use_ttt:
            # We use the train examples to 'teach' the model the rule
            ttt_prompt = self.encoder.task_to_prompt(task_json, include_test_output=False)
            self.test_time_train(ttt_prompt)

        # Generate the output grid
        input_ids = torch.tensor(self.tokenizer.encode(prompt), device=self.device).unsqueeze(0)
        generated = []
        
        # ARC grids are small, max 128 tokens is usually enough
        for _ in range(128):
            _, logits = self.model(input_ids[:, -self.cfg.max_seq_length:])
            next_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_id], dim=-1)
            
            # Stop if we hit end of sequence or newline
            if next_id.item() == 50256 or next_id.item() == 198: 
                break
            generated.append(next_id.item())
            
        return self.tokenizer.decode(generated)

    def run_benchmark(self, tasks_path):
        """Runs the LNN against a directory of ARC JSON files."""
        tasks = [f for f in os.listdir(tasks_path) if f.endswith('.json')]
        correct = 0
        
        print(f"🚀 Benchmarking SKAI-LNN on {len(tasks)} ARC-AGI-1 tasks...")
        
        for task_file in tqdm(tasks):
            with open(os.path.join(tasks_path, task_file), 'r') as f:
                task = json.load(f)
            
            prediction = self.solve_task(task)
            ground_truth = self.encoder.grid_to_tokens(task['test'][0]['output'])
            
            # Clean strings for comparison
            if prediction.strip() == ground_truth.strip():
                correct += 1
                
        accuracy = (correct / len(tasks)) * 100
        print(f"\n📊 Benchmark Result: {accuracy:.2f}% ({correct}/{len(tasks)})")
        return accuracy

# ========================== EXECUTION ==========================

if __name__ == "__main__":
    # 1. Setup Config (Must match your training config)
    eval_cfg = Config()
    eval_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Automatically load ARC-AGI-1 
    # This downloads the data if it doesn't exist (no manual git clone needed)
    print("📥 Loading ARC-AGI-1 dataset via arckit...")
    train_set, eval_set = arckit.load_data("arcagi")
    
    # 3. Initialize Evaluator
    evaluator = ARCEvaluator("liq_parallel_final.pt", eval_cfg)
    
    # 4. Custom Benchmark Loop for arckit objects
    correct = 0
    print(f"🚀 Benchmarking SKAI-LNN on {len(eval_set)} tasks...")
    
    for task in tqdm(eval_set):
        # arckit tasks have a .to_dict() method compatible with our solver
        prediction = evaluator.solve_task(task.to_dict())
        ground_truth = evaluator.encoder.grid_to_tokens(task.test[0][1]) # Index 1 is the output grid
        
        if prediction.strip() == ground_truth.strip():
            correct += 1
            
    accuracy = (correct / len(eval_set)) * 100
    print(f"\n📊 Final SKAI-LNN Reasoning Score: {accuracy:.2f}%")
