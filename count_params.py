from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class Pinard(nn.Module):

    def __init__(self, in_features, out_features, N, config):
        super().__init__()

        self.k = nn.Parameter(torch.randn((N, in_features)))
        self.v = nn.Parameter(torch.randn((N, out_features)))

        torch.nn.init.normal_(self.k, mean=0., std=0.02)
        torch.nn.init.normal_(self.v, mean=0., std=2/config.n_layer/math.sqrt(config.n_embd))

    def _norm_scores(self, scores):
        norm_outputs = scores / torch.norm(scores, p=2, dim=-1, keepdim=True) * math.sqrt(scores.shape[-1])
        return F.gelu(norm_outputs)

    def forward(self, q):
        # same shapes and overall computations as standard attention with T=1 or N, n_head=1, head_dim=d1 or d2
        # q: (B, T, n_head, head_dim) = (B, 1, 1, d1) = (B, d1), k: (T, 1, d1), v: (T, 1, d2)
        scores = q @ self.k.T # (B, N)
        out = self._norm_scores(scores) @ self.v # (B, d2)
        return out
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key in ['k', 'v']:
            full_key = prefix + key
            if full_key in state_dict:
                param = state_dict[full_key]
                curr_param = getattr(self, key)
                curr_N = curr_param.size(0)
                loaded_N = param.size(0)
                assert loaded_N <= curr_N
                new_param = torch.zeros_like(curr_param)
                new_param[:loaded_N] = param[:loaded_N]
                state_dict[full_key] = new_param

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = Pinard(self.n_embd, self.n_embd, config.n_param_attn, config) # todo : create var
        self.c_k = Pinard(self.n_embd, self.n_embd, config.n_param_attn, config)
        self.c_v = Pinard(self.n_embd, self.n_embd, config.n_param_attn, config)
        # output projection
        self.c_proj = Pinard(self.n_embd, self.n_embd, config.n_param_attn, config)
        #self.c_proj.RESIDUAL_SCALE_FLAG = 1
        #self.c_proj.k.data.zero_() # zero init suggested by @Grad62304977
        #self.c_proj.v.data.zero_()
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = Pinard(config.n_embd, config.n_embd, config.n_param_mlp, config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
    n_param_attn : int = 576
    n_param_mlp : int = 2304 

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

        self.apply(self._init_weights)

    def forward(self, idx, targets=None):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        logits = self.lm_head(x)
        logits = logits.float() # use tf32/fp32 for logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768, n_param_attn=144, n_param_mlp=576))
print(f"Model initialized. Number of parameters : {sum([p.numel()/1000000 for p in model.parameters()]):.2f}M.")
