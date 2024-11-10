import time
import os
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch._inductor.config as config
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def rmsnorm(x0):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + torch.finfo(x0.dtype).eps)
    return x.type_as(x0)

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000, max_seq_len=1024):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = max_seq_len

        # Pré-calcul des fréquences pour T_max
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer('cos_cached', freqs.cos().bfloat16(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin().bfloat16(), persistent=False)

    def forward(self, x):
        seq_len = x.shape[1]
        
        # Assurer que cos_cached et sin_cached sont sur le même device que x
        cos = self.cos_cached[:seq_len]#.to(x.device)
        sin = self.sin_cached[:seq_len]#.to(x.device)
        
        return cos[None, :, None, :], sin[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = rmsnorm(q), rmsnorm(k) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        #self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        #x = x + self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight.data.zero_() # @Grad62304977

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = rmsnorm(x)
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

def get_seqlen(it):
    if it<8:
        return 8
    elif it<16:
        return 16
    elif it<24:
        return 24
    elif it<32:
        return 32
    elif it<40:
        return 40
    elif it<48:
        return 48
    elif it<56:
        return 56
    elif it<64:
        return 64
    elif it<72:
        return 72
    elif it<80:
        return 80
    return 1024

def get_batch_size(it):
    return int(it*2)

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

print("initialisation du modèle...")

gptconfig = GPTConfig(vocab_size=50304, n_layer=12, n_head=6, n_embd=768)
#gptconfig = GPTConfig(vocab_size=128, n_layer=4, n_head=1, n_embd=64)
model = GPT(gptconfig).to("cuda")

#inputs = torch.randint(0, gptconfig.vocab_size, (2, 1024,), device="cuda")
#print(inputs.shape)
#print("BEFORE MARK_DYNAMIC ------------------------------")
#torch._dynamo.mark_dynamic(inputs, index=0, min=1, max=21)
#print("AFTER MARK_DYNAMIC ------------------------------")
#print(inputs.shape)

if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
#print("BEFORE TORCH COMPILE ------------------------------")
model = torch.compile(model, dynamic=False)
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
#print("AFTER TORCH COMPILE ------------------------------")

# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
#from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
#enable_cudnn_sdp(True)
#enable_flash_sdp(False)
#enable_mem_efficient_sdp(False)
#enable_math_sdp(False)

#print(inputs.shape)
#model(inputs)

#print("warmup")
#for l in [8, 16, 64]:
#    inputs = torch.randint(0, gptconfig.vocab_size, (16, l,), device="cuda")
#    targets = torch.randint(0, gptconfig.vocab_size, (16, l,), device="cuda")
#    _, loss = model(inputs, targets, return_logits=False)

optimizer = optim.Adam(raw_model.parameters(), lr=0.001)

print("lancement du training...")
start_time = time.time()
last_time = start_time

for epoch in range(100):  # Nombre d'époques
    print(f"[{(time.time()-start_time):.4f}][{(time.time()-last_time):.4f}] epoch {epoch}, size: {get_seqlen(epoch)}")
    #print(f"[{(time.time()-start_time):.4f}][{(time.time()-last_time):.4f}] epoch {epoch}, size: {get_batch_size(epoch)}")
    last_time = time.time()

    inputs = torch.randint(0, gptconfig.vocab_size, (16, get_seqlen(epoch),), device="cuda")
    targets = torch.randint(0, gptconfig.vocab_size, (16, get_seqlen(epoch),), device="cuda")

    with ctx:
        _, loss = model(inputs, targets, return_logits=False)
    
    loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)

    print(loss.item())
