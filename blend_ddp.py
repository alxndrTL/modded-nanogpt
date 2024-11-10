import os
import math
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
torch.set_float32_matmul_precision("high")

N_LAYERS = 12
VOCAB_SIZE = 8000
D_MODEL = 768
device = "cuda:0"

# setup DDP
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0)

def get_batch_size(it):
    return min(30, int(2*it))

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

def rmsnorm(x0):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + torch.finfo(x0.dtype).eps)
    return x.type_as(x0)

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

model = GPT(GPTConfig()).to(device)
model = torch.compile(model, dynamic=False)
model = DDP(model, device_ids=[ddp_local_rank])

start_time = time.time()
for epoch in range(100):
    print(f"[{(time.time()-start_time):.2f}] epoch {epoch}. len={get_seqlen(epoch)}.")

    inputs = torch.randint(low=0, high=VOCAB_SIZE, size=(16, get_seqlen(epoch)), device=device)
    a, b = model(inputs)

dist.destroy_process_group()