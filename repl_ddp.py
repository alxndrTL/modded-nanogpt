import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
torch.set_float32_matmul_precision("high")

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
    return 16+int(2*it)

def get_seqlen(it):
    return 16+int(2*it)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_0 = nn.Linear(D_MODEL, D_MODEL)
        self.fc_1 = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        x = self.fc_0(x)
        x = self.fc_1(x)
        return x

model = MyModel().to(device)
model = torch.compile(model, dynamic=None)
model = DDP(model, device_ids=[ddp_local_rank])

start_time = time.time()
for epoch in range(100):
    print(f"[{(time.time()-start_time):.2f}] epoch {epoch}. len={get_seqlen(epoch)}.")

    inputs = torch.randn((16, get_seqlen(epoch), D_MODEL), device=device)
    logits = model(inputs)

dist.destroy_process_group()