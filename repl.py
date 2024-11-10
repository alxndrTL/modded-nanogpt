import time

import torch
import torch.nn as nn
torch.set_float32_matmul_precision("high")

D_MODEL = 768
device = "cuda:0"

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

start_time = time.time()
for epoch in range(50):
    print(f"[{(time.time()-start_time):.2f}] epoch {epoch}. bsz={get_seqlen(epoch)}.")

    inputs = torch.randn((16, get_seqlen(epoch), D_MODEL), device=device)
    logits = model(inputs)
