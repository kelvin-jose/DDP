from litgpt.model import GPT
from litgpt.config import Config
from dataloader import DistributedDataLoader

import os
import tqdm
import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


assert torch.cuda.is_available(), "GPU is not available!"    
is_ddp = os.getenv('RANK', -1) != -1
print('[x] is DDP:', is_ddp)

if is_ddp:
    rank = os.getenv('RANK')
    local_rank = int(os.getenv('LOCAL_RANK'))
    world_size = int(os.getenv('WORLD_SIZE'))
    dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)

micro_llama = dict(
        name="micro-llama-4M",
        hf_config=dict(org="3rdplanet", name="MicroLlama"),
        block_size=256,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=4,
        n_head=4,
        n_embd=64,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",  # original TinyLlama and MicroLlama use FusedRMSNorm
        norm_eps=1e-5,
        mlp_class_name="LLaMAMLP",
        intermediate_size=256,
        n_query_groups=4)

config = Config(**micro_llama)
model = GPT(config)
model.cuda()
model = DistributedDataParallel(model, device_ids=[local_rank])
loss_fn = torch.nn.CrossEntropyLoss()
bias_params = [p for name, p in model.named_parameters() if 'bias' in name]
others = [p for name, p in model.named_parameters() if 'bias' not in name]
total_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(params=[
                {'params': others, 'weight_decay': 1e-4},
                {'params': bias_params, 'weight_decay': 0}
            ], lr=5e-3)

print('number of params:', sum(p.numel() for p in model.parameters()))

T = 32
B = 64
train_dataloader = DistributedDataLoader("/home/machine-x/Downloads/archive/train.pkl", B, T, rank, world_size)
val_dataloader = DistributedDataLoader("/home/machine-x/Downloads/archive/val.pkl", T = 32)

epochs = 10
num_iters = 1000
for e in range(epochs):
    model.train()
    for iter in tqdm.tqdm(range(num_iters), desc=f"Epoch {e}"): 
        batch = train_dataloader.get_batch()
        if batch['eod']:
            break
        X = batch['x'].cuda()
        y = batch['y'].cuda()
        optimizer.zero_grad()
        y_pred = model(X)
        b,t,c = y_pred.shape
        loss = loss_fn(y_pred.view(b*t, c), y.view(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    valid_loss = []
    with torch.no_grad():
        for iter in range(num_iters):
            vbatch = val_dataloader.get_batch()
            vX = vbatch['x'].cuda()
            vy = vbatch['y'].cuda()
        vy_pred = model(vX)
        b,t,c = vy_pred.shape
        vloss = loss_fn(vy_pred.view(b*t, c), vy.view(-1))
        valid_loss.append(vloss.detach().cpu().numpy())

    print(f'[x] epoch: {e} | train loss: {loss.detach().cpu().numpy():.6f} | valid loss: {np.mean(valid_loss): .6f}')

dist.destroy_process_group()