import os
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

X_train = np.load("/mnt/ddp/sample/X_train.npy")
y_train = np.load("/mnt/ddp/sample/y_train.npy")
X_val = np.load("/mnt/ddp/sample/X_val.npy")
y_val = np.load("/mnt/ddp/sample/y_val.npy")

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return {
            'X': torch.tensor(self.X[index], dtype=torch.float32),
            'y': torch.tensor(self.y[index], dtype=torch.long)
        }
    
train_dataset = SimpleDataset(X_train, y_train)
val_dataset = SimpleDataset(X_val, y_val)

class SimpleModel(torch.nn.Module):
    def __init__(self, fin, fout) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(fin, fout)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.linear(x)
        return self.softmax(logits)

assert torch.cuda.is_available(), "GPU is not available!"    
is_ddp = os.getenv('RANK', -1) != -1
print(f'[x] DDP available: {is_ddp}')

global_rank = 0
local_rank = 0
world_size = 1
sampler = None

if is_ddp:
    global_rank = int(os.getenv('RANK'))
    local_rank = int(os.getenv('LOCAL_RANK'))
    world_size = int(os.getenv('WORLD_SIZE'))
    dist.init_process_group(backend="nccl")

torch.cuda.set_device(local_rank)
    
if global_rank == 0:
    print(f'[x] Local Rank: {local_rank}')
    print(f'[x] World Size: {world_size}')

if is_ddp:
    sampler = DistributedSampler(train_dataset, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, sampler=sampler)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

fin = X_train.shape[1]
fout = len(np.unique(y_train))

device = f"cuda:{local_rank}"
model = SimpleModel(fin=fin, fout=fout).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
bias_params = [p for name, p in model.named_parameters() if 'bias' in name]
others = [p for name, p in model.named_parameters() if 'bias' not in name]
total_params = sum(p.numel() for p in model.parameters())
print(f'[x] total params: {total_params}')
optimizer = torch.optim.AdamW(params=[
                {'params': others, 'weight_decay': 1e-4},
                {'params': bias_params, 'weight_decay': 0}
            ], lr=1e-4)

if is_ddp:
    model = DistributedDataParallel(model, device_ids=[local_rank])

n_epochs = 100
for e in range(n_epochs):
    model.train()
    for idx, batch in enumerate(train_dataloader):
        X = batch['X'].to(device)
        y = batch['y'].to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    model.eval()
    valid_loss = []
    with torch.no_grad():
        for vidx, vbatch in enumerate(val_dataloader):
            vX = vbatch['X'].to(device)
            vy = vbatch['y'].to(device)
        vy_pred = model(vX)
        vloss = loss_fn(vy_pred, vy).detach().cpu().numpy()
        valid_loss.append(vloss)
    print(f'[x] epoch: {e} | train loss: {loss.detach().cpu().numpy():.6f} | valid loss: {np.mean(valid_loss):.6f}')

dist.destroy_process_group()
         
        


        