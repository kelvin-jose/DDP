import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

X_train = np.load("/mnt/data/sample/X_train.npy")
y_train = np.load("/mnt/data/sample/y_train.npy")
X_val = np.load("/mnt/data/sample/X_val.npy")
y_val = np.load("/mnt/data/sample/y_val.npy")

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
            'y': torch.tensor(self.y[index], dtype=torch.int16)
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
    
is_ddp = os.getenv('RANK', -1) != -1
global_rank = 0
local_rank = 0
world_size = 1
sampler = None

if is_ddp:
    global_rank = os.getenv('RANK')
    local_rank = os.getenv('LOCAL_RANK')
    world_size = os.getnv('WORLD_SIZE')
    sampler = DistributedSampler(train_dataset, shuffle=True)

print(f'[x] DDP available: {is_ddp}')

if global_rank == 0:
    print(f'[x] Local Rank: {local_rank}')
    print(f'[x] World Size: {world_size}')

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=sampler)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)





         
        


        