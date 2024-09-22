import torch
from pickle import load

class DistributedDataLoader:
    def __init__(self, filename: str, B: int = 1, T: int = 8, global_rank: int = 0, world_size: int = 1):
        self.global_rank = global_rank
        self.world_size = world_size
        self.B = B
        self.T = T
        # self.bos = [1]
        # self.eos = [2]
        self.curr_index = int(self.global_rank * self.B * self.T)
        with open(filename, 'rb') as f:
            self.tokens = load(f)
    
    def get_batch(self):
        batch = dict(eod=False,x=None,y=None)
        buffer = self.tokens[self.curr_index: self.curr_index + self.B * self.T + 1]
        x = torch.tensor(buffer[:-1], dtype=torch.long).view(self.B, self.T)
        y = torch.tensor(buffer[1:], dtype=torch.long).view(self.B, self.T)
        self.curr_index += self.world_size * self.B * self.T
        if (self.curr_index + self.B * self.T + 1) > len(self.tokens):
            self.curr_index = int(self.global_rank * self.B * self.T)
            batch['eod'] = True
        batch['x'] = x
        batch['y'] = y
        return batch




