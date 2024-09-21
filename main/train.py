from litgpt.model import GPT
from litgpt.config import Config
from dataloader import DistributedDataLoader

import os
import torch
import torch.distributed as dist

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
print('number of params:', sum(p.numel() for p in model.parameters()))
