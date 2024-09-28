from litgpt.model import GPT
from litgpt.config import Config
from dataloader import DistributedDataLoader

import os
import tqdm
import torch
import numpy as np
import torch.distributed as dist
from argparse import ArgumentParser
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

if __name__ == '__main__':
        parser = ArgumentParser(description="DDP training of a small Llama on a toy dataset.")
        parser.add_argument("--block_size", type=int, default=256, help="Maximum sequence length (max: 256)")
        parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size.")
        parser.add_argument('--padding_multiple', type=int, default=64, help='Padding multiple (default: 64)')
        parser.add_argument('--n_layer', type=int, default=4, help='Number of layers (default: 4)')
        parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads (default: 4)')
        parser.add_argument('--n_embd', type=int, default=64, help='Embedding dimension (default: 64)')
        parser.add_argument('--rotary_percentage', type=float, default=1.0, help='Rotary percentage (default: 1.0)')
        parser.add_argument('--bias', type=bool, default=False, help='Use bias in layers (default: False)')
        parser.add_argument('--intermediate_size', type=int, default=256, help='Intermediate size (default: 256)')
        parser.add_argument('--n_query_groups', type=int, default=4, help='Number of query groups (default: 4)')
        parser.add_argument('--train_file', type=str, required=True, help="Path to the training pkl file.")
        parser.add_argument('--valid_file', type=str, required=True, help="Path to the validation pkl file.")
        parser.add_argument('--train_seq_len', type=int, default=32, help="Sequence length of the training input.")
        parser.add_argument('--train_batch_size', type=int, default=64, help="Training batch size.")
        parser.add_argument('--valid_seq_len', type=int, default=32, help="Sequence length of the validation input.")
        parser.add_argument('--valid_batch_size', type=int, default=64, help="Validation batch size.")
        parser.add_argument('--train_epochs', type=int, default=10, help="Number of epochs.")
        parser.add_argument('--train_steps', type=int, default=1000, help="Number of training steps.")
        parser.add_argument('--valid_steps', type=int, default=1000, help="Number of validation steps.")

        args = parser.parse_args()
        
        micro_llama = dict(
        name="micro-llama-4M",
        hf_config=dict(org="3rdplanet", name="MicroLlama"),
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        padding_multiple=args.padding_multiple,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        rotary_percentage=args.rotary_percentage,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",  # original TinyLlama and MicroLlama use FusedRMSNorm
        norm_eps=1e-5,
        mlp_class_name="LLaMAMLP",
        intermediate_size=args.intermediate_size,
        n_query_groups=args.n_query_groups)

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

        T = args.train_seq_len
        B = args.train_batch_size
        
        train_dataloader = DistributedDataLoader(args.train_file, B, T, rank, world_size)
        val_dataloader = DistributedDataLoader(args.valid_file, T = args.valid_seq_len)

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
                if rank == 0:
                        print(f'[x] epoch: {e} | train loss: {loss.detach().cpu().numpy():.6f} | valid loss: {np.mean(valid_loss): .6f}')

        dist.destroy_process_group()