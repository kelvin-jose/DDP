from litgpt.model import GPT
from litgpt.config import Config

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
