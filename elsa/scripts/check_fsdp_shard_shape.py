import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

MODEL = "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True)
model.to(device)

_block_cls = type(model.model.layers[0])
_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={_block_cls})
_mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
model = FSDP(model, auto_wrap_policy=_wrap_policy, mixed_precision=_mp,
             sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True,
             device_id=torch.cuda.current_device())

for name, p in model.named_parameters():
    if "layers.0." in name and "weight" in name and "mlp" in name:
        print(f"[rank {local_rank}] {name}: shape={tuple(p.shape)} data.shape={tuple(p.data.shape)} numel={p.data.numel()} dim={p.data.dim()}")

dist.barrier()
dist.destroy_process_group()
