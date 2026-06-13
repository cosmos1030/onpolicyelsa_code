"""
Pinpoint the bug: check what position_ids shape the model uses for bs=2 no-pad,
and what cu_seqlens _prepare_from_posids generates.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_flash_attention_utils import _prepare_from_posids

MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

TEXT = "The answer to 1+1 is 2. What is 2+2?"
enc = tok(TEXT, return_tensors="pt")
input_ids = enc["input_ids"]
S = input_ids.shape[1]
B = 2
input_ids_b2 = input_ids.repeat(B, 1).cuda()
attn_mask_b2 = enc["attention_mask"].repeat(B, 1).cuda()

print(f"B={B}, S={S}")
print(f"attention_mask.all() = {attn_mask_b2.all()}")

# Simulate what the model does for position_ids when none provided
import torch
cache_position = torch.arange(0, S, device="cuda")
position_ids_broadcast = cache_position.unsqueeze(0)  # (1, S) — what model creates
position_ids_full = cache_position.unsqueeze(0).expand(B, -1)  # (B, S)

print(f"\nposition_ids_broadcast shape: {position_ids_broadcast.shape}")
print(f"position_ids_full shape:      {position_ids_full.shape}")

# What cu_seqlens does _prepare_from_posids generate for each?
B_h, H, D = B, 12, 128
Hkv = 2
q = torch.randn(B, S, H,   D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device="cuda")

_, _, _, _, (cu_q_broad, cu_k_broad), (mq_broad, _) = _prepare_from_posids(q, k, v, position_ids_broadcast)
_, _, _, _, (cu_q_full, cu_k_full), (mq_full, _) = _prepare_from_posids(q.clone(), k.clone(), v.clone(), position_ids_full)

print(f"\n_prepare_from_posids with (1,S) position_ids:")
print(f"  cu_seqlens = {cu_q_broad.tolist()}, max_seqlen = {mq_broad}")

print(f"\n_prepare_from_posids with (B,S) position_ids:")
print(f"  cu_seqlens = {cu_q_full.tolist()}, max_seqlen = {mq_full}")

# Now verify that the wrong cu_seqlens causes wrong output
from flash_attn import flash_attn_varlen_func
q_flat = q.reshape(B*S, H, D)
k_flat = k.reshape(B*S, Hkv, D)
v_flat = v.reshape(B*S, Hkv, D)

out_wrong = flash_attn_varlen_func(q_flat, k_flat, v_flat,
    cu_seqlens_q=cu_q_broad, cu_seqlens_k=cu_k_broad, max_seqlen_q=mq_broad, max_seqlen_k=mq_broad, causal=True)
out_correct = flash_attn_varlen_func(q_flat, k_flat, v_flat,
    cu_seqlens_q=cu_q_full, cu_seqlens_k=cu_k_full, max_seqlen_q=mq_full, max_seqlen_k=mq_full, causal=True)

print(f"\nout_wrong  shape={out_wrong.shape}  sum={out_wrong.float().sum().item():.2f}")
print(f"out_correct shape={out_correct.shape}  sum={out_correct.float().sum().item():.2f}")
print(f"Max diff: {(out_wrong[:S] - out_correct[:S]).abs().max().item():.4f}")

# Now check what actual model forward uses as position_ids
# by hooking the attention forward
model_fa2 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
).cuda().eval()

captured = {}
orig_prepare = _prepare_from_posids.__wrapped__ if hasattr(_prepare_from_posids, '__wrapped__') else None

import transformers.modeling_flash_attention_utils as fa_utils
orig_fn = fa_utils._prepare_from_posids

def patched_prepare(query, key, value, position_ids):
    captured['position_ids_shape'] = position_ids.shape
    captured['position_ids_vals'] = position_ids.flatten()[:8].tolist()
    return orig_fn(query, key, value, position_ids)

fa_utils._prepare_from_posids = patched_prepare

with torch.no_grad():
    out = model_fa2(input_ids=input_ids_b2, attention_mask=attn_mask_b2)

if captured:
    print(f"\nActual position_ids in model forward:")
    print(f"  shape = {captured['position_ids_shape']}")
    print(f"  first 8 values = {captured['position_ids_vals']}")
else:
    print("\n_prepare_from_posids was NOT called (dense path used)")
    # Check with attention_mask=None
    out2 = model_fa2(input_ids=input_ids_b2)
    if captured:
        print(f"With attention_mask=None:")
        print(f"  position_ids shape = {captured['position_ids_shape']}")

fa_utils._prepare_from_posids = orig_fn
print("\nDone.")
