"""
Bypass transformers entirely. Call flash_attn_varlen_func and flash_attn_func directly
with manually constructed Q/K/V and compare against torch SDPA.
Isolates whether the bug is in flash_attn or in transformers' wrapper.
"""
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input

torch.manual_seed(42)
B, S, H, Hkv, D = 2, 31, 12, 2, 128
dtype = torch.bfloat16

q = torch.randn(B, S, H,   D, dtype=dtype, device="cuda") * 0.1
k = torch.randn(B, S, Hkv, D, dtype=dtype, device="cuda") * 0.1
v = torch.randn(B, S, Hkv, D, dtype=dtype, device="cuda") * 0.1

# ── Reference: torch SDPA ─────────────────────────────────────────────────
# expand kv for GQA
k_exp = k.repeat_interleave(H // Hkv, dim=2)  # (B, S, H, D)
v_exp = v.repeat_interleave(H // Hkv, dim=2)

q_t = q.transpose(1, 2)       # (B, H, S, D)
k_t = k_exp.transpose(1, 2)
v_t = v_exp.transpose(1, 2)
mask_causal = torch.ones(S, S, device="cuda").tril().bool()
sdpa_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
sdpa_out = sdpa_out.transpose(1, 2)  # (B, S, H, D)

# ── FA2: dense flash_attn_func (bs=2, no-padding, no attention_mask) ───────
fa2_dense = flash_attn_func(q, k, v, causal=True)

# ── FA2: varlen with no-padding cu_seqlens (bs=2 seqs of len 31) ──────────
q_flat = q.reshape(B * S, H,   D)
k_flat = k.reshape(B * S, Hkv, D)
v_flat = v.reshape(B * S, Hkv, D)
cu_seqlens = torch.tensor([0, S, 2 * S], dtype=torch.int32, device="cuda")
fa2_varlen_nopad = flash_attn_varlen_func(
    q_flat, k_flat, v_flat,
    cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
    max_seqlen_q=S, max_seqlen_k=S, causal=True,
)
fa2_varlen_nopad = fa2_varlen_nopad.reshape(B, S, H, D)

# ── FA2: varlen with padded case (seq1=31, seq2=8, padded to 31) ──────────
S2 = 8
# build padded batch: seq1 full (31), seq2 only first 8 tokens, rest zero-padded
q2 = torch.cat([q[1:2, :S2], torch.zeros(1, S - S2, H,   D, dtype=dtype, device="cuda")], dim=1)
k2 = torch.cat([k[1:2, :S2], torch.zeros(1, S - S2, Hkv, D, dtype=dtype, device="cuda")], dim=1)
v2 = torch.cat([v[1:2, :S2], torch.zeros(1, S - S2, Hkv, D, dtype=dtype, device="cuda")], dim=1)
q_padded = torch.cat([q[0:1], q2], dim=0)  # (2, 31, H, D)
k_padded = torch.cat([k[0:1], k2], dim=0)
v_padded = torch.cat([v[0:1], v2], dim=0)
mask_pad = torch.zeros(2, S, dtype=torch.long, device="cuda")
mask_pad[0, :] = 1
mask_pad[1, :S2] = 1  # seq2 only has S2 real tokens

cu_seqlens_pad = torch.tensor([0, S, S + S2], dtype=torch.int32, device="cuda")
q_up, _, _, _, _ = unpad_input(q_padded, mask_pad)
k_up, _, _, _, _ = unpad_input(k_padded, mask_pad)
v_up, _, _, _, _ = unpad_input(v_padded, mask_pad)
fa2_varlen_padded_out = flash_attn_varlen_func(
    q_up, k_up, v_up,
    cu_seqlens_q=cu_seqlens_pad, cu_seqlens_k=cu_seqlens_pad,
    max_seqlen_q=S, max_seqlen_k=S, causal=True,
)
fa2_varlen_padded_seq1 = fa2_varlen_padded_out[:S]  # first S tokens = seq1

# ── Compare first sequence output ──────────────────────────────────────────
ref = sdpa_out[0]  # (S, H, D)
d_dense      = (fa2_dense[0] - ref).abs().mean().item()
d_varlen     = (fa2_varlen_nopad[0] - ref).abs().mean().item()
d_varlen_pad = (fa2_varlen_padded_seq1 - ref).abs().mean().item()

print("=== direct flash_attn vs SDPA (first sequence of batch) ===")
print(f"  flash_attn_func    (dense, bs=2):            mean_abs_diff = {d_dense:.6f}  {'OK' if d_dense < 0.01 else 'WRONG'}")
print(f"  flash_attn_varlen  (no-pad, cu=[0,31,62]):   mean_abs_diff = {d_varlen:.6f}  {'OK' if d_varlen < 0.01 else 'WRONG'}")
print(f"  flash_attn_varlen  (padded, cu=[0,31,39]):   mean_abs_diff = {d_varlen_pad:.6f}  {'OK' if d_varlen_pad < 0.01 else 'WRONG'}")

print(f"\nSDPA            output[0] sum: {ref.float().sum().item():.4f}")
print(f"FA2 dense              [0] sum: {fa2_dense[0].float().sum().item():.4f}")
print(f"FA2 varlen_nopad       [0] sum: {fa2_varlen_nopad[0].float().sum().item():.4f}")
print(f"FA2 varlen_padded seq1    sum: {fa2_varlen_padded_seq1.float().sum().item():.4f}")
