"""
Isolate flash_attn_2 vs SDPA correctness.
Loads same model twice with different attn_impl, runs identical forward,
compares outputs and loss. Batch=1 (no padding), then batch=2.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "right"

# Fixed input - same tokens every time
TEXT = "The answer to 1+1 is 2. The answer to 2+2 is 4. The answer to 3+3 is"
enc = tok(TEXT, return_tensors="pt")
input_ids = enc["input_ids"]         # (1, seq)
attention_mask = enc["attention_mask"]
labels = input_ids.clone()

print(f"Input shape: {input_ids.shape}, tokens: {input_ids.shape[1]}")
print(f"pad_token_id={tok.pad_token_id}, eos_token_id={tok.eos_token_id}")

def run_forward(impl, batch_ids, batch_mask, batch_labels, tag):
    print(f"\n--- {tag} (attn_impl={impl}, batch={batch_ids.shape[0]}) ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        attn_implementation=impl,
        low_cpu_mem_usage=True,
    ).cuda().eval()

    with torch.no_grad():
        out = model(
            input_ids=batch_ids.cuda(),
            attention_mask=batch_mask.cuda(),
            labels=batch_labels.cuda(),
        )
    loss = out.loss.item()
    logits_sum = out.logits.float().sum().item()

    # Check for NaN/Inf in logits
    nan_count = out.logits.isnan().sum().item()
    inf_count = out.logits.isinf().sum().item()
    print(f"  loss={loss:.4f}  logits_sum={logits_sum:.2f}  nan={nan_count}  inf={inf_count}")
    del model
    torch.cuda.empty_cache()
    return loss

# ── batch=1, no padding ────────────────────────────────────────────────────
b1_ids   = input_ids
b1_mask  = attention_mask
b1_label = labels

loss_sdpa_b1  = run_forward("sdpa",              b1_ids, b1_mask, b1_label, "SDPA  bs=1")
loss_flash_b1 = run_forward("flash_attention_2", b1_ids, b1_mask, b1_label, "FA2   bs=1")

# ── batch=2, no padding (same sequence duplicated) ─────────────────────────
b2_ids   = input_ids.repeat(2, 1)
b2_mask  = attention_mask.repeat(2, 1)
b2_label = labels.repeat(2, 1)

loss_sdpa_b2  = run_forward("sdpa",              b2_ids, b2_mask, b2_label, "SDPA  bs=2")
loss_flash_b2 = run_forward("flash_attention_2", b2_ids, b2_mask, b2_label, "FA2   bs=2")

# ── batch=2, WITH padding (different lengths) ──────────────────────────────
TEXT2 = "What is 5+5?"
enc2 = tok(TEXT2, return_tensors="pt")
# pad shorter to match longer
max_len = max(input_ids.shape[1], enc2["input_ids"].shape[1])
pad_len = max_len - enc2["input_ids"].shape[1]
ids_pad   = torch.cat([enc2["input_ids"], torch.full((1, pad_len), tok.pad_token_id)], dim=1)
mask_pad  = torch.cat([enc2["attention_mask"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1)
label_pad = torch.cat([enc2["input_ids"].clone(), torch.full((1, pad_len), -100)], dim=1)

b2p_ids   = torch.cat([input_ids, ids_pad], dim=0)
b2p_mask  = torch.cat([attention_mask, mask_pad], dim=0)
b2p_label = torch.cat([labels, label_pad], dim=0)

print(f"\nPadded batch shapes: ids={b2p_ids.shape}, mask_sum={b2p_mask.sum(dim=1).tolist()}")
loss_sdpa_b2p  = run_forward("sdpa",              b2p_ids, b2p_mask, b2p_label, "SDPA  bs=2+pad")
loss_flash_b2p = run_forward("flash_attention_2", b2p_ids, b2p_mask, b2p_label, "FA2   bs=2+pad")

print("\n=== SUMMARY ===")
print(f"  SDPA  bs=1:       {loss_sdpa_b1:.4f}")
print(f"  FA2   bs=1:       {loss_flash_b1:.4f}  {'OK' if abs(loss_flash_b1 - loss_sdpa_b1) < 0.1 else 'WRONG'}")
print(f"  SDPA  bs=2:       {loss_sdpa_b2:.4f}")
print(f"  FA2   bs=2:       {loss_flash_b2:.4f}  {'OK' if abs(loss_flash_b2 - loss_sdpa_b2) < 0.1 else 'WRONG'}")
print(f"  SDPA  bs=2+pad:   {loss_sdpa_b2p:.4f}")
print(f"  FA2   bs=2+pad:   {loss_flash_b2p:.4f}  {'OK' if abs(loss_flash_b2p - loss_sdpa_b2p) < 0.1 else 'WRONG'}")
