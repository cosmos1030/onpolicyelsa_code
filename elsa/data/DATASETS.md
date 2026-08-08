# OT3/FineWeb dataset naming

Three distinct OpenThoughts3+FineWeb-Edu builds exist in this directory and get
confused easily. Use these short names when discussing them; symlinks with the
same names point at the real files below.

| Short name | Real file | Tokenizer | Notes |
|---|---|---|---|
| **LEGACY-20K** | `ot3_fineweb_20k.jsonl` | DeepSeek-R1-Distill-Qwen-1.5B | Built pre-08-04. Build script never got committed to git and is lost — reverse-engineered by diffing against raw OpenThoughts3-1.2M. ~62% of rows keep the full `<think>` block as-is; the other ~38% have `<think>...</think>` stripped down to just the post-think text (verified byte-for-byte against raw OT3). Median ~2450 tokens, 51.0% truncate at seqlen=2048. Incompatible with Qwen3 (wrong special tokens) — do not use for training, kept only as a reference/comparison point. |
| **PLAIN-200K** | `ot3_fineweb_200k_qwen3_train.jsonl` | Qwen3 | Built 2026-08-01/04. No truncation mitigation — median ~16.8k tokens (OT3 responses commonly run very long), 80.3% truncate at seqlen=2048, and ~26.5% of truncated rows lose the final `\boxed{}` answer entirely. This is the dataset behind the "TR-GMP got worse after 08-04" regression. |
| **THINKSTRIP-200K** | `ot3_fineweb_200k_qwen3_thinkstrip.jsonl` | Qwen3 | Built 2026-08-08 via `scripts/build_ot3_fineweb_dataset.py --strip_think_if_long`, reproducing the LEGACY-20K mechanism on purpose: if a rendered conversation exceeds seqlen(2048) tokens, drop `<think>...</think>` and keep only the post-think write-up. No length-based filtering, so domain mix (math/code/science) is untouched. Measured 48.5% truncation at seqlen=2048 (slightly better than LEGACY-20K's 51.0%). ~62% of rows have no short "clean final answer" to fall back to (the whole response is reasoning with no separate wrap-up) and stay long even after stripping — expected, not a bug. **Use this one for new training runs.**

Symlinks (`data/OT3FW_LEGACY-20K.jsonl`, `data/OT3FW_PLAIN-200K.jsonl`,
`data/OT3FW_THINKSTRIP-200K.jsonl`) exist purely for readability; the
underlying files/paths above are what scripts actually reference.
