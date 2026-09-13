# Rollout divergence vs representation displacement — plot data

Qwen3-4B. self_gen: **30** held-out OpenThoughts3 prompts x 64 rollouts per model.
fixed_cot: the first 6 of those prompts (it needs one CoT per prompt, not a cloud).
Two axes, deliberately mirrored:

| | fixed | varies | measures |
|---|---|---|---|
| `self_gen/` | the **encoder** (always dense) | the text (each model's own rollouts) | where each model's generated text lands |
| `fixed_cot/` | the **text** (the dataset's CoT) | the encoder (each model in turn) | how far pruning moves the representation |

The overlay in `fixed_cot` is legitimate because ALPS, SparseGPT and ours all
mask and update the dense weights rather than retraining, so neuron *i* keeps its
meaning. It would be meaningless for two independently trained models.

---

## self_gen/pooled_L18_late_pca256_fp16.npz
Key: `p{0..29}|{model}|L18|late` → `(n_rollout, 256) float16`
One row per rollout: its hidden state averaged over continuation tokens
1024–2048, encoded by the **dense** model, then projected to 256 dims by a PCA
fit **per prompt over all models together**, so every cloud in a prompt shares
one basis.

The projection is what makes this transferable — 10x smaller — and it costs
nothing measurable: 256 components hold 99.9% of the variance and MMD agrees
with the full 2560 dims to three decimals (e.g. ALPS s70 prompt 0: 0.7648 full
vs 0.7661 projected). Going coarser starts to drift (PCA-64 reads 0.7910).
The untrimmed float32 originals are on disk, see the bottom of this file.

Models: `dense`, and `alps|sparsegpt|alps_sft|ours` at `s50|s60|s70`, plus `teacher`.
(`alps_sft` = ALPS followed by sparse SFT: the recovery-matched control. It is
kept here even though the figure composition drops it.)

`teacher` is the dataset's own CoT and has **1 row** — OpenThoughts3 ships one
trace per problem. Draw it as a landmark; it cannot carry an MMD.

### Recomputing MMD² on a common scale
`divergence.json` was computed with a bandwidth per comparison pair, which gives
each model its own kernel — a model whose cloud is more spread gets a wider one,
which shrinks its MMD. Use one bandwidth per prompt instead:

```python
import numpy as np
z = np.load("self_gen/pooled_L18_late_pca256_fp16.npz")
pi, sp = 0, "s70"
g = lambda k: z[f"p{pi}|{k}|L18|late"].astype(np.float32)

def mmd2(A, B, med):
    Z = np.concatenate([A, B]); sq = (Z*Z).sum(1)
    d2 = np.maximum(sq[:,None] + sq[None,:] - 2*(Z@Z.T), 0.0)
    K = np.exp(-d2/med); n = len(A)
    return float(K[:n,:n].mean() + K[n:,n:].mean() - 2*K[:n,n:].mean())

models = ["dense", f"alps:{sp}", f"sparsegpt:{sp}", f"ours:{sp}"]
Z = np.concatenate([g(m) for m in models]); sq = (Z*Z).sum(1)
d2 = np.maximum(sq[:,None] + sq[None,:] - 2*(Z@Z.T), 0.0)
med = np.median(d2[d2 > 0])
for m in models[1:]:
    print(m, round(mmd2(g("dense"), g(m), med), 4))
```

## fixed_cot/cot_states_L18_fp16.npz
Key: `p{0..5}|{model}|L18|cot` → `(n_seg, 2560)` — the dataset CoT cut into
256-token segments, each pooled, read by that model.
Key: `p{0..5}|dense_rollouts|L18|scale` → dense's own rollouts pooled over the
same span. This is the scale bar: a displacement only means something next to
how much dense's own generations scatter.

`cot_displacement.json` holds the derived numbers, keyed `L{layer}/{model}`,
as a per-prompt list in units of that scale.

---

## What the numbers say (layer 18)

Fixed-CoT representation displacement (units of the dense rollout cloud's spread):

| | s50 | s60 | s70 |
|---|---|---|---|
| ALPS | **2.11** | **2.69** | **3.43** |
| SparseGPT | 2.32 | 3.37 | 5.80 |
| SCOUT w/o OPD | 3.77 | 4.04 | 5.30 |
| SCOUT | 2.58 | 3.99 | 5.18 |

Self-gen MMD² to dense (shared bandwidth per prompt, n=30, ±SEM):

| | s50 | s60 | s70 | growth s50→s70 |
|---|---|---|---|---|
| SCOUT | **0.071±0.009** | **0.099±0.011** | **0.145±0.023** | 2.0x |
| SCOUT w/o OPD | 0.137±0.017 | 0.145±0.018 | 0.173±0.024 | **1.3x** |
| SFT | 0.106±0.014 | 0.166±0.025 | 0.200±0.026 | 1.9x |
| ALPS | 0.133±0.019 | 0.270±0.040 | 0.558±0.049 | 4.2x |
| SparseGPT | 0.195±0.023 | 0.359±0.041 | 0.835±0.063 | 4.3x |

An earlier 6-prompt pass gave lower values throughout (SCOUT s70 read 0.095);
those six happened to be a favourable draw. Use the 30-prompt numbers.

### What the w/o-OPD ablation says
`SCOUT w/o OPD` is SCOUT with the on-policy distillation term set to zero. With
no OPD loss no vLLM engine is built, so the PGD trust-region screening also
falls back to the fixed CoT data -- the run is off-policy in both the loss and
the mask screening, with the growth mechanism otherwise untouched.

Two things follow, and neither is the obvious one:

1. **The resistance to sparsity comes from the growth mechanism, not from OPD.**
   w/o OPD barely moves across sparsity (1.3x) while the one-shot baselines
   quadruple. Whatever keeps the generated distribution near dense as sparsity
   rises, it is support adaptation.

2. **OPD's contribution shrinks as sparsity rises**, on both metrics:

   | | s50 | s60 | s70 |
   |---|---|---|---|
   | MMD² gap (w/o OPD − SCOUT) | +0.066 | +0.046 | +0.028 |
   | paired p | 0.0005 | 0.021 | **0.18** |
   | SCOUT wins | 27/30 | 22/30 | 19/30 |
   | avg5 gap | +2.69 | +1.08 | (eval running) |

   At 70% the OPD effect is not significant. Claiming OPD as the central
   contribution invites exactly that challenge.

Representation displacement tells them apart too: w/o OPD and SCOUT are
indistinguishable there (s70: 5.30 vs 5.18), because they share the growth
mechanism. Same displacement, different MMD -- so proximity in representation
space does not explain the generation-distribution result, and that can now be
shown *within one method* rather than across families.

The ordering flips between the two. ALPS moves the representation least at every
sparsity, and its own generations land furthest from dense — 5.7x further than
ours at 70%. SCOUT moves the representation most and generates closest.

**Read this at layer 18 only.** At layer 36 the flip survives at s50 but not
beyond: ours is lowest on *both* axes at s60 and s70 (3.08 vs ALPS 3.70 at s70).
The last layer sits next to the output and tracks the self-generation ordering
instead, which is why the mid-layer is the informative place to look — and also
the more conservative one, since the last layer is dominated by next-token
prediction and would reflect lexical differences directly.

## Caveats worth carrying into any caption
- MMD² is between **rollout embeddings under a shared dense encoder**, not a
  distance between text distributions. Say so explicitly.
- AUC (also in `divergence.json`) saturates at ~1.0 from s50 upward and carries
  no information — rollouts from different models are trivially separable.
- self_gen is n = 30 prompts; fixed_cot is n = 6. Do not quote a shared n.
- t-SNE is for illustration; the MMD curve is the evidence.

## Full, untrimmed sources on disk
```
elsa/logs/policy_divergence/n30_k64_core/       30 prompts, both layers/windows, float32, full 2560 dims
elsa/logs/policy_divergence/n6_k96_base/        6 prompts x 96 rollouts, all 17 models incl. s30/s40
elsa/logs/policy_divergence/cot_through_models/ both layers
```
