# Rollout divergence vs representation displacement — plot data

Qwen3-4B, 6 held-out OpenThoughts3 prompts, 96 rollouts per model per prompt.
Two axes, deliberately mirrored:

| | fixed | varies | measures |
|---|---|---|---|
| `self_gen/` | the **encoder** (always dense) | the text (each model's own rollouts) | where each model's generated text lands |
| `fixed_cot/` | the **text** (the dataset's CoT) | the encoder (each model in turn) | how far pruning moves the representation |

The overlay in `fixed_cot` is legitimate because ALPS, SparseGPT and ours all
mask and update the dense weights rather than retraining, so neuron *i* keeps its
meaning. It would be meaningless for two independently trained models.

---

## self_gen/pooled_L18_late_fp16.npz
Key: `p{0..5}|{model}|L18|late` → `(n_rollout, 2560) float16`
One row per rollout: its hidden state averaged over continuation tokens
1024–2048, encoded by the **dense** model.

Models: `dense`, `alps:s50|s60|s70`, `sparsegpt:s50|s60|s70`, `ours:s50|s60|s70`,
and `teacher`.

`teacher` is the dataset's own CoT and has **1 row** — OpenThoughts3 ships one
trace per problem. Draw it as a landmark; it cannot carry an MMD.

### Recomputing MMD² on a common scale
`divergence.json` was computed with a bandwidth per comparison pair, which gives
each model its own kernel — a model whose cloud is more spread gets a wider one,
which shrinks its MMD. Use one bandwidth per prompt instead:

```python
import numpy as np
z = np.load("self_gen/pooled_L18_late_fp16.npz")
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

| | fixed-CoT displacement (s70) | self-gen MMD² s50 / s60 / s70 |
|---|---|---|
| ALPS | **3.43** (smallest) | 0.085 / 0.200 / 0.507 |
| SparseGPT | 5.80 | 0.155 / 0.378 / 0.931 |
| SCOUT (ours) | 5.18 | **0.059 / 0.083 / 0.089** |

ALPS moves the representation least on fixed text and its own generations land
furthest from dense; ours is the reverse.

**Read this at layer 18 only.** At layer 36 the reversal is absent — ALPS 3.70 vs
ours 3.08 — because the last layer sits next to the output and tracks the
self-generation ordering instead.

## Caveats worth carrying into any caption
- MMD² is between **rollout embeddings under a shared dense encoder**, not a
  distance between text distributions. Say so explicitly.
- AUC (also in `divergence.json`) saturates at ~1.0 from s50 upward and carries
  no information — rollouts from different models are trivially separable.
- n = 6 prompts. A 30-prompt rerun is in progress; prefer it for a final figure.
- t-SNE is for illustration; the MMD curve is the evidence.

## Full, untrimmed sources on disk
```
elsa/logs/policy_divergence/n6_k96_base/        both layers, both windows, all 17 models, float32
elsa/logs/policy_divergence/cot_through_models/ both layers
elsa/logs/policy_divergence/n30_k64_core/       the 30-prompt rerun (in progress)
```
