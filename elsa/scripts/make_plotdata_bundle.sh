#!/bin/bash
# One bundle with both axes of the rollout/representation analysis.
#   self-gen  : each model's own rollouts, all embedded by the DENSE model
#   fixed-CoT : the dataset's CoT, read by each model in turn
# Trimmed to layer 18 / late window and float16 so it stays under the 30 MB
# transfer limit; the untrimmed sources stay on disk under logs/.
set -e
P=/home1/doyoonkim/miniconda3/envs/rac/bin/python
ROOT=/home1/doyoonkim/projects/elsa
SELF=$ROOT/logs/policy_divergence/n6_k96_base
COT=$ROOT/logs/policy_divergence/cot_through_models
STAGE=/tmp/plotdata_bundle_$$
OUT=$ROOT/figures/plotdata.zip

rm -rf "$STAGE"; mkdir -p "$STAGE"/{self_gen,fixed_cot,scripts}

$P - "$SELF" "$STAGE" <<'PY'
import numpy as np, json, sys, os
src, stage = sys.argv[1], sys.argv[2]
z = np.load(os.path.join(src, "pooled.npz"))
WANT = {"dense", "teacher"} | {f"{f}:{s}" for f in ("alps", "sparsegpt", "ours")
                               for s in ("s50", "s60", "s70")}
keep = {k: z[k].astype(np.float16) for k in z.files
        if "|L18|late" in k and k.split("|")[1] in WANT}
np.savez_compressed(os.path.join(stage, "self_gen", "pooled_L18_late_fp16.npz"), **keep)
m = json.load(open(os.path.join(src, "pooled_meta.json")))
m.update({"subset": "layer 18, late window (continuation tokens 1024-2048), float16",
          "windows": {"late": m["windows"]["late"]}, "layers": [18]})
json.dump(m, open(os.path.join(stage, "self_gen", "pooled_meta.json"), "w"), indent=2)
print(f"  self_gen: {len(keep)} arrays")
PY

cp "$SELF/prompts.json" "$SELF/divergence.json" "$STAGE/self_gen/"

if [ -f "$COT/cot_states.npz" ]; then
  $P - "$COT" "$STAGE" <<'PY'
import numpy as np, json, sys, os
src, stage = sys.argv[1], sys.argv[2]
z = np.load(os.path.join(src, "cot_states.npz"))
keep = {k: z[k].astype(np.float16) for k in z.files if "|L18|" in k}
np.savez_compressed(os.path.join(stage, "fixed_cot", "cot_states_L18_fp16.npz"), **keep)
print(f"  fixed_cot: {len(keep)} arrays")
PY
  cp "$COT/cot_states_meta.json" "$STAGE/fixed_cot/" 2>/dev/null || true
fi
cp "$COT/cot_displacement.json" "$STAGE/fixed_cot/" 2>/dev/null || true

cp "$ROOT/scripts/make_bylevel_figure.py" "$ROOT/scripts/cot_through_models.py" \
   "$ROOT/scripts/policy_divergence_tsne.py" "$STAGE/scripts/"

cp "$ROOT/figures/BUNDLE_README.md" "$STAGE/README.md"

cd "$(dirname "$STAGE")" && rm -f "$OUT" && zip -qr "$OUT" "$(basename "$STAGE")" && \
  cd / && rm -rf "$STAGE"
ls -la "$OUT" | awk '{printf "  %s  %.1f MB\n", $9, $5/1048576}'
