#!/bin/bash
# One bundle with both axes of the rollout/representation analysis.
#   self-gen  : each model's own rollouts, all embedded by the DENSE model
#   fixed-CoT : the dataset's CoT, read by each model in turn
# Trimmed to layer 18 / late window and float16 so it stays under the 30 MB
# transfer limit; the untrimmed sources stay on disk under logs/.
set -e
P=/home1/doyoonkim/miniconda3/envs/rac/bin/python
ROOT=/home1/doyoonkim/projects/elsa
SELF=${SELF_DIR:-$ROOT/logs/policy_divergence/n30_k64_core}
COT=$ROOT/logs/policy_divergence/cot_through_models
STAGE=/tmp/pdb_$$/plotdata
OUT=$ROOT/figures/plotdata.zip

rm -rf "$STAGE"; mkdir -p "$STAGE"/{self_gen,fixed_cot,scripts}

$P - "$SELF" "$STAGE" <<'PY'
import numpy as np, json, sys, os
from sklearn.decomposition import PCA
src, stage = sys.argv[1], sys.argv[2]
import glob
z = np.load(os.path.join(src, "pooled.npz"))
# Models added later live in sidecar files (see add_model_to_pooled.py); without
# these the w/o-OPD ablation silently drops out of the bundle.
extras = [np.load(f) for f in sorted(glob.glob(os.path.join(src, "pooled_extra_*.npz")))]
def fetch(key):
    if key in z.files:
        return z[key]
    for e in extras:
        if key in e.files:
            return e[key]
    return None
meta = json.load(open(os.path.join(src, "pooled_meta.json")))
P = meta["prompts"]
WANT = ["dense"] + [f"{f}:{s}" for f in ("alps", "sparsegpt", "alps_sft", "noopd", "ours")
                    for s in ("s50", "s60", "s70")] + ["teacher"]
# PCA to 256 dims, fit per prompt on all models together so every cloud shares
# one basis. This is what makes the bundle transferable: 10x smaller, and MMD
# agrees with the full 2560 dims to three decimals (256 comps hold 99.9% of the
# variance). Anything coarser starts to drift.
keep, D = {}, 256
for pi in range(P):
    mats, names = [], []
    for lab in WANT:
        v = fetch(f"p{pi}|{lab}|L18|late")
        if v is not None:
            mats.append(v.astype(np.float32)); names.append(lab)
    if not mats:
        continue
    X = np.concatenate(mats)
    d = min(D, X.shape[1], len(X) - 1)
    pca = PCA(n_components=d, random_state=0).fit(X)
    off = 0
    for lab, m in zip(names, mats):
        keep[f"p{pi}|{lab}|L18|late"] = pca.transform(m).astype(np.float16)
        off += len(m)
np.savez_compressed(os.path.join(stage, "self_gen", "pooled_L18_late_pca256_fp16.npz"), **keep)
meta.update({"subset": "layer 18, late window (continuation tokens 1024-2048); "
                       "PCA to 256 dims fit per prompt over all models; float16",
             "windows": {"late": meta["windows"]["late"]}, "layers": [18],
             "pca_dims": D})
json.dump(meta, open(os.path.join(stage, "self_gen", "pooled_meta.json"), "w"), indent=2)
print(f"  self_gen: {len(keep)} arrays, PCA-{D}")
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

cd "$(dirname "$STAGE")" && rm -f "$OUT" && zip -qr "$OUT" plotdata
cd / && rm -rf "$(dirname "$STAGE")"
ls -la "$OUT" | awk '{printf "  %s  %.1f MB\n", $9, $5/1048576}'
