"""Swap one whole decoder layer's projections back to their original dense
weights inside an already-pruned ALPS checkpoint, without re-running any
pruning. Cheap ablation for the KL-diagnostic finding that a specific layer
(e.g. L35's mlp.up_proj) is the single largest incremental-KL spike --
if that's really the bottleneck, patching just that layer back to dense
should measurably move downstream benchmarks.

Usage:
    python patch_dense_layer.py <pruned_model_path> <dense_model_path> <layer_idx> --save <out_path>
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ_NAMES = [
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('pruned_model', type=str)
    p.add_argument('dense_model', type=str)
    p.add_argument('layer_idx', type=int)
    p.add_argument('--save', type=str, required=True)
    args = p.parse_args()

    print(f'Loading pruned model from {args.pruned_model} ...')
    pruned = AutoModelForCausalLM.from_pretrained(args.pruned_model, torch_dtype='auto')
    print(f'Loading dense model from {args.dense_model} ...')
    dense = AutoModelForCausalLM.from_pretrained(args.dense_model, torch_dtype='auto')

    layer_p = pruned.model.layers[args.layer_idx]
    layer_d = dense.model.layers[args.layer_idx]

    with torch.no_grad():
        for name in PROJ_NAMES:
            w_p = layer_p.get_submodule(name).weight
            w_d = layer_d.get_submodule(name).weight
            nnz_before = (w_p != 0).sum().item()
            total = w_p.numel()
            w_p.copy_(w_d.to(w_p.dtype))
            print(f'  {name}: sparsity before={1 - nnz_before/total:.4f} -> now dense (0.0000)')

    del dense
    print(f'Saving patched model to {args.save} ...')
    pruned.save_pretrained(args.save)
    tokenizer = AutoTokenizer.from_pretrained(args.pruned_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.save)
    print('Done.')
