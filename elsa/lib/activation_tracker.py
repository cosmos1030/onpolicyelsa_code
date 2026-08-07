"""
Global activation-covariance tracker for pruned nn.Linear layers.

Registers ONE global forward hook (torch.nn.modules.module.register_module_forward_hook)
that fires for every nn.Module in every model in the process -- this avoids needing
the optimizer to hold a reference to the model. For every pruned nn.Linear
(any zero entries in its weight), it accumulates a running per-group activation
covariance A_g = X_g^T X_g, block-diagonal with block size `group_size` (4 for 2:4).

ActivationMetricProjectedSGD looks up `COV.get(id(weight_tensor))` during its step().

Ported from opt_baseline_run/sparsegpt_lib/activation_tracker.py.
"""
import torch
import torch.nn as nn

GROUP_SIZE = 4
EMA_MOMENTUM = 0.99

COV = {}          # id(weight) -> running [num_groups, group_size, group_size] covariance
_SEEN_WARMUP = set()  # id(weight) -> True once cov has been initialized (skip EMA blend on first batch)
_registered = False


def _is_pruned_linear(module):
    if not isinstance(module, nn.Linear) or module.weight.is_meta:
        return False
    return (module.weight.data == 0).any().item()


def _hook(module, inputs, output):
    if not _is_pruned_linear(module):
        return
    if not inputs or inputs[0] is None or inputs[0].is_meta:
        return
    p = module.weight
    in_features = p.shape[1]
    gs = GROUP_SIZE
    if in_features % gs != 0:
        return  # skip layers whose in_features isn't a multiple of the block size
    num_groups = in_features // gs

    x = inputs[0]
    if x is None:
        return
    x = x.detach().reshape(-1, in_features).float()
    if x.shape[0] == 0:
        return
    xg = x.view(x.shape[0], num_groups, gs)  # [n, groups, gs]
    batch_cov = torch.einsum("ngi,ngj->gij", xg, xg) / x.shape[0]  # [groups, gs, gs]

    key = id(p)
    if key not in COV:
        COV[key] = batch_cov.detach().clone()
        _SEEN_WARMUP.add(key)
    else:
        COV[key].mul_(EMA_MOMENTUM).add_(batch_cov, alpha=1 - EMA_MOMENTUM)


def enable():
    global _registered
    if _registered:
        return
    nn.modules.module.register_module_forward_hook(_hook)
    _registered = True


def set_group_size(new_size):
    """Change the block size covariance is tracked at. `_hook` re-reads the
    GROUP_SIZE global on every call, so this takes effect on the next forward
    pass. Any already-accumulated covariance is the wrong shape for a new
    block size, so it's cleared."""
    global GROUP_SIZE
    if new_size != GROUP_SIZE:
        COV.clear()
        _SEEN_WARMUP.clear()
        GROUP_SIZE = new_size


def get_covariance(weight_tensor):
    return COV.get(id(weight_tensor))


enable()
