"""
MaskedAdam optimizer for sparse fine-tuning.

After pruning, standard Adam will gradually fill back zeroed weights.
MaskedAdam prevents this by:
  1. Masking p.data and p.grad at pruned positions before each step
  2. Masking exp_avg (1st moment) to prevent momentum buildup at pruned positions

The mask is built lazily on the first step from the initial parameter values
(zero = pruned, non-zero = active), so no explicit mask argument is needed —
just load the pruned model and use this optimizer.
"""

import torch
from torch.optim.optimizer import _get_scalar_dtype, _device_dtype_check_for_fused

class MaskedAdam(torch.optim.Adam):
    """
    A variant of Adam that applies a fixed mask to the parameters after each
    optimizer step. This is useful for retraining pruned models, ensuring that
    the pruned weights remain zero.
    """
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)

    def _lazy_init_mask(self, p,group):
        st = self.state[p]
        if len(st) == 0: ## optimizer state init
            mask = (p.to(torch.float32)!=0.0).bool()
            st['mask'] = mask
            if group["fused"]:
                _device_dtype_check_for_fused(p)
            st["step"] = (
                torch.zeros(
                    (),
                    dtype=_get_scalar_dtype(is_fused=group["fused"]),
                    device=p.device,
                )
                if group["capturable"] or group["fused"]
                else torch.tensor(0.0, dtype=_get_scalar_dtype())
            )
            # Exponential moving average of gradient values
            st["exp_avg"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            # Exponential moving average of squared gradient values
            st["exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            if group["amsgrad"]:
                # Maintains max of all exp. moving avg. of sq. grad. values
                st["max_exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                    )

    @torch.no_grad()
    def step(self, closure=None):
        ## apply mask before step
        for group in self.param_groups:
            for p in group['params']:
                self._lazy_init_mask(p,group)
                if 'mask' in self.state[p]:
                    mask = self.state[p]['mask']
                    p.data.mul_(mask) ## param masking
                    if p.grad is not None:
                        p.grad.data.mul_(mask) ## grad masking
                    if 'exp_avg' in self.state[p]:
                        self.state[p]['exp_avg'].mul_(mask) ## first moment masking
        super().step(closure)

