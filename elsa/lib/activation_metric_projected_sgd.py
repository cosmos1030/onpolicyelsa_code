"""
Activation-Metric Projected SGD -- generalized to arbitrary (not just 2:4) sparsity.

Consistent-metric derivation: a dense step under the activation metric A = X^T X
satisfies A u = -lr*g; projecting u onto {d : d_I = 0} in the same A-metric
reduces to a form that needs no u_I at all:

    d_S = -lr * (A_SS + lam*I)^{-1} g_S
    d_I = 0

per group of `group_size` input-dim columns, using the running activation
covariance from `activation_tracker` (global forward hook, no model reference
needed here). `S`/`I` (active/inactive) can be ANY size per (row, group) --
not fixed at 2 -- so this covers unstructured sparsity too, not just 2:4.

Implementation: rather than branching per active-count k (0..group_size), each
group's (A_SS + lam*I) system is embedded in a full `group_size x group_size`
matrix M where inactive rows/cols are replaced by a trivial identity equation
(1*x_i = 0) -- solving the full M is then mathematically identical to solving
the k x k active-only system, but avoids needing separate code paths per k.
Two exceptions handled explicitly:
  - k=0 (nothing active in this block for this row): g is already masked to 0
    there, so M's identity rows just give x=0 -- no special-casing needed.
  - k=group_size (nothing pruned in this block): falls back to a plain
    (unprojected) gradient step, since there's nothing to compensate for --
    applying full-block preconditioning there isn't part of the mechanism
    being tested here.

Falls back to plain masked SGD for any parameter with no tracked covariance
yet (e.g. before the first forward pass) or non-2D / group_size-indivisible
parameters (biases, norms).

Optional classical momentum (`momentum` > 0): safe to combine here, unlike
Adam's diagonal 1/sqrt(v) scaling, because momentum is just an EMA of the
gradient itself, not a competing metric -- it's applied BEFORE the projection,
so the projection still consistently uses the single activation metric `A`.

Ported from opt_baseline_run/sparsegpt_lib/activation_metric_projected_sgd.py.
"""
import torch

from .activation_tracker import get_covariance, set_group_size, GROUP_SIZE


class ActivationMetricProjectedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, lam=1e-3, group_size=GROUP_SIZE, trust_ratio=5.0, momentum=0.0,
                 max_intermediate_gb=1.5):
        # keep the global activation-covariance tracker's block size in sync with
        # whatever this optimizer is configured for (it defaults to 4, matching 2:4).
        set_group_size(group_size)
        defaults = dict(lr=lr, lam=lam, group_size=group_size, trust_ratio=trust_ratio, momentum=momentum,
                         max_intermediate_gb=max_intermediate_gb)
        super().__init__(params, defaults)

    def _lazy_init(self, p, group_size, max_intermediate_gb):
        st = self.state[p]
        if "mask" in st:
            return st
        mask = (p.data != 0.0)
        st["mask"] = mask
        st["momentum_buf"] = torch.zeros_like(p.data)
        if p.dim() != 2 or p.shape[1] % group_size != 0:
            st["prunable"] = False
            return st

        out_features, in_features = p.shape
        # The per-group [out, groups, gs, gs] intermediates (M, A_exp, diag_embed, ...)
        # cost ~out_features * in_features * group_size * 4 bytes each -- independent of
        # batch size. For huge-out_features layers (e.g. lm_head, out~150k) this alone
        # can be several GB and OOM regardless of micro_batch_size. Fall back to plain
        # masked SGD for any layer whose intermediate would exceed the budget.
        intermediate_gb = out_features * in_features * group_size * 4 / 1e9
        if intermediate_gb > max_intermediate_gb:
            st["prunable"] = False
            return st

        num_groups = in_features // group_size
        mask_g = mask.view(out_features, num_groups, group_size)
        st["prunable"] = True
        st["mask_g"] = mask_g
        st["n_active"] = mask_g.sum(dim=-1)  # [out, groups], values 0..group_size
        st["num_groups"] = num_groups
        return st

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, lam, gs = group["lr"], group["lam"], group["group_size"]
            trust_ratio = group["trust_ratio"]
            momentum = group["momentum"]
            max_intermediate_gb = group["max_intermediate_gb"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._lazy_init(p, gs, max_intermediate_gb)
                mask = st["mask"]
                p.data.mul_(mask)

                g = p.grad * mask
                if momentum > 0:
                    buf = st["momentum_buf"]
                    buf.mul_(momentum).add_(g)
                    g = buf

                if not st.get("prunable", False):
                    p.data.add_(g, alpha=-lr)
                    continue

                A = get_covariance(p)
                if A is None:
                    p.data.add_(g, alpha=-lr)
                    continue

                out_features = p.shape[0]
                num_groups = st["num_groups"]
                active = st["mask_g"]  # [out, groups, gs] bool
                n_active = st["n_active"]  # [out, groups]

                g_g = g.view(out_features, num_groups, gs).float()  # [out, groups, gs]

                # Embed each group's active-only system in a full gs x gs matrix:
                # zero cross-terms touching inactive coords, identity on inactive diag.
                A_exp = A.unsqueeze(0).expand(out_features, num_groups, gs, gs)
                row_active = active.unsqueeze(-1)
                col_active = active.unsqueeze(-2)
                M = torch.where(row_active & col_active, A_exp, torch.zeros_like(A_exp))

                # Relative (trace-scaled) damping using only the active diagonal's mean --
                # a fixed absolute `lam` isn't safe since real activation covariances vary
                # hugely in scale across layers (e.g. RoPE-paired Q/K channels can be
                # near-perfectly correlated, making some blocks near-singular regardless
                # of overall magnitude).
                diag = A_exp.diagonal(dim1=-2, dim2=-1)  # [out, groups, gs]
                active_f = active.float()
                n_active_f = active_f.sum(-1).clamp(min=1)
                diag_scale = ((diag * active_f).sum(-1).abs() / n_active_f) + 1e-6  # [out, groups]
                lam_eff = lam * diag_scale
                diag_add = torch.where(active, lam_eff.unsqueeze(-1).expand(-1, -1, gs),
                                        torch.ones_like(lam_eff.unsqueeze(-1).expand(-1, -1, gs)))
                M = M + torch.diag_embed(diag_add)

                rhs = (-lr * g_g * active_f).unsqueeze(-1)  # [out, groups, gs, 1]

                x = torch.linalg.solve(M.reshape(-1, gs, gs), rhs.reshape(-1, gs, 1))
                x = x.view(out_features, num_groups, gs)

                # Trust-region safety net: an occasional near-singular block can still
                # produce a wildly oversized step even with relative damping. Cap the
                # preconditioned step's magnitude at `trust_ratio`x the plain-SGD step in
                # that same coordinate, so a single bad block can't blow up training.
                plain_step = -lr * g_g
                cap = trust_ratio * plain_step.abs() + 1e-12
                x = torch.clamp(x, -cap, cap) * active_f

                # Fully-active blocks (nothing pruned there): plain step, no projection.
                full_active = (n_active == gs).unsqueeze(-1)
                update = torch.where(full_active, plain_step, x)

                p.data.add_(update.view_as(p.data).to(p.dtype))
