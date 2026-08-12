"""
BEST-style Gradual Magnitude Pruning trainer.

Key components (from "The State of Sparsity in LLMs"):
  1. Fisher-weighted importance: score_i = F_hat_ii * w_i^2
     where F_hat_ii = running avg of g_i^2 (empirical Fisher diagonal)
  2. Cubic gradual sparsity schedule: s_t = s_final * (1 - (1 - t/T)^3)
  3. LR warmup + cosine decay
  4. Periodic mask update every `mask_update_interval` steps
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from absl import logging
from contextlib import nullcontext
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    _FSDP_AVAILABLE = True
except ImportError:
    _FSDP_AVAILABLE = False

try:
    from torch.distributed.tensor import DTensor, Replicate
    _DTENSOR_AVAILABLE = True
except ImportError:
    _DTENSOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_decoder_layers(model):
    core = getattr(model, "model", model)
    return getattr(core, "decoder", core).layers


def _find_linear_weights(model):
    """Return {name: param} for transformer block Linear weights (matches SparseGPT scope)."""
    result = {}
    for block_idx, layer in enumerate(_get_decoder_layers(model)):
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"model.layers.{block_idx}.{name}.weight"
                result[full_name] = module.weight
    return result


def _structured_l1_loss(named_params: dict, masks: dict, prune_n: int, prune_m: int) -> torch.Tensor:
    """2:4 structured L1 regularization (mean-normalized).

    Penalizes the mean abs value of the bottom-(M-N) alive weights per group-of-M
    that has NOT yet reached its prune_n cap (see _open_group_mask) -- once a
    group is down to exactly prune_n alive weights, topk(..., largest=False)
    over a metric where dead positions are +inf trivially selects those same
    prune_n survivors every time (they're the only finite entries left), so
    without this gating the loss would keep shrinking already-decided,
    supposed-to-survive weights indefinitely instead of only pressuring
    still-undecided candidates -- actively damaging the model in the endgame
    when most groups have already closed.
    Already-pruned (mask=0) positions are excluded — penalizing zeros is meaningless
    and would bias the gradient signal.
    Normalized by alive element count so scale stays comparable to per-token NTP loss.

    named_params must hold full, unsharded [rows, cols] tensors -- under FSDP1
    (classic FullyShardedDataParallel, this file does not use FSDP2/DTensor)
    each rank's plain param is a flat, differently-sized local shard, so the
    reshape-into-groups-of-M logic below is only valid inside a
    FSDP.summon_full_params(fsdp_model, with_grads=True) block; see the call
    site for how the resulting loss's backward() is isolated from the main
    training loss's backward() to avoid interleaving with FSDP's own
    sharded forward/backward hooks.
    """
    total = None
    count = 0
    for name, param in named_params.items():
        w = param
        mask = masks.get(name)
        if w.dim() < 2:
            alive = w[mask] if mask is not None else w
            term = alive.abs().sum()
            n = alive.numel()
        else:
            n_rows, n_cols = w.shape
            n_full = n_cols // prune_m
            n_nm_cols = n_full * prune_m
            w_nm = w[:, :n_nm_cols].reshape(n_rows * n_full, prune_m)
            alive_nm = mask[:, :n_nm_cols].reshape(n_rows * n_full, prune_m) if mask is not None \
                       else torch.ones_like(w_nm, dtype=torch.bool)
            # exclude groups already at their prune_n cap -- nothing left to
            # decide there, so they must never contribute to this loss
            open_nm = (alive_nm.sum(dim=1, keepdim=True) > prune_n).expand_as(alive_nm)
            # within each group, only consider alive weights for bottom-k selection
            metric = w_nm.abs()
            metric[~alive_nm] = float('inf')  # dead weights can't be bottom-k
            n_pruned = prune_m - prune_n
            bottom_idx = torch.topk(metric, n_pruned, dim=1, largest=False).indices
            selected = w_nm.abs().gather(1, bottom_idx)
            # only count positions that are actually alive AND in a still-open group
            alive_selected = alive_nm.gather(1, bottom_idx) & open_nm.gather(1, bottom_idx)
            term = selected[alive_selected].sum()
            n = int(alive_selected.sum().item())
        total = term if total is None else total + term
        count += n
    if total is None or count == 0:
        return torch.tensor(0.0)
    return total / count


def _open_group_mask(alive: torch.Tensor, prune_n: int, prune_m: int) -> torch.Tensor:
    """For a 2D alive mask, return a same-shape bool mask marking weights that belong
    to a group-of-prune_m which has NOT yet reached its 2:4 cap (i.e. still has more
    than prune_n alive weights, so more pruning is still expected there).

    Groups already at cap (alive_count <= prune_n) are excluded entirely — there's
    nothing left to prune there, so regularizing them serves no purpose and only
    dilutes the L1 signal on the groups that actually still need to shed weight.
    """
    n_rows, n_cols = alive.shape
    n_full = n_cols // prune_m
    n_nm_cols = n_full * prune_m
    out = torch.zeros_like(alive)
    if n_full == 0:
        return out
    alive_nm = alive[:, :n_nm_cols].reshape(n_rows * n_full, prune_m)
    group_alive_count = alive_nm.sum(dim=1, keepdim=True)
    group_open = (group_alive_count > prune_n).expand_as(alive_nm)
    out[:, :n_nm_cols] = group_open.reshape(n_rows, n_nm_cols)
    return out


def _gmp_l1_regularizer(named_params, maskmgr, fisher, mode="plain",
                        clip_min=0.1, clip_max=10.0,
                        open_groups_only=False, prune_n=0, prune_m=0):
    """L1 regularization term for GMP training.

    mode="plain":
        mean |w_i| over alive weights (mean-normalized across layers)

    mode="inv_fisher_sqrt":
        mean  |w_i| / sqrt(clamp(f_i / mean(f_alive), clip_min, clip_max))
        Weights with high Fisher (important) get lower penalty,
        weights with low Fisher (pruning candidates) get higher penalty.
        Falls back to plain L1 if Fisher state not yet available.

    open_groups_only: restrict the alive set to weights in 2:4 groups that
        haven't reached their prune_n cap yet (see _open_group_mask). Use this
        to concentrate L1 pressure on the shrinking pool of still-prunable
        weights instead of diluting it across already-finished groups —
        intended to help the last few percent of a 2:4 schedule converge under
        a tight TR-KL budget.
    """
    reg_terms = []
    for name, param in named_params.items():
        if param.ndim != 2:
            continue
        mask = maskmgr.masks.get(name)
        if mask is None:
            continue
        alive = mask.bool()
        if open_groups_only and prune_n > 0 and prune_m > 0:
            alive = alive & _open_group_mask(alive, prune_n, prune_m)
        if alive.sum() == 0:
            continue

        w_abs = param.abs()

        if mode == "plain":
            reg_terms.append(w_abs[alive].mean())

        elif mode == "inv_fisher_sqrt":
            f = fisher.fisher_factor(param)
            if f is None:
                # fallback: no Adam state yet
                reg_terms.append(w_abs[alive].mean())
                continue
            f = f.detach()
            f_alive = f[alive]
            f_mean = f_alive.mean().clamp_min(1e-12)
            f_norm = (f / f_mean).clamp(min=clip_min, max=clip_max)
            weight = 1.0 / torch.sqrt(f_norm)
            reg_terms.append((w_abs * weight)[alive].mean())

        else:
            raise ValueError(f"Unknown gmp_l1_mode: {mode}")

    if not reg_terms:
        return None
    return torch.stack(reg_terms).mean()


def _cubic_sparsity(step, total_steps, final_sparsity, warmup_steps=0):
    """Cubic schedule: s_t = s_final * (1 - (1 - (t-warmup)/(T-warmup))^3)."""
    if step < warmup_steps:
        return 0.0
    t = step - warmup_steps
    T = max(total_steps - warmup_steps, 1)
    return final_sparsity * (1.0 - (1.0 - min(t / T, 1.0)) ** 3)


def _apply_mask(param, mask):
    with torch.no_grad():
        param.data.mul_(mask)


# ---------------------------------------------------------------------------
# Fisher accumulator
# ---------------------------------------------------------------------------

class FisherAccumulator:
    """Fisher diagonal from Adam's exp_avg_sq (== empirical Fisher diagonal).

    Adam's second moment v_t is an EMA of g² with the same semantics as the
    hand-rolled FisherAccumulator it replaces — no separate bookkeeping needed.
    """

    def __init__(self, named_params, optimizer, beta=0.999, saliency='fisher'):
        self.named_params = named_params  # {name: param}
        self.optimizer = optimizer
        self._step = 0
        self.saliency = saliency  # 'fisher', 'magnitude', 'spa', 'sqrt_fisher', or 'wanda'
        # Per-parameter group lookup for 'spa' -- correct even if some params
        # (e.g. embeddings, no-decay groups) use different lr/wd/betas/eps
        # than param_groups[0].
        self.param_to_group = {
            id(p): group for group in optimizer.param_groups for p in group['params']
        }
        self._wanda_scaler = {}  # name -> per-input-column activation L2-norm^2 (float32)

    def capture_wanda_stats(self, model, cal_batch, device, chunk_size=8):
        """Wanda-style activation scaler: scaler_row[j] = sum_tokens x_j^2,
        accumulated over `cal_batch` in `chunk_size`-sequence forward passes
        (same hook pattern as _pcg_correct_masked_weights, but chunked since
        cal_batch here can be the full OPKD rollout pool -- e.g. 256
        sequences -- which would OOM as a single forward pass). Only
        meaningful per-layer (the resulting importance() scores are NOT
        comparable across layers -- pair with --gmp_pruning_scope=layer,
        see _compute_tr_kl / candidate_masks)."""
        name_to_module = dict(model.named_modules())
        accum = {}
        handles = []

        def _make_hook(pname):
            def hook(module, inp, out):
                x = inp[0]
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                sq = (x.detach().float() ** 2).sum(dim=0)
                if pname in accum:
                    accum[pname] += sq
                else:
                    accum[pname] = sq
            return hook

        for name in self.named_params:
            mod_name = name[:-len('.weight')] if name.endswith('.weight') else None
            module = name_to_module.get(mod_name) if mod_name else None
            if not isinstance(module, nn.Linear):
                continue
            handles.append(module.register_forward_hook(_make_hook(name)))

        if not handles:
            return

        was_training = model.training
        model.eval()
        input_ids_full = cal_batch['input_ids']
        attn_mask_full = cal_batch.get('attention_mask')
        n = input_ids_full.shape[0]
        with torch.no_grad():
            for i in range(0, n, chunk_size):
                input_ids = input_ids_full[i:i + chunk_size].to(device)
                attn_mask = attn_mask_full[i:i + chunk_size].to(device) if attn_mask_full is not None else None
                model(input_ids=input_ids, attention_mask=attn_mask)
        if was_training:
            model.train()
        for h in handles:
            h.remove()

        for name, sq in accum.items():
            self._wanda_scaler[name] = sq

    def update(self):
        """No-op: Adam updates exp_avg_sq automatically in optimizer.step()."""
        self._step += 1

    def fisher_factor(self, param):
        """Return bias-corrected Adam second moment f_i (empirical Fisher diagonal).

        Returns None before the first optimizer step (no state yet).
        """
        st = self.optimizer.state.get(param, {})
        v = st.get('exp_avg_sq', None)
        if v is None:
            return None
        if _DTENSOR_AVAILABLE and isinstance(v, DTensor):
            v = v.redistribute(placements=[Replicate()]).to_local()
        f = v.float()
        step = st.get('step', self._step)
        if torch.is_tensor(step):
            step = step.item()
        beta2 = self.optimizer.param_groups[0].get('betas', (0.9, 0.999))[1]
        if step > 0:
            f = f / (1.0 - beta2 ** step)
        return f

    def importance(self, name, param):
        """Importance score for pruning. 'fisher': F_hat*w^2, 'magnitude': w^2,
        'spa': h*u^2 (Sparse Projected Adam -- see _spa_importance), 'sqrt_fisher':
        sqrt(F_hat)*w^2 (the lr->0 limit of 'spa': u->w and h->sqrt(v_hat)+eps
        as the momentum/decay terms vanish, so imp->sqrt(v_hat)*w^2 -- same cost
        as 'fisher', just with an extra sqrt(), no momentum state needed).
        'wanda': |w|*sqrt(scaler_row) (Wanda-style weight*activation-norm, see
        capture_wanda_stats) -- NOT comparable across layers (activation scale
        varies wildly layer-to-layer), so only meaningful with
        --gmp_pruning_scope=layer, never 'global'."""
        if self.saliency == 'magnitude':
            return param.data.float() ** 2
        if self.saliency == 'spa':
            return self._spa_importance(param)
        if self.saliency == 'wanda':
            scaler = self._wanda_scaler.get(name)
            if scaler is None or param.dim() != 2:
                return param.data.float() ** 2  # fallback before first capture / non-2D param
            return param.data.float().abs() * scaler.to(param.device).sqrt().reshape(1, -1)
        f = self.fisher_factor(param)
        if f is None:
            return param.data.float() ** 2  # fallback before first optimizer step
        if self.saliency == 'sqrt_fisher':
            f = f.clamp(min=0).sqrt()
        imp = f * param.data.float() ** 2
        if imp.sum() == 0:
            imp = param.data.float() ** 2
        return imp

    def _spa_importance(self, param):
        """Sparse Projected Adam(W) saliency: cost of pruning coordinate i in
        the Adam-metric projection of the next unconstrained AdamW iterate u
        onto a sparse support. u_i = (1-lr*wd)*w_i - lr*m_hat_i/h_i,
        h_i = sqrt(v_hat_i)+eps; s_i = h_i * u_i^2 (keeping costs 0, pruning
        costs h_i*u_i^2 exactly, since the projection decomposes
        coordinate-wise under diagonal H). Uses the param's actual optimizer
        group (lr/betas/eps/weight_decay), not param_groups[0], so this stays
        correct if decay/no-decay or per-layer-LR groups are ever introduced."""
        st = self.optimizer.state.get(param, {})
        v = st.get('exp_avg_sq', None)
        m = st.get('exp_avg', None)
        if v is None or m is None:
            return param.data.float() ** 2  # fallback before first optimizer step
        if _DTENSOR_AVAILABLE and isinstance(v, DTensor):
            v = v.redistribute(placements=[Replicate()]).to_local()
        if _DTENSOR_AVAILABLE and isinstance(m, DTensor):
            m = m.redistribute(placements=[Replicate()]).to_local()
        step = st.get('step', self._step)
        if torch.is_tensor(step):
            step = step.item()
        group = self.param_to_group[id(param)]
        beta1, beta2 = group.get('betas', (0.9, 0.999))
        eps = group.get('eps', 1e-8)
        lr = group.get('lr', 0.0)
        wd = group.get('weight_decay', 0.0)
        # Each step below reuses/overwrites its own buffer in-place instead of
        # keeping every intermediate (v, m, v_hat, m_hat, h, u...) alive at
        # once -- the naive out-of-place version held ~7 full-model-sized fp32
        # tensors simultaneously (peak memory 3-4x fisher's), which OOM'd a
        # 1.7B model on an 80GB GPU. `.float()` already makes a private copy
        # (states are bf16), so mutating it in place is safe.
        h = v.float()  # private copy; becomes h in place
        if step > 0:
            h.div_(1.0 - beta2 ** step)
        h.sqrt_().add_(eps)  # h = sqrt(v_hat) + eps; v_hat's buffer freed
        u = m.float()  # private copy; becomes u in place
        if step > 0:
            u.div_(1.0 - beta1 ** step)
        u.div_(h).mul_(-lr).add_(param.data.float(), alpha=(1.0 - lr * wd))
        # u = (1-lr*wd)*w - lr*m_hat/h ; m_hat's buffer freed
        imp = u.pow_(2).mul_(h)  # imp = h * u^2, written into u's buffer
        if imp.sum() == 0:
            imp = param.data.float() ** 2
        return imp


class EmpiricalFisherAccumulator:
    """Empirical Fisher diagonal computed from a calibration batch via grad².

    Replaces Adam exp_avg_sq with F_ii = mean_n(g_i^2) over the cal_batch.
    Call update_from_batch() before each TR mask update; importance() interface
    is identical to FisherAccumulator so it plugs in without other changes.
    """

    def __init__(self, named_params, saliency='fisher'):
        self.named_params = named_params  # {name: param}
        self.saliency = saliency
        self._fisher: dict = {}  # name -> Fisher diagonal tensor (float32, CPU)

    def update_from_batch(self, model: 'nn.Module', cal_batch: dict, device: str):
        """Compute empirical Fisher from cal_batch: F_ii = mean(g_i^2) over samples."""
        input_ids = cal_batch['input_ids'].to(device)
        attn_mask = cal_batch['attention_mask'].to(device)

        model.eval()
        saved_grads = {n: [] for n in self.named_params}

        B = input_ids.shape[0]
        for b in range(B):
            ids_b = input_ids[b:b+1]
            msk_b = attn_mask[b:b+1]
            model.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(input_ids=ids_b, attention_mask=msk_b)
            logits = out.logits[:, :-1, :].float()
            labels = ids_b[:, 1:]
            valid  = (msk_b[:, 1:] == 1)
            if not valid.any():
                continue
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100 if 'labels' in cal_batch else -1,
                reduction='none',
            )
            valid_loss = loss[valid.reshape(-1)].mean()
            valid_loss.backward()
            for n, p in self.named_params.items():
                if p.grad is not None:
                    saved_grads[n].append(p.grad.detach().float().cpu() ** 2)

        model.zero_grad()
        model.train()

        self._fisher = {}
        for n, grads in saved_grads.items():
            if grads:
                self._fisher[n] = torch.stack(grads).mean(0)

    def fisher_factor(self, param):
        for n, p in self.named_params.items():
            if p is param:
                return self._fisher.get(n, None)
        return None

    def importance(self, name, param):
        if self.saliency == 'magnitude':
            return param.data.float() ** 2
        f = self._fisher.get(name, None)
        if f is None:
            return param.data.float() ** 2
        f = f.to(param.device)
        imp = f * param.data.float() ** 2
        if imp.sum() == 0:
            imp = param.data.float() ** 2
        return imp

    def update(self):
        pass  # no-op; use update_from_batch() instead


# ---------------------------------------------------------------------------
# Mask manager
# ---------------------------------------------------------------------------

class GradualMaskManager:
    """Maintains binary masks and updates them on a schedule."""

    def __init__(self, named_params, fsdp_model=None, prune_n=0, prune_m=0, pruning_scope='global'):
        self.named_params = named_params
        self.prune_n = prune_n  # N for N:M semi-structured sparsity (0 = unstructured)
        self.prune_m = prune_m  # M for N:M semi-structured sparsity
        self.pruning_scope = pruning_scope  # 'global' or 'layer' (per-layer)
        # With FSDP, p.data is a local shard — masks live at local shard shape.
        # summon_full_params is NOT used here: importance scoring and mask application
        # operate on local shards directly (all-gather used for global threshold only).
        self.masks = {n: torch.ones(p.data.shape, dtype=torch.bool, device=p.data.device)
                      for n, p in named_params.items()}

    @torch.no_grad()
    def init_from_weights(self, fsdp_model=None):
        """Initialize mask from existing zero pattern (for sparse SFT on pre-pruned models)."""
        for n, p in self.named_params.items():
            self.masks[n] = (p.data != 0)

    @torch.no_grad()
    def _nm_mask(self, imp: torch.Tensor, current_mask: torch.Tensor, sparsity: float) -> torch.Tensor:
        """N:M semi-structured mask for a single weight matrix.

        Ported from log_efficient_qwen_competition/lib/gmp.py.
        Protects top-N weights per group of M, then globally prunes remaining
        positions to reach target sparsity (gradual schedule).
        """
        prune_n, prune_m = self.prune_n, self.prune_m
        if imp.numel() == 0:
            # FSDP shard-of-this-param is empty on this rank (param fully resides
            # in another rank's shard) — nothing to mask, keep=True is a no-op.
            return torch.ones_like(imp, dtype=torch.bool)
        if imp.dim() < 2:
            # 1-D param (bias etc.) — fall back to unstructured
            return imp > torch.kthvalue(imp.flatten(), max(1, int(imp.numel() * sparsity))).values

        n_rows, n_cols = imp.shape
        n_full_chunks = n_cols // prune_m
        n_nm_cols = n_full_chunks * prune_m

        metric_nm     = imp[:, :n_nm_cols].reshape(n_rows * n_full_chunks, prune_m)
        already_zero  = current_mask[:, :n_nm_cols].reshape(n_rows * n_full_chunks, prune_m)

        # Protect top-N per group; already-zero weights cannot consume a protected slot.
        metric_protect = metric_nm.clone()
        metric_protect[already_zero] = -float('inf')
        _, top_idx = torch.topk(metric_protect, prune_n, dim=1, largest=True)
        protect_mask = torch.zeros_like(metric_nm, dtype=torch.bool)
        protect_mask.scatter_(1, top_idx, True)

        # Find how many additional positions to prune.
        n_total      = n_rows * n_full_chunks * prune_m
        n_max_pruned = n_rows * n_full_chunks * (prune_m - prune_n)
        n_already    = int(already_zero.sum().item())
        n_target     = min(int(n_total * sparsity), n_max_pruned)
        n_new        = max(0, n_target - n_already)

        W_mask = current_mask.clone()
        if n_new > 0:
            metric_thresh = metric_nm.clone()
            metric_thresh[protect_mask]  = float('inf')
            metric_thresh[already_zero]  = float('inf')
            flat = metric_thresh.flatten()
            n_avail = int((flat < float('inf')).sum().item())
            n_new = min(n_new, n_avail)
            if n_new > 0:
                _, prune_idx = torch.topk(flat, n_new, largest=False)
                prune_flat = torch.zeros(n_total, dtype=torch.bool, device=imp.device)
                prune_flat[prune_idx] = True
                W_mask[:, :n_nm_cols] |= prune_flat.reshape(n_rows, n_nm_cols)
        return ~W_mask  # mask=True means KEEP (consistent with unstructured path)

    @torch.no_grad()
    def candidate_masks(self, fisher: 'FisherAccumulator', sparsity: float, fsdp_model=None) -> dict:
        """Compute candidate masks at target sparsity without modifying self.masks or weights.

        Returns a dict {name: bool_tensor} where True=KEEP, same convention as self.masks.
        """
        if sparsity <= 0.0:
            return {n: m.clone() for n, m in self.masks.items()}

        use_fsdp = _FSDP_AVAILABLE and fsdp_model is not None

        if self.prune_n > 0 and self.prune_m > 0:
            new_masks = {}
            for name, param in self.named_params.items():
                imp = fisher.importance(name, param)
                if torch.isnan(imp).any() or torch.isinf(imp).any():
                    new_masks[name] = self.masks[name].clone()
                    continue
                current_pruned = ~self.masks[name]
                new_masks[name] = self._nm_mask(imp, current_pruned, sparsity)
            return new_masks
        else:
            local_imps = {}
            for name, param in self.named_params.items():
                local_imps[name] = fisher.importance(name, param)

            if use_fsdp:
                import torch.distributed as _dist

                # --- memory-efficient FSDP path: iterate over per-param tensors,
                # never concatenate all scores into one large GPU tensor ---
                # With FSDP FULL_SHARD + use_orig_params=True, some param shards may
                # have 0 elements on a given rank (param fully resides in another rank's
                # shard). Filter these out before any reduction.
                _dev = next(iter(local_imps.values())).device
                local_imps = {n: v for n, v in local_imps.items() if v.numel() > 0}

                # NaN/Inf check via min/max scalars (no boolean tensor)
                if local_imps:
                    _lmin = min(v.min().item() for v in local_imps.values())
                    _lmax = max(v.max().item() for v in local_imps.values())
                else:
                    _lmin, _lmax = 0.0, 0.0
                has_nan_t = torch.tensor(
                    1.0 if (math.isnan(_lmin) or math.isnan(_lmax) or
                            math.isinf(_lmin) or math.isinf(_lmax)) else 0.0,
                    device=_dev)
                _dist.all_reduce(has_nan_t, op=_dist.ReduceOp.MAX)
                if has_nan_t.item() > 0:
                    logging.warning("NaN/Inf in Fisher importance scores, skipping candidate mask")
                    return {n: m.clone() for n, m in self.masks.items()}

                # Global element count
                n_local = sum(v.numel() for v in local_imps.values())
                n_local_t = torch.tensor([n_local], dtype=torch.long, device=_dev)
                _dist.all_reduce(n_local_t, op=_dist.ReduceOp.SUM)
                n_total = n_local_t.item()
                k = int(n_total * sparsity)
                if k == 0:
                    return {n: m.clone() for n, m in self.masks.items()}

                # Global min/max for binary search bounds
                lo_t = torch.tensor(_lmin, dtype=torch.float32, device=_dev)
                hi_t = torch.tensor(_lmax, dtype=torch.float32, device=_dev)
                _dist.all_reduce(lo_t, op=_dist.ReduceOp.MIN)
                _dist.all_reduce(hi_t, op=_dist.ReduceOp.MAX)
                lo, hi = lo_t.item(), hi_t.item()

                # Chunked binary search: 50M-element boolean chunks to cap temp GPU alloc at 50 MB
                _CHUNK = 50_000_000
                for _ in range(64):
                    mid = (lo + hi) / 2.0
                    cnt = torch.zeros(1, dtype=torch.long, device=_dev)
                    for imp_v in local_imps.values():
                        flat = imp_v.flatten()
                        for ci in range(0, flat.numel(), _CHUNK):
                            cnt += (flat[ci:ci + _CHUNK] <= mid).sum(dtype=torch.long)
                    _dist.all_reduce(cnt, op=_dist.ReduceOp.SUM)
                    if cnt.item() < k:
                        lo = mid
                    else:
                        hi = mid
                threshold = torch.tensor(hi, device=_dev, dtype=next(iter(local_imps.values())).dtype)
                logging.info(f"  [Fisher/FSDP] global threshold={hi:.4e} (n_total={n_total}, k={k})")
            else:
                # ── per-layer pruning: each param independently hits target sparsity ──
                if self.pruning_scope == 'layer':
                    new_masks = {}
                    for name, param in self.named_params.items():
                        imp = local_imps[name]
                        if torch.isnan(imp).any() or torch.isinf(imp).any():
                            new_masks[name] = self.masks[name].clone()
                            continue
                        n_elems = imp.numel()
                        k_l = int(n_elems * sparsity)
                        if k_l == 0:
                            new_masks[name] = torch.ones_like(imp, dtype=torch.bool)
                            continue
                        if k_l >= n_elems:
                            new_masks[name] = torch.zeros_like(imp, dtype=torch.bool)
                            continue
                        threshold = torch.kthvalue(imp.flatten(), k_l).values
                        new_masks[name] = imp > threshold
                    return new_masks

                # ── global pruning: single threshold across all layers ─────────────
                # Never materialize one torch.cat'd tensor over every param in the model
                # (e.g. ~13.5GB of fp32 scores for Qwen3-4B's ~3.6B linear weights, on top
                # of whatever's already resident -- OOMs on an 80GB GPU well before the
                # model + optimizer + activations even get close to the card's limit).
                # Stay chunked over the per-layer dict throughout, same as the FSDP branch
                # above; torch.kthvalue's int32/2B-element ceiling is moot here too since
                # no per-layer tensor is anywhere near that size.
                imp_tensors = list(local_imps.values())
                if any(torch.isnan(v).any() or torch.isinf(v).any() for v in imp_tensors):
                    logging.warning("NaN/Inf in Fisher importance scores, skipping candidate mask")
                    return {n: m.clone() for n, m in self.masks.items()}
                n_total = sum(v.numel() for v in imp_tensors)
                k = int(n_total * sparsity)
                if k == 0:
                    return {n: m.clone() for n, m in self.masks.items()}
                lo = min(v.min().item() for v in imp_tensors)
                hi = max(v.max().item() for v in imp_tensors)
                for _ in range(48):
                    mid = (lo + hi) / 2.0
                    cnt = sum((v <= mid).sum().item() for v in imp_tensors)
                    if cnt < k:
                        lo = mid
                    else:
                        hi = mid
                threshold = torch.tensor(hi, device=imp_tensors[0].device, dtype=imp_tensors[0].dtype)
                actual = sum((v <= threshold).sum().item() for v in imp_tensors)
                logging.info(f"  [Fisher] chunked binary-search threshold={threshold.item():.4e} "
                             f"(n_total={n_total}, actual_below={actual}, target={k})")

            # For params with empty local shard (filtered out above), keep existing mask.
            return {
                name: (local_imps[name] > threshold if name in local_imps
                       else self.masks[name].clone())
                for name in self.named_params
            }

    @torch.no_grad()
    def update(self, fisher: 'FisherAccumulator', sparsity: float, fsdp_model=None):
        """Recompute global mask at target sparsity using Fisher importance.

        FSDP note: importance scoring and mask application run on local shards directly.
        For global unstructured pruning, scores are all-gathered across ranks to compute
        a consistent threshold. summon_full_params is NOT used — it causes shape mismatch
        because optimizer exp_avg_sq retains local shard shape while param.data becomes
        full shape inside the context manager.
        """
        if sparsity <= 0.0:
            return
        new_masks = self.candidate_masks(fisher, sparsity, fsdp_model)
        self.masks = new_masks
        for name, param in self.named_params.items():
            _apply_mask(param, self.masks[name])

    def apply(self, fsdp_model=None):
        """Zero out masked weights (call after every optimizer step).

        Applies masks to local shards directly — FSDP all-gathers params before each
        forward pass, so zeroing local shards is sufficient to enforce sparsity globally.
        """
        for name, param in self.named_params.items():
            _apply_mask(param, self.masks[name])

    def current_sparsity(self):
        total = sum(m.numel() for m in self.masks.values())
        zeros = sum((~m).sum().item() for m in self.masks.values())
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                # NCCL requires CUDA tensor — get device from masks
                _dev = next(iter(self.masks.values())).device if self.masks else 'cpu'
                t = torch.tensor([total, zeros], dtype=torch.long, device=_dev)
                _dist.all_reduce(t, op=_dist.ReduceOp.SUM)
                total, zeros = t[0].item(), t[1].item()
        except Exception:
            pass
        return zeros / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def _hidden_loss(s_hidden, t_hidden, labels, attention_mask, mode="cosine", mask_mode="cot"):
    """Hidden state reconstruction loss between student and teacher.

    s_hidden, t_hidden: (B, T, D) — last transformer layer output before lm_head.
    mask_mode:
      'cot' — only CoT positions (labels != -100)
      'all' — all non-padding positions (attention_mask == 1)
    mode: 'cosine', 'nmse', or 'mse'.
    """
    if mask_mode == "all":
        mask = attention_mask.float()
    else:
        mask = (labels != -100).float()

    denom = mask.sum().clamp(min=1)
    if denom == 0:
        return s_hidden.new_tensor(0.0)

    if mode == "cosine":
        per_token = 1.0 - F.cosine_similarity(s_hidden, t_hidden, dim=-1)
    elif mode == "nmse":
        diff = (s_hidden - t_hidden).pow(2).sum(dim=-1)
        den  = t_hidden.pow(2).sum(dim=-1).clamp_min(1e-6)
        per_token = diff / den
    else:  # mse
        per_token = (s_hidden - t_hidden).pow(2).mean(dim=-1)

    return (per_token * mask).sum() / denom


def _hidden_loss_layerwise(s_hidden_states, t_hidden_states, labels, attention_mask,
                           mode="nmse", mask_mode="all", step=0, total_steps=1):
    """Coarse-to-fine layerwise hidden loss with normalized annealing weights.

    All-layer average at the start, final-layer-only at the end.
    Weights always sum to 1 so loss scale stays comparable to final-only.

    s_hidden_states, t_hidden_states: tuple of (B, T, D) per layer.
      Pass hidden_states[1:] from model output to skip embedding layer.
    """
    if mask_mode == "all":
        mask = attention_mask.float()
    else:
        mask = (labels != -100).float()
    denom = mask.sum().clamp(min=1)

    layer_losses = []
    for s_h, t_h in zip(s_hidden_states, t_hidden_states):
        s_h = s_h.float()
        t_h = t_h.float()
        if mode == "cosine":
            per_token = 1.0 - F.cosine_similarity(s_h, t_h, dim=-1)
        elif mode == "nmse":
            diff = (s_h - t_h).pow(2).sum(dim=-1)
            den  = t_h.pow(2).sum(dim=-1).clamp_min(1e-6)
            per_token = diff / den
        else:  # mse
            per_token = (s_h - t_h).pow(2).mean(dim=-1)
        layer_losses.append((per_token * mask).sum() / denom)

    layer_losses = torch.stack(layer_losses)  # (L,)
    L = layer_losses.numel()

    alpha = min(step / max(total_steps, 1), 1.0)
    weights = layer_losses.new_full((L,), (1.0 - alpha) / L)
    weights[-1] = weights[-1] + alpha  # final layer gets extra weight

    return (weights * layer_losses).sum()


def _kl_loss(s_logits, t_logits, labels, temperature, topk, reverse=False):
    """Token-level KL divergence on CoT positions (labels != -100).

    reverse=False: forward KL D(T||S) over teacher top-K tokens (default)
    reverse=True:  reverse KL D(S||T) full vocab, always >= 0
    topk used for forward KL and for diag metrics in both modes.
    """
    # align: logit at position t predicts token at t+1
    s_logits = s_logits[:, :-1, :]       # (B, T-1, V) (batch size, seq len-1, vocab size)
    t_logits = t_logits[:, :-1, :]
    labels   = labels[:, 1:]             # (B, T-1)
    mask = (labels != -100).float()
    denom = mask.sum().clamp(min=1)
    if mask.sum() == 0:
        return s_logits.new_tensor(0.0), {}

    s_logp_full = F.log_softmax(s_logits / temperature, dim=-1)
    t_logp_full = F.log_softmax(t_logits / temperature, dim=-1)

    if reverse:
        # D(S||T) = sum_x S(x) * (log S(x) - log T(x)), always >= 0
        # F.kl_div is a fused kernel: mathematically identical to
        # (s_logp.exp() * (s_logp - t_logp)).sum(-1) but doesn't materialize
        # exp()/subtract/multiply as separate (B,T,V) tensors -- at
        # seqlen=8192 x full vocab (~152k) each such tensor is ~5GB, so the
        # naive elementwise chain was keeping 3-4 of them alive at once and
        # eating most of an 80GB GPU's headroom for this single loss term.
        kl = F.kl_div(t_logp_full, s_logp_full, log_target=True, reduction='none').sum(dim=-1)
    elif topk > 0:
        t_topk_idx = t_logits.topk(topk, dim=-1).indices     # (B, T-1, K)
        t_logp = t_logp_full.gather(-1, t_topk_idx)
        s_logp = s_logp_full.gather(-1, t_topk_idx)
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
    else:
        kl = F.kl_div(s_logp_full, t_logp_full, log_target=True, reduction='none').sum(dim=-1)

    diag = {}
    if topk > 0:
        with torch.no_grad():
            t_topk_idx = t_logits.topk(topk, dim=-1).indices
            s_topk_idx = s_logits.topk(topk, dim=-1).indices
            overlap = (s_topk_idx.unsqueeze(-1) == t_topk_idx.unsqueeze(-2)).any(dim=-1)
            diag["kd/overlap_ratio"] = ((overlap.float().mean(dim=-1) * mask).sum() / denom).item()
            s_logp_s = s_logp_full.gather(-1, s_topk_idx)
            t_logp_t = t_logp_full.gather(-1, t_topk_idx)
            s_ent = -(s_logp_s.exp() * s_logp_s).sum(dim=-1)
            t_ent = -(t_logp_t.exp() * t_logp_t).sum(dim=-1)
            diag["kd/entropy_gap"] = (((s_ent - t_ent).abs() * mask).sum() / denom).item()

    loss = (kl * mask).sum() / denom
    return loss, diag


def _mixed_sample(student, teacher, prompt_ids, prompt_mask,
                  max_new_tokens, alpha, temperature, pad_id, eos_id):
    """Token-by-token generation sampling from α*p_T + (1-α)*q_S at each step.

    Adapted from MiniLLM dpkd/transformers generation/utils.py:2964-2997.
    IS weight is computed post-hoc from full forward passes (sampler.py:112-114).

    Returns:
        generated : (B, prompt_len + gen_len)  full token ids
    """
    B, L = prompt_ids.shape
    device = prompt_ids.device
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    past_s, past_t = None, None
    gen_ids_list = []

    cur_input = prompt_ids
    cur_mask  = prompt_mask

    with torch.no_grad():
        for step_i in range(max_new_tokens):
            inp = cur_input if step_i == 0 else cur_input[:, -1:]

            s_out = student(input_ids=inp, attention_mask=cur_mask,
                            past_key_values=past_s, use_cache=True)
            t_out = teacher(input_ids=inp, attention_mask=cur_mask,
                            past_key_values=past_t, use_cache=True)
            past_s = s_out.past_key_values
            past_t = t_out.past_key_values

            s_logits = s_out.logits[:, -1, :].float() / temperature
            t_logits = t_out.logits[:, -1, :].float() / temperature

            # MiniLLM utils.py:2997 — mix distributions then sample
            s_probs = F.softmax(s_logits, dim=-1)
            t_probs = F.softmax(t_logits, dim=-1)
            mixed_probs = (1.0 - alpha) * s_probs + alpha * t_probs

            next_tok = torch.multinomial(mixed_probs, num_samples=1)  # (B, 1)
            next_tok = next_tok.masked_fill(finished.unsqueeze(-1), pad_id)
            finished = finished | (next_tok.squeeze(-1) == eos_id)
            gen_ids_list.append(next_tok)

            cur_input = next_tok
            cur_mask  = torch.cat(
                [cur_mask, torch.ones(B, 1, dtype=cur_mask.dtype, device=device)], dim=1)

            if finished.all():
                break

    gen_new  = torch.cat(gen_ids_list, dim=1)            # (B, gen_len)
    generated = torch.cat([prompt_ids, gen_new], dim=1)  # (B, L + gen_len)
    return generated


class RolloutBuffer:
    """Stores rollout data for PPO reuse (MiniLLM PPOSampler-style).

    Per-rollout tensors (all stored on CPU):
      generated   : (B, seq_len) full token ids
      gen_labels  : (B, seq_len) labels (-100 for prompt/pad positions)
      rewards     : (B, T-1) log p_T(y_t) - log q_S_old(y_t)
      old_s_logp  : (B, T-1) log q_S_old(y_t) — used for PPO ratio
      is_log_w    : (B, T-1) log IS weight = log q_S - log p̃ (0 if no mixed sampling)
    """

    def __init__(self):
        self.generated:  list = []
        self.gen_labels: list = []
        self.rewards:    list = []
        self.old_s_logp: list = []
        self.is_log_w:   list = []

    def add(self, generated, gen_labels, rewards, old_s_logp, is_log_w):
        self.generated.append(generated.cpu())
        self.gen_labels.append(gen_labels.cpu())
        self.rewards.append(rewards.cpu())
        self.old_s_logp.append(old_s_logp.cpu())
        self.is_log_w.append(is_log_w.cpu())

    def __len__(self):
        return len(self.generated)

    def clear(self):
        self.generated.clear()
        self.gen_labels.clear()
        self.rewards.clear()
        self.old_s_logp.clear()
        self.is_log_w.clear()


def _pg_loss(s_logits, t_logits, gen_labels, is_log_w=None, old_s_logp=None,
             stored_rewards=None, cliprange=0.2, gamma=0.99,
             reward_clip=10.0, reward_scale=0.0):
    """MiniLLM-style long-term policy gradient loss with PPO clip.

    r_t = log p_T(y_t) - log q_S_old(y_t) for generated tokens.
    advantages = future-only reversed cumsum A_t = Σ_{t'>t} r_{t'},
    since local reverse KL already covers r_t. Length-normalized, whitened.

    is_log_w      : (B, T-1) log IS weight = log q_S - log p̃, MiniLLM sampler.py:114.
    old_s_logp    : (B, T-1) log q_S_old per position. Used for PPO ratio.
                    Falls back to current logp if None.
    stored_rewards: (B, T-1) pre-computed rewards from rollout buffer (bypasses
                    teacher logit reward computation — MiniLLM ppo_loss pattern).
    cliprange     : PPO clip range ε, MiniLLM losses.py:89-94.
    """
    s_logits_shift = s_logits[:, :-1, :]          # (B, T-1, V)
    t_logits_shift = t_logits[:, :-1, :]
    gen_ids        = gen_labels[:, 1:]             # (B, T-1)
    gen_mask       = (gen_ids != -100).float()

    if gen_mask.sum() == 0:
        return s_logits.new_tensor(0.0)

    with torch.no_grad():
        s_logp = F.log_softmax(s_logits_shift.detach().float(), dim=-1)
        s_logp_tok = s_logp.gather(-1, gen_ids.clamp(min=0).unsqueeze(-1)).squeeze(-1)

        if stored_rewards is not None:
            # buffer PPO mode: rewards pre-computed during rollout collection
            rewards = stored_rewards.to(s_logits.device) * gen_mask
            s_old = (old_s_logp.to(s_logits.device)
                     if old_s_logp is not None else s_logp_tok)
        else:
            # inline mode: compute rewards from current teacher logits
            t_logp = F.log_softmax(t_logits_shift.float(), dim=-1)
            t_logp_tok = t_logp.gather(-1, gen_ids.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            s_old = old_s_logp if old_s_logp is not None else s_logp_tok
            rewards = (t_logp_tok - s_old) * gen_mask   # (B, T-1)

        if reward_scale > 0:
            rewards = rewards / reward_scale
        if reward_clip > 0:
            rewards = rewards.clamp(-reward_clip, reward_clip)

        # future-only discounted reversed cumsum → A_t = Σ_{t'>t} γ^(t'-t-1) r_{t'}
        B, T = rewards.shape
        last = rewards.new_zeros(B)
        adv_list = []
        for t in reversed(range(T)):
            adv_list.append(last)
            last = rewards[:, t] + gamma * last
        advantages = torch.stack(adv_list[::-1], dim=1)

        # length normalization (MiniLLM losses.py:39-49)
        lens = gen_mask.cumsum(dim=-1)
        lens = gen_mask - lens + lens[:, -1:]
        lens = lens.masked_fill(lens == 0, 1)
        advantages = advantages / lens

        # whitening
        n = gen_mask.sum().clamp(min=1)
        adv_mean = (advantages * gen_mask).sum() / n
        adv_var  = ((advantages - adv_mean) ** 2 * gen_mask).sum() / n
        advantages = ((advantages - adv_mean) / (adv_var.sqrt() + 1e-8) * gen_mask).detach()

        # IS weight (MiniLLM sampler.py:115)
        is_w = is_log_w.exp().detach() if is_log_w is not None else 1.0

    # PPO ratio = exp(new_logp - old_logp) * IS weight  (MiniLLM losses.py:72-74)
    s_logp_grad    = F.log_softmax(s_logits_shift.float(), dim=-1)
    s_logp_tok_new = s_logp_grad.gather(-1, gen_ids.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    s_old_det      = s_old.detach() if old_s_logp is not None else s_logp_tok.detach()
    log_ratio      = (s_logp_tok_new - s_old_det) * gen_mask
    ratio          = log_ratio.exp() * is_w

    # PPO clip objective (MiniLLM losses.py:88-94)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    loss = (torch.max(pg_loss1, pg_loss2) * gen_mask).sum() / n
    return loss


def _offload_optimizer_state(optimizer) -> None:
    """Move AdamW's exp_avg/exp_avg_sq (bf16, same size as the model itself --
    ~32GB for 8B params) to CPU in-place, freeing that GPU memory for the
    vLLM engine's wake_up() to actually have room to remap its offloaded
    weights back onto the GPU. Model weights/grads are left alone (grads are
    already None between steps via zero_grad(); only optimizer state is big
    enough here to matter). Single-GPU path only -- FSDP shards this
    per-rank already and isn't what's tight on memory."""
    for state in optimizer.state.values():
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state and state[key].is_cuda:
                state[key] = state[key].to('cpu', non_blocking=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _reload_optimizer_state(optimizer, device) -> None:
    """Undo _offload_optimizer_state after the vLLM rollout + sleep finishes,
    moving exp_avg/exp_avg_sq back onto GPU before the next optimizer.step()."""
    for state in optimizer.state.values():
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state and not state[key].is_cuda:
                state[key] = state[key].to(device, non_blocking=True)
    torch.cuda.synchronize()


def _opkd_vllm_wake(vllm_engine) -> None:
    """Wake the OPKD vLLM engine before a rollout batch, if it supports vLLM's
    sleep/wake_up (the direct in-process single-GPU vllm.LLM object created
    with enable_sleep_mode=True -- not the FSDP sidecar adapter, which has its
    own dedicated/shared GPU budget and isn't put to sleep between rollouts).

    Tracks asleep/awake ourselves (via an attribute stashed on the engine
    object) rather than probing vLLM for its state or swallowing whatever
    wake_up() raises: a wake_up() that fails when we *know* it should be
    asleep is a real (usually GPU-memory) failure and must propagate --
    silently continuing here once already caused a call into
    _sync_opkd_weights_to_vllm() to write into memory vLLM had unmapped,
    segfaulting instead of raising a catchable Python OOM."""
    if hasattr(vllm_engine, 'wake_up') and getattr(vllm_engine, '_opkd_asleep', False):
        vllm_engine.wake_up()
        vllm_engine._opkd_asleep = False


def _opkd_vllm_sleep(vllm_engine) -> None:
    """Sleep (level 1: offload weights to CPU, drop KV cache) the OPKD vLLM
    engine after a rollout batch, releasing its GPU memory for the training
    steps until the next rollout -- this is what let an 8B single-GPU run
    OOM in the KD full-vocab loss (vLLM's ~21GB stayed permanently resident
    otherwise, see slurm_gmp_tr_ntpkd_opd_qwen3_4b_general.sh jobs 41450-55).
    A failure here (e.g. not enough room to even start offloading) should
    also propagate rather than leave training running with an engine we
    believe is asleep but isn't."""
    if hasattr(vllm_engine, 'sleep'):
        vllm_engine.sleep(1)
        vllm_engine._opkd_asleep = True


def _sync_opkd_weights_to_vllm(model: nn.Module, vllm_engine) -> None:
    """Sync current student weights into the OPKD vLLM engine.

    Must be called while inside FSDP.summon_full_params context (or non-FSDP).
    Only rank 0 needs to call this; other ranks just participate in summon_full_params.
    """
    engine = vllm_engine.llm_engine
    # vLLM 0.10+ V1 engine: model_executor lives under engine_core
    executor = engine.engine_core.model_executor if hasattr(engine, 'engine_core') else engine.model_executor
    vllm_model = executor.driver_worker.model_runner.model
    vllm_state = {k: v for k, v in vllm_model.named_parameters()}
    for name, param in model.named_parameters():
        if name in vllm_state:
            vllm_state[name].data.copy_(param.data.to(vllm_state[name].dtype))


def _opkd_broadcast_pool(pool: list, is_distributed: bool, device) -> list:
    """Broadcast OPKD rollout pool from rank 0 to all ranks.

    pool items: {"full_seq": LongTensor[1,T], "prompt_len": int}
    Non-distributed: returns pool unchanged.
    """
    if not is_distributed:
        return pool
    import torch.distributed as _dist
    obj = [pool]
    _dist.broadcast_object_list(obj, src=0)
    result = []
    for item in obj[0]:
        result.append({
            "full_seq": item["full_seq"].to(device),
            "prompt_len": item["prompt_len"],
        })
    return result


@torch.no_grad()
def _opkd_pool_to_batch(pool_items: list, device: str) -> dict:
    """Convert a list of OPKD pool items to a cal_batch dict.

    Each item: {'full_seq': LongTensor[1, T], 'prompt_len': int}
    Returns a batch with input_ids/attention_mask (no labels — valid = all positions).
    """
    seqs = [item['full_seq'] for item in pool_items]
    max_len = max(s.shape[1] for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    attn   = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.shape[1]
        padded[i, :L] = s[0]
        attn[i, :L]   = 1
    return {'input_ids': padded.to(device), 'attention_mask': attn.to(device)}


def _compute_tr_kl(model: nn.Module, cal_batch: dict, cand_masks: dict,
                   maskmgr: 'GradualMaskManager', device: str,
                   kl_reduce: str = 'mean', kl_quantile: float = 0.95) -> float:
    """Compute KL(old || cand) over valid token positions.

    kl_reduce: 'mean' (default) or 'quantile' (uses kl_quantile percentile).
    cal_batch may come from prompt_iter (has 'labels') or from OPKD pool
    (no 'labels' — all non-padding positions are valid).
    Temporarily applies candidate masks, runs two forward passes, then restores.
    """
    input_ids = cal_batch['input_ids'].to(device)
    attn_mask = cal_batch['attention_mask'].to(device)
    # labels may be absent (prompt-only / OPKD rollout batches)
    if 'labels' in cal_batch:
        labels = cal_batch['labels'].to(device)
        valid  = (labels[:, 1:] != -100)  # [B, T-1]
    else:
        valid  = (attn_mask[:, 1:] == 1)  # [B, T-1]

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        old_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits.detach()

    # Temporarily zero newly-pruned weights (old_mask=True & cand_mask=False)
    saved = {}
    for name, param in maskmgr.named_params.items():
        newly_pruned = maskmgr.masks[name] & ~cand_masks[name]
        if newly_pruned.any():
            saved[name] = (newly_pruned, param.data[newly_pruned].clone())
            param.data[newly_pruned] = 0.0

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        cand_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits.detach()

    # Restore
    for name, (mask_idx, vals) in saved.items():
        maskmgr.named_params[name].data[mask_idx] = vals

    if not valid.any():
        return 0.0, None

    # Keep in bfloat16 and delete logits immediately after log_softmax to avoid
    # materializing 5 × [B,T,V] float32 tensors simultaneously (~1.86 GB peak).
    old_lp  = F.log_softmax(old_logits[:, :-1, :], dim=-1)   # [B, T-1, V] bf16
    del old_logits
    cand_lp = F.log_softmax(cand_logits[:, :-1, :], dim=-1)  # [B, T-1, V] bf16
    del cand_logits
    old_p   = old_lp.exp()
    kl_tok  = (old_p * (old_lp - cand_lp)).sum(dim=-1)       # [B, T-1]
    del old_p, cand_lp, old_lp
    kl_vals = kl_tok[valid].float()
    if kl_reduce == 'quantile':
        result = torch.quantile(kl_vals, kl_quantile).item()
    else:
        result = kl_vals.mean().item()
    return max(result, 0.0), kl_vals  # (scalar, per-token KL tensor)


@torch.no_grad()
def _cg_batch(A: torch.Tensor, B: torch.Tensor, A_supp: torch.Tensor,
              X0: torch.Tensor, rtol: float = 1e-3, atol: float = 0.,
              maxiter: int = 10) -> torch.Tensor:
    """Solve A X = B via identity-preconditioned conjugate gradient, with the
    residual masked by A_supp every iteration so entries outside the support
    never move off X0 -- i.e. the pruning mask is preserved exactly, only the
    already-nonzero entries get updated. Ported/simplified from ALPS's
    cg_batch (mazumder-lab/ALPS, alps.py) with the verbose/error-tracking
    scaffolding stripped out; validated to reproduce that implementation's
    numerics in an offline post-hoc test (math500 64.8 -> 73.8 on the TR-GMP
    KD+OPD 1.7B s50 checkpoint, job 700758).
    """
    X_k = X0
    R_k = (B - A @ X_k) * A_supp
    P_k = torch.zeros_like(R_k)
    R_k1 = R_k
    B_norm = torch.norm(B, dim=1)
    stopping = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))
    for k in range(1, maxiter + 1):
        if k == 1:
            P_k = R_k
            R_k1 = R_k
        else:
            R_k2 = R_k1
            P_k1 = P_k
            R_k1 = R_k
            denom = (R_k2 * R_k2).sum(0)
            denom = torch.where(denom == 0, torch.full_like(denom, 1e-8), denom)
            beta = (R_k1 * R_k1).sum(0) / denom
            P_k = R_k1 + beta.unsqueeze(0) * P_k1
        AP = A @ P_k
        denom = (P_k * AP).sum(0)
        denom = torch.where(denom == 0, torch.full_like(denom, 1e-8), denom)
        alpha = (R_k1 * R_k1).sum(0) / denom
        X_k = X_k + alpha.unsqueeze(0) * P_k
        R_k = (R_k1 - alpha.unsqueeze(0) * AP) * A_supp
        resid = torch.norm(A @ X_k - B, dim=1)
        if (resid <= stopping).all():
            break
    return X_k


@torch.no_grad()
def _pcg_correct_masked_weights(model: nn.Module, teacher_model: nn.Module,
                                 maskmgr: 'GradualMaskManager', cal_batch: dict,
                                 device: str, maxiter: int = 5, damp_coef: float = 0.01,
                                 global_step: int = 0, use_wandb: bool = False):
    """ALPS-style PCG backsolve applied to the CURRENT mask, right after a TR-GMP
    mask update, using the already-loaded dense `teacher_model` as the
    reconstruction target -- no extra model load needed.

    Unlike ALPS's own one-shot pipeline (sequential layer-by-layer, each
    layer's calibration input re-derived from the previous layer's
    just-corrected output -- see pcg_correct_gmp_checkpoint.py), this hooks
    ALL target Linear layers at once and captures every layer's input from a
    SINGLE forward pass on the current (pre-correction) weights, then solves
    every layer's correction from that one snapshot. This trades the
    sequential/exact-input-propagation precision for O(1) forward passes
    instead of O(num_layers) -- necessary to make this cheap enough to run
    every mask_interval steps instead of a several-hour one-shot job.

    Only entries maskmgr already kept nonzero are touched (support is fixed
    from the CURRENT mask, not re-derived) -- this never changes sparsity or
    which positions are pruned, only what the surviving weights are worth.
    """
    named_params = maskmgr.named_params
    teacher_params = dict(teacher_model.named_parameters())
    name_to_module = dict(model.named_modules())

    captured = {}
    handles = []

    def _make_hook(pname):
        def hook(module, inp, out):
            x = inp[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            captured[pname] = x.detach().float()
        return hook

    for name in named_params:
        mod_name = name[:-len('.weight')] if name.endswith('.weight') else None
        module = name_to_module.get(mod_name) if mod_name else None
        if not isinstance(module, nn.Linear):
            continue
        handles.append(module.register_forward_hook(_make_hook(name)))

    if not handles:
        return

    was_training = model.training
    model.eval()
    input_ids = cal_batch['input_ids'].to(device)
    attn_mask = cal_batch.get('attention_mask')
    attn_mask = attn_mask.to(device) if attn_mask is not None else None
    model(input_ids=input_ids, attention_mask=attn_mask)
    if was_training:
        model.train()
    for h in handles:
        h.remove()

    n_corrected, worst_resid_ratio = 0, 0.0
    for name, W in named_params.items():
        if name not in captured or name not in teacher_params:
            continue
        X = captured[name]
        if X.shape[0] < 2 or W.dim() != 2:
            continue
        W_dense = teacher_params[name].detach().float().to(W.device)
        W_cur = W.data.detach().float()

        XtX = X.t() @ X
        damp = damp_coef * torch.mean(torch.diag(XtX)).item()
        diag_idx = torch.arange(XtX.shape[0], device=XtX.device)
        XtX[diag_idx, diag_idx] += damp
        X_norm = torch.diag(XtX).sqrt() + 1e-8
        XtX = XtX / X_norm
        XtX = (XtX.T / X_norm).T

        YtX = torch.matmul(W_dense * X_norm, XtX)
        B0 = (W_cur * X_norm).t().contiguous()
        A_supp = (B0 != 0).float()

        B = _cg_batch(XtX, YtX.t(), A_supp, X0=B0, maxiter=maxiter)
        new_w = (B.t() / X_norm).reshape(W.shape).to(W.dtype)

        resid_before = torch.norm(B0)
        resid_after = torch.norm(B - B0)
        if resid_before > 0:
            worst_resid_ratio = max(worst_resid_ratio, (resid_after / resid_before).item())
        W.data.copy_(new_w)
        n_corrected += 1

    captured.clear()
    logging.info(f"  PCG mask correction @ step {global_step}: {n_corrected} layers, "
                 f"max relative weight shift {worst_resid_ratio:.4f}")
    if use_wandb:
        wandb.log({"train/pcg_layers_corrected": n_corrected,
                   "train/pcg_max_relative_shift": worst_resid_ratio, "step": global_step})


@torch.no_grad()
def _pcg_correct_one_weight(W: torch.Tensor, X: torch.Tensor, W_dense: torch.Tensor,
                             maxiter: int, damp_coef: float) -> tuple:
    """Shared per-weight CG backsolve: given captured input activations X and
    the dense reference weight, re-solve W's nonzero entries via _cg_batch.
    Returns (new_weight, relative_shift) or (None, 0.0) if X is degenerate."""
    if X.shape[0] < 2 or W.dim() != 2:
        return None, 0.0
    W_cur = W.data.detach().float()

    XtX = X.t() @ X
    damp = damp_coef * torch.mean(torch.diag(XtX)).item()
    diag_idx = torch.arange(XtX.shape[0], device=XtX.device)
    XtX[diag_idx, diag_idx] += damp
    X_norm = torch.diag(XtX).sqrt() + 1e-8
    XtX = XtX / X_norm
    XtX = (XtX.T / X_norm).T

    YtX = torch.matmul(W_dense.float() * X_norm, XtX)
    B0 = (W_cur * X_norm).t().contiguous()
    A_supp = (B0 != 0).float()

    B = _cg_batch(XtX, YtX.t(), A_supp, X0=B0, maxiter=maxiter)
    new_w = (B.t() / X_norm).reshape(W.shape).to(W.dtype)

    resid_before = torch.norm(B0)
    resid_after = torch.norm(B - B0)
    rel_shift = (resid_after / resid_before).item() if resid_before > 0 else 0.0
    return new_w, rel_shift


@torch.no_grad()
def _pcg_correct_masked_weights_sequential(model: nn.Module, teacher_model: nn.Module,
                                            maskmgr: 'GradualMaskManager', cal_batch: dict,
                                            device: str, maxiter: int = 5, damp_coef: float = 0.01,
                                            global_step: int = 0, use_wandb: bool = False):
    """Sequential (ALPS-style) variant of _pcg_correct_masked_weights: corrects
    decoder layer 0, re-forwards it with the NEW weights to get the actual
    hidden_states layer 1 will see, corrects layer 1 using THAT input, and so
    on -- so each layer's correction accounts for how every earlier
    correction changed its input, unlike the single-snapshot version (which
    captures every layer's input from one forward pass on the
    PRE-correction weights and is blind to upstream corrections).

    Costs one extra forward pass PER DECODER LAYER (~28 for Qwen3-1.7B)
    instead of one forward pass total -- meaningfully slower, which is why
    the single-snapshot version is the default for per-mask-update use.
    """
    named_params = maskmgr.named_params
    teacher_params = dict(teacher_model.named_parameters())
    name_to_module = dict(model.named_modules())

    layers = model.model.layers
    was_training = model.training
    model.eval()

    input_ids = cal_batch['input_ids'].to(device)
    attn_mask = cal_batch.get('attention_mask')
    attn_mask = attn_mask.to(device) if attn_mask is not None else None

    # Capture the exact kwargs (attention_mask/position_ids/position_embeddings)
    # Qwen3Model.forward() passes into decoder layers, plus layer 0's actual
    # input hidden_states, via a pre-hook -- avoids re-deriving rotary
    # embeddings / causal mask construction by hand.
    _cache = {}

    def _catch_layer0(module, args, kwargs):
        _cache['hidden_states'] = args[0] if args else kwargs.get('hidden_states')
        _cache['attention_mask'] = kwargs.get('attention_mask')
        _cache['position_ids'] = kwargs.get('position_ids')
        _cache['position_embeddings'] = kwargs.get('position_embeddings')

    _h0 = layers[0].register_forward_pre_hook(_catch_layer0, with_kwargs=True)
    model(input_ids=input_ids, attention_mask=attn_mask)
    _h0.remove()

    hidden_states = _cache['hidden_states']
    layer_kwargs = {k: v for k, v in _cache.items()
                    if k != 'hidden_states' and v is not None}

    n_corrected_total, worst_resid_ratio = 0, 0.0

    for layer_idx, layer in enumerate(layers):
        prefix = f'model.layers.{layer_idx}.'
        layer_param_names = [n for n in named_params if n.startswith(prefix)]
        if not layer_param_names:
            hidden_states = layer(hidden_states, **layer_kwargs)
            continue

        captured = {}
        handles = []

        def _make_hook(pname):
            def hook(module, inp, out):
                x = inp[0]
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                captured[pname] = x.detach().float()
            return hook

        for name in layer_param_names:
            mod_name = name[:-len('.weight')] if name.endswith('.weight') else None
            module = name_to_module.get(mod_name) if mod_name else None
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(_make_hook(name)))

        layer(hidden_states, **layer_kwargs)  # forward only to trigger hooks
        for h in handles:
            h.remove()

        for name in layer_param_names:
            if name not in captured or name not in teacher_params:
                continue
            W = named_params[name]
            new_w, rel_shift = _pcg_correct_one_weight(
                W, captured[name], teacher_params[name].detach().to(W.device), maxiter, damp_coef)
            if new_w is None:
                continue
            W.data.copy_(new_w)
            worst_resid_ratio = max(worst_resid_ratio, rel_shift)
            n_corrected_total += 1

        # Re-forward with the now-corrected weights -- this is what makes it
        # "sequential": layer_idx+1 will see the ACTUAL post-correction output.
        hidden_states = layer(hidden_states, **layer_kwargs)
        captured.clear()

    if was_training:
        model.train()

    logging.info(f"  Sequential PCG correction @ step {global_step}: {n_corrected_total} weights, "
                 f"max relative weight shift {worst_resid_ratio:.4f}")
    if use_wandb:
        wandb.log({"train/pcg_seq_layers_corrected": n_corrected_total,
                   "train/pcg_seq_max_relative_shift": worst_resid_ratio, "step": global_step})


def _tr_mask_update(maskmgr: 'GradualMaskManager', fisher: 'FisherAccumulator',
                    fsdp_model, model: nn.Module, cal_batch: dict,
                    final_sparsity: float, tr_delta: float,
                    kl_threshold: float, delta_min: float,
                    device: str, max_iters: int = 16,
                    kl_reduce: str = 'mean', kl_quantile: float = 0.95,
                    use_wandb: bool = False, global_step: int = 0) -> tuple:
    """Trust-region mask update via KL-constrained binary search.

    Finds the largest delta s.t. KL(old||cand) <= kl_threshold.
    Returns (new_sparsity, new_tr_delta, reached_target).

    FSDP note: current_sparsity() is LOCAL (per-shard), so break conditions are
    all_reduced across ranks to keep all ranks executing the same NCCL collectives.
    """
    # Init dist early — needed before any all_reduce / early-return.
    _tr_dist = None
    try:
        import torch.distributed as _td
        if _td.is_available() and _td.is_initialized():
            _tr_dist = _td
    except Exception:
        pass
    _tr_rank = _tr_dist.get_rank() if _tr_dist else 0

    current_sp = maskmgr.current_sparsity()

    # Early return: all_reduce so all ranks agree (local shard sparsity can differ).
    _early = int(current_sp >= final_sparsity - 1e-4)
    if _tr_dist:
        _et = torch.tensor([_early], dtype=torch.int32, device=device)
        _tr_dist.all_reduce(_et, op=_tr_dist.ReduceOp.MAX)
        _early = _et.item()
    if _early:
        maskmgr.apply(fsdp_model)
        return current_sp, tr_delta, True, {}

    delta               = tr_delta
    last_accepted_masks = None
    last_accepted_sp    = current_sp
    last_accepted_delta = 0.0
    last_kl             = float('inf')
    prev_accepted       = False  # True if the previous iter was accepted

    for i in range(max_iters):
        if _tr_dist:
            _tr_dist.barrier()
            logging.info(f"  [BARRIER] TR-GMP iter {i} start (rank={_tr_rank})")
        try_sp   = min(current_sp + delta, final_sparsity)
        cand     = maskmgr.candidate_masks(fisher, try_sp, fsdp_model)
        if _tr_dist:
            _tr_dist.barrier()
            logging.info(f"  [BARRIER] after candidate_masks iter {i} (rank={_tr_rank})")
        kl, kl_vals = _compute_tr_kl(model, cal_batch, cand, maskmgr, device,
                                      kl_reduce=kl_reduce, kl_quantile=kl_quantile)
        if _tr_dist:
            _tr_dist.barrier()
            logging.info(f"  [BARRIER] after _compute_tr_kl iter {i} (rank={_tr_rank})")
        accepted = kl <= kl_threshold  # kl is globally reduced → same on all ranks

        logging.info(f"  TR-GMP iter {i}: try_sp={try_sp:.4f} delta={delta:.5f} "
                     f"KL={kl:.5f} {'✓' if accepted else '✗'}")

        if use_wandb and kl_vals is not None:
            import wandb as _wandb
            _wandb.log({
                "tr/kl_hist":    _wandb.Histogram(kl_vals.cpu().numpy()),
                "tr/kl_mean":    kl_vals.mean().item(),
                "tr/kl_max":     kl_vals.max().item(),
                "tr/kl_p50":     torch.quantile(kl_vals, 0.50).item(),
                "tr/kl_p90":     torch.quantile(kl_vals, 0.90).item(),
                "tr/kl_p95":     torch.quantile(kl_vals, 0.95).item(),
                "tr/kl_p99":     torch.quantile(kl_vals, 0.99).item(),
                "tr/kl_reduce_val": kl,
                "tr/try_sp":     try_sp,
                "tr/accepted":   int(accepted),
                "tr/iter":       i,
            }, step=global_step)

        # Compute local break signal, then all_reduce so all ranks break together.
        # try_sp is per-rank (local shard sparsity differs) so "target reached" can
        # fire on some ranks before others without this synchronization.
        _break_now = 0
        if accepted:
            last_accepted_masks = cand
            last_accepted_sp    = try_sp
            last_accepted_delta = delta
            last_kl             = kl
            if try_sp >= final_sparsity - 1e-4:
                _break_now = 1  # target reached on this rank
            else:
                prev_accepted = True
                delta = min(delta * 2.0, final_sparsity - current_sp)
        else:
            if prev_accepted:
                _break_now = 1  # ✓ → ✗ boundary found
            else:
                prev_accepted = False
                delta /= 2.0
                if delta < delta_min:
                    _break_now = 1  # delta exhausted

        if _tr_dist:
            _bt = torch.tensor([_break_now], dtype=torch.int32, device=device)
            _tr_dist.all_reduce(_bt, op=_tr_dist.ReduceOp.MAX)
            _break_now = _bt.item()

        if _break_now:
            break

    if last_accepted_masks is not None:
        # Compute delta BEFORE applying — old values still intact at newly-pruned positions.
        _mask_delta = {}
        for name, param in maskmgr.named_params.items():
            newly_pruned = maskmgr.masks[name] & ~last_accepted_masks[name]
            if newly_pruned.any():
                _mask_delta[name] = (newly_pruned, param.data[newly_pruned].clone())
        maskmgr.masks = last_accepted_masks
        maskmgr.apply(fsdp_model)
        new_sp      = maskmgr.current_sparsity()
        # Also check last_accepted_sp: subsampling threshold can make actual sparsity
        # land slightly below target even when the candidate was accepted at target.
        reached     = new_sp >= final_sparsity - 5e-3 or last_accepted_sp >= final_sparsity - 5e-3
        # Carry forward the delta that worked (doubled for next step)
        new_delta   = min(last_accepted_delta * 2.0, 0.10)
        logging.info(f"  TR-GMP: {current_sp:.4f} → {new_sp:.4f} "
                     f"(delta={last_accepted_delta:.5f}, KL={last_kl:.5f}, next_delta={new_delta:.5f})")
    else:
        # Nothing accepted — keep current masks, shrink delta for next step
        _mask_delta = {}
        maskmgr.apply(fsdp_model)
        new_sp    = current_sp
        reached   = False
        new_delta = max(delta / 2.0, delta_min)
        logging.info(f"  TR-GMP: no accepted delta (current_sp={current_sp:.4f}), "
                     f"shrinking delta to {new_delta:.5f}")

    # All_reduce `reached` so all ranks agree on whether to stop the training loop.
    # new_sp / last_accepted_sp are local shard values → can differ per rank.
    if _tr_dist:
        _rt = torch.tensor([int(reached)], dtype=torch.int32, device=device)
        _tr_dist.all_reduce(_rt, op=_tr_dist.ReduceOp.MAX)
        reached = bool(_rt.item())

    return new_sp, new_delta, reached, _mask_delta


def globalprune_gmp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    FLAGS,
    teacher_model: AutoModelForCausalLM = None,
    dpo_dense_model: AutoModelForCausalLM = None,
    eval_fn=None,        # optional callable(model) → dict of metrics
    prebuilt_vllm_engine=None,   # pre-initialized vLLM engine (FSDP+OPKD: built before dist.init)
    prebuilt_vllm_params=None,   # corresponding SamplingParams
):
    """
    BEST-style GMP training loop with optional token-level KD.

    FLAGS expected attributes:
      steps               int    total training steps
      gmp_batch_size          int    per-device batch size
      gmp_grad_accum          int    gradient accumulation steps
      lr                  float  peak learning rate
      gmp_warmup_ratio        float  fraction of steps for LR warmup
      gmp_mask_interval       int    steps between mask updates
      gmp_fisher_beta         float  EMA beta for Fisher accumulation (0.999)
      gmp_kd_lambda           float  weight for KD loss (0 = NTP only)
      gmp_kd_temperature      float  KD temperature
      gmp_kd_topk             int    top-k for KL (0 = full vocab)
      sparsity_ratio          float  final target sparsity
      gmp_save_path           str    directory to save pruned model
      save_model              bool
      wandb                   bool
    """
    import os as _os_dbg
    print(f"[DBG gmp_train ENTER] pid={_os_dbg.getpid()}", flush=True)
    device = next(model.parameters()).device
    named_params = _find_linear_weights(model)

    # ── FSDP detection ─────────────────────────────────────────────────────────
    fsdp_model = None
    is_fsdp = False
    if _FSDP_AVAILABLE:
        _root = next((m for m in model.modules() if isinstance(m, FSDP)), None)
        if _root is not None:
            fsdp_model = _root
            is_fsdp = True
            logging.info("FSDP detected — enabling summon_full_params for mask updates")

    # Distributed state (DDP or FSDP)
    import torch.distributed as _dist
    is_distributed = _dist.is_available() and _dist.is_initialized()
    local_rank = _dist.get_rank() if is_distributed else 0
    world_size = _dist.get_world_size() if is_distributed else 1
    is_main_process = (local_rank == 0)

    total_steps    = FLAGS.steps
    batch_size     = getattr(FLAGS, 'gmp_batch_size', 1)
    grad_accum     = getattr(FLAGS, 'gmp_grad_accum', 8)
    lr             = getattr(FLAGS, 'lr', 1e-5)
    warmup_ratio        = getattr(FLAGS, 'gmp_warmup_ratio', 0.05)
    lr_schedule         = getattr(FLAGS, 'lr_scheduler', 'cosine')
    mask_interval       = getattr(FLAGS, 'gmp_mask_interval', 32)
    log_interval        = getattr(FLAGS, 'gmp_log_interval', 1)
    fisher_beta         = getattr(FLAGS, 'gmp_fisher_beta', 0.999)
    final_sparsity      = FLAGS.sparsity_ratio
    # Step-based warmup takes priority over the ratio (applies to both the
    # cosine and constant LR schedules); lr_warmup_steps=0 falls back to
    # gmp_warmup_ratio * steps.
    lr_warmup_steps_override = getattr(FLAGS, 'lr_warmup_steps', 0)
    warmup_steps        = lr_warmup_steps_override if lr_warmup_steps_override > 0 else int(total_steps * warmup_ratio)
    constant_warmup_steps    = warmup_steps
    # Step-based takes priority: reserve the last `gmp_sparse_train_steps` steps
    # for fixed-mask sparse training (pruning/cubic ramp already done by then),
    # instead of deriving the cutoff from gmp_pruning_end_ratio.
    sparse_train_steps  = getattr(FLAGS, 'gmp_sparse_train_steps', 0)
    if sparse_train_steps > 0:
        pruning_end_steps = max(0, total_steps - sparse_train_steps)
    else:
        pruning_end_ratio = getattr(FLAGS, 'gmp_pruning_end_ratio', 1.0)
        pruning_end_steps = int(total_steps * pruning_end_ratio)
    # Gates mask application, TR-GMP growth, the cubic sparsity ramp, PGD, and
    # DPO-queue refill alike (was previously coupled to the LR warmup_steps
    # variable at the _cubic_sparsity call sites instead, and only checked in
    # one of the two call sites — now unified on this single flag everywhere).
    dense_warmup_steps  = getattr(FLAGS, 'gmp_dense_warmup_steps', 0)
    # TR-GMP flags
    tr_enabled      = getattr(FLAGS, 'gmp_tr_enabled', False)
    tr_kl_threshold = getattr(FLAGS, 'gmp_tr_kl_threshold', 0.01)
    tr_delta_init   = getattr(FLAGS, 'gmp_tr_delta_init', 0.05)
    tr_delta_min    = getattr(FLAGS, 'gmp_tr_delta_min', 0.005)
    tr_kl_reduce    = getattr(FLAGS, 'gmp_tr_kl_reduce', 'mean')
    tr_kl_quantile  = getattr(FLAGS, 'gmp_tr_kl_quantile', 0.95)
    use_wandb      = getattr(FLAGS, 'wandb', False) and is_main_process
    ntp_lambda     = getattr(FLAGS, 'gmp_ntp_lambda', 1.0)
    kd_lambda      = getattr(FLAGS, 'gmp_kd_lambda', 0.0)
    kd_temperature = getattr(FLAGS, 'gmp_kd_temperature', 2.0)
    kd_topk        = getattr(FLAGS, 'gmp_kd_topk', 0)
    kd_only        = getattr(FLAGS, 'gmp_kd_only', False)
    hidden_lambda  = getattr(FLAGS, 'gmp_hidden_lambda', 0.0)
    hidden_only    = getattr(FLAGS, 'gmp_hidden_only', False)
    hidden_mode    = getattr(FLAGS, 'gmp_hidden_mode', 'cosine')
    hidden_mask    = getattr(FLAGS, 'gmp_hidden_mask', 'cot')
    hidden_layers  = getattr(FLAGS, 'gmp_hidden_layers', 'final')  # 'final' or 'anneal_all_to_final'
    onpolicy_lambda     = getattr(FLAGS, 'gmp_onpolicy_kd_lambda', 0.0)
    onpolicy_interval   = getattr(FLAGS, 'gmp_onpolicy_kd_interval', 32)
    opkd_reuse_ipo         = getattr(FLAGS, 'gmp_opkd_reuse_ipo_rollouts', False)
    opkd_vllm_gpu_mem      = getattr(FLAGS, 'gmp_opkd_vllm_gpu_mem', 0.35)
    opkd_prev_mask_teacher  = getattr(FLAGS, 'gmp_opkd_prev_mask_teacher', False)
    prevmask_opkd_lambda    = getattr(FLAGS, 'gmp_prevmask_opkd_lambda', 0.0)
    measure_grad_conflict       = getattr(FLAGS, 'gmp_measure_grad_conflict', False)
    filter_grad_conflict        = getattr(FLAGS, 'gmp_filter_grad_conflict', False)
    project_opkd_onto_combined  = getattr(FLAGS, 'gmp_opkd_project_onto_combined', False)
    filter_opkd_combined        = getattr(FLAGS, 'gmp_opkd_filter_combined', False)
    onpolicy_max_new    = getattr(FLAGS, 'gmp_onpolicy_max_new_tokens', 256)
    onpolicy_topk       = getattr(FLAGS, 'gmp_onpolicy_kd_topk', 0)
    onpolicy_temp       = getattr(FLAGS, 'gmp_onpolicy_temperature', 0.6)
    onpolicy_grad_accum = max(1, getattr(FLAGS, 'gmp_onpolicy_grad_accum', 1))
    onpolicy_grad_clip  = getattr(FLAGS, 'gmp_onpolicy_grad_clip', 1.0)
    onpolicy_reverse_kl = getattr(FLAGS, 'gmp_onpolicy_reverse_kl', False)
    onpolicy_pg           = getattr(FLAGS, 'gmp_onpolicy_pg', False)
    onpolicy_mixed_alpha  = getattr(FLAGS, 'gmp_onpolicy_mixed_alpha', 0.0)
    onpolicy_pg_cliprange = getattr(FLAGS, 'gmp_onpolicy_pg_cliprange', 0.2)
    onpolicy_pg_gamma     = getattr(FLAGS, 'gmp_onpolicy_pg_gamma', 0.99)
    rollout_buffer_size   = getattr(FLAGS, 'gmp_rollout_buffer_size', 0)
    ppo_epochs            = getattr(FLAGS, 'gmp_ppo_epochs', 2)
    pg_reward_clip        = getattr(FLAGS, 'gmp_pg_reward_clip', 10.0)
    pg_reward_scale       = getattr(FLAGS, 'gmp_pg_reward_scale', 0.0)
    use_rollout = onpolicy_pg and rollout_buffer_size > 0
    anchor_lambda     = getattr(FLAGS, 'gmp_anchor_kd_lambda', 0.0)
    anchor_interval   = getattr(FLAGS, 'gmp_anchor_kd_interval', 32)
    anchor_prefix_len = getattr(FLAGS, 'gmp_anchor_prefix_len', 1536)
    anchor_max_new    = getattr(FLAGS, 'gmp_anchor_max_new_tokens', 512)
    teacher_seqkd      = getattr(FLAGS, 'gmp_teacher_seqkd', False)
    teacher_seqkd_temp = getattr(FLAGS, 'gmp_onpolicy_temperature', 1.0)
    teacher_seqkd_max_new = getattr(FLAGS, 'gmp_onpolicy_max_new_tokens', 512)
    pgd_enabled    = getattr(FLAGS, 'gmp_pgd', False)

    use_kd         = (teacher_model is not None) and (kd_lambda > 0.0)
    use_hidden     = (teacher_model is not None) and (hidden_lambda > 0.0)
    use_teacher_gen_kd_flag = getattr(FLAGS, 'gmp_teacher_gen_kd', False)
    # Teacher-gen KD (forward KL, prompts pre-generated once from data_path)
    # and on-policy/OPD (reverse KL, live student rollouts from gmp_prompt_path)
    # draw from independently-configurable prompt sources and don't share any
    # generation state, so they can run together -- both are weighted by
    # gmp_onpolicy_kd_lambda (same knob, applied to each loss term separately).
    use_onpolicy   = (teacher_model is not None) and (onpolicy_lambda > 0.0)
    use_anchor     = (teacher_model is not None) and (anchor_lambda > 0.0)
    use_teacher_seqkd = (teacher_model is not None) and teacher_seqkd

    # DPO flags
    dpo_lambda        = getattr(FLAGS, 'gmp_dpo_lambda', 0.0)
    dpo_beta          = getattr(FLAGS, 'gmp_dpo_beta', 0.1)
    dpo_n_pairs       = getattr(FLAGS, 'gmp_dpo_n_pairs', 1024)
    dpo_gen_batch     = getattr(FLAGS, 'gmp_dpo_gen_batch', 8)
    dpo_max_new       = getattr(FLAGS, 'gmp_dpo_max_new_tokens', 512)
    dpo_temperature   = getattr(FLAGS, 'gmp_dpo_temperature', 0.7)
    dpo_start_step    = getattr(FLAGS, 'gmp_dpo_start_step', 0)
    dpo_reference_free = getattr(FLAGS, 'gmp_dpo_reference_free', False)
    dpo_loss_type      = getattr(FLAGS, 'gmp_dpo_loss_type', 'sigmoid')
    use_dpo_loss = (dpo_lambda > 0.0) and (dpo_dense_model is not None)
    _use_vllm_rejected = getattr(FLAGS, 'gmp_dpo_use_vllm_rejected', False)
    use_dpo_queue = use_dpo_loss or (_use_vllm_rejected and use_onpolicy)
    use_dpo = use_dpo_loss  # alias for IPO-loss-specific paths

    # ── Offline IPO (UltraFeedback-style preference pairs) ────────────────────
    offline_ipo_lambda   = getattr(FLAGS, 'gmp_offline_ipo_lambda', 0.0)
    offline_ipo_beta     = getattr(FLAGS, 'gmp_offline_ipo_beta', 0.1)
    offline_ipo_datasets = getattr(FLAGS, 'gmp_offline_ipo_datasets',
                                   'HuggingFaceH4/ultrafeedback_binarized')
    offline_ipo_splits   = getattr(FLAGS, 'gmp_offline_ipo_splits', 'train_prefs')
    offline_ipo_per_max  = getattr(FLAGS, 'gmp_offline_ipo_per_max', 8000)
    offline_ipo_max_len  = getattr(FLAGS, 'gmp_offline_ipo_max_length', 2048)
    offline_ipo_max_prompt = getattr(FLAGS, 'gmp_offline_ipo_max_prompt_length', 1024)
    use_offline_ipo = (offline_ipo_lambda > 0.0)
    offline_ipo_iter = None
    use_teacher_gen_kd = getattr(FLAGS, 'gmp_teacher_gen_kd', False) and (dpo_dense_model is not None or teacher_model is not None)

    if use_kd or use_hidden or use_onpolicy or use_teacher_seqkd or use_teacher_gen_kd:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    if use_dpo_loss:
        dpo_dense_model.eval()
        for p in dpo_dense_model.parameters():
            p.requires_grad_(False)

    if use_offline_ipo:
        from lib.gmp_dpo import (build_offline_ipo_dataset, OfflineIPOCollator,
                                  dpo_loss as _offline_ipo_loss_fn, concatenated_forward)
        _ds_names = [d.strip() for d in offline_ipo_datasets.split(",")]
        _splits   = [s.strip() for s in offline_ipo_splits.split(",")]
        _per_maxs = [int(x) for x in str(offline_ipo_per_max).split(",")]
        # Broadcast single per_max to all datasets
        if len(_per_maxs) == 1:
            _per_maxs = _per_maxs * len(_ds_names)
        if len(_splits) == 1:
            _splits = _splits * len(_ds_names)
        _offline_ds = build_offline_ipo_dataset(
            _ds_names, _splits, _per_maxs, tokenizer,
            max_length=offline_ipo_max_len,
            max_prompt_length=offline_ipo_max_prompt,
            seed=getattr(FLAGS, 'seed', 42),
        )
        _pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        _offline_loader = DataLoader(
            _offline_ds,
            batch_size=1,
            shuffle=True,
            collate_fn=OfflineIPOCollator(_pad_val),
            drop_last=True,
        )
        offline_ipo_iter = _infinite(_offline_loader)
        # Use dense model as ref if available, else frozen copy of initial student
        if dpo_dense_model is not None:
            _offline_ipo_ref = dpo_dense_model
        else:
            import copy as _copy
            _offline_ipo_ref = _copy.deepcopy(model)
            _offline_ipo_ref.eval()
            for p in _offline_ipo_ref.parameters():
                p.requires_grad_(False)
        logging.info(f"[offline_ipo] lambda={offline_ipo_lambda}, beta={offline_ipo_beta}, "
                     f"datasets={_ds_names}, per_max={_per_maxs}")

    # Prompt dataset for on-policy generation or teacher SeqKD
    prompt_iter = None
    if use_onpolicy or use_teacher_seqkd or tr_enabled:
        from lib.gkd_admm_trainer import MixedPromptDataset
        prompt_path = getattr(FLAGS, 'gmp_prompt_path', None) or getattr(FLAGS, 'data_path', None)
        prompt_max_len = getattr(FLAGS, 'gmp_max_prompt_len', 512)
        _prompt_ds = MixedPromptDataset(
            jsonl_path=prompt_path,
            tokenizer=tokenizer,
            max_prompt_len=prompt_max_len,
        )
        from lib.gkd_admm_trainer import collate_prompts
        _prompt_loader = DataLoader(
            _prompt_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_prompts(tokenizer.pad_token_id or 0),
        )
        prompt_iter = _infinite(_prompt_loader)
        logging.info(f"  On-policy KD: lambda={onpolicy_lambda}, interval={onpolicy_interval}, "
                     f"max_new_tokens={onpolicy_max_new}, topk={onpolicy_topk}")

    _opkd_vllm_engine = None
    _opkd_vllm_params = None
    _opkd_standalone_pool: list = []
    _opkd_standalone_pool_ptr: int = 0
    _opkd_prev_delta = None  # {name: (positions, old_values)} — prev-mask weight delta for OPKD teacher
    if use_onpolicy:
        import os as _os
        _os.environ['VLLM_USE_V1'] = '0'
        print(f"[DBG use_onpolicy] rank={local_rank} pid={_os.getpid()} before vllm import", flush=True)
        from vllm.inputs import TokensPrompt as _TokensPrompt
        print(f"[DBG use_onpolicy] rank={local_rank} after vllm import, prebuilt={prebuilt_vllm_engine is not None}", flush=True)
        if prebuilt_vllm_engine is not None:
            # vLLM was pre-initialized in main.py BEFORE dist.init_process_group (FSDP path).
            # vLLM calls torch.distributed.new_group() internally — a global collective that
            # requires ALL world ranks. Pre-init avoids the deadlock by running vLLM before
            # dist is initialized, so new_group() is a no-op.
            if is_main_process:
                _opkd_vllm_engine = prebuilt_vllm_engine
                _opkd_vllm_params = prebuilt_vllm_params
                logging.info("  OPKD vLLM: using pre-built engine (standalone init before dist.init_process_group)")
        elif not is_distributed:
            # Single-GPU path: no FSDP, no dist conflict — init vLLM normally here.
            from vllm import LLM, SamplingParams as _VLLMSamplingParams
            _opkd_vllm_enforce_eager = getattr(FLAGS, 'gmp_opkd_vllm_enforce_eager', False)
            logging.info(f"  OPKD vLLM: initializing engine (single-GPU, enforce_eager={_opkd_vllm_enforce_eager}) gpu_mem={opkd_vllm_gpu_mem} ...")
            _opkd_vllm_engine = LLM(
                getattr(FLAGS, 'model', None),
                dtype="bfloat16",
                gpu_memory_utilization=opkd_vllm_gpu_mem,
                trust_remote_code=True,
                max_model_len=onpolicy_max_new + getattr(FLAGS, 'gmp_max_prompt_len', 512),
                enforce_eager=_opkd_vllm_enforce_eager,
                # enables .sleep(1)/.wake_up() -- offload weights to CPU + drop
                # KV cache between rollouts so this engine's ~gpu_mem-fraction
                # GPU footprint isn't permanently resident on a single shared
                # GPU (see _opkd_vllm_sleep/_opkd_vllm_wake call sites below).
                enable_sleep_mode=True,
            )
            _opkd_vllm_params = _VLLMSamplingParams(
                max_tokens=onpolicy_max_new,
                temperature=onpolicy_temp,
                top_p=0.95,
            )
            logging.info("  OPKD vLLM: engine ready")
        else:
            # FSDP multi-GPU but no pre-built engine — should not happen (main.py always pre-inits).
            logging.warning("  OPKD vLLM: no pre-built engine in FSDP mode — disabling on-policy KD.")

        # Pre-fill pool: sync weights to vLLM, then rank 0 generates rollouts.
        # FSDP: summon_full_params is a collective — all ranks must enter together.
        # Rank 0 then sends the gathered CPU state_dict to the vLLM subprocess.
        _in_fsdp = (fsdp_model is not None and _FSDP_AVAILABLE)
        _fsdp_ctx = (FSDP.summon_full_params(fsdp_model, writeback=False, offload_to_cpu=True, rank0_only=True)
                     if _in_fsdp else nullcontext())
        with _fsdp_ctx:
            if is_main_process and _opkd_vllm_engine is not None:
                _opkd_vllm_wake(_opkd_vllm_engine)
                if _in_fsdp and hasattr(_opkd_vllm_engine, 'sync_weights'):
                    _sd = {n: p.data.cpu() for n, p in model.named_parameters()}
                    logging.info("  OPKD vLLM: syncing weights (initial pool, FSDP→subprocess)")
                    _opkd_vllm_engine.sync_weights(_sd)
                    del _sd
                elif not _in_fsdp:
                    _sync_opkd_weights_to_vllm(model, _opkd_vllm_engine)
        if is_main_process:
            _n_pool = mask_interval * grad_accum
            _pool_batches = [next(prompt_iter) for _ in range(_n_pool)]
            _vllm_inputs = [
                _TokensPrompt(prompt_token_ids=b['input_ids'][0][:int(b['prompt_len'].item())].tolist())
                for b in _pool_batches
            ]
            _vllm_outs = _opkd_vllm_engine.generate(_vllm_inputs, _opkd_vllm_params)
            _opkd_vllm_sleep(_opkd_vllm_engine)
            for _pb, _vo in zip(_pool_batches, _vllm_outs):
                _plen = int(_pb['prompt_len'].item())
                _p_ids = _pb['input_ids'][:, :_plen].cpu()
                _gen_ids = torch.tensor([_vo.outputs[0].token_ids], dtype=torch.long)
                _full_seq = torch.cat([_p_ids, _gen_ids], dim=1)
                _opkd_standalone_pool.append({"full_seq": _full_seq, "prompt_len": _plen})
            logging.info(f"  OPKD vLLM: initial pool filled with {len(_opkd_standalone_pool)} rollouts")
        logging.info(f"[rank {local_rank}] pre-broadcast: pool={len(_opkd_standalone_pool)}")
        _opkd_standalone_pool = _opkd_broadcast_pool(_opkd_standalone_pool, is_distributed, device)
        logging.info(f"[rank {local_rank}] post-broadcast: pool={len(_opkd_standalone_pool)}")

    rollout_buffer = RolloutBuffer() if use_rollout else None

    fixed_mask       = getattr(FLAGS, 'gmp_fixed_mask', False)
    l1_lambda        = getattr(FLAGS, 'gmp_l1_lambda', 0.0)
    l1_structured    = getattr(FLAGS, 'gmp_l1_structured', True)
    l1_mode          = getattr(FLAGS, 'gmp_l1_mode', 'plain')
    l1_fisher_cmin   = getattr(FLAGS, 'gmp_l1_fisher_clip_min', 0.1)
    l1_fisher_cmax   = getattr(FLAGS, 'gmp_l1_fisher_clip_max', 10.0)
    l1_open_only     = getattr(FLAGS, 'gmp_l1_open_groups_only', False)

    # N:M semi-structured sparsity support (e.g. "2:4")
    sparsity_type = getattr(FLAGS, 'sparsity_type', 'unstructured')
    prune_n, prune_m = 0, 0
    if sparsity_type != 'unstructured':
        prune_n, prune_m = map(int, sparsity_type.split(':'))
        logging.info(f"  N:M semi-structured sparsity: {prune_n}:{prune_m}")
    use_l1 = l1_lambda > 0.0
    if use_l1:
        use_structured_l1 = l1_structured and prune_n > 0 and prune_m > 0
        if use_structured_l1:
            logging.info(f"  Bottom-2 structured L1: lambda={l1_lambda}")
        else:
            logging.info(f"  L1 mode={l1_mode}: lambda={l1_lambda}"
                         f"{' (open-groups-only)' if l1_open_only else ''}")

    if getattr(FLAGS, 'gmp_gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        logging.info("  Gradient checkpointing ENABLED (reduces activation memory)")

    logging.info(f"[rank {local_rank}] creating optimizer")
    base_optimizer = getattr(FLAGS, 'gmp_base_optimizer', 'adamw')
    if base_optimizer == 'activation_metric_pgd':
        from .activation_metric_projected_sgd import ActivationMetricProjectedSGD
        optimizer = ActivationMetricProjectedSGD(
            model.parameters(), lr=lr,
            lam=getattr(FLAGS, 'gmp_pgd_lam', 1e-3),
            group_size=getattr(FLAGS, 'gmp_pgd_group_size', 4),
            trust_ratio=getattr(FLAGS, 'gmp_pgd_trust_ratio', 5.0),
            momentum=getattr(FLAGS, 'gmp_pgd_momentum', 0.0),
        )
        logging.info(f"  Base optimizer: ActivationMetricProjectedSGD (lr={lr}, lam={FLAGS.gmp_pgd_lam}, "
                     f"group_size={FLAGS.gmp_pgd_group_size}, trust_ratio={FLAGS.gmp_pgd_trust_ratio}, "
                     f"momentum={FLAGS.gmp_pgd_momentum})")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    logging.info(f"[rank {local_rank}] optimizer created")
    _fisher_source = getattr(FLAGS, 'gmp_fisher_source', 'adam')
    if _fisher_source == 'opd_empirical':
        fisher = EmpiricalFisherAccumulator(named_params, saliency=FLAGS.gmp_saliency)
        logging.info("Fisher source: opd_empirical (grad^2 on OPD cal_batch)")
    else:
        fisher = FisherAccumulator(named_params, optimizer, saliency=FLAGS.gmp_saliency)
        if base_optimizer == 'activation_metric_pgd':
            logging.info("Fisher source: adam (exp_avg_sq) -- no-op with activation_metric_pgd "
                         "(no such state), fine for gmp_fixed_mask=true where it's never consulted")
        else:
            logging.info("Fisher source: adam (exp_avg_sq)")
    maskmgr = GradualMaskManager(named_params, fsdp_model, prune_n=prune_n, prune_m=prune_m,
                                  pruning_scope=getattr(FLAGS, 'gmp_pruning_scope', 'global'))
    if fixed_mask:
        maskmgr.init_from_weights()
        maskmgr.apply(fsdp_model)
    if lr_schedule in ('constant', 'constant_with_warmup'):
        if constant_warmup_steps > 0:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=constant_warmup_steps)
            logging.info(f"  LR schedule: constant with {constant_warmup_steps}-step linear warmup, no decay")
        else:
            scheduler = get_constant_schedule(optimizer)
            logging.info("  LR schedule: constant (no warmup, no decay)")
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    _pad_tok = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    _collate_fn = lambda b: _collate(b, pad_token_id=_pad_tok)

    if is_distributed:
        _train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True
        )
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=_train_sampler,
            collate_fn=_collate_fn,
        )
    else:
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
        )
    data_iter = _infinite(loader, sampler=_train_sampler if is_distributed else None)

    # Anchor KD: separate iterator over CoT dataset (batch_size=1)
    anchor_iter = None
    if use_anchor:
        _anchor_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=_collate_fn,
        )
        anchor_iter = _infinite(_anchor_loader)
        logging.info(f"  Anchor KD: lambda={anchor_lambda}, interval={anchor_interval}, "
                     f"prefix_len={anchor_prefix_len}, max_new_tokens={anchor_max_new}")

    # ── DPO setup ──────────────────────────────────────────────────────────────
    import copy as _copy
    dpo_chosen_cache   = None
    dpo_ref_model      = None
    dpo_rejected_queue = None

    use_ca_ipo = (dpo_loss_type == "ca_ipo")

    if use_dpo_queue and is_main_process:
        from lib.gmp_dpo import (RejectedQueue, generate_chosen_cache,
                                  concatenated_forward, dpo_loss as _dpo_loss,
                                  ca_ipo_loss as _ca_ipo_loss,
                                  get_completion_token_logps as _get_token_logps)
        from lib.gkd_admm_trainer import NTPPromptWrapper
        _dpo_cache_dir = getattr(FLAGS, 'gmp_dpo_cache_dir', '') or None
        _sync_n_pairs = total_steps * batch_size * grad_accum
        _dpo_data_path = getattr(FLAGS, 'data_path', '') or ''
        _dense_model_path = getattr(FLAGS, 'model', None) or getattr(FLAGS, 'model_path', None)
        _use_vllm_chosen = getattr(FLAGS, 'gmp_dpo_use_vllm_chosen', False)

        if use_dpo_loss:
            # Full IPO path: generate teacher chosen cache + ref model
            _dpo_prompt_ds = NTPPromptWrapper(train_dataset)
            dpo_chosen_cache = generate_chosen_cache(
                dpo_dense_model, tokenizer, _dpo_prompt_ds,
                n_pairs=_sync_n_pairs, gen_batch_size=dpo_gen_batch,
                max_new_tokens=dpo_max_new, temperature=dpo_temperature,
                device=device,
                cache_dir=_dpo_cache_dir,
                prompt_path=f"{_dpo_data_path}|ntp_prompt_wrapper|gbs={batch_size*grad_accum}",
                store_teacher_logps=use_ca_ipo,
                use_vllm=_use_vllm_chosen,
                model_path=_dense_model_path,
            )
            dpo_ref_model = _copy.deepcopy(model).eval()
            for p in dpo_ref_model.parameters():
                p.requires_grad_(False)
            _cache_for_queue = dpo_chosen_cache
            logging.info(f"  DPO: lambda={dpo_lambda}, beta={dpo_beta}, loss_type={dpo_loss_type}, "
                         f"n_pairs={_sync_n_pairs} (={total_steps}*{batch_size*grad_accum}), "
                         f"max_new={dpo_max_new}, start_step={dpo_start_step}")
        else:
            # OPKD-only path: build pseudo cache from NTP prompts (no teacher response needed)
            _dpo_prompt_ds = NTPPromptWrapper(train_dataset)
            _pseudo_dl = DataLoader(_dpo_prompt_ds, batch_size=1, shuffle=False)
            _cache_for_queue = []
            for _i, _b in enumerate(_pseudo_dl):
                if _i >= _sync_n_pairs:
                    break
                _cache_for_queue.append({
                    "prompt_input_ids":      _b["input_ids"][0:1],
                    "prompt_attention_mask": _b["attention_mask"][0:1],
                })
            logging.info(f"  OPKD-vLLM queue (no IPO): pseudo_cache={len(_cache_for_queue)} prompts")

        dpo_rejected_queue = RejectedQueue(
            _cache_for_queue, mask_interval,
            gen_batch_size=dpo_gen_batch,
            max_new_tokens=dpo_max_new,
            temperature=dpo_temperature,
            teacher_model=dpo_dense_model if use_ca_ipo else None,
            grad_accum=batch_size * grad_accum,
            use_vllm=_use_vllm_rejected,
            model_path=_dense_model_path if _use_vllm_rejected else None,
            vllm_gpu_memory_utilization=getattr(FLAGS, 'gmp_dpo_vllm_gpu_mem', 0.35),
        )
        model.eval()
        dpo_rejected_queue.refill(model, tokenizer, str(device))
        model.train()
        maskmgr.apply(fsdp_model)
    elif use_dpo_loss:
        from lib.gmp_dpo import (concatenated_forward, dpo_loss as _dpo_loss,
                                  ca_ipo_loss as _ca_ipo_loss,
                                  get_completion_token_logps as _get_token_logps)

    # ── Teacher-gen KD cache (fixed, no refill) ───────────────────────────────
    tgkd_cache = None
    _tgkd_ptr  = 0
    if use_teacher_gen_kd and is_main_process:
        from lib.gmp_dpo import generate_chosen_cache as _gen_chosen
        from lib.gkd_admm_trainer import NTPPromptWrapper
        _tgkd_n    = total_steps * batch_size * grad_accum
        _tgkd_path = getattr(FLAGS, 'data_path', '') or ''
        _tgkd_model_path = getattr(FLAGS, 'model', None) or getattr(FLAGS, 'model_path', None)
        _tgkd_cache_dir  = getattr(FLAGS, 'gmp_dpo_cache_dir', '') or None

        if dpo_chosen_cache is not None:
            # Reuse IPO chosen cache — no extra generation needed
            tgkd_cache = dpo_chosen_cache
            logging.info(f"  TGKD: reusing IPO chosen cache ({len(tgkd_cache)} entries), lambda={onpolicy_lambda}")
        else:
            _tgkd_ds    = NTPPromptWrapper(train_dataset)
            _tgkd_dense = dpo_dense_model if dpo_dense_model is not None else teacher_model
            tgkd_cache = _gen_chosen(
                _tgkd_dense, tokenizer, _tgkd_ds,
                n_pairs=_tgkd_n, gen_batch_size=dpo_gen_batch,
                max_new_tokens=dpo_max_new, temperature=dpo_temperature,
                device=device,
                cache_dir=_tgkd_cache_dir,
                prompt_path=f"{_tgkd_path}|ntp_prompt_wrapper|gbs={batch_size*grad_accum}",
                store_teacher_logps=False,
                use_vllm=getattr(FLAGS, 'gmp_dpo_use_vllm_chosen', False),
                model_path=_tgkd_model_path,
            )
            logging.info(f"  TGKD: generated new cache ({len(tgkd_cache)} entries), lambda={onpolicy_lambda}")

        # Pre-compute teacher top-K logits for forward KL (one-time, no teacher forward during training)
        _tgkd_topk_k = onpolicy_topk if onpolicy_topk > 0 else 100
        if "teacher_topk_logits" not in tgkd_cache[0]:
            _tgkd_teacher = dpo_dense_model if dpo_dense_model is not None else teacher_model
            _tgkd_teacher.to(device).eval()
            logging.info(f"  TGKD: pre-computing teacher top-{_tgkd_topk_k} logits for {len(tgkd_cache)} entries ...")
            for _entry in tgkd_cache:
                _tc_ids_e  = _entry["chosen_input_ids"].to(device)
                _tc_msk_e  = _entry["chosen_attention_mask"].to(device)
                _tc_plen_e = _entry["prompt_input_ids"].shape[1]
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _t_logits_e = _tgkd_teacher(_tc_ids_e, attention_mask=_tc_msk_e).logits
                # logit[i] predicts token[i+1]; completion = tokens[plen:]
                _comp_e = _t_logits_e[0, _tc_plen_e - 1:-1].float()   # [comp_len, V]
                _tv, _ti = _comp_e.topk(_tgkd_topk_k, dim=-1)
                _entry["teacher_topk_logits"]  = _tv.cpu()             # [comp_len, K]
                _entry["teacher_topk_indices"] = _ti.cpu()             # [comp_len, K]
                del _t_logits_e, _comp_e
            torch.cuda.empty_cache()
            # free GPU if teacher not needed for training
            if not (use_kd or use_hidden or use_onpolicy or use_anchor or use_dpo_loss):
                _tgkd_teacher.to('cpu')
                torch.cuda.empty_cache()
            logging.info("  TGKD: teacher top-K logits pre-computed.")

    model.train()
    optimizer.zero_grad()

    start_time = time.time()
    step = 0
    tr_delta        = tr_delta_init   # current TR step size, adapted each mask update
    tr_reached      = False           # set True when target sparsity achieved
    _tr_reached_step = None           # step at which tr_reached first flipped True
    _post_target_steps = getattr(FLAGS, 'gmp_post_target_steps', -1)
    if _post_target_steps < 0:
        _post_target_steps = mask_interval  # default: stop after exactly one more mask-update cycle

    do_save = getattr(FLAGS, 'save_model', False) and getattr(FLAGS, 'gmp_save_path', None)

    # Milestone checkpointing (TR-GMP multi-target): save model at each milestone sparsity,
    # then eval post-hoc after training. Avoids vLLM mid-training memory conflicts.
    # Recovery: wait `mask_interval` steps after first crossing milestone before saving,
    # so the checkpoint reflects one full mask-interval of training at the new sparsity.
    _milestone_sparsities = []
    _ms_str = getattr(FLAGS, 'gmp_milestone_sparsities', '')
    if _ms_str:
        _milestone_sparsities = sorted([float(x) for x in str(_ms_str).split(',') if x.strip()])
        logging.info(f"  Milestone sparsities: {_milestone_sparsities}")
    _passed_milestones: dict = {}   # sp -> saved_path
    _milestone_reached_at: dict = {}  # sp -> step when first crossed
    accum_loss      = 0.0
    accum_ntp       = 0.0
    accum_kd        = 0.0
    accum_l1        = 0.0
    accum_grad_norm = 0.0
    accum_dpo_loss  = 0.0
    accum_dpo_acc   = 0.0
    accum_dpo_chosen_logp       = 0.0
    accum_dpo_rejected_logp     = 0.0
    accum_dpo_ref_chosen_logp   = 0.0
    accum_dpo_ref_rejected_logp = 0.0
    accum_dpo_margin            = 0.0
    accum_ca_ipo_diag: dict     = {}
    accum_diag: dict = {}
    accum_diag_n = 0
    accum_onpolicy_diag: dict = {}

    logging.info("***** Running GMP Training *****")
    logging.info(f"  Total steps = {total_steps}")
    logging.info(f"  Batch size  = {batch_size}, grad_accum = {grad_accum}")
    logging.info(f"  LR = {lr}, warmup = {constant_warmup_steps if lr_schedule in ('constant', 'constant_with_warmup') else warmup_steps} steps ({lr_schedule})")
    logging.info(f"  Target sparsity = {final_sparsity}, mask_interval = {mask_interval}")
    if use_kd:
        logging.info(f"  KD: lambda={kd_lambda}, temperature={kd_temperature}, topk={kd_topk}")

    while step < total_steps:
        accum_onpolicy = 0.0
        accum_offline_ipo = 0.0
        _measure_generated  = None   # OPKD reuse sequence for grad conflict measurement
        _measure_prompt_len = None
        _measure_pair       = None   # IPO pair for grad conflict measurement
        _g_ntp_snap         = None   # NTP grad snapshot for filter mode
        _g_opkd_filter      = None   # extracted OPKD grad for filter mode
        _opkd_reuse_fired   = False  # whether OPKD reuse ran this step

        # ── Teacher SeqKD: teacher generates, forward KL(T||S) on generated seq ─
        if use_teacher_seqkd:
            _pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            p_batch = next(prompt_iter)
            prompt_ids  = p_batch['input_ids'].to(device)
            prompt_mask = p_batch['attention_mask'].to(device)
            prompt_len  = prompt_ids.shape[1]

            teacher_model.config.use_cache = True
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    generated = teacher_model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        max_new_tokens=teacher_seqkd_max_new,
                        do_sample=True,
                        temperature=teacher_seqkd_temp,
                        pad_token_id=_pad_id,
                    )
            teacher_model.config.use_cache = False

            gen_labels = generated.clone()
            gen_labels[:, :prompt_len] = -100
            gen_labels[generated == _pad_id] = -100

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                s_out = model(input_ids=generated)
                with torch.no_grad():
                    t_out = teacher_model(input_ids=generated)
                loss, _ = _kl_loss(s_out.logits, t_out.logits, gen_labels,
                                   kd_temperature, kd_topk, reverse=False)

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf SeqKD loss at step {step}, skipping")
            else:
                loss.backward()
                accum_loss += loss.item()
                accum_kd   += loss.item()

        # ── NTP + offline KD micro-steps ──────────────────────────────────────
        else:
          for micro_step in range(grad_accum):
            batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                fwd_inputs = {k: v for k, v in batch.items()}
                out = model(**fwd_inputs, output_hidden_states=use_hidden)
                ntp_loss = out.loss

                if use_kd or use_hidden:
                    t_inputs = {k: v for k, v in batch.items() if k != 'labels'}
                    with torch.no_grad():
                        t_out = teacher_model(
                            **t_inputs,
                            output_hidden_states=use_hidden,
                        )

                    if use_hidden:
                        if hidden_layers == "anneal_all_to_final":
                            h_loss = _hidden_loss_layerwise(
                                out.hidden_states[1:], t_out.hidden_states[1:],
                                batch['labels'], batch['attention_mask'],
                                mode=hidden_mode, mask_mode=hidden_mask,
                                step=step, total_steps=total_steps,
                            )
                        else:
                            h_loss = _hidden_loss(
                                out.hidden_states[-1], t_out.hidden_states[-1],
                                batch['labels'], batch['attention_mask'],
                                mode=hidden_mode, mask_mode=hidden_mask,
                            )
                        accum_kd += h_loss.item() / grad_accum
                    if use_kd:
                        kl, kd_diag = _kl_loss(out.logits, t_out.logits, batch['labels'],
                                               kd_temperature, kd_topk)
                        accum_kd += kl.item() / grad_accum
                        for k, v in kd_diag.items():
                            accum_diag[k] = accum_diag.get(k, 0.0) + v
                        accum_diag_n += 1

                    # build total loss
                    aux = (hidden_lambda * h_loss if use_hidden else ntp_loss.new_tensor(0.0)) + \
                          (kd_lambda * kl if use_kd else ntp_loss.new_tensor(0.0))
                    skip_ntp = (hidden_only or kd_only)
                    if skip_ntp:
                        loss = aux / grad_accum
                    else:
                        loss = (ntp_lambda * ntp_loss + aux) / grad_accum
                    if not skip_ntp:
                        accum_ntp += ntp_loss.item() / grad_accum
                else:
                    loss = ntp_lambda * ntp_loss / grad_accum
                    accum_ntp += ntp_loss.item() / grad_accum

                # ── Teacher-gen KD: forward KL on teacher rollouts using pre-stored top-K logits ──
                if use_teacher_gen_kd and tgkd_cache:
                    _tgkd_entry = tgkd_cache[_tgkd_ptr % len(tgkd_cache)]
                    _tgkd_ptr  += 1
                    _tc_ids  = _tgkd_entry["chosen_input_ids"].to(device)   # [1, seq_len]
                    _tc_msk  = _tgkd_entry["chosen_attention_mask"].to(device)
                    _tc_plen = _tgkd_entry["prompt_input_ids"].shape[1]
                    _tc_lbl  = _tc_ids.clone()
                    _tc_lbl[:, :_tc_plen] = -100
                    _tc_pad  = tokenizer.pad_token_id or tokenizer.eos_token_id
                    _tc_lbl[_tc_ids == _tc_pad] = -100
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        _tc_s_logits = model(_tc_ids, attention_mask=_tc_msk).logits  # [1, L, V]
                    _tc_t_tv = _tgkd_entry["teacher_topk_logits"].to(device).float()   # [comp_len, K]
                    _tc_t_ti = _tgkd_entry["teacher_topk_indices"].to(device)           # [comp_len, K]
                    _tc_comp_len = _tc_t_tv.shape[0]
                    # student logits at completion positions (shift-by-1: logit[plen-1] predicts token[plen])
                    _tc_s_comp = _tc_s_logits[0, _tc_plen - 1:_tc_plen - 1 + _tc_comp_len].float()  # [comp_len, V]
                    # forward KL D(T||S) restricted to teacher top-K
                    _tc_mask = (_tc_lbl[0, _tc_plen:_tc_plen + _tc_comp_len] != -100).float()  # [comp_len]
                    if _tc_mask.sum() > 0:
                        _t_logp = F.log_softmax(_tc_t_tv / onpolicy_temp, dim=-1)                   # [comp_len, K]
                        _s_logp = F.log_softmax(_tc_s_comp / onpolicy_temp, dim=-1).gather(
                            1, _tc_t_ti)                                                              # [comp_len, K]
                        _kl_tok = (_t_logp.exp() * (_t_logp - _s_logp)).sum(-1)                     # [comp_len]
                        _tgkd_kl = (_kl_tok * _tc_mask).sum() / _tc_mask.sum()
                    else:
                        _tgkd_kl = loss.new_tensor(0.0)
                    loss = loss + onpolicy_lambda * _tgkd_kl / grad_accum
                    accum_onpolicy += _tgkd_kl.item() / grad_accum

            _structured_l1_pending_fsdp = False
            if use_l1:
                if use_structured_l1:
                    if is_fsdp:
                        # Under FSDP1 each rank only holds a flat, unevenly-sized
                        # local shard (not a full [rows, cols] tensor, and not a
                        # DTensor either -- this file uses classic
                        # FullyShardedDataParallel) so the reshape-into-groups-of-M
                        # logic in _structured_l1_loss needs the params
                        # temporarily un-sharded. Deferred to right before
                        # loss.backward() below (still inside this micro_step) and
                        # given its OWN backward() call inside the summon block,
                        # rather than folded into `loss` here, because summoning
                        # full params while the main loss's forward graph (built
                        # from FSDP's own sharded forward hooks) is still
                        # unresolved risks interleaving two different notions of
                        # "current param shape" that FSDP's autograd hooks aren't
                        # written to expect.
                        _structured_l1_pending_fsdp = True
                    else:
                        l1 = _structured_l1_loss(named_params, maskmgr.masks, prune_n, prune_m)
                else:
                    l1 = _gmp_l1_regularizer(named_params, maskmgr, fisher,
                                             mode=l1_mode,
                                             clip_min=l1_fisher_cmin,
                                             clip_max=l1_fisher_cmax,
                                             open_groups_only=l1_open_only,
                                             prune_n=prune_n, prune_m=prune_m)
                if not _structured_l1_pending_fsdp and l1 is not None:
                    l1_term = l1_lambda * l1 / grad_accum
                    loss = loss + l1_term
                    accum_l1 += l1_term.item()

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss at micro_step {micro_step}, skipping")
                continue
            _bwd_ctx = (model.no_sync()
                        if is_fsdp and micro_step < grad_accum - 1
                        else nullcontext())
            with _bwd_ctx:
                loss.backward()
            accum_loss += loss.item()

            if _structured_l1_pending_fsdp:
                # Main loss's backward is fully resolved now (FSDP's hooks have
                # already reduce-scattered its gradients into each shard) --
                # safe to summon full params for the structured-L1 term's own,
                # separate forward+backward. with_grads=True accumulates the
                # resulting gradient into each rank's local .grad on exit, on
                # top of whatever the main loss.backward() already put there.
                with _bwd_ctx, FSDP.summon_full_params(fsdp_model, writeback=False, with_grads=True):
                    l1 = _structured_l1_loss(named_params, maskmgr.masks, prune_n, prune_m)
                    l1_term = l1_lambda * l1 / grad_accum
                    l1_term.backward()
                accum_l1 += l1_term.item()

        # anchored KD contributes to the NTP optimizer step
        if use_anchor and (step + 1) % anchor_interval == 0:
            a_batch = next(anchor_iter)
            a_ids   = a_batch['input_ids'].to(device)
            a_mask  = a_batch['attention_mask'].to(device)
            seq_len = a_ids.shape[1]

            if seq_len > anchor_prefix_len:
                prefix_ids  = a_ids[:, :anchor_prefix_len]
                prefix_mask = a_mask[:, :anchor_prefix_len]

                model.config.use_cache = True
                model.eval()
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=prefix_ids,
                        attention_mask=prefix_mask,
                        max_new_tokens=anchor_max_new,
                        do_sample=True,
                        temperature=onpolicy_temp,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                model.train()
                model.config.use_cache = False
                if step > dense_warmup_steps:
                    maskmgr.apply(fsdp_model)

                anc_mask = (generated != (tokenizer.pad_token_id or tokenizer.eos_token_id)).long()
                anc_labels = generated.clone()
                anc_labels[:, :anchor_prefix_len] = -100
            else:
                generated  = a_ids
                anc_mask   = a_mask
                anc_labels = a_batch['labels'].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                anc_s_out = model(input_ids=generated, attention_mask=anc_mask)
                with torch.no_grad():
                    anc_t_out = teacher_model(input_ids=generated, attention_mask=anc_mask)
                anc_kl, _ = _kl_loss(anc_s_out.logits, anc_t_out.logits, anc_labels,
                                     kd_temperature, onpolicy_topk)
                anc_loss = anchor_lambda * anc_kl / grad_accum
            anc_loss.backward()
            accum_onpolicy_diag.update({"anchor/kl_loss": anc_kl.item()})

        step += 1

        # periodic mask update (freeze mask after pruning_end_steps)
        if step % mask_interval == 0:
            # Pruning-aware DPO: snapshot ref BEFORE mask update.
            # ref = π_{k-1} (pre-mask stable policy)
            # rejected will be generated AFTER mask update → π̃_k (damaged policy)
            # This ensures ref ≠ rejected_generator, giving non-zero DPO margin.
            # (v3 bug: ref was snapshotted AFTER mask update → ref ≈ rejected → margin ≈ 0)
            if use_dpo_loss and is_main_process and dpo_rejected_queue is not None:
                del dpo_ref_model
                dpo_ref_model = _copy.deepcopy(model).eval()   # π_{k-1}: pre-mask
                for p in dpo_ref_model.parameters():
                    p.requires_grad_(False)

            # OPKD vLLM pool refill BEFORE mask update when TR-GMP is enabled,
            # so rollouts (generated with pre-mask weights) serve as TR calibration.
            # When TR-GMP is off, refill happens after mask update as before.
            _opkd_refilled_pre_mask = False
            if use_onpolicy and tr_enabled and not tr_reached:
                # Sync current student weights to vLLM subprocess (FSDP: collective
                # summon_full_params + sync_weights; non-FSDP: direct internal API
                # via _sync_opkd_weights_to_vllm, same as the initial pool fill above).
                _in_fsdp_refill = fsdp_model is not None and _FSDP_AVAILABLE
                _fsdp_sync_ctx = (FSDP.summon_full_params(fsdp_model, writeback=False, offload_to_cpu=True, rank0_only=True)
                                  if _in_fsdp_refill else nullcontext())
                with _fsdp_sync_ctx:
                    if is_main_process and _opkd_vllm_engine is not None:
                        if fsdp_model is None:
                            _offload_optimizer_state(optimizer)
                        _opkd_vllm_wake(_opkd_vllm_engine)
                        if _in_fsdp_refill and hasattr(_opkd_vllm_engine, 'sync_weights'):
                            _sd = {n: p.data.cpu() for n, p in model.named_parameters()}
                            _opkd_vllm_engine.sync_weights(_sd)
                            del _sd
                        elif not _in_fsdp_refill:
                            _sync_opkd_weights_to_vllm(model, _opkd_vllm_engine)
                if is_main_process and _opkd_vllm_engine is not None:
                    _n_pool = mask_interval * grad_accum
                    _pool_batches = [next(prompt_iter) for _ in range(_n_pool)]
                    _vllm_inputs = [
                        _TokensPrompt(prompt_token_ids=b['input_ids'][0][:int(b['prompt_len'].item())].tolist())
                        for b in _pool_batches
                    ]
                    _vllm_outs = _opkd_vllm_engine.generate(_vllm_inputs, _opkd_vllm_params)
                    _opkd_vllm_sleep(_opkd_vllm_engine)
                    if fsdp_model is None:
                        _reload_optimizer_state(optimizer, device)
                    _opkd_standalone_pool = []
                    for _pb, _vo in zip(_pool_batches, _vllm_outs):
                        _plen = int(_pb['prompt_len'].item())
                        _p_ids = _pb['input_ids'][:, :_plen].cpu()
                        _gen_ids = torch.tensor([_vo.outputs[0].token_ids], dtype=torch.long)
                        _full_seq = torch.cat([_p_ids, _gen_ids], dim=1)
                        _opkd_standalone_pool.append({"full_seq": _full_seq, "prompt_len": _plen})
                    logging.info(f"  OPKD vLLM pool refilled (pre-mask): {len(_opkd_standalone_pool)} rollouts (step={step})")
                _opkd_standalone_pool = _opkd_broadcast_pool(_opkd_standalone_pool, is_distributed, device)
                _opkd_standalone_pool_ptr = 0
                _opkd_refilled_pre_mask = True

            if step <= dense_warmup_steps:
                pass  # dense warmup: no mask update or apply
            elif fixed_mask or (tr_enabled and tr_reached):
                # TR-GMP already hit target sparsity (or a fixed pre-pruned mask
                # was loaded): keep the mask frozen and just continue training
                # (sparse training) for the remaining steps instead of stopping.
                maskmgr.apply(fsdp_model)
            elif tr_enabled and not tr_reached:
                # Use OPKD rollouts as calibration if available, else fall back to prompt_iter
                if _opkd_refilled_pre_mask and _opkd_standalone_pool:
                    _n_cal = min(8, len(_opkd_standalone_pool))
                    _cal_batch = _opkd_pool_to_batch(_opkd_standalone_pool[:_n_cal], str(device))
                else:
                    _cal_batch = next(prompt_iter)
                if getattr(FLAGS, 'gmp_fisher_source', 'adam') == 'opd_empirical':
                    fisher.update_from_batch(fsdp_model if fsdp_model is not None else model, _cal_batch, str(device))
                if getattr(fisher, 'saliency', None) == 'wanda':
                    # Use the FULL OPKD rollout pool (mask_interval*grad_accum
                    # sequences, e.g. 256) for the activation-norm snapshot when
                    # available -- more samples than Wanda's own paper (128) --
                    # instead of the small 8-sequence _cal_batch used for the KL
                    # check, since scaler_row benefits from more tokens while
                    # the KL check itself doesn't need to be this expensive.
                    if _opkd_refilled_pre_mask and _opkd_standalone_pool:
                        _wanda_batch = _opkd_pool_to_batch(_opkd_standalone_pool, str(device))
                    else:
                        _wanda_batch = _cal_batch
                    fisher.capture_wanda_stats(fsdp_model if fsdp_model is not None else model, _wanda_batch, str(device))
                current_sparsity, tr_delta, tr_reached, _tr_mask_delta = _tr_mask_update(
                    maskmgr, fisher, fsdp_model, model, _cal_batch,
                    final_sparsity=final_sparsity,
                    tr_delta=tr_delta,
                    kl_threshold=tr_kl_threshold,
                    delta_min=tr_delta_min,
                    device=str(device),
                    kl_reduce=tr_kl_reduce,
                    kl_quantile=tr_kl_quantile,
                    use_wandb=use_wandb,
                    global_step=step,
                )
                if is_distributed:
                    import torch.distributed as _td2
                    _td2.barrier()
                    logging.info(f"  [BARRIER] after _tr_mask_update step={step} (rank={_td2.get_rank()})")
                if getattr(FLAGS, 'gmp_pcg_correct', False) and teacher_model is not None and fsdp_model is None:
                    _pcg_fn = (_pcg_correct_masked_weights_sequential
                               if getattr(FLAGS, 'gmp_pcg_sequential', False)
                               else _pcg_correct_masked_weights)
                    _pcg_fn(
                        model, teacher_model, maskmgr, _cal_batch, str(device),
                        maxiter=getattr(FLAGS, 'gmp_pcg_maxiter', 5),
                        damp_coef=getattr(FLAGS, 'gmp_pcg_damp', 0.01),
                        global_step=step, use_wandb=use_wandb,
                    )
                if (opkd_prev_mask_teacher or prevmask_opkd_lambda > 0) and use_onpolicy:
                    _opkd_prev_delta = _tr_mask_delta
                if use_wandb:
                    wandb.log({"train/sparsity": current_sparsity,
                               "train/tr_delta": tr_delta, "step": step})
                if tr_reached:
                    logging.info(f"TR-GMP: target sparsity {final_sparsity} reached at step {step}, "
                                 f"switching to sparse training (mask frozen) for remaining steps.")
                    if _tr_reached_step is None:
                        _tr_reached_step = step
            else:
                current_sparsity = 0.0 if step <= dense_warmup_steps else _cubic_sparsity(
                    min(step, pruning_end_steps), pruning_end_steps, final_sparsity, dense_warmup_steps)
                if step <= pruning_end_steps:
                    maskmgr.update(fisher, current_sparsity, fsdp_model)
                else:
                    maskmgr.apply(fsdp_model)

            # Refill rejected AFTER mask update → rejected ~ π̃_k (post-mask damaged policy)
            if use_dpo_queue and is_main_process and dpo_rejected_queue is not None and step > dense_warmup_steps:
                model.eval()
                dpo_rejected_queue.refill(model, tokenizer, str(device))
                model.train()
                maskmgr.apply(fsdp_model)

                # Δ_T diagnostic: teacher logprob gap on fresh pairs
                _pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                _diag_pairs = dpo_rejected_queue.peek_n(min(64, len(dpo_rejected_queue)))
                if _diag_pairs and dpo_dense_model is not None:
                    from lib.gmp_dpo import compute_teacher_delta as _compute_delta
                    _deltas = _compute_delta(dpo_dense_model, _diag_pairs, _pad_val, str(device))
                    _delta_mean = _deltas.mean().item()
                    _delta_std  = _deltas.std().item()
                    _delta_pos  = (_deltas > 0).float().mean().item()
                    logging.info(f"  Δ_T: mean={_delta_mean:.4f} std={_delta_std:.4f} P(>0)={_delta_pos:.3f} (step={step})")
                    if use_wandb:
                        import wandb as _wandb
                        _wandb.log({"train/delta_T_mean": _delta_mean,
                                    "train/delta_T_std":  _delta_std,
                                    "train/delta_T_pos_rate": _delta_pos,
                                    "step": step})

            # OPKD vLLM pool refill AFTER mask update (only when TR-GMP is off)
            if use_onpolicy and not _opkd_refilled_pre_mask:
                _in_fsdp_refill2 = fsdp_model is not None and _FSDP_AVAILABLE
                _fsdp_sync_ctx2 = (FSDP.summon_full_params(fsdp_model, writeback=False, offload_to_cpu=True, rank0_only=True)
                                   if _in_fsdp_refill2 else nullcontext())
                with _fsdp_sync_ctx2:
                    if is_main_process and _opkd_vllm_engine is not None:
                        if fsdp_model is None:
                            _offload_optimizer_state(optimizer)
                        _opkd_vllm_wake(_opkd_vllm_engine)
                        if _in_fsdp_refill2 and hasattr(_opkd_vllm_engine, 'sync_weights'):
                            _sd = {n: p.data.cpu() for n, p in model.named_parameters()}
                            _opkd_vllm_engine.sync_weights(_sd)
                            del _sd
                        elif not _in_fsdp_refill2:
                            _sync_opkd_weights_to_vllm(model, _opkd_vllm_engine)
                if is_main_process and _opkd_vllm_engine is not None:
                    _n_pool = mask_interval * grad_accum
                    _pool_batches = [next(prompt_iter) for _ in range(_n_pool)]
                    _vllm_inputs = [
                        _TokensPrompt(prompt_token_ids=b['input_ids'][0][:int(b['prompt_len'].item())].tolist())
                        for b in _pool_batches
                    ]
                    _vllm_outs = _opkd_vllm_engine.generate(_vllm_inputs, _opkd_vllm_params)
                    _opkd_vllm_sleep(_opkd_vllm_engine)
                    if fsdp_model is None:
                        _reload_optimizer_state(optimizer, device)
                    _opkd_standalone_pool = []
                    for _pb, _vo in zip(_pool_batches, _vllm_outs):
                        _plen = int(_pb['prompt_len'].item())
                        _p_ids = _pb['input_ids'][:, :_plen].cpu()
                        _gen_ids = torch.tensor([_vo.outputs[0].token_ids], dtype=torch.long)
                        _full_seq = torch.cat([_p_ids, _gen_ids], dim=1)
                        _opkd_standalone_pool.append({"full_seq": _full_seq, "prompt_len": _plen})
                    logging.info(f"  OPKD vLLM pool refilled: {len(_opkd_standalone_pool)} rollouts (step={step})")
                _opkd_standalone_pool = _opkd_broadcast_pool(_opkd_standalone_pool, is_distributed, device)
                _opkd_standalone_pool_ptr = 0

            # Milestone checkpoint: save model after `mask_interval` recovery steps past milestone.
            # Two-phase: (1) record step when sparsity first crosses milestone,
            #            (2) save checkpoint one mask_interval later so weights have adapted.
            # Eval is run post-hoc after training to avoid vLLM memory conflicts mid-training.
            if _milestone_sparsities and is_main_process:
                _real_sp_now = maskmgr.current_sparsity()
                for _ms in _milestone_sparsities:
                    if _ms not in _passed_milestones:
                        # Phase 1: first time sparsity crosses the milestone threshold
                        if _ms not in _milestone_reached_at and _real_sp_now >= _ms - 5e-3:
                            _milestone_reached_at[_ms] = step
                            logging.info(f"[Milestone] sparsity={_real_sp_now:.4f} >= {_ms:.2f} at step={step}"
                                         f" — saving in {mask_interval} recovery steps (step {step + mask_interval})")
                        # Phase 2: mask_interval steps after crossing → save checkpoint
                        elif _ms in _milestone_reached_at and step >= _milestone_reached_at[_ms] + mask_interval:
                            _passed_milestones[_ms] = None
                            _ms_tag = f"sp{int(_ms * 100):02d}"
                            logging.info(f"[Milestone] saving checkpoint at step={step}"
                                         f" ({mask_interval} steps after {_ms:.2f} milestone)")
                            if do_save:
                                _ms_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                _ms_path = f"{FLAGS.gmp_save_path}/{_run_tag(FLAGS)}_{_ms_tag}_{_ms_ts}"
                                model.save_pretrained(_ms_path)
                                tokenizer.save_pretrained(_ms_path)
                                _passed_milestones[_ms] = _ms_path
                                logging.info(f"[Milestone] saved to {_ms_path}")

        # Early stop N steps after TR-GMP first reaches target sparsity, instead of
        # continuing for the full remaining budget with the mask frozen (gmp_post_target_steps=0
        # keeps the old behavior of training all the way to `steps`).
        if _post_target_steps > 0 and _tr_reached_step is not None and step >= _tr_reached_step + _post_target_steps:
            logging.info(f"TR-GMP: stopping {_post_target_steps} steps after reaching target sparsity "
                         f"(reached at step {_tr_reached_step}, stopping at step {step}).")
            break

        # Snapshot NTP grads before OPKD (for gradient conflict filter / projection)
        if ((filter_grad_conflict or project_opkd_onto_combined or filter_opkd_combined)
                and use_onpolicy and step % onpolicy_interval == 0
                and opkd_reuse_ipo and is_main_process):
            _g_ntp_snap = [p.grad.detach().clone() if p.grad is not None else None
                           for p in model.parameters()]
            # zero_grad so OPKD backward gives pure g_OPKD
            optimizer.zero_grad()

        # ── On-policy: rollout collection + RL grad accumulation (combined step fires below) ──
        # When opkd_use_vllm is active and pool has data, fire every step (pool makes it cheap).
        _opkd_fires = use_onpolicy and (
            step % onpolicy_interval == 0
            or bool(_opkd_standalone_pool)
        )
        if is_distributed and _opkd_fires:
            import torch.distributed as _td3
            _td3.barrier()
            logging.info(f"  [BARRIER] before OPKD training step={step} (rank={_td3.get_rank()})")
        if _opkd_fires:
            _pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            _eos_id = tokenizer.eos_token_id or _pad_id
            use_mixed = onpolicy_pg and (onpolicy_mixed_alpha > 0.0)

            if use_rollout:
                # ── ROLLOUT BUFFER PATH ──────────────────────────────────────
                _total_gen_tok = 0
                _total_r = 0.0
                _t_gen = time.time()
                _n_collect = onpolicy_grad_accum  # prompts per collection step (default 1)

                _p_batches = [next(prompt_iter) for _ in range(_n_collect)]
                _p_ids_list  = [b['input_ids'].to(device)  for b in _p_batches]
                _p_mask_list = [b['attention_mask'].to(device) for b in _p_batches]
                _max_plen = max(p.shape[1] for p in _p_ids_list)
                _batch_ids = torch.cat([
                    torch.cat([torch.full((1, _max_plen - p.shape[1]), _pad_id,
                                         dtype=torch.long, device=device), p], dim=1)
                    for p in _p_ids_list
                ], dim=0)  # (_n_collect, _max_plen)
                _batch_mask = torch.cat([
                    torch.cat([torch.zeros(1, _max_plen - m.shape[1],
                                          dtype=torch.long, device=device), m], dim=1)
                    for m in _p_mask_list
                ], dim=0)  # (_n_collect, _max_plen)

                model.config.use_cache = True
                model.eval()
                if use_mixed: # mix logits of student and teacher for sampling
                    generated = _mixed_sample(
                        model, teacher_model, _batch_ids, _batch_mask,
                        onpolicy_max_new, onpolicy_mixed_alpha, onpolicy_temp,
                        _pad_id, _eos_id,
                    )
                else:
                    with torch.no_grad():
                        generated = model.generate(
                            input_ids=_batch_ids,
                            attention_mask=_batch_mask,
                            max_new_tokens=onpolicy_max_new,
                            do_sample=True,
                            temperature=onpolicy_temp,
                            pad_token_id=_pad_id,
                        )
                _total_gen_time = time.time() - _t_gen
                model.train()
                model.config.use_cache = False
                if step > dense_warmup_steps:
                    maskmgr.apply(fsdp_model)

                gen_labels = generated.clone()  # (_n_collect, _max_plen + gen_len)
                gen_labels[:, :_max_plen] = -100
                gen_labels[generated == _pad_id] = -100

                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        _s_fwd = model(input_ids=generated)
                        _t_fwd = teacher_model(input_ids=generated)
                    _gen_pos_mask = (gen_labels[:, 1:] != -100).float()
                    _gids  = gen_labels[:, 1:].clamp(min=0)
                    _s_lp  = F.log_softmax(_s_fwd.logits[:, :-1].float(), dim=-1)
                    _t_lp  = F.log_softmax(_t_fwd.logits[:, :-1].float(), dim=-1)
                    _s_tok = _s_lp.gather(-1, _gids.unsqueeze(-1)).squeeze(-1)
                    _t_tok = _t_lp.gather(-1, _gids.unsqueeze(-1)).squeeze(-1)
                    _buf_rewards = (_t_tok - _s_tok) * _gen_pos_mask
                    if use_mixed:
                        _mix_prob = ((1 - onpolicy_mixed_alpha) * _s_tok.exp()
                                     + onpolicy_mixed_alpha * _t_tok.exp()).clamp(min=1e-10)
                        _buf_is_log_w = (_s_tok - _mix_prob.log()) * _gen_pos_mask
                    else:
                        _buf_is_log_w = torch.zeros_like(_gen_pos_mask)
                    for _i in range(_n_collect):
                        rollout_buffer.add(
                            generated[_i:_i+1], gen_labels[_i:_i+1],
                            _buf_rewards[_i:_i+1], _s_tok[_i:_i+1], _buf_is_log_w[_i:_i+1],
                        )
                    _total_gen_tok = int(_gen_pos_mask.sum().item())
                    _total_r = (_buf_rewards.sum(dim=1) / _gen_pos_mask.sum(dim=1).clamp(min=1)).mean().item()

                logging.info(f"  [buf {len(rollout_buffer)}/{rollout_buffer_size}] "
                             f"step={step} gen={_total_gen_tok}tok "
                             f"r={_total_r:.3f} t={_total_gen_time:.1f}s")

                # ── RL update: accumulate into NTP grads (combined step fires below) ──
                if len(rollout_buffer) >= rollout_buffer_size:
                    _n_buf = len(rollout_buffer)
                    _last_kl = 0.0
                    for _ppo_epoch in range(ppo_epochs):
                        for _bi in range(_n_buf):
                            _gen2        = rollout_buffer.generated[_bi].to(device)
                            _glabels     = rollout_buffer.gen_labels[_bi].to(device)
                            _stored_rew  = rollout_buffer.rewards[_bi].to(device)
                            _s_old_lp    = rollout_buffer.old_s_logp[_bi].to(device)
                            _is_log_w_b  = rollout_buffer.is_log_w[_bi].to(device)

                            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                                _s_out2 = model(input_ids=_gen2)
                                with torch.no_grad():
                                    _t_out2 = teacher_model(input_ids=_gen2)
                                _op_kl2, _ = _kl_loss(
                                    _s_out2.logits, _t_out2.logits, _glabels,
                                    kd_temperature, onpolicy_topk,
                                    reverse=onpolicy_reverse_kl,
                                )
                                _pg2 = _pg_loss(
                                    _s_out2.logits, _t_out2.logits, _glabels,
                                    is_log_w=_is_log_w_b,
                                    old_s_logp=_s_old_lp,
                                    stored_rewards=_stored_rew,
                                    cliprange=onpolicy_pg_cliprange,
                                    gamma=onpolicy_pg_gamma,
                                    reward_clip=pg_reward_clip,
                                    reward_scale=pg_reward_scale,
                                )
                                _buf_loss = (onpolicy_lambda * _op_kl2 + onpolicy_lambda * _pg2) / (grad_accum * ppo_epochs * _n_buf)
                            if not (torch.isnan(_buf_loss) or torch.isinf(_buf_loss)):
                                _buf_loss.backward()
                            _last_kl = _op_kl2.item()

                    accum_onpolicy = _last_kl
                    accum_onpolicy_diag.update({
                        "onpolicy/kl_loss":      _last_kl,
                        "onpolicy/buffer_items": _n_buf,
                        "onpolicy/ppo_epochs":   ppo_epochs,
                    })
                    rollout_buffer.clear()

            else:
                # ── INLINE PATH (original, no buffer) ────────────────────────────
                _total_gen_time = 0.0
                _total_gen_tokens = 0
                _diag_kl = 0.0
                _diag_kl_prev = 0.0
                _diag_s_ent = 0.0
                _diag_t_ent = 0.0
                _diag_overlap = 0.0

                for _op_i in range(onpolicy_grad_accum * batch_size * grad_accum):
                    # reuse IPO rejected rollouts or standalone vLLM pool
                    _reuse_dpo = (opkd_reuse_ipo
                                  and dpo_rejected_queue is not None
                                  and dpo_rejected_queue.rollout_pool)
                    _reuse_standalone = bool(_opkd_standalone_pool)
                    if _reuse_standalone:
                        _pooled = _opkd_standalone_pool[_opkd_standalone_pool_ptr % len(_opkd_standalone_pool)]
                        _opkd_standalone_pool_ptr += 1
                    elif _reuse_dpo:
                        _pooled = dpo_rejected_queue.sample_from_pool()
                    else:
                        raise RuntimeError("OPKD: vLLM pool is empty and no DPO queue — pool should have been filled before this point.")
                    generated = _pooled["full_seq"].to(device)
                    prompt_len = _pooled["prompt_len"]
                    _total_gen_tokens += generated.shape[1] - prompt_len
                    if measure_grad_conflict and _op_i == 0:
                        _measure_generated  = generated
                        _measure_prompt_len = prompt_len
                    if filter_grad_conflict or project_opkd_onto_combined or filter_opkd_combined:
                        _opkd_reuse_fired = True

                    gen_labels = generated.clone()
                    gen_labels[:, :prompt_len] = -100
                    gen_labels[generated == _pad_id] = -100

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        s_out = model(input_ids=generated)

                        # Dense teacher forward (primary OPKD teacher, always used)
                        with torch.no_grad():
                            t_out = teacher_model(input_ids=generated)

                        # Prev-mask teacher: dual mode (both losses combined)
                        t_prev_out = None
                        if prevmask_opkd_lambda > 0 and _opkd_prev_delta:
                            for _pn, (_pos, _vals) in _opkd_prev_delta.items():
                                maskmgr.named_params[_pn].data[_pos] = _vals
                            with torch.no_grad():
                                t_prev_out = model(input_ids=generated)
                            for _pn, (_pos, _) in _opkd_prev_delta.items():
                                maskmgr.named_params[_pn].data[_pos] = 0.0
                        elif opkd_prev_mask_teacher and _opkd_prev_delta:
                            # Legacy: replace dense teacher entirely with prev-mask teacher
                            for _pn, (_pos, _vals) in _opkd_prev_delta.items():
                                maskmgr.named_params[_pn].data[_pos] = _vals
                            with torch.no_grad():
                                t_out = model(input_ids=generated)
                            for _pn, (_pos, _) in _opkd_prev_delta.items():
                                maskmgr.named_params[_pn].data[_pos] = 0.0

                        op_kl, op_diag = _kl_loss(s_out.logits, t_out.logits, gen_labels,
                                                  kd_temperature, onpolicy_topk,
                                                  reverse=onpolicy_reverse_kl)
                        if t_prev_out is not None:
                            op_kl_prev, _ = _kl_loss(s_out.logits, t_prev_out.logits, gen_labels,
                                                      kd_temperature, onpolicy_topk,
                                                      reverse=onpolicy_reverse_kl)
                        else:
                            op_kl_prev = None

                        with torch.no_grad():
                            _gen_pos_mask = (gen_labels[:, 1:] != -100).float()
                            _s_logp = F.log_softmax(s_out.logits[:, :-1] / kd_temperature, dim=-1)
                            _t_logp = F.log_softmax(t_out.logits[:, :-1] / kd_temperature, dim=-1)
                            _s_ent = -(_s_logp.exp() * _s_logp).sum(dim=-1)
                            _t_ent = -(_t_logp.exp() * _t_logp).sum(dim=-1)
                            _denom = _gen_pos_mask.sum().clamp(min=1)
                            _s_ent_mean = (_s_ent * _gen_pos_mask).sum() / _denom
                            _t_ent_mean = (_t_ent * _gen_pos_mask).sum() / _denom
                            _K = 100
                            _s_top = s_out.logits[:, :-1].topk(_K, dim=-1).indices
                            _t_top = t_out.logits[:, :-1].topk(_K, dim=-1).indices
                            _overlap = (_s_top.unsqueeze(-1) == _t_top.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
                            _overlap_mean = (_overlap * _gen_pos_mask).sum() / _denom

                            is_log_w = None
                            if use_mixed:
                                _s_lp = F.log_softmax(s_out.logits[:, :-1].detach().float(), dim=-1)
                                _t_lp = F.log_softmax(t_out.logits[:, :-1].float(), dim=-1)
                                _gids  = gen_labels[:, 1:].clamp(min=0)
                                _s_tok = _s_lp.gather(-1, _gids.unsqueeze(-1)).squeeze(-1)
                                _t_tok = _t_lp.gather(-1, _gids.unsqueeze(-1)).squeeze(-1)
                                _mix_prob = ((1 - onpolicy_mixed_alpha) * _s_tok.exp()
                                            + onpolicy_mixed_alpha * _t_tok.exp()).clamp(min=1e-10)
                                is_log_w = (_s_tok - _mix_prob.log()) * _gen_pos_mask

                        if onpolicy_pg:
                            pg = _pg_loss(s_out.logits, t_out.logits, gen_labels,
                                          is_log_w=is_log_w,
                                          old_s_logp=_s_tok if use_mixed else None,
                                          cliprange=onpolicy_pg_cliprange,
                                          gamma=onpolicy_pg_gamma,
                                          reward_clip=pg_reward_clip,
                                          reward_scale=pg_reward_scale)
                            op_loss = onpolicy_lambda * op_kl + onpolicy_lambda * pg
                        else:
                            op_loss = onpolicy_lambda * op_kl
                        if op_kl_prev is not None:
                            op_loss = op_loss + prevmask_opkd_lambda * op_kl_prev
                        op_loss = op_loss / (batch_size * grad_accum * onpolicy_grad_accum)

                    if torch.isnan(op_loss) or torch.isinf(op_loss):
                        logging.warning(f"NaN/Inf on-policy loss at step {step} micro {_op_i}, skipping")
                    else:
                        op_loss.backward()
                        accum_onpolicy += op_kl.item()
                        if op_kl_prev is not None:
                            _diag_kl_prev += op_kl_prev.item()

                    _diag_kl      += op_kl.item()
                    _diag_s_ent   += _s_ent_mean.item()
                    _diag_t_ent   += _t_ent_mean.item()
                    _diag_overlap += _overlap_mean.item()

                accum_onpolicy /= (onpolicy_grad_accum * batch_size * grad_accum)
                _n = onpolicy_grad_accum * batch_size * grad_accum
                accum_onpolicy_diag.update({
                    "onpolicy/kl_loss":              _diag_kl / _n,
                    "onpolicy/gen_tokens":           _total_gen_tokens / _n,
                    "onpolicy/gen_time_sec":         _total_gen_time,
                    "onpolicy/student_entropy":      _diag_s_ent / _n,
                    "onpolicy/teacher_entropy":      _diag_t_ent / _n,
                    "onpolicy/entropy_gap":          (_diag_s_ent - _diag_t_ent) / _n,
                    "onpolicy/overlap_ratio_top100": _diag_overlap / _n,
                })
                if _diag_kl_prev > 0:
                    accum_onpolicy_diag["onpolicy/prevmask_kl_loss"] = _diag_kl_prev / _n
                accum_onpolicy_diag.update({f"onpolicy/{k.split('/')[-1]}": v
                                            for k, v in op_diag.items()})

        # Extract pure g_OPKD, then zero_grad so IPO backward gives clean g_IPO
        if _g_ntp_snap is not None and _opkd_reuse_fired:
            _g_opkd_filter = [p.grad.detach().clone() if p.grad is not None else None
                              for p in model.parameters()]
            optimizer.zero_grad()

        # ── DPO loss (grad_accum pairs per optimizer step, matching NTP) ──────
        if use_dpo_loss and step >= dpo_start_step and is_main_process and dpo_rejected_queue:
            for _dpo_i in range(batch_size * grad_accum):
                pair = dpo_rejected_queue.pop()
                if pair is not None:
                    if measure_grad_conflict and _dpo_i == 0:
                        _measure_pair = pair
                    _pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    pair_dev = {k: v.to(device) for k, v in pair.items()}
                    try:
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            policy_out = concatenated_forward(model, pair_dev, padding_value=_pad_val)
                            if not dpo_reference_free:
                                with torch.no_grad():
                                    ref_out = concatenated_forward(dpo_ref_model, pair_dev, padding_value=_pad_val, is_ref_model=True)
                            else:
                                # reference-free: zero logratios from ref
                                ref_out = {
                                    "chosen_logps":   torch.zeros_like(policy_out["chosen_logps"]),
                                    "rejected_logps": torch.zeros_like(policy_out["rejected_logps"]),
                                }
                        if use_ca_ipo:
                            # CA-IPO: get token-level logps for policy and ref
                            _p_ids   = pair_dev["prompt_input_ids"]
                            _p_msk   = pair_dev["prompt_attention_mask"]
                            _c_ids   = pair_dev["chosen_input_ids"]
                            _c_msk   = pair_dev["chosen_attention_mask"]
                            _r_ids   = pair_dev["rejected_input_ids"]
                            _r_msk   = pair_dev["rejected_attention_mask"]
                            policy_chosen_tok  = _get_token_logps(model,   _p_ids, _p_msk, _c_ids, _c_msk)
                            policy_rej_tok     = _get_token_logps(model,   _p_ids, _p_msk, _r_ids, _r_msk)
                            with torch.no_grad():
                                ref_chosen_tok     = _get_token_logps(dpo_ref_model, _p_ids, _p_msk, _c_ids, _c_msk)
                                ref_rej_tok        = _get_token_logps(dpo_ref_model, _p_ids, _p_msk, _r_ids, _r_msk)
                            teacher_chosen_tok  = pair_dev["teacher_chosen_token_logps"].to(device)
                            teacher_rej_tok     = pair_dev["teacher_rejected_token_logps"].to(device)
                            _eps_credit = getattr(FLAGS, 'gmp_ca_ipo_eps_credit', 1e-6)
                            dpo_l, _ca_metrics = _ca_ipo_loss(
                                policy_chosen_tok, policy_rej_tok,
                                ref_chosen_tok,    ref_rej_tok,
                                teacher_chosen_tok, teacher_rej_tok,
                                _c_msk.float(), _r_msk.float(),
                                ref_chosen_logps_avg=ref_out["chosen_logps"],
                                ref_rejected_logps_avg=ref_out["rejected_logps"],
                                policy_chosen_logps_avg=policy_out["chosen_logps"],
                                policy_rejected_logps_avg=policy_out["rejected_logps"],
                                beta=dpo_beta,
                                eps_credit=_eps_credit,
                            )
                        else:
                            losses, chosen_rew, rejected_rew = _dpo_loss(
                                policy_out["chosen_logps"], policy_out["rejected_logps"],
                                ref_out["chosen_logps"],   ref_out["rejected_logps"],
                                beta=dpo_beta,
                                loss_type=dpo_loss_type,
                                reference_free=dpo_reference_free,
                            )
                            dpo_l = losses.mean()

                        if not (torch.isnan(dpo_l) or torch.isinf(dpo_l)):
                            _gbs = batch_size * grad_accum
                            (dpo_lambda * dpo_l / _gbs).backward()
                            accum_dpo_loss += dpo_l.item() / _gbs
                            if use_ca_ipo:
                                for k, v in _ca_metrics.items():
                                    accum_ca_ipo_diag[k] = accum_ca_ipo_diag.get(k, 0.0) + v.item() / _gbs
                            _dpo_margin = dpo_beta * (
                                (policy_out["chosen_logps"] - ref_out["chosen_logps"]) -
                                (policy_out["rejected_logps"] - ref_out["rejected_logps"])
                            )
                            accum_dpo_acc += (_dpo_margin > 0).float().mean().item() / _gbs
                            # diagnostic accumulators (v2: sanity check logprob scale)
                            accum_dpo_chosen_logp   += policy_out["chosen_logps"].mean().item() / _gbs
                            accum_dpo_rejected_logp += policy_out["rejected_logps"].mean().item() / _gbs
                            accum_dpo_ref_chosen_logp   += ref_out["chosen_logps"].mean().item() / _gbs
                            accum_dpo_ref_rejected_logp += ref_out["rejected_logps"].mean().item() / _gbs
                            accum_dpo_margin        += _dpo_margin.mean().item() / _gbs
                    except Exception as e:
                        logging.warning(f"DPO loss error at step {step} dpo_i={_dpo_i}: {e}")

        # ── Offline IPO loss (UltraFeedback preference pairs) ─────────────────
        if use_offline_ipo and offline_ipo_iter is not None:
            _pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            _gbs = batch_size * grad_accum
            for _ in range(_gbs):
                pair = next(offline_ipo_iter)
                pair_dev = {k: v.to(device) for k, v in pair.items()}
                try:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        policy_out = concatenated_forward(model, pair_dev, padding_value=_pad_val)
                        with torch.no_grad():
                            ref_out = concatenated_forward(
                                _offline_ipo_ref, pair_dev,
                                padding_value=_pad_val, is_ref_model=True,
                            )
                    losses, _, _ = _offline_ipo_loss_fn(
                        policy_out["chosen_logps"], policy_out["rejected_logps"],
                        ref_out["chosen_logps"],    ref_out["rejected_logps"],
                        beta=offline_ipo_beta,
                        loss_type="ipo",
                    )
                    ipo_l = losses.mean()
                    if not (torch.isnan(ipo_l) or torch.isinf(ipo_l)):
                        (offline_ipo_lambda * ipo_l / _gbs).backward()
                        accum_offline_ipo += ipo_l.item() / _gbs
                except Exception as e:
                    logging.warning(f"Offline IPO loss error at step {step}: {e}")

        # Gradient conflict filter / half-space projection
        # p.grad is now pure g_IPO (zero_grad was called after OPKD backward)
        if _g_opkd_filter is not None:
            if filter_opkd_combined or project_opkd_onto_combined:
                # Half-space projection of g_OPKD onto {x: x·(g_NTP+g_DPO) >= 0}
                # g_combined = g_NTP + g_IPO; if dot(g_OPKD, g_combined) < 0,
                # remove the g_combined component from g_OPKD.
                _dot_oc = _norm_c_sq = _norm_op = 0.0
                _g_combined = []
                for p, g_ntp in zip(model.parameters(), _g_ntp_snap):
                    g_c = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p.data)  # g_IPO (0 if no IPO)
                    if g_ntp is not None:
                        g_c = g_c + g_ntp.to(g_c.dtype)
                    _g_combined.append(g_c)
                    _norm_c_sq += g_c.float().pow(2).sum().item()
                for g_op, g_c in zip(_g_opkd_filter, _g_combined):
                    if g_op is None:
                        continue
                    _dot_oc  += (g_op.float() * g_c.float()).sum().item()
                    _norm_op += g_op.float().pow(2).sum().item()
                _cos_sim_filter = _dot_oc / (math.sqrt(_norm_op * _norm_c_sq) + 1e-10)
                if use_wandb and wandb.run is not None:
                    wandb.log({"grad_conflict/cos_sim":     _cos_sim_filter,
                               "grad_conflict/norm_opkd":  math.sqrt(_norm_op),
                               "grad_conflict/norm_combined": math.sqrt(_norm_c_sq)}, step=step)
                if _dot_oc >= 0:
                    # no conflict: add g_OPKD as-is
                    for p, g_op in zip(model.parameters(), _g_opkd_filter):
                        if g_op is not None:
                            p.grad = (p.grad + g_op) if p.grad is not None else g_op
                elif filter_opkd_combined:
                    # conflict: drop g_OPKD entirely
                    pass
                else:
                    # conflict: project out g_combined component from g_OPKD
                    _proj_scalar = _dot_oc / (_norm_c_sq + 1e-10)
                    for p, g_op, g_c in zip(model.parameters(), _g_opkd_filter, _g_combined):
                        if g_op is None:
                            continue
                        g_op_proj = g_op - _proj_scalar * g_c.to(g_op.dtype)
                        p.grad = (p.grad + g_op_proj) if p.grad is not None else g_op_proj
                # add g_NTP back
                for p, g_ntp in zip(model.parameters(), _g_ntp_snap):
                    if g_ntp is not None:
                        p.grad = (p.grad + g_ntp) if p.grad is not None else g_ntp
                del _g_combined
            else:
                # filter_grad_conflict: half-space projection wrt g_IPO only
                _dot = _norm1 = _norm2 = 0.0
                for p, g_op in zip(model.parameters(), _g_opkd_filter):
                    if g_op is None or p.grad is None:
                        continue
                    g_op_f  = g_op.float().flatten()
                    g_ref_f = p.grad.detach().float().flatten()
                    _dot   += (g_op_f * g_ref_f).sum().item()
                    _norm1 += g_op_f.pow(2).sum().item()
                    _norm2 += g_ref_f.pow(2).sum().item()
                _cos_sim_filter = _dot / (math.sqrt(_norm1 * _norm2) + 1e-10)
                if use_wandb and wandb.run is not None:
                    wandb.log({"grad_conflict/cos_sim":   _cos_sim_filter,
                               "grad_conflict/norm_opkd": math.sqrt(_norm1),
                               "grad_conflict/norm_ipo":  math.sqrt(_norm2)}, step=step)
                if _cos_sim_filter >= 0:
                    for p, g_op in zip(model.parameters(), _g_opkd_filter):
                        if g_op is not None:
                            p.grad = (p.grad + g_op) if p.grad is not None else g_op
                else:
                    # g̃_OPKD = g_OPKD - (g_OPKD·g_IPO / |g_IPO|²) * g_IPO
                    _proj_scalar = _dot / (_norm2 + 1e-10)
                    for p, g_op in zip(model.parameters(), _g_opkd_filter):
                        if g_op is None or p.grad is None:
                            continue
                        g_op_proj = g_op - _proj_scalar * p.grad.detach().to(g_op.dtype)
                        p.grad = p.grad + g_op_proj
                # add g_NTP back
                for p, g_ntp in zip(model.parameters(), _g_ntp_snap):
                    if g_ntp is not None:
                        p.grad = (p.grad + g_ntp) if p.grad is not None else g_ntp
            del _g_opkd_filter, _g_ntp_snap
            _g_opkd_filter = _g_ntp_snap = None

        # ── Combined optimizer step (NTP + RL grads) ─────────────────────────
        # Use FSDP-aware clip_grad_norm_ so all ranks get the same global norm.
        # torch.nn.utils.clip_grad_norm_ on sharded params returns a LOCAL norm
        # which can differ across ranks → the PGD skip condition (NaN/Inf check)
        # would then diverge and cause a NCCL collective mismatch deadlock.
        _in_fsdp_pgd = _FSDP_AVAILABLE and fsdp_model is not None
        if _in_fsdp_pgd:
            grad_norm = fsdp_model.clip_grad_norm_(1.0).item()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            logging.warning(f"NaN/Inf grad_norm at step {step}, skipping optimizer step")
            optimizer.zero_grad()
        else:
            fisher.update()
            accum_grad_norm += grad_norm
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad()

        # ── PGD projection (fisher-saliency, FSDP-aware) ─────────────────────
        if pgd_enabled and step > dense_warmup_steps and not math.isnan(grad_norm) and not math.isinf(grad_norm):
            _pgd_revivals = 0
            _pgd_prunings = 0
            _pgd_use_fsdp = _FSDP_AVAILABLE and fsdp_model is not None
            if _pgd_use_fsdp:
                import torch.distributed as _dist

            # importance scores (v_t * w^2), skip empty FSDP shards
            _pgd_imps = {}
            for _n, _p in maskmgr.named_params.items():
                _t = fisher.importance(_n, _p)
                if _t.numel() > 0:
                    _pgd_imps[_n] = _t

            if _pgd_imps:
                _pgd_dev = next(iter(_pgd_imps.values())).device

                # n_keep / n_total — one all_reduce for FSDP
                _pgd_stats = torch.tensor(
                    [sum(maskmgr.masks[n].sum().item() for n in _pgd_imps),
                     sum(v.numel() for v in _pgd_imps.values())],
                    dtype=torch.long, device=_pgd_dev)
                if _pgd_use_fsdp:
                    _dist.all_reduce(_pgd_stats, op=_dist.ReduceOp.SUM)
                _pgd_k_prune = int(_pgd_stats[1].item() - _pgd_stats[0].item())

                # global min/max — two all_reduces for FSDP
                _pgd_lo_t = torch.tensor(
                    min(v.min().item() for v in _pgd_imps.values()),
                    dtype=torch.float32, device=_pgd_dev)
                _pgd_hi_t = torch.tensor(
                    max(v.max().item() for v in _pgd_imps.values()),
                    dtype=torch.float32, device=_pgd_dev)
                if _pgd_use_fsdp:
                    _dist.all_reduce(_pgd_lo_t, op=_dist.ReduceOp.MIN)
                    _dist.all_reduce(_pgd_hi_t, op=_dist.ReduceOp.MAX)
                _pgd_lo, _pgd_hi = _pgd_lo_t.item(), _pgd_hi_t.item()

                # binary search — _pgd_cnt_t reused in-place, one all_reduce/iter for FSDP
                _pgd_cnt_t = torch.zeros(1, dtype=torch.long, device=_pgd_dev)
                for _ in range(48):
                    _pgd_mid = (_pgd_lo + _pgd_hi) / 2.0
                    _pgd_cnt_t.zero_()
                    for _v in _pgd_imps.values():
                        _pgd_cnt_t += (_v <= _pgd_mid).sum(dtype=torch.long)
                    if _pgd_use_fsdp:
                        _dist.all_reduce(_pgd_cnt_t, op=_dist.ReduceOp.SUM)
                    if _pgd_cnt_t.item() < _pgd_k_prune:
                        _pgd_lo = _pgd_mid
                    else:
                        _pgd_hi = _pgd_mid
                _pgd_thr = _pgd_hi

                # apply new mask, count revivals/prunings
                for _n in maskmgr.named_params:
                    _old = maskmgr.masks[_n]
                    _new = (_pgd_imps[_n] > _pgd_thr) if _n in _pgd_imps else _old.clone()
                    _pgd_revivals += int((_new & ~_old).sum().item())
                    _pgd_prunings += int((~_new & _old).sum().item())
                    maskmgr.masks[_n] = _new
                maskmgr.apply(fsdp_model)

                # sum revival/pruning counts across ranks (FSDP only)
                if _pgd_use_fsdp:
                    _pgd_rv_t = torch.tensor([_pgd_revivals, _pgd_prunings],
                                             dtype=torch.long, device=_pgd_dev)
                    _dist.all_reduce(_pgd_rv_t, op=_dist.ReduceOp.SUM)
                    _pgd_revivals, _pgd_prunings = int(_pgd_rv_t[0].item()), int(_pgd_rv_t[1].item())

            if use_wandb and is_main_process:
                wandb.log({"pgd/revivals": _pgd_revivals, "pgd/prunings": _pgd_prunings,
                           "step": step})

        # ── Gradient conflict measurement (OPKD vs IPO on same sequence) ──────
        if (measure_grad_conflict and not filter_grad_conflict
                and _measure_generated is not None
                and _measure_pair is not None and is_main_process):
            _pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            _mg = _measure_generated
            _mpl = _measure_prompt_len
            _mp = {k: v.to(device) for k, v in _measure_pair.items()}

            # 1. OPKD gradient
            model.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _s_out_m = model(input_ids=_mg)
                with torch.no_grad():
                    _t_out_m = teacher_model(input_ids=_mg)
                _m_labels = _mg.clone()
                _m_labels[:, :_mpl] = -100
                _m_labels[_mg == _pad_val] = -100
                _op_kl_m, _ = _kl_loss(_s_out_m.logits, _t_out_m.logits, _m_labels,
                                        kd_temperature, onpolicy_topk, reverse=onpolicy_reverse_kl)
            (onpolicy_lambda * _op_kl_m).backward()
            _opkd_grads = [p.grad.detach().clone() if p.grad is not None else None
                           for p in model.parameters()]

            # 2. IPO gradient
            model.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _pol_out_m = concatenated_forward(model, _mp, padding_value=_pad_val)
                if not dpo_reference_free:
                    with torch.no_grad():
                        _ref_out_m = concatenated_forward(dpo_ref_model, _mp, padding_value=_pad_val, is_ref_model=True)
                else:
                    _ref_out_m = {
                        "chosen_logps":   torch.zeros_like(_pol_out_m["chosen_logps"]),
                        "rejected_logps": torch.zeros_like(_pol_out_m["rejected_logps"]),
                    }
                _losses_m, _, _ = _dpo_loss(
                    _pol_out_m["chosen_logps"], _pol_out_m["rejected_logps"],
                    _ref_out_m["chosen_logps"], _ref_out_m["rejected_logps"],
                    beta=dpo_beta, loss_type=dpo_loss_type, reference_free=dpo_reference_free,
                )
            (dpo_lambda * _losses_m.mean()).backward()

            # 3. Streaming cosine similarity (no large flat tensor)
            _dot = _norm1 = _norm2 = 0.0
            for _g1, _p in zip(_opkd_grads, model.parameters()):
                if _g1 is None or _p.grad is None:
                    continue
                _g1f = _g1.float().flatten()
                _g2f = _p.grad.detach().float().flatten()
                _dot  += (_g1f * _g2f).sum().item()
                _norm1 += _g1f.pow(2).sum().item()
                _norm2 += _g2f.pow(2).sum().item()
            _cos_sim = _dot / (math.sqrt(_norm1 * _norm2) + 1e-10)
            del _opkd_grads

            if use_wandb and wandb.run is not None:
                wandb.log({"grad_conflict/cos_sim": _cos_sim,
                           "grad_conflict/norm_opkd": math.sqrt(_norm1),
                           "grad_conflict/norm_ipo":  math.sqrt(_norm2)}, step=step)
            model.zero_grad()

        if step > dense_warmup_steps:
            maskmgr.apply(fsdp_model)

        # periodic logging
        if step % log_interval == 0:
            real_sparsity = maskmgr.current_sparsity()
            current_sparsity = 0.0 if step <= dense_warmup_steps else _cubic_sparsity(
                min(step, pruning_end_steps), pruning_end_steps, final_sparsity, dense_warmup_steps)
            log_dict = {
                "train/loss": accum_loss,
                "train/ntp_loss": accum_ntp,
                "train/sparsity": real_sparsity,
                "train/target_sparsity": current_sparsity,
                "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": accum_grad_norm / log_interval,
                "step": step,
            }
            if use_l1:
                log_dict["train/l1_loss"] = accum_l1
            if use_kd or use_hidden:
                log_dict["train/aux_loss"] = accum_kd
                if accum_diag_n > 0:
                    log_dict.update({k: v / accum_diag_n for k, v in accum_diag.items()})
            if use_onpolicy:
                if accum_onpolicy > 0 or not use_rollout:
                    log_dict["train/onpolicy_kd_loss"] = accum_onpolicy
                log_dict.update(accum_onpolicy_diag)
            if use_dpo_loss:
                log_dict["train/dpo_loss"]              = accum_dpo_loss
                log_dict["train/dpo_acc"]               = accum_dpo_acc
                # v2 diagnostics: sanity-check logprob scale & ref alignment
                log_dict["train/dpo_chosen_logp"]       = accum_dpo_chosen_logp
                log_dict["train/dpo_rejected_logp"]     = accum_dpo_rejected_logp
                log_dict["train/dpo_ref_chosen_logp"]   = accum_dpo_ref_chosen_logp
                log_dict["train/dpo_ref_rejected_logp"] = accum_dpo_ref_rejected_logp
                log_dict["train/dpo_margin"]            = accum_dpo_margin
                if use_ca_ipo and accum_ca_ipo_diag:
                    log_dict.update({k: v / log_interval for k, v in accum_ca_ipo_diag.items()})
                if use_offline_ipo:
                    log_dict["train/offline_ipo_loss"] = accum_offline_ipo
            logging.info(f"Step {step}/{total_steps} | loss={accum_loss:.4f} | "
                         f"sparsity={real_sparsity:.3f} | lr={scheduler.get_last_lr()[0]:.2e}"
                         + (f" | dpo_loss={accum_dpo_loss:.4f} acc={accum_dpo_acc:.3f} "
                            f"margin={accum_dpo_margin:.4f}" if use_dpo else "")
                         + (f" | offline_ipo={accum_offline_ipo:.4f}" if use_offline_ipo else ""))
            if use_wandb and wandb.run is not None and is_main_process:
                wandb.log(log_dict, step=step)
            accum_loss           = 0.0
            accum_ntp            = 0.0
            accum_kd             = 0.0
            accum_l1             = 0.0
            accum_grad_norm      = 0.0
            accum_dpo_loss               = 0.0
            accum_dpo_acc                = 0.0
            accum_dpo_chosen_logp        = 0.0
            accum_dpo_rejected_logp      = 0.0
            accum_dpo_ref_chosen_logp    = 0.0
            accum_dpo_ref_rejected_logp  = 0.0
            accum_dpo_margin             = 0.0
            accum_offline_ipo            = 0.0
            accum_diag           = {}
            accum_diag_n         = 0
            accum_onpolicy_diag  = {}
            accum_ca_ipo_diag    = {}

    # final mask at full sparsity
    maskmgr.update(fisher, final_sparsity, fsdp_model)
    logging.info(f"Final sparsity: {maskmgr.current_sparsity():.4f}")

    if is_main_process:
        # Gradient fine-tuning: ~6*N*tokens (forward+backward+update), vs ~2*N*tokens
        # for forward-only one-shot calibration (ALPS/SparseGPT/Wanda/SparseLLM).
        n_params = sum(p.numel() for p in model.parameters())
        global_batch = batch_size * grad_accum * world_size
        n_tokens = step * global_batch * FLAGS.seqlen
        flops = 6 * n_params * n_tokens
        logging.info(f"Training FLOPs: {flops:.3e} ({n_params} params x {n_tokens} tokens)")
        if use_wandb and wandb.run is not None:
            wandb.log({"flops": flops})

    # save model
    # FSDP: summon_full_params is a collective — ALL ranks must enter it together.
    # Only rank-0 actually writes to disk.
    saved_path = None
    if is_fsdp and do_save:
        if is_main_process:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_path = f"{FLAGS.gmp_save_path}/{_run_tag(FLAGS)}_{ts}"
        with FSDP.summon_full_params(fsdp_model, writeback=False, recurse=True):
            if is_main_process:
                model.save_pretrained(saved_path)
        if is_main_process:
            tokenizer.save_pretrained(saved_path)
            logging.info(f"Saved pruned model to {saved_path}")
    elif not is_fsdp and is_main_process and do_save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = f"{FLAGS.gmp_save_path}/{_run_tag(FLAGS)}_{ts}"
        model.save_pretrained(saved_path)
        tokenizer.save_pretrained(saved_path)
        logging.info(f"Saved pruned model to {saved_path}")

    # optional downstream eval (rank-0 only)
    if eval_fn is not None and is_main_process:
        metrics = eval_fn(model)
        if use_wandb and wandb.run is not None:
            wandb.log(metrics, step=step)

    # Post-hoc milestone eval: reload each saved milestone checkpoint and run eval.
    # Done after final eval so the training model is no longer needed in GPU memory.
    if eval_fn is not None and is_main_process and _passed_milestones:
        import gc as _gc
        for _ms, _ms_path in sorted(_passed_milestones.items()):
            if _ms_path is None:
                continue
            _ms_tag = f"sp{int(_ms * 100):02d}"
            logging.info(f"[Milestone] running eval on {_ms_tag} checkpoint: {_ms_path}")
            try:
                from transformers import AutoModelForCausalLM as _AutoModel
                _ms_model = _AutoModel.from_pretrained(_ms_path, torch_dtype=model.dtype,
                                                       device_map=str(device))
                _ms_metrics = eval_fn(_ms_model)
                _ms_metrics_tagged = {f"milestone_{_ms_tag}/{k}": v for k, v in _ms_metrics.items()}
                if use_wandb and wandb.run is not None:
                    wandb.log(_ms_metrics_tagged)
                logging.info(f"[Milestone] {_ms_tag} eval: {_ms_metrics}")
                del _ms_model
                _gc.collect()
                torch.cuda.empty_cache()
            except Exception as _e:
                logging.warning(f"[Milestone] eval failed for {_ms_tag}: {_e}")

    total_time = time.time() - start_time
    logging.info(f"GMP training done in {total_time/3600:.2f}h")
    return saved_path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _collate(batch, pad_token_id=0):
    # Only use fields needed for NTP forward pass
    ntp_keys = [k for k in batch[0].keys() if k in ('input_ids', 'attention_mask', 'labels')]
    max_len = max(len(b['input_ids']) if isinstance(b['input_ids'], list) else b['input_ids'].shape[0] for b in batch)
    result = {}
    for k in ntp_keys:
        tensors = []
        for b in batch:
            t = b[k]
            if isinstance(t, list):
                t = torch.tensor(t, dtype=torch.long)
            pad_val = -100 if k == 'labels' else pad_token_id
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                t = torch.cat([t, torch.full((pad_len,), pad_val, dtype=t.dtype)])
            tensors.append(t)
        result[k] = torch.stack(tensors)
    return result


def _infinite(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def _run_tag(FLAGS):
    lr  = getattr(FLAGS, 'lr', 0)
    sp  = getattr(FLAGS, 'sparsity_ratio', 0)
    tag = f"gmp_s{int(sp*100)}pct_lr{lr}"
    if getattr(FLAGS, 'gmp_anchor_kd_lambda', 0.0) > 0:
        tag += f"_anchor_lmda{FLAGS.gmp_anchor_kd_lambda}_pfx{FLAGS.gmp_anchor_prefix_len}"
    elif getattr(FLAGS, 'gmp_onpolicy_kd_lambda', 0.0) > 0:
        tag += f"_onpol_lmda{FLAGS.gmp_onpolicy_kd_lambda}"
    return tag
