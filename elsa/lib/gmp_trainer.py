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
from transformers.optimization import get_cosine_schedule_with_warmup
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

    Penalizes the mean abs value of the bottom-(M-N) alive weights per group-of-M.
    Already-pruned (mask=0) positions are excluded — penalizing zeros is meaningless
    and would bias the gradient signal.
    Normalized by alive element count so scale stays comparable to per-token NTP loss.
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
            # within each group, only consider alive weights for bottom-k selection
            metric = w_nm.abs()
            metric[~alive_nm] = float('inf')  # dead weights can't be bottom-k
            n_pruned = prune_m - prune_n
            bottom_idx = torch.topk(metric, n_pruned, dim=1, largest=False).indices
            selected = w_nm.abs().gather(1, bottom_idx)
            # only count positions that are actually alive
            alive_selected = alive_nm.gather(1, bottom_idx)
            term = selected[alive_selected].sum()
            n = int(alive_selected.sum().item())
        total = term if total is None else total + term
        count += n
    if total is None or count == 0:
        return torch.tensor(0.0)
    return total / count


def _gmp_l1_regularizer(named_params, maskmgr, fisher, mode="plain",
                        clip_min=0.1, clip_max=10.0):
    """L1 regularization term for GMP training.

    mode="plain":
        mean |w_i| over alive weights (mean-normalized across layers)

    mode="inv_fisher_sqrt":
        mean  |w_i| / sqrt(clamp(f_i / mean(f_alive), clip_min, clip_max))
        Weights with high Fisher (important) get lower penalty,
        weights with low Fisher (pruning candidates) get higher penalty.
        Falls back to plain L1 if Fisher state not yet available.
    """
    reg_terms = []
    for name, param in named_params.items():
        if param.ndim != 2:
            continue
        mask = maskmgr.masks.get(name)
        if mask is None:
            continue
        alive = mask.bool()
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
        self.saliency = saliency  # 'fisher' or 'magnitude'

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
        """Importance score for pruning. 'fisher': F_hat*w^2, 'magnitude': w^2."""
        if self.saliency == 'magnitude':
            return param.data.float() ** 2
        f = self.fisher_factor(param)
        if f is None:
            return param.data.float() ** 2  # fallback before first optimizer step
        imp = f * param.data.float() ** 2
        if imp.sum() == 0:
            imp = param.data.float() ** 2
        return imp


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
                all_scores = torch.cat([v.flatten() for v in local_imps.values()])
                if torch.isnan(all_scores).any() or torch.isinf(all_scores).any():
                    logging.warning("NaN/Inf in Fisher importance scores, skipping candidate mask")
                    return {n: m.clone() for n, m in self.masks.items()}
                k = int(all_scores.numel() * sparsity)
                if k == 0:
                    return {n: m.clone() for n, m in self.masks.items()}
                # torch.kthvalue uses int32 internally and overflows for tensors with
                # >2B elements (Qwen3-4B has ~3.6B linear weights). Use chunked binary
                # search: count in 100M-element chunks to avoid allocating a 27GB
                # temporary comparison tensor. Same accuracy as full binary search.
                _MAX_KTHVALUE = 100_000_000
                if all_scores.numel() > _MAX_KTHVALUE:
                    _CHUNK = 100_000_000
                    lo = all_scores.min().item()
                    hi = all_scores.max().item()
                    for _ in range(48):
                        mid = (lo + hi) / 2.0
                        cnt = 0
                        for _s in range(0, all_scores.numel(), _CHUNK):
                            cnt += (all_scores[_s:_s + _CHUNK] <= mid).sum().item()
                        if cnt < k:
                            lo = mid
                        else:
                            hi = mid
                    threshold = torch.tensor(hi, device=all_scores.device, dtype=all_scores.dtype)
                    actual = sum(
                        (all_scores[_s:_s + _CHUNK] <= threshold).sum().item()
                        for _s in range(0, all_scores.numel(), _CHUNK)
                    )
                    logging.info(f"  [Fisher] large model ({all_scores.numel()} params): "
                                 f"chunked binary-search threshold={threshold.item():.4e} "
                                 f"(actual_below={actual}, target={k})")
                else:
                    threshold = torch.kthvalue(all_scores, k).values

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
        kl = (s_logp_full.exp() * (s_logp_full - t_logp_full)).sum(dim=-1)
    elif topk > 0:
        t_topk_idx = t_logits.topk(topk, dim=-1).indices     # (B, T-1, K)
        t_logp = t_logp_full.gather(-1, t_topk_idx)
        s_logp = s_logp_full.gather(-1, t_topk_idx)
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
    else:
        kl = (t_logp_full.exp() * (t_logp_full - s_logp_full)).sum(dim=-1)

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


def _sync_opkd_weights_to_vllm(model: nn.Module, vllm_engine) -> None:
    """Sync current student weights into the OPKD vLLM engine."""
    engine = vllm_engine.llm_engine
    # vLLM 0.10+ V1 engine: model_executor lives under engine_core
    executor = engine.engine_core.model_executor if hasattr(engine, 'engine_core') else engine.model_executor
    vllm_model = executor.driver_worker.model_runner.model
    vllm_state = {k: v for k, v in vllm_model.named_parameters()}
    for name, param in model.named_parameters():
        if name in vllm_state:
            vllm_state[name].data.copy_(param.data.to(vllm_state[name].dtype))


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
        old_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits.detach().float()

    # Temporarily zero newly-pruned weights (old_mask=True & cand_mask=False)
    saved = {}
    for name, param in maskmgr.named_params.items():
        newly_pruned = maskmgr.masks[name] & ~cand_masks[name]
        if newly_pruned.any():
            saved[name] = (newly_pruned, param.data[newly_pruned].clone())
            param.data[newly_pruned] = 0.0

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        cand_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits.detach().float()

    # Restore
    for name, (mask_idx, vals) in saved.items():
        maskmgr.named_params[name].data[mask_idx] = vals

    if not valid.any():
        return 0.0, None

    old_lp   = F.log_softmax(old_logits[:, :-1, :], dim=-1)   # [B, T-1, V]
    cand_lp  = F.log_softmax(cand_logits[:, :-1, :], dim=-1)  # [B, T-1, V]
    old_p    = old_lp.exp()
    kl_tok   = (old_p * (old_lp - cand_lp)).sum(dim=-1)       # [B, T-1]
    kl_vals  = kl_tok[valid].float()
    if kl_reduce == 'quantile':
        result = torch.quantile(kl_vals, kl_quantile).item()
    else:
        result = kl_vals.mean().item()
    return max(result, 0.0), kl_vals  # (scalar, per-token KL tensor)


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
    """
    current_sp = maskmgr.current_sparsity()

    if current_sp >= final_sparsity - 1e-4:
        maskmgr.apply(fsdp_model)
        return current_sp, tr_delta, True

    delta               = tr_delta
    last_accepted_masks = None
    last_accepted_sp    = current_sp
    last_accepted_delta = 0.0
    last_kl             = float('inf')
    prev_accepted       = False  # True if the previous iter was accepted

    for i in range(max_iters):
        try_sp   = min(current_sp + delta, final_sparsity)
        cand     = maskmgr.candidate_masks(fisher, try_sp, fsdp_model)
        kl, kl_vals = _compute_tr_kl(model, cal_batch, cand, maskmgr, device,
                                      kl_reduce=kl_reduce, kl_quantile=kl_quantile)
        accepted = kl <= kl_threshold

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

        if accepted:
            last_accepted_masks = cand
            last_accepted_sp    = try_sp
            last_accepted_delta = delta
            last_kl             = kl
            if try_sp >= final_sparsity - 1e-4:
                break  # target reached, done
            prev_accepted = True
            delta = min(delta * 2.0, final_sparsity - current_sp)
        else:
            if prev_accepted:
                break  # ✓ → ✗: found boundary, use last accepted
            prev_accepted = False
            delta /= 2.0
            if delta < delta_min:
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

    return new_sp, new_delta, reached, _mask_delta


def globalprune_gmp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    FLAGS,
    teacher_model: AutoModelForCausalLM = None,
    dpo_dense_model: AutoModelForCausalLM = None,
    eval_fn=None,        # optional callable(model) → dict of metrics
):
    """
    BEST-style GMP training loop with optional token-level KD.

    FLAGS expected attributes:
      gmp_steps               int    total training steps
      gmp_batch_size          int    per-device batch size
      gmp_grad_accum          int    gradient accumulation steps
      gmp_lr                  float  peak learning rate
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

    total_steps    = FLAGS.gmp_steps
    batch_size     = getattr(FLAGS, 'gmp_batch_size', 1)
    grad_accum     = getattr(FLAGS, 'gmp_grad_accum', 8)
    lr             = getattr(FLAGS, 'gmp_lr', 1e-5)
    warmup_ratio        = getattr(FLAGS, 'gmp_warmup_ratio', 0.05)
    mask_interval       = getattr(FLAGS, 'gmp_mask_interval', 32)
    log_interval        = getattr(FLAGS, 'gmp_log_interval', 1)
    fisher_beta         = getattr(FLAGS, 'gmp_fisher_beta', 0.999)
    final_sparsity      = FLAGS.sparsity_ratio
    warmup_steps        = int(total_steps * warmup_ratio)
    pruning_end_ratio   = getattr(FLAGS, 'gmp_pruning_end_ratio', 1.0)
    pruning_end_steps   = int(total_steps * pruning_end_ratio)
    dense_warmup_steps  = getattr(FLAGS, 'gmp_dense_warmup_steps', 0)
    # TR-GMP flags
    tr_enabled      = getattr(FLAGS, 'gmp_tr_enabled', False)
    tr_kl_threshold = getattr(FLAGS, 'gmp_tr_kl_threshold', 0.01)
    tr_delta_init   = getattr(FLAGS, 'gmp_tr_delta_init', 0.05)
    tr_delta_min    = getattr(FLAGS, 'gmp_tr_delta_min', 0.005)
    tr_kl_reduce    = getattr(FLAGS, 'gmp_tr_kl_reduce', 'mean')
    tr_kl_quantile  = getattr(FLAGS, 'gmp_tr_kl_quantile', 0.95)
    use_wandb      = getattr(FLAGS, 'wandb', False)
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
    use_kd         = (teacher_model is not None) and (kd_lambda > 0.0)
    use_hidden     = (teacher_model is not None) and (hidden_lambda > 0.0)
    use_teacher_gen_kd_flag = getattr(FLAGS, 'gmp_teacher_gen_kd', False)
    use_onpolicy   = (teacher_model is not None) and (onpolicy_lambda > 0.0) and not use_teacher_gen_kd_flag
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
        from lib.gkd_admm_trainer import MathPromptDataset
        prompt_path = getattr(FLAGS, 'gmp_prompt_path', None) or getattr(FLAGS, 'data_path', None)
        prompt_max_len = getattr(FLAGS, 'gmp_max_prompt_len', 512)
        _prompt_ds = MathPromptDataset(
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
        _os.environ['VLLM_USE_V1'] = '0'  # V1 uses SyncMPClient; V0 allows direct weight access
        from vllm import LLM, SamplingParams as _VLLMSamplingParams
        from vllm.inputs import TokensPrompt as _TokensPrompt
        _model_path_for_opkd = getattr(FLAGS, 'model', None)
        logging.info(f"  OPKD vLLM: initializing engine gpu_mem={opkd_vllm_gpu_mem} ...")
        _opkd_vllm_enforce_eager = getattr(FLAGS, 'gmp_opkd_vllm_enforce_eager', False)
        _opkd_vllm_engine = LLM(
            _model_path_for_opkd,
            dtype="bfloat16",
            gpu_memory_utilization=opkd_vllm_gpu_mem,
            trust_remote_code=True,
            max_model_len=onpolicy_max_new + getattr(FLAGS, 'gmp_max_prompt_len', 512),
            enforce_eager=_opkd_vllm_enforce_eager,
        )
        _opkd_vllm_params = _VLLMSamplingParams(
            max_tokens=onpolicy_max_new,
            temperature=onpolicy_temp,
            top_p=0.95,
        )
        logging.info(f"  OPKD vLLM: engine ready")
        # Pre-fill pool before training loop starts so step 1 has rollouts available
        _sync_opkd_weights_to_vllm(model, _opkd_vllm_engine)
        _n_pool = mask_interval * grad_accum
        _pool_batches = [next(prompt_iter) for _ in range(_n_pool)]
        _vllm_inputs = [
            _TokensPrompt(prompt_token_ids=b['input_ids'][0][:int(b['prompt_len'].item())].tolist())
            for b in _pool_batches
        ]
        _vllm_outs = _opkd_vllm_engine.generate(_vllm_inputs, _opkd_vllm_params)
        for _pb, _vo in zip(_pool_batches, _vllm_outs):
            _plen = int(_pb['prompt_len'].item())
            _p_ids = _pb['input_ids'][:, :_plen].cpu()
            _gen_ids = torch.tensor([_vo.outputs[0].token_ids], dtype=torch.long)
            _full_seq = torch.cat([_p_ids, _gen_ids], dim=1)
            _opkd_standalone_pool.append({"full_seq": _full_seq, "prompt_len": _plen})
        logging.info(f"  OPKD vLLM: initial pool filled with {len(_opkd_standalone_pool)} rollouts")

    rollout_buffer = RolloutBuffer() if use_rollout else None

    fixed_mask       = getattr(FLAGS, 'gmp_fixed_mask', False)
    l1_lambda        = getattr(FLAGS, 'gmp_l1_lambda', 0.0)
    l1_structured    = getattr(FLAGS, 'gmp_l1_structured', True)
    l1_mode          = getattr(FLAGS, 'gmp_l1_mode', 'plain')
    l1_fisher_cmin   = getattr(FLAGS, 'gmp_l1_fisher_clip_min', 0.1)
    l1_fisher_cmax   = getattr(FLAGS, 'gmp_l1_fisher_clip_max', 10.0)

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
            logging.info(f"  L1 mode={l1_mode}: lambda={l1_lambda}")

    if getattr(FLAGS, 'gmp_gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        logging.info("  Gradient checkpointing ENABLED (reduces activation memory)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    fisher  = FisherAccumulator(named_params, optimizer, saliency=FLAGS.gmp_saliency)
    maskmgr = GradualMaskManager(named_params, fsdp_model, prune_n=prune_n, prune_m=prune_m,
                                  pruning_scope=getattr(FLAGS, 'gmp_pruning_scope', 'global'))
    if fixed_mask:
        maskmgr.init_from_weights()
        maskmgr.apply(fsdp_model)
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
            shuffle=False,
            collate_fn=_collate_fn,
        )
    data_iter = _infinite(loader)

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
    logging.info(f"  LR = {lr}, warmup = {warmup_steps} steps")
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

            if use_l1:
                if use_structured_l1:
                    l1 = _structured_l1_loss(named_params, maskmgr.masks, prune_n, prune_m)
                else:
                    l1 = _gmp_l1_regularizer(named_params, maskmgr, fisher,
                                             mode=l1_mode,
                                             clip_min=l1_fisher_cmin,
                                             clip_max=l1_fisher_cmax)
                if l1 is not None:
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
            if (use_onpolicy and _opkd_vllm_engine is not None
                    and is_main_process and tr_enabled and not tr_reached):
                _sync_opkd_weights_to_vllm(model, _opkd_vllm_engine)
                _n_pool = mask_interval * grad_accum
                _pool_batches = [next(prompt_iter) for _ in range(_n_pool)]
                _vllm_inputs = [
                    _TokensPrompt(prompt_token_ids=b['input_ids'][0][:int(b['prompt_len'].item())].tolist())
                    for b in _pool_batches
                ]
                _vllm_outs = _opkd_vllm_engine.generate(_vllm_inputs, _opkd_vllm_params)
                _opkd_standalone_pool = []
                for _pb, _vo in zip(_pool_batches, _vllm_outs):
                    _plen = int(_pb['prompt_len'].item())
                    _p_ids = _pb['input_ids'][:, :_plen].cpu()
                    _gen_ids = torch.tensor([_vo.outputs[0].token_ids], dtype=torch.long)
                    _full_seq = torch.cat([_p_ids, _gen_ids], dim=1)
                    _opkd_standalone_pool.append({"full_seq": _full_seq, "prompt_len": _plen})
                _opkd_standalone_pool_ptr = 0
                _opkd_refilled_pre_mask = True
                logging.info(f"  OPKD vLLM pool refilled (pre-mask): {len(_opkd_standalone_pool)} rollouts (step={step})")

            if step <= dense_warmup_steps:
                pass  # dense warmup: no mask update or apply
            elif fixed_mask:
                maskmgr.apply(fsdp_model)
            elif tr_enabled and not tr_reached:
                # Use OPKD rollouts as calibration if available, else fall back to prompt_iter
                if _opkd_refilled_pre_mask and _opkd_standalone_pool:
                    _n_cal = min(8, len(_opkd_standalone_pool))
                    _cal_batch = _opkd_pool_to_batch(_opkd_standalone_pool[:_n_cal], str(device))
                else:
                    _cal_batch = next(prompt_iter)
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
                if (opkd_prev_mask_teacher or prevmask_opkd_lambda > 0) and use_onpolicy:
                    _opkd_prev_delta = _tr_mask_delta
                if use_wandb:
                    wandb.log({"train/sparsity": current_sparsity,
                               "train/tr_delta": tr_delta, "step": step})
                if tr_reached:
                    logging.info(f"TR-GMP: target sparsity {final_sparsity} reached at step {step}, stopping.")
                    break
            else:
                current_sparsity = _cubic_sparsity(min(step, pruning_end_steps), pruning_end_steps, final_sparsity, warmup_steps)
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
            if (use_onpolicy and _opkd_vllm_engine is not None
                    and is_main_process and not _opkd_refilled_pre_mask):
                _sync_opkd_weights_to_vllm(model, _opkd_vllm_engine)
                _n_pool = mask_interval * grad_accum
                _pool_batches = [next(prompt_iter) for _ in range(_n_pool)]
                _vllm_inputs = [
                    _TokensPrompt(prompt_token_ids=b['input_ids'][0][:int(b['prompt_len'].item())].tolist())
                    for b in _pool_batches
                ]
                _vllm_outs = _opkd_vllm_engine.generate(_vllm_inputs, _opkd_vllm_params)
                _opkd_standalone_pool = []
                for _pb, _vo in zip(_pool_batches, _vllm_outs):
                    _plen = int(_pb['prompt_len'].item())
                    _p_ids = _pb['input_ids'][:, :_plen].cpu()
                    _gen_ids = torch.tensor([_vo.outputs[0].token_ids], dtype=torch.long)
                    _full_seq = torch.cat([_p_ids, _gen_ids], dim=1)
                    _opkd_standalone_pool.append({"full_seq": _full_seq, "prompt_len": _plen})
                _opkd_standalone_pool_ptr = 0
                logging.info(f"  OPKD vLLM pool refilled: {len(_opkd_standalone_pool)} rollouts (step={step})")

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
            current_sparsity = 0.0 if step <= dense_warmup_steps else _cubic_sparsity(min(step, pruning_end_steps), pruning_end_steps, final_sparsity, warmup_steps)
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


def _infinite(loader):
    while True:
        yield from loader


def _run_tag(FLAGS):
    lr  = getattr(FLAGS, 'gmp_lr', 0)
    sp  = getattr(FLAGS, 'sparsity_ratio', 0)
    tag = f"gmp_s{int(sp*100)}pct_lr{lr}"
    if getattr(FLAGS, 'gmp_anchor_kd_lambda', 0.0) > 0:
        tag += f"_anchor_lmda{FLAGS.gmp_anchor_kd_lambda}_pfx{FLAGS.gmp_anchor_prefix_len}"
    elif getattr(FLAGS, 'gmp_onpolicy_kd_lambda', 0.0) > 0:
        tag += f"_onpol_lmda{FLAGS.gmp_onpolicy_kd_lambda}"
    return tag
