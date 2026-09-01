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
import re
import time
import types
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


def _find_linear_shapes(model):
    """Companion to _find_linear_weights: {name: (out_features, in_features)}
    using nn.Linear's own plain-int attributes. Unlike param.shape or
    param.data.shape, these are set once at module construction and are
    completely unaffected by how FSDP subsequently shards the underlying
    parameter storage -- verified empirically that under this codebase's
    FSDP setup (transformer_auto_wrap_policy, use_orig_params=True),
    param.shape itself reports the LOCAL FLAT SHARD's size (not the true
    logical 2D shape, and not even consistent per-parameter -- a single
    Linear's weight can straddle a rank boundary or be entirely absent on a
    rank), so it cannot be used as a source of truth for reconstruction.
    See _fsdp_nm_reconstruct, which needs the real shape to do so."""
    result = {}
    for block_idx, layer in enumerate(_get_decoder_layers(model)):
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"model.layers.{block_idx}.{name}.weight"
                result[full_name] = (module.out_features, module.in_features)
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

    named_params must hold full, unsharded [rows, cols] tensors -- not valid
    directly under FSDP1 (classic FullyShardedDataParallel; this file does
    not use FSDP2/DTensor), where each rank only holds a flat, differently
    -sized local shard. See _register_structured_l1_hooks for the FSDP path,
    which computes the same per-layer term via _structured_l1_layer_term
    from inside a forward pre-hook (params are guaranteed fully gathered
    there) instead.
    """
    total = None
    count = 0
    for name, param in named_params.items():
        mask = masks.get(name)
        term, n = _structured_l1_layer_term(param, mask, prune_n, prune_m)
        if term is None:
            continue
        total = term if total is None else total + term
        count += n
    if total is None or count == 0:
        return torch.tensor(0.0)
    return total / count


def _structured_l1_layer_term(w: torch.Tensor, mask, prune_n: int, prune_m: int):
    """Single-layer structured-L1 contribution (sum, alive-count) -- the per-
    parameter body of _structured_l1_loss, factored out so the FSDP forward-
    hook path (_register_structured_l1_hooks) can call it per-layer while
    `w` is momentarily the full gathered weight, without needing the whole
    named_params dict to be full-shape at once. Returns (None, 0) if the
    layer contributes nothing (e.g. n_full == 0)."""
    if w.dim() < 2:
        alive = w[mask] if mask is not None else w
        if alive.numel() == 0:
            return None, 0
        return alive.abs().sum(), alive.numel()
    n_rows, n_cols = w.shape
    n_full = n_cols // prune_m
    if n_full == 0:
        return None, 0
    n_nm_cols = n_full * prune_m
    w_nm = w[:, :n_nm_cols].reshape(n_rows * n_full, prune_m)
    alive_nm = mask[:, :n_nm_cols].reshape(n_rows * n_full, prune_m) if mask is not None \
               else torch.ones_like(w_nm, dtype=torch.bool)
    # exclude groups already at their prune_n cap -- nothing left to
    # decide there, so they must never contribute to this loss
    open_nm = (alive_nm.sum(dim=1, keepdim=True) > prune_n).expand_as(alive_nm)
    # within each group, only consider alive weights for bottom-k selection
    metric = w_nm.abs()
    metric = metric.masked_fill(~alive_nm, float('inf'))  # dead weights can't be bottom-k
    n_pruned = prune_m - prune_n
    bottom_idx = torch.topk(metric, n_pruned, dim=1, largest=False).indices
    selected = w_nm.abs().gather(1, bottom_idx)
    # only count positions that are actually alive AND in a still-open group
    alive_selected = alive_nm.gather(1, bottom_idx) & open_nm.gather(1, bottom_idx)
    term = selected[alive_selected].sum()
    n = int(alive_selected.sum().item())
    if n == 0:
        return None, 0
    return term, n


def _register_structured_l1_hooks(model, named_params: dict, masks: dict, prune_n: int, prune_m: int):
    """Register forward PRE-hooks on each target Linear so its structured-L1
    contribution is computed at the moment FSDP has that layer's parameters
    fully gathered (pre-hook fires before the layer's own forward math, i.e.
    strictly after the enclosing FSDP unit's all-gather and strictly before
    its post-forward reshard) -- riding FSDP's own, already-correct forward/
    backward machinery instead of a separate summon_full_params + backward()
    pass, which torch's exit-time grad-resharding doesn't support for a
    freshly computed (not pre-existing) gradient (see call site history).

    Returns (handles, terms) -- terms is a list this call appends
    (term_tensor, alive_count) tuples to as hooks fire during the NEXT
    forward pass; caller sums it into the main loss BEFORE calling
    loss.backward(), then removes the handles.
    """
    name_to_module = dict(model.named_modules())
    handles = []
    terms: list = []
    for name, param in named_params.items():
        mod_name = name[:-len('.weight')] if name.endswith('.weight') else None
        module = name_to_module.get(mod_name) if mod_name else None
        if not isinstance(module, nn.Linear):
            continue
        mask = masks.get(name)

        def _hook(module, args, mask=mask):
            term, n = _structured_l1_layer_term(module.weight, mask, prune_n, prune_m)
            if term is not None:
                terms.append((term, n))

        handles.append(module.register_forward_pre_hook(_hook))
    return handles, terms


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


def _nm_fully_closed(masks: dict, prune_n: int, prune_m: int, shapes: dict = None) -> bool:
    """Explicit, structural N:M completeness check: True only if EVERY
    group-of-prune_m in EVERY masked layer has been pruned down to exactly
    prune_n alive weights -- i.e. the mask is an actually-valid, hardware-
    deployable N:M pattern. Used instead of comparing the aggregate sparsity
    fraction against final_sparsity (even with 0 tolerance a match there is
    only necessary, not sufficient in principle, and reasoning about "is the
    pattern actually done" via a single scalar ratio is exactly the kind of
    inference this function exists to avoid needing) -- checks per-layer,
    per-group state directly instead.

    shapes (optional, {name: (out_features, in_features)}): under FSDP,
    `masks` holds flat local shards (1D, classic FSDP1 flat-buffer chunking
    -- see _fsdp_nm_reconstruct), not the true 2D [rows, cols] this check
    needs. When given, each local shard is gathered across ranks and
    reshaped to its real shape before the per-group check runs -- same
    gather-then-check pattern as candidate_masks/_pgd_nm_pre_target/
    _pgd_nm_post_target's FSDP branches. Without this, every mask here is
    1D and the function would always hit the "nothing 2D to check" guard
    below (which correctly refuses to silently report vacuous closure) --
    that's a real dead end, not a false alarm, so this reconstruction is
    required for TR-GMP's early-exit check to ever return True under FSDP.
    """
    _checked = 0
    for name, alive in masks.items():
        _shape = shapes.get(name) if shapes else None
        if _shape is not None and tuple(_shape) != tuple(alive.shape):
            _, alive = _fsdp_nm_reconstruct(alive.float(), alive, _shape)
        if alive.dim() < 2:
            continue
        n_rows, n_cols = alive.shape
        n_full = n_cols // prune_m
        if n_full == 0:
            continue
        n_nm_cols = n_full * prune_m
        alive_nm = alive[:, :n_nm_cols].reshape(n_rows * n_full, prune_m)
        group_alive_count = alive_nm.sum(dim=1)
        _checked += 1
        if bool((group_alive_count > prune_n).any()):
            return False
    if _checked == 0:
        # Nothing was actually inspected -- either `masks` is empty or every
        # entry was skipped (dim<2 or n_full==0). Silently returning True
        # here would be a false "fully closed" positive (exactly the vacuous
        # -truth bug this function exists to avoid for the aggregate-ratio
        # check) -- fail loud instead so a real shape/wiring problem (e.g.
        # under some FSDP configuration) surfaces immediately rather than
        # silently reporting the mask as done when nothing was pruned.
        raise RuntimeError(
            f"_nm_fully_closed: no 2D masks with n_full>0 found among {len(masks)} "
            f"entries (prune_n={prune_n}, prune_m={prune_m}) -- refusing to report "
            f"vacuously 'fully closed'. Sample shapes: "
            f"{[(n, tuple(m.shape)) for n, m in list(masks.items())[:3]]}"
        )
    return True


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


def _cosine_sparsity(step, total_steps, final_sparsity, warmup_steps=0):
    """Cosine schedule: s_t = s_final * 0.5 * (1 - cos(pi * (t-warmup)/(T-warmup))).
    Slower start/end than cubic, steepest in the middle of the ramp."""
    if step < warmup_steps:
        return 0.0
    t = step - warmup_steps
    T = max(total_steps - warmup_steps, 1)
    frac = min(t / T, 1.0)
    return final_sparsity * 0.5 * (1.0 - math.cos(math.pi * frac))


def _pgd_topk_mask(imps_by_name, cand_by_name, k, want_highest, dev, use_fsdp, global_lo, global_hi):
    """Select exactly k positions (FSDP-global) from the True positions of
    cand_by_name, ranked by imps_by_name -- the k LARGEST values if
    want_highest (used to pick which revival candidates are most confidently
    due), else the k SMALLEST (which pruning candidates are most confidently
    due). Same binary-search-on-threshold pattern as the main PGD threshold
    search above, so it composes with FSDP local shards the same way.
    Positions outside cand_by_name are pushed just past the global value
    range (global_lo-1 / global_hi+1) so they can never be selected.

    Two prior fix attempts failed empirically and are recorded here so they
    don't get retried: (1) a tiny per-element jitter to break exact ties --
    zero measurable effect; (2) rebracketing the linear search to this
    candidate pool's own min/max instead of the global one -- helped on
    SOME steps (one step's overshoot dropped from ~330M to ~9M) but relapsed
    to 15x-460x overshoot on others, because the candidate pool ITSELF can
    still span many orders of magnitude (importance = fisher_factor *
    weight^2 is a heavy-tailed, non-negative quantity), and 48 *linear*
    bisections give resolution (hi-lo)/2^48 -- coarser than the gaps
    between genuinely distinct values near the true k-th quantile whenever
    hi-lo spans that many orders of magnitude, regardless of how tightly
    [lo, hi] is bracketed. The actual fix: bisect in LOG-SPACE instead of
    raw value space, so resolution is PROPORTIONAL (a fixed number of
    significant digits) rather than absolute -- the standard technique for
    precisely thresholding a low (here ~0.07%) percentile of a heavy-tailed
    non-negative distribution. Values clamped to a tiny floor before
    logging (so exact-zero entries, e.g. weights already crushed by
    structured-L1, map to one specific very-negative-but-finite value
    instead of -inf)."""
    if use_fsdp:
        import torch.distributed as _dist
    # Floor must sit BELOW the smallest real importance value actually seen
    # in practice, or genuinely-distinct tiny values get clamped together
    # into one artificial tie cluster -- worse than the original problem.
    # Verified empirically: real fisher*weight^2 values here range down to
    # ~1e-37 (float32 denormals), so an earlier floor of 1e-30 clamped a
    # huge swath of legitimately-different near-zero importances to the
    # exact same value and made the overshoot WORSE (one run jumped to an
    # impossible 0.70 sparsity for a 2:4 pattern, whose true max is 0.50).
    # 1e-40 sits safely below the observed range with margin, while still
    # being representable in float32 (denormal floor ~1.4e-45).
    vals, lo, hi = _pgd_build_topk_vals(imps_by_name, cand_by_name, want_highest, global_lo, global_hi)
    return _pgd_topk_mask_from_vals(vals, lo, hi, k, dev, use_fsdp, want_highest)


def _pgd_build_topk_vals(imps_by_name, cand_by_name, want_highest, global_lo, global_hi):
    """One-time setup half of _pgd_topk_mask -- builds the log-space `vals`
    dict (the expensive part: one full-model-sized tensor allocation per
    param) plus the initial (lo, hi) search bracket. Split out so a caller
    that needs the SAME candidate set at several different k (e.g. a
    bisection search over k itself, see gmp_pgd_kl_budget) can build this
    once and reuse it via _pgd_topk_mask_from_vals -- calling the combined
    _pgd_topk_mask() repeatedly instead was observed to OOM (each call
    re-allocates ~2x the full importance-tensor footprint just for this
    setup, on top of everything else already live at a mask_interval
    boundary step)."""
    _floor = 1e-40
    log_imps = {n: torch.log(imps_by_name[n].clamp(min=_floor)) for n in imps_by_name}
    _log_floor = math.log(_floor)
    _log_hi_bound = math.log(max(global_hi, _floor))
    sentinel = (_log_floor - 1.0) if want_highest else (_log_hi_bound + 1.0)
    vals = {n: torch.where(cand_by_name[n], log_imps[n], torch.full_like(log_imps[n], sentinel))
            for n in imps_by_name}
    lo, hi = _log_floor - 1.0, _log_hi_bound + 1.0
    return vals, lo, hi


def _pgd_topk_mask_from_vals(vals, lo, hi, k, dev, use_fsdp, want_highest):
    """Bisection half of _pgd_topk_mask, given a precomputed `vals` dict
    (see _pgd_build_topk_vals) -- cheap (no new full-sized tensors), so
    safe to call many times in a row for different k against the same vals."""
    if use_fsdp:
        import torch.distributed as _dist
    if k <= 0:
        return {n: torch.zeros_like(v, dtype=torch.bool) for n, v in vals.items()}
    cnt_t = torch.zeros(1, dtype=torch.long, device=dev)
    for _ in range(64):
        mid = (lo + hi) / 2.0
        cnt_t.zero_()
        for v in vals.values():
            cnt_t += (v >= mid).sum(dtype=torch.long) if want_highest else (v <= mid).sum(dtype=torch.long)
        if use_fsdp:
            _dist.all_reduce(cnt_t, op=_dist.ReduceOp.SUM)
        c = cnt_t.item()
        if want_highest:
            # count(>=mid) == k target; raising mid shrinks count
            if c < k:
                hi = mid
            else:
                lo = mid
        else:
            # count(<=mid) == k target; raising mid grows count
            if c < k:
                lo = mid
            else:
                hi = mid
    thr = lo if want_highest else hi
    sel = {n: ((v >= thr) if want_highest else (v <= thr)) for n, v in vals.items()}

    # Tie-breaking: a naive threshold cut can massively overshoot k when many
    # candidates share the EXACT value at `thr` -- e.g. fisher*weight^2==0
    # for every weight whose Adam exp_avg_sq is still zero (never touched by
    # a gradient), very common for a large embedding/lm_head table at early
    # steps. Verified empirically: step 1 of a gmp_pgd_grow_to_target run
    # requested k=63 and the naive threshold cut selected 680,136 (a
    # ~10,800x overshoot, beyond this function's own previously-documented
    # 15x-460x worst case) -- 680k embedding rows tied at exactly zero
    # importance, all swept in together since the cut can't split a tied
    # block. Thin the AT-threshold tie cluster with elementwise random
    # selection instead (one extra full-tensor boolean compare + one random
    # draw, done ONCE here -- not per bisection iteration -- so cheap
    # relative to the 64-iteration search already above) so the total
    # selected count lands close to k instead of "all-or-nothing" on the
    # tie. Approximate, not exact (especially under FSDP, where each rank
    # thins its own local slice of the tie independently without cross-rank
    # coordination on WHICH elements) -- but for a tie cluster this large
    # the binomial variance around the target count is tiny (~sqrt(N)
    # candidates), vastly tighter than accepting the entire tie or none of it.
    # DIAGNOSTIC A/B toggle (env var, not a real flag -- temporary, for
    # isolating this tie-breaking block's own wall-clock cost from
    # everything else): GMP_PGD_SKIP_TIEBREAK=1 reverts to the old
    # all-or-nothing behavior with none of the extra tensor ops below.
    import os as _os_tiebreak
    if _os_tiebreak.environ.get('GMP_PGD_SKIP_TIEBREAK') == '1':
        return sel
    tied = {n: (v == thr) for n, v in vals.items()}
    n_tied_t = torch.zeros(1, dtype=torch.long, device=dev)
    for t in tied.values():
        n_tied_t += t.sum(dtype=torch.long)
    if use_fsdp:
        _dist.all_reduce(n_tied_t, op=_dist.ReduceOp.SUM)
    n_tied = int(n_tied_t.item())
    if n_tied > 0:
        n_strict_t = torch.zeros(1, dtype=torch.long, device=dev)
        for n in vals:
            n_strict_t += (sel[n] & ~tied[n]).sum(dtype=torch.long)
        if use_fsdp:
            _dist.all_reduce(n_strict_t, op=_dist.ReduceOp.SUM)
        n_strict = int(n_strict_t.item())
        n_needed = max(0, k - n_strict)
        if n_needed < n_tied:
            keep_frac = n_needed / n_tied
            sel = {n: (sel[n] & ~tied[n]) | (tied[n] & (torch.rand_like(v) < keep_frac))
                   for n, v in vals.items()}
        # else n_needed >= n_tied: keep the whole tie cluster (already true in `sel`) -- not an overshoot, k genuinely requires all of it.
    return sel


def _pgd_topk_mask_from_vals_kthvalue(vals, k, dev, want_highest):
    """DIAGNOSTIC A/B alternative to _pgd_topk_mask_from_vals (--gmp_pgd_topk_impl=kthvalue,
    default is 'bisect' -- this is not the default, analysis-phase only).
    Same contract (given the same precomputed `vals` dict from
    _pgd_build_topk_vals, return the boolean selection mask for the k
    highest/lowest values), but finds the threshold via one torch.kthvalue
    call on a one-time flat concat of the candidate pool instead of 64
    sequential value-threshold bisection iterations. Ties at the exact
    threshold value are NOT thinned here (unlike _pgd_topk_mask_from_vals's
    tie-breaking) -- kthvalue's own tie-breaking (whichever elements happen
    to land on the boundary) governs, so this diagnostic path can still
    overshoot/undershoot k on a large tie cluster; that's a separate
    question from the timing comparison this exists for.
    Non-FSDP only (single-rank local concat) -- not wired up for the FSDP
    collective case."""
    if k <= 0:
        return {n: torch.zeros_like(v, dtype=torch.bool) for n, v in vals.items()}
    names = list(vals.keys())
    flat = torch.cat([vals[n].reshape(-1) for n in names])
    total = flat.numel()
    if want_highest:
        thr = torch.kthvalue(flat, max(1, total - k + 1)).values
    else:
        thr = torch.kthvalue(flat, min(total, k)).values
    del flat
    return {n: ((v >= thr) if want_highest else (v <= thr)) for n, v in vals.items()}


def _fsdp_gather_flat(local_flat, group=None):
    """All-gather a flat local FSDP shard (arbitrary, possibly-zero, per-rank
    length -- classic FSDP1 concatenates ALL of a wrapped unit's parameters
    into one flat buffer and splits it into equal contiguous byte-range
    chunks across ranks, so a single param can land entirely on one rank,
    span a rank boundary, or be empty on a rank -- verified empirically,
    not row/column-aligned in any way) into the true global flat tensor in
    rank order, which reconstructs the original pre-shard flat layout
    exactly since FSDP's own chunking is contiguous-by-rank-order."""
    import torch.distributed as _dist
    world_size = _dist.get_world_size(group)
    local_size = torch.tensor([local_flat.numel()], device=local_flat.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    _dist.all_gather(sizes, local_size, group=group)
    sizes = [int(s.item()) for s in sizes]
    max_size = max(sizes) if sizes else 0
    padded = local_flat.new_zeros(max_size)
    if local_flat.numel() > 0:
        padded[:local_flat.numel()] = local_flat
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    _dist.all_gather(gathered, padded, group=group)
    pieces = [gathered[r][:sizes[r]] for r in range(world_size) if sizes[r] > 0]
    if not pieces:
        return local_flat.new_zeros(0)
    return torch.cat(pieces, dim=0)


def _fsdp_scatter_flat(full_flat, local_numel, rank, group=None):
    """Inverse of _fsdp_gather_flat: slice out this rank's own contiguous
    chunk from the reconstructed full-flat tensor. Needs every rank's local
    size (to know offsets), gathered the same way as the forward direction."""
    import torch.distributed as _dist
    world_size = _dist.get_world_size(group)
    local_size = torch.tensor([local_numel], device=full_flat.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    _dist.all_gather(sizes, local_size, group=group)
    sizes = [int(s.item()) for s in sizes]
    offset = sum(sizes[:rank])
    return full_flat[offset:offset + sizes[rank]]


def _fsdp_nm_reconstruct(local_imp, local_mask, full_shape, group=None):
    """Gather a param's local FSDP-sharded (importance, mask) pair into their
    TRUE logical [out_features, in_features] shape so the existing,
    unchanged N:M group-of-prune_m logic (which needs real row/col structure
    -- see GradualMaskManager._nm_mask, _pgd_nm_pre_target,
    _pgd_nm_post_target) can run correctly instead of silently falling into
    their dim<2 "give up, treat as 1D" fallback on every call under FSDP.

    full_shape must come from an independent, FSDP-storage-agnostic source
    -- e.g. GradualMaskManager.named_shapes, built from nn.Linear's own
    out_features/in_features attributes -- NOT param.shape/param.data.shape:
    verified empirically that under this codebase's FSDP setup, param.shape
    itself reports the LOCAL FLAT SHARD's size (classic FSDP1 concatenates a
    whole wrapped unit's parameters into one flat buffer and chunks it by
    contiguous byte-range per rank, not per-parameter or row/col-aligned --
    a single Linear's weight can straddle a rank boundary or be entirely
    absent on a rank), not the true logical shape.

    Returns (full_imp_2d, full_mask_2d) or (None, None) if full_shape isn't
    2D (nothing to do). Pair with _fsdp_nm_scatter_back to write a resulting
    full mask back out to this rank's own local shard slice."""
    if full_shape is None or len(tuple(full_shape)) != 2:
        return None, None
    full_shape = tuple(full_shape)
    full_imp_flat  = _fsdp_gather_flat(local_imp.reshape(-1), group=group)
    full_mask_flat = _fsdp_gather_flat(local_mask.reshape(-1), group=group)
    return full_imp_flat.reshape(full_shape), full_mask_flat.reshape(full_shape)


def _fsdp_nm_scatter_back(full_mask_2d, local_numel, rank, local_shape, group=None):
    """Inverse of _fsdp_nm_reconstruct: take the resulting full [rows,cols]
    mask (identical on every rank -- gather+compute was redundant but
    read-only/deterministic, so no cross-rank disagreement is possible) and
    slice out this rank's own local shard, reshaped back to its original
    local (possibly-flat) shape for storage in maskmgr.masks[name]."""
    full_flat = full_mask_2d.reshape(-1)
    local_flat = _fsdp_scatter_flat(full_flat, local_numel, rank, group=group)
    return local_flat.reshape(local_shape)


def _pgd_nm_pre_target_2d(imp, mask, desired, prune_n, prune_m, max_dead):
    """Core per-tensor body of _pgd_nm_pre_target -- requires a genuine 2D
    [rows, cols] tensor (real group structure). Factored out so both the
    plain (non-FSDP) path and the FSDP gather/scatter path in
    _pgd_nm_pre_target call the identical, unchanged math.

    desired (bool, True=should be alive): the SAME global-threshold mask
    unstructured PGD's prune_cand already uses (_pgd_desired). Without this,
    eligibility here was PURELY the per-group structural budget -- "this
    group still has spare cap room" -- with no notion of whether a candidate
    is actually a weight the model wants gone. Since most groups have spare
    room for most of training (any group growth hasn't visited yet), that
    made the eligible pool balloon to ~half the model's live weights
    regardless of the self-KL budget, so gmp_pgd_kl_budget's bisection was
    saturating almost every step (measured: >99% of PGD-active steps fully
    capped at 2:4, vs ~18% for unstructured at the same nominal budget).
    ANDing with `desired` narrows eligibility back down to "this group has
    spare room AND the global threshold also wants this weight gone" --
    same semantics as unstructured's prune_cand, just intersected with the
    N:M structural cap that unstructured doesn't need."""
    n_rows, n_cols = imp.shape
    n_full = n_cols // prune_m
    n_nm = n_full * prune_m
    imp_g = imp[:, :n_nm].reshape(n_rows * n_full, prune_m)
    mask_g = mask[:, :n_nm].reshape(n_rows * n_full, prune_m)  # True = alive
    desired_g = desired[:, :n_nm].reshape(n_rows * n_full, prune_m)  # True = should be alive
    dead_g = ~mask_g
    dead_count = dead_g.sum(dim=1, keepdim=True)
    budget = (max_dead - dead_count).clamp(min=0)  # more this group can lose this step

    imp_rank_src = imp_g.masked_fill(dead_g, float('inf'))
    rank = imp_rank_src.argsort(dim=1).argsort(dim=1)  # ascending; dead -> highest ranks
    elig_g = (rank < budget) & mask_g & ~desired_g  # lowest-importance alive slots, within budget, AND globally undesired

    elig_full = torch.zeros_like(mask)
    elig_full[:, :n_nm] = elig_g.reshape(n_rows, n_nm)
    return elig_full


def _pgd_nm_post_target_2d(imp, prune_n, prune_m):
    """Core per-tensor body of _pgd_nm_post_target -- see _pgd_nm_pre_target_2d."""
    n_rows, n_cols = imp.shape
    n_full = n_cols // prune_m
    n_nm = n_full * prune_m
    imp_g = imp[:, :n_nm].reshape(n_rows * n_full, prune_m)
    rank = imp_g.argsort(dim=1).argsort(dim=1)  # ascending: 0=lowest .. prune_m-1=highest
    keep_g = rank >= (prune_m - prune_n)
    keep_full = torch.zeros(n_rows, imp.shape[1], dtype=torch.bool, device=imp.device)
    keep_full[:, :n_nm] = keep_g.reshape(n_rows, n_nm)
    return keep_full


def _pgd_nm_pre_target(imps, masks, desired, prune_n, prune_m, k_prune, dev, use_fsdp, shapes=None):
    """N:M-aware PGD swap for use BEFORE TR-GMP growth has reached
    final_sparsity (--gmp_pgd with sparsity_type=N:M). Growth's own
    candidate_masks()/_nm_mask already guarantees every group of prune_m
    never has more than (prune_m-prune_n) dead -- but PGD's plain global-
    threshold reprojection has no group awareness at all, so left unchanged
    it silently breaks the N:M pattern (verified empirically: ~1.5-2.4% of
    groups ended up with the wrong dead-count on a 2:4 checkpoint). This caps
    how much any single group can be pruned THIS step at its remaining
    budget (prune_m-prune_n minus its current dead count) -- revivals are
    always safe (they only reduce a group's dead count) but are throttled to
    match however many prunes actually clear the cap, so the aggregate
    revive/prune counts stay equal and overall sparsity doesn't drift.

    desired (dict of bool tensors, True=should be alive, same keys as
    imps/masks): the caller's already-computed _pgd_desired (global-
    threshold mask). Eligibility here is ANDed with "~desired" (globally
    undesired) -- see _pgd_nm_pre_target_2d's docstring for why this matters
    (without it, eligibility was pure structural-budget, ballooning the
    candidate pool to ~half the model and saturating gmp_pgd_kl_budget's
    bisection almost every step).

    shapes (optional, {name: (out_features, in_features)}): under FSDP,
    imp/mask are flat local shards (classic FSDP1 flat-buffer chunking, not
    row/column-aligned at all -- verified empirically, param.shape itself is
    NOT reliable here, see _fsdp_nm_reconstruct) instead of the true
    [rows,cols] the group logic above needs. When a shape is given (from
    GradualMaskManager.named_shapes, sourced from nn.Linear's own
    out_features/in_features), gather the full tensor across ranks
    (_fsdp_nm_reconstruct), run the SAME _pgd_nm_pre_target_2d unchanged,
    then scatter this rank's own slice back out (_fsdp_nm_scatter_back) --
    no change to the group-selection math itself.
    Returns eligible_prune dict (same local/flat shard shape as the inputs)."""
    if use_fsdp:
        import torch.distributed as _dist
        _rank = _dist.get_rank()
    eligible_prune = {}
    # Drive iteration off `shapes` (GradualMaskManager.named_shapes), NOT
    # imps.keys(): imps only contains names whose LOCAL shard is non-empty on
    # THIS rank (see the `if _t.numel() > 0` filter that builds _pgd_imps in
    # globalprune_gmp), and under FSDP1's arbitrary flat-buffer chunking
    # different ranks have DIFFERENT empty/non-empty params -- so looping
    # over imps.items() made each rank call the collective ops inside
    # _fsdp_nm_reconstruct/_fsdp_nm_scatter_back for a DIFFERENT name at the
    # same iteration, silently pairing up unrelated tensors across ranks
    # (verified empirically: reshape errors mixing sizes/shapes from
    # unrelated params). named_shapes is built from static nn.Linear module
    # attributes, so it is identical (same names, same order) on every rank
    # regardless of sharding -- safe to drive collective-calling loops from.
    names = shapes.keys() if shapes else imps.keys()
    for n in names:
        mask = masks[n]
        imp = imps.get(n)
        if imp is None:
            imp = mask.new_zeros(mask.shape, dtype=torch.float32)
        des = desired.get(n) if desired is not None else None
        if des is None:
            des = mask  # no global-desired info for this param (e.g. FSDP-empty-shard fallback) -- treat as "keep as-is", i.e. never additionally eligible
        shape = shapes.get(n) if shapes else None
        if shape is not None and tuple(shape) != tuple(imp.shape):
            full_imp, full_mask = _fsdp_nm_reconstruct(imp, mask, shape)
            full_des_flat = _fsdp_gather_flat(des.reshape(-1))
            full_des = full_des_flat.reshape(tuple(shape))
            full_elig = _pgd_nm_pre_target_2d(full_imp, full_mask, full_des, prune_n, prune_m, prune_m - prune_n)
            eligible_prune[n] = _fsdp_nm_scatter_back(full_elig, imp.numel(), _rank, imp.shape)
            continue
        if imp.dim() < 2 or imp.numel() == 0:
            eligible_prune[n] = torch.zeros_like(mask)
            continue
        eligible_prune[n] = _pgd_nm_pre_target_2d(imp, mask, des, prune_n, prune_m, prune_m - prune_n)
    return eligible_prune


def _pgd_nm_post_target(imps, masks, prune_n, prune_m, shapes=None):
    """N:M-aware PGD swap for use AFTER TR-GMP growth has reached
    final_sparsity. Every group must stay at EXACTLY (prune_m-prune_n) dead
    from here on (drifting either direction in one group forces the opposite
    drift somewhere else, globally, to keep overall sparsity fixed) -- so
    the pre-target cap-only approach can't do anything useful here: once
    every group is already at its cap, no group ever has spare prune budget,
    which would freeze the mask completely (zero revivals, zero prunings)
    instead of letting PGD keep refining WHICH prune_n survive per group as
    importance shifts. Independently recomputes each group's top-prune_n
    alive set from scratch every step -- no cross-group bookkeeping needed.

    shapes: see _pgd_nm_pre_target -- same FSDP gather/scatter shim, same
    unchanged core math (_pgd_nm_post_target_2d)."""
    if shapes:
        import torch.distributed as _dist
        _rank = _dist.get_rank()
    new_masks = {}
    # See the matching comment in _pgd_nm_pre_target: must drive this loop
    # off `shapes` (rank-identical), not imps.keys() (rank-varying, since
    # zero-numel local FSDP shards are filtered out of imps per-rank) --
    # otherwise the collective ops inside _fsdp_nm_reconstruct/
    # _fsdp_nm_scatter_back get called out of step across ranks.
    names = shapes.keys() if shapes else imps.keys()
    for n in names:
        mask = masks[n]
        imp = imps.get(n)
        if imp is None:
            imp = mask.new_zeros(mask.shape, dtype=torch.float32)
        shape = shapes.get(n) if shapes else None
        if shape is not None and tuple(shape) != tuple(imp.shape):
            full_imp, full_mask = _fsdp_nm_reconstruct(imp, mask, shape)
            full_keep = _pgd_nm_post_target_2d(full_imp, prune_n, prune_m)
            new_masks[n] = _fsdp_nm_scatter_back(full_keep, imp.numel(), _rank, imp.shape)
            continue
        if imp.dim() < 2 or imp.numel() == 0:
            new_masks[n] = mask.clone()
            continue
        new_masks[n] = _pgd_nm_post_target_2d(imp, prune_n, prune_m)
    return new_masks


def _pgd_nm_group_finished_2d(mask, prune_n, prune_m):
    """Per-group boolean, broadcast to every coordinate of that group: True
    where the group's CURRENT alive-count already equals prune_n (a valid
    N:M group). Pure structural check on the mask alone, no importance
    needed -- used by gmp_pgd_grow_to_target's N:M path to split candidates
    into "unfinished" groups (free coordinate-level movement toward target,
    same as unstructured) vs "finished" groups (already-valid N:M groups,
    restricted to paired swaps only so they never get pushed away from N:M
    by an unpaired partial accept -- see _pgd_nm_finished_swap below)."""
    n_rows, n_cols = mask.shape
    n_full = n_cols // prune_m
    n_nm = n_full * prune_m
    mask_g = mask[:, :n_nm].reshape(n_rows * n_full, prune_m)
    alive_count = mask_g.sum(dim=1)
    finished_g = (alive_count == prune_n).unsqueeze(1).expand(-1, prune_m)
    finished_full = torch.zeros(n_rows, mask.shape[1], dtype=torch.bool, device=mask.device)
    finished_full[:, :n_nm] = finished_g.reshape(n_rows, n_nm)
    return finished_full


def _pgd_nm_group_finished(masks, prune_n, prune_m, shapes=None):
    """Wrapper matching _pgd_nm_post_target's FSDP gather/scatter shim --
    see _pgd_nm_group_finished_2d for the core per-tensor logic."""
    if shapes:
        import torch.distributed as _dist
        _rank = _dist.get_rank()
    result = {}
    names = shapes.keys() if shapes else masks.keys()
    for n in names:
        mask = masks[n]
        shape = shapes.get(n) if shapes else None
        if shape is not None and tuple(shape) != tuple(mask.shape):
            _dummy_imp = mask.new_zeros(mask.shape, dtype=torch.float32)
            _, full_mask = _fsdp_nm_reconstruct(_dummy_imp, mask, shape)
            full_fin = _pgd_nm_group_finished_2d(full_mask, prune_n, prune_m)
            result[n] = _fsdp_nm_scatter_back(full_fin, mask.numel(), _rank, mask.shape)
            continue
        if mask.dim() < 2 or mask.numel() == 0:
            result[n] = torch.zeros_like(mask)
            continue
        result[n] = _pgd_nm_group_finished_2d(mask, prune_n, prune_m)
    return result


def _pgd_nm_directional_2d(imp, mask, desired, prune_n, prune_m):
    """For groups NOT at exactly prune_n alive yet (over- or under-shooting):
    per-coordinate prune/revive ELIGIBILITY, directionally and per-group-
    count restricted so a single accept can never push a group's alive-count
    past prune_n in either direction, and never mixes prune+revive within
    the SAME group in one step (a group is either overshoot-prune-eligible
    XOR undershoot-revive-eligible XOR neither -- never both):
      - alive_g > prune_n (overshoot): prune candidates only, capped
        per-group at (alive_g - prune_n), lowest-importance-first within
        the group (so if the step's budget can't clear the whole group,
        the most confidently-unimportant excess goes first).
      - alive_g < prune_n (undershoot): revive candidates only, capped
        per-group at (prune_n - alive_g), highest-importance-first.
      - alive_g == prune_n (finished): excluded entirely here -- handled
        by _pgd_nm_finished_swap_2d instead, which enforces atomic pairing
        (this function's per-group cap alone is NOT enough for finished
        groups, since a finished group needs coordinated prune+revive, not
        one-directional movement)."""
    n_rows, n_cols = imp.shape
    n_full = n_cols // prune_m
    n_nm = n_full * prune_m
    imp_g = imp[:, :n_nm].reshape(n_rows * n_full, prune_m)
    mask_g = mask[:, :n_nm].reshape(n_rows * n_full, prune_m)
    desired_g = desired[:, :n_nm].reshape(n_rows * n_full, prune_m)
    alive_g = mask_g.sum(dim=1)
    over_g = alive_g > prune_n
    under_g = alive_g < prune_n
    budget_over = (alive_g - prune_n).clamp(min=0)
    budget_under = (prune_n - alive_g).clamp(min=0)

    prune_cand_g = mask_g & ~desired_g
    revive_cand_g = (~mask_g) & desired_g

    prune_key = torch.where(prune_cand_g, imp_g, imp_g.new_full((), float('inf')))
    prune_rank = prune_key.argsort(dim=1).argsort(dim=1)  # rank0 = lowest importance
    prune_elig_g = prune_cand_g & over_g.unsqueeze(1) & (prune_rank < budget_over.unsqueeze(1))

    revive_key = torch.where(revive_cand_g, imp_g, imp_g.new_full((), float('-inf')))
    revive_rank = (-revive_key).argsort(dim=1).argsort(dim=1)  # rank0 = highest importance
    revive_elig_g = revive_cand_g & under_g.unsqueeze(1) & (revive_rank < budget_under.unsqueeze(1))

    prune_full = torch.zeros(n_rows, imp.shape[1], dtype=torch.bool, device=imp.device)
    revive_full = torch.zeros(n_rows, imp.shape[1], dtype=torch.bool, device=imp.device)
    prune_full[:, :n_nm] = prune_elig_g.reshape(n_rows, n_nm)
    revive_full[:, :n_nm] = revive_elig_g.reshape(n_rows, n_nm)
    return prune_full, revive_full


def _pgd_nm_directional(imps, masks, desired, prune_n, prune_m, shapes=None):
    """FSDP-aware wrapper -- see _pgd_nm_directional_2d. Same gather/scatter
    shim as _pgd_nm_pre_target (needs `desired` gathered too, unlike
    _pgd_nm_post_target/_pgd_nm_group_finished which don't take a desired
    arg)."""
    if shapes:
        import torch.distributed as _dist
        _rank = _dist.get_rank()
    prune_out, revive_out = {}, {}
    names = shapes.keys() if shapes else imps.keys()
    for n in names:
        mask = masks[n]
        imp = imps.get(n)
        if imp is None:
            imp = mask.new_zeros(mask.shape, dtype=torch.float32)
        des = desired.get(n) if desired is not None else None
        if des is None:
            des = mask
        shape = shapes.get(n) if shapes else None
        if shape is not None and tuple(shape) != tuple(imp.shape):
            full_imp, full_mask = _fsdp_nm_reconstruct(imp, mask, shape)
            full_des_flat = _fsdp_gather_flat(des.reshape(-1))
            full_des = full_des_flat.reshape(tuple(shape))
            full_p, full_r = _pgd_nm_directional_2d(full_imp, full_mask, full_des, prune_n, prune_m)
            prune_out[n] = _fsdp_nm_scatter_back(full_p, imp.numel(), _rank, imp.shape)
            revive_out[n] = _fsdp_nm_scatter_back(full_r, imp.numel(), _rank, imp.shape)
            continue
        if imp.dim() < 2 or imp.numel() == 0:
            prune_out[n] = torch.zeros_like(mask)
            revive_out[n] = torch.zeros_like(mask)
            continue
        prune_out[n], revive_out[n] = _pgd_nm_directional_2d(imp, mask, des, prune_n, prune_m)
    return prune_out, revive_out


def _pgd_nm_finished_swap_2d(imp, mask, desired, prune_n, prune_m):
    """For groups ALREADY at exactly prune_n alive: identify which need a
    swap (mask_g != desired_g) and score each such group as ONE ATOMIC UNIT
    -- selection must happen at the group level (via the score returned
    here), never independently on the flattened prune/revive coordinates,
    or a selected prune from one group could pair with a selected revive
    from an unrelated group and break BOTH (a real bug in an earlier
    version of this code, caught by user review: independent top-k on
    flat _prune_cand_fin/_revive_cand_fin pools has no per-group coupling
    at all, even though global prune count == global revive count).

    Returns (group_score [n_rows*n_full], prune_mask_g, revive_mask_g
    [both n_rows*n_full, prune_m] bool, n_rows, n_full, n_nm) -- group
    index order is row-major, caller reshapes back for expansion. score is
    -inf for groups that are unfinished or already match desired (not
    swap candidates)."""
    n_rows, n_cols = imp.shape
    n_full = n_cols // prune_m
    n_nm = n_full * prune_m
    imp_g = imp[:, :n_nm].reshape(n_rows * n_full, prune_m)
    mask_g = mask[:, :n_nm].reshape(n_rows * n_full, prune_m)
    desired_g = desired[:, :n_nm].reshape(n_rows * n_full, prune_m)
    alive_g = mask_g.sum(dim=1)
    finished_g = alive_g == prune_n
    prune_mask_g = mask_g & ~desired_g
    revive_mask_g = (~mask_g) & desired_g
    needs_swap_g = prune_mask_g.any(dim=1) | revive_mask_g.any(dim=1)
    eligible_g = finished_g & needs_swap_g

    prune_cnt = prune_mask_g.sum(dim=1).clamp(min=1).float()
    revive_cnt = revive_mask_g.sum(dim=1).clamp(min=1).float()
    prune_imp_mean = (imp_g * prune_mask_g.float()).sum(dim=1) / prune_cnt
    revive_imp_mean = (imp_g * revive_mask_g.float()).sum(dim=1) / revive_cnt
    # "value" of doing this group's swap in one shot: how much more
    # important is what we'd revive than what we'd prune -- bigger gap =
    # safer/more-confident swap, accepted first under the self-KL budget.
    score_g = revive_imp_mean - prune_imp_mean
    score_g = torch.where(eligible_g, score_g, score_g.new_full((), float('-inf')))
    return score_g, prune_mask_g, revive_mask_g, n_rows, n_full, n_nm


def _pgd_nm_finished_swap_build(imps, masks, desired, prune_n, prune_m, shapes=None):
    """FSDP-aware wrapper around _pgd_nm_finished_swap_2d. Unlike
    _pgd_nm_directional's shim (gather -> compute -> scatter-back
    per-tensor immediately), this returns the per-name (score_g,
    prune_mask_g, revive_mask_g, meta) BEFORE any scatter-back, because
    group-level top-k selection (done by the caller, across ALL names at
    once) must run on the same full-shape group layout that produced the
    scores -- scattering back to local shards first would fragment groups
    across ranks and make the group index bookkeeping impossible to
    reassemble correctly. meta[n] = (n_rows, n_full, n_nm, full_shape,
    local_numel, local_shape) for the caller's later scatter-back of the
    FINAL (post-selection) coordinate-level result.
    FSDP note: this means every rank computes the FULL group score tensor
    identically post-gather -- fine for correctness (all ranks agree) but
    not yet perf-optimized for large FSDP world sizes; validated so far
    only on the non-FSDP (single-GPU) path."""
    if shapes:
        import torch.distributed as _dist
        _rank = _dist.get_rank()
    scores, prune_g, revive_g, meta = {}, {}, {}, {}
    names = shapes.keys() if shapes else imps.keys()
    for n in names:
        mask = masks[n]
        imp = imps.get(n)
        if imp is None:
            imp = mask.new_zeros(mask.shape, dtype=torch.float32)
        des = desired.get(n) if desired is not None else None
        if des is None:
            des = mask
        shape = shapes.get(n) if shapes else None
        if shape is not None and tuple(shape) != tuple(imp.shape):
            full_imp, full_mask = _fsdp_nm_reconstruct(imp, mask, shape)
            full_des_flat = _fsdp_gather_flat(des.reshape(-1))
            full_des = full_des_flat.reshape(tuple(shape))
            s, p, r, nr, nf, nnm = _pgd_nm_finished_swap_2d(full_imp, full_mask, full_des, prune_n, prune_m)
            scores[n] = s; prune_g[n] = p; revive_g[n] = r
            meta[n] = (nr, nf, nnm, tuple(shape), imp.numel(), tuple(imp.shape))
            continue
        if imp.dim() < 2 or imp.numel() == 0:
            continue
        s, p, r, nr, nf, nnm = _pgd_nm_finished_swap_2d(imp, mask, des, prune_n, prune_m)
        scores[n] = s; prune_g[n] = p; revive_g[n] = r
        meta[n] = (nr, nf, nnm, tuple(imp.shape), imp.numel(), tuple(imp.shape))
    return scores, prune_g, revive_g, meta


def _pgd_topk_groups_from_scores(scores, k, dev, use_fsdp):
    """Select the k highest-scoring groups (FSDP-global count) from `scores`
    (dict of per-name 1D float tensors, -inf for ineligible groups). Plain
    LINEAR-space bisection, unlike _pgd_topk_mask_from_vals's log-space
    search -- scores here are importance DIFFERENCES (revive_imp - prune_imp,
    can be negative), so log-space doesn't apply; the heavy-tailed-range
    problem log-space was built to fix is a property of raw non-negative
    importance values, not of this bounded difference. Same tie-breaking
    (random thinning) as _pgd_topk_mask_from_vals, for the same reason --
    many groups can share the exact same score (e.g. 0.0 for untouched
    embedding-table groups early in training)."""
    if use_fsdp:
        import torch.distributed as _dist
    if k <= 0:
        return {n: torch.zeros_like(v, dtype=torch.bool) for n, v in scores.items()}
    finite_lo = min((v[torch.isfinite(v)].min().item() if torch.isfinite(v).any() else 0.0) for v in scores.values())
    finite_hi = max((v[torch.isfinite(v)].max().item() if torch.isfinite(v).any() else 0.0) for v in scores.values())
    if use_fsdp:
        _lo_t = torch.tensor(finite_lo, device=dev)
        _hi_t = torch.tensor(finite_hi, device=dev)
        _dist.all_reduce(_lo_t, op=_dist.ReduceOp.MIN)
        _dist.all_reduce(_hi_t, op=_dist.ReduceOp.MAX)
        finite_lo, finite_hi = _lo_t.item(), _hi_t.item()
    thr_lo, thr_hi = finite_lo, finite_hi
    cnt_t = torch.zeros(1, dtype=torch.long, device=dev)
    for _ in range(48):
        mid = (thr_lo + thr_hi) / 2.0
        cnt_t.zero_()
        for v in scores.values():
            cnt_t += (v >= mid).sum(dtype=torch.long)
        if use_fsdp:
            _dist.all_reduce(cnt_t, op=_dist.ReduceOp.SUM)
        if cnt_t.item() > k:
            thr_lo = mid
        else:
            thr_hi = mid
    thr = thr_hi
    sel = {n: (v >= thr) for n, v in scores.items()}
    tied = {n: (v == thr) for n, v in scores.items()}
    n_tied_t = torch.zeros(1, dtype=torch.long, device=dev)
    for t in tied.values():
        n_tied_t += t.sum(dtype=torch.long)
    if use_fsdp:
        _dist.all_reduce(n_tied_t, op=_dist.ReduceOp.SUM)
    n_tied = int(n_tied_t.item())
    if n_tied > 0:
        n_strict_t = torch.zeros(1, dtype=torch.long, device=dev)
        for n in scores:
            n_strict_t += (sel[n] & ~tied[n]).sum(dtype=torch.long)
        if use_fsdp:
            _dist.all_reduce(n_strict_t, op=_dist.ReduceOp.SUM)
        n_strict = int(n_strict_t.item())
        n_needed = max(0, k - n_strict)
        if n_needed < n_tied:
            keep_frac = n_needed / n_tied
            sel = {n: (sel[n] & ~tied[n]) | (tied[n] & (torch.rand_like(v) < keep_frac))
                   for n, v in scores.items()}
    return sel


def _pgd_nm_expand_group_sel(sel_g, prune_g, revive_g, meta, rank=None):
    """Expand a group-level selection (from _pgd_topk_groups_from_scores)
    back into full coordinate-level prune/revive boolean masks, scattering
    back to local FSDP shards if meta indicates this tensor was gathered
    (full_shape != local_shape)."""
    prune_out, revive_out = {}, {}
    for n, sel in sel_g.items():
        n_rows, n_full, n_nm, full_shape, local_numel, local_shape = meta[n]
        p_g = prune_g[n] & sel.unsqueeze(1)
        r_g = revive_g[n] & sel.unsqueeze(1)
        p_full = torch.zeros(full_shape, dtype=torch.bool, device=sel.device)
        r_full = torch.zeros(full_shape, dtype=torch.bool, device=sel.device)
        p_full[:, :n_nm] = p_g.reshape(n_rows, n_nm)
        r_full[:, :n_nm] = r_g.reshape(n_rows, n_nm)
        if tuple(full_shape) != tuple(local_shape):
            prune_out[n] = _fsdp_nm_scatter_back(p_full, local_numel, rank, local_shape)
            revive_out[n] = _fsdp_nm_scatter_back(r_full, local_numel, rank, local_shape)
        else:
            prune_out[n] = p_full
            revive_out[n] = r_full
    return prune_out, revive_out


def _pgd_nm_check_invariant(before_masks, after_masks, prune_n, prune_m, step, shapes=None):
    """Hard runtime check (always on, not a debug flag -- this exists
    specifically because the 3-phase gmp_pgd_grow_to_target N:M design had
    two real per-group-atomicity bugs caught by user review before this
    check was added, and prose claims of correctness were not trusted
    afterward, for good reason). Verifies, for every N:M group in every
    prunable tensor, that ONE PGD step's Phase A+B+C changes never made a
    group's alive-count move further from prune_n than it started, and that
    a group already exactly at prune_n before the step is STILL exactly at
    prune_n after -- i.e. the invariant |a_g - prune_n| is non-increasing,
    and finished groups stay finished. Raises RuntimeError (not a bare
    `assert`, which can be compiled out with -O) with the offending tensor
    name, group index, and before/after alive-counts on any violation, so a
    breakage crashes the run loudly instead of silently shipping a
    non-N:M checkpoint."""
    if shapes:
        import torch.distributed as _dist
    names = shapes.keys() if shapes else before_masks.keys()
    for n in names:
        b_mask = before_masks[n]
        a_mask = after_masks[n]
        shape = shapes.get(n) if shapes else None
        if shape is not None and tuple(shape) != tuple(b_mask.shape):
            _dummy = b_mask.new_zeros(b_mask.shape, dtype=torch.float32)
            _, b_full = _fsdp_nm_reconstruct(_dummy, b_mask, shape)
            _, a_full = _fsdp_nm_reconstruct(_dummy, a_mask, shape)
        else:
            b_full, a_full = b_mask, a_mask
        if b_full.dim() < 2 or b_full.numel() == 0:
            continue
        n_rows, n_cols = b_full.shape
        n_full = n_cols // prune_m
        n_nm = n_full * prune_m
        if n_nm == 0:
            continue
        b_g = b_full[:, :n_nm].reshape(n_rows * n_full, prune_m).sum(dim=1)
        a_g = a_full[:, :n_nm].reshape(n_rows * n_full, prune_m).sum(dim=1)
        b_dist = (b_g - prune_n).abs()
        a_dist = (a_g - prune_n).abs()
        worse = a_dist > b_dist
        if worse.any():
            idx = int(torch.nonzero(worse, as_tuple=False)[0].item())
            raise RuntimeError(
                f"[pgd_nm_invariant] step={step} tensor={n} group={idx}: "
                f"alive-count moved AWAY from prune_n={prune_n} "
                f"(before={int(b_g[idx].item())}, after={int(a_g[idx].item())}) -- "
                f"a Phase A/B/C bug let a group's |alive-prune_n| increase.")
        was_finished = b_g == prune_n
        broke_finished = was_finished & (a_g != prune_n)
        if broke_finished.any():
            idx = int(torch.nonzero(broke_finished, as_tuple=False)[0].item())
            raise RuntimeError(
                f"[pgd_nm_invariant] step={step} tensor={n} group={idx}: "
                f"was already a valid N:M group (alive={prune_n}) but ISN'T "
                f"anymore (after={int(a_g[idx].item())}) -- Phase C's atomic "
                f"pairing didn't hold.")


def _apply_mask(param, mask, ste=False):
    if ste:
        # STE mode: param.data is never hard-reset -- masking is enforced only
        # in the forward pass (see _STEMaskFn / install_ste_forward_hooks), so
        # Adam sees a true, continuously-compounding trajectory for masked
        # weights instead of a one-step-from-zero snapshot every step.
        return
    with torch.no_grad():
        param.data.mul_(mask)


class _STEMaskFn(torch.autograd.Function):
    """Straight-through estimator for masked weights: forward computes
    weight*mask (so sparsity is respected in the actual computation), but
    backward passes the gradient straight through unmasked to `weight` --
    the real parameter is never touched/reset, so Adam naturally accumulates
    the true trajectory underneath the mask with no manual replay needed."""

    @staticmethod
    def forward(ctx, weight, mask):
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def install_ste_forward_hooks(model, maskmgr):
    """Patch each masked nn.Linear's forward to route through _STEMaskFn,
    reading maskmgr.masks[name] fresh on every call (so later mask updates --
    in-place item writes or whole-dict reassignment -- are picked up with no
    re-registration). Opt-in only (--gmp_ste=true) -- forward is 100% stock
    HF Linear when this is never called."""
    name_to_module = dict(model.named_modules())
    for full_name in maskmgr.named_params:
        assert full_name.endswith('.weight')
        module = name_to_module[full_name[:-len('.weight')]]

        def _make_forward(mask_name):
            def _ste_forward(self, x):
                w = _STEMaskFn.apply(self.weight, maskmgr.masks[mask_name])
                return F.linear(x, w, self.bias)
            return _ste_forward

        module.forward = types.MethodType(_make_forward(full_name), module)


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

    def __init__(self, named_params, fsdp_model=None, prune_n=0, prune_m=0, pruning_scope='global', ste=False,
                 named_shapes=None):
        self.named_params = named_params
        self.named_shapes = named_shapes  # {name: (out_features, in_features)}, FSDP-storage-agnostic; see _fsdp_nm_reconstruct
        self.prune_n = prune_n  # N for N:M semi-structured sparsity (0 = unstructured)
        self.prune_m = prune_m  # M for N:M semi-structured sparsity
        self.pruning_scope = pruning_scope  # 'global' or 'layer' (per-layer)
        self.ste = ste  # opt-in STE mode: apply()/update() skip the hard param.data reset (see _apply_mask)
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
    def candidate_masks(self, fisher: 'FisherAccumulator', sparsity: float, fsdp_model=None, block_size=None) -> dict:
        """Compute candidate masks at target sparsity without modifying self.masks or weights.

        Returns a dict {name: bool_tensor} where True=KEEP, same convention as self.masks.

        block_size: only used when self.pruning_scope == 'block' -- groups
        named_params by decoder layer index // block_size (matching the
        SquareHead blockwise anchor spacing, see _squarehead_anchor_layers),
        so growth happens with its own independent threshold PER GROUP
        instead of one pooled global threshold. Passed at call time (not
        stored at construction) since block_size changes during training as
        --gmp_blockwise_squarehead widens it.
        """
        if sparsity <= 0.0:
            return {n: m.clone() for n, m in self.masks.items()}

        use_fsdp = _FSDP_AVAILABLE and fsdp_model is not None

        if self.pruning_scope == 'block' and self.prune_n == 0:
            if use_fsdp:
                raise NotImplementedError("gmp_pruning_scope='block' is not yet implemented under FSDP")
            if block_size is None:
                raise ValueError("gmp_pruning_scope='block' requires block_size to be passed to candidate_masks")
            return self._block_candidate_masks(fisher, sparsity, block_size)

        if self.prune_n > 0 and self.prune_m > 0:
            import torch.distributed as _dist
            new_masks = {}
            for name, param in self.named_params.items():
                imp = fisher.importance(name, param)
                if torch.isnan(imp).any() or torch.isinf(imp).any():
                    new_masks[name] = self.masks[name].clone()
                    continue
                current_pruned = ~self.masks[name]
                _shape = self.named_shapes.get(name) if self.named_shapes else None
                if use_fsdp and _shape is not None and tuple(_shape) != tuple(imp.shape):
                    # Local shard isn't the param's true logical shape (classic
                    # FSDP1 flat-buffer sharding, verified to not even respect
                    # per-parameter or row/col boundaries -- see
                    # _fsdp_gather_flat; param.shape itself is NOT reliable
                    # here either, see _fsdp_nm_reconstruct) -- reconstruct
                    # the full 2D tensor so the group-of-prune_m logic below
                    # sees real structure instead of silently degrading to
                    # its dim<2 fallback.
                    _full_imp, _full_pruned = _fsdp_nm_reconstruct(imp, current_pruned, _shape)
                    _full_new = self._nm_mask(_full_imp, _full_pruned, sparsity)
                    new_masks[name] = _fsdp_nm_scatter_back(
                        _full_new, imp.numel(), _dist.get_rank(), imp.shape)
                else:
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
    def _param_block_group(self, name: str, block_size: int) -> int:
        """Which block group a param belongs to, matching the SquareHead
        anchor spacing (decoder layer index // block_size). Non-layer params
        (embeddings, final norm, lm_head) get group -1, pooled together
        since they don't belong to any single anchor's span.

        If the total layer count doesn't divide evenly by block_size, the
        leftover tail is folded into the last FULL group instead of forming
        its own small orphan group (e.g. 28 layers, block_size=8 -> groups
        [0-7],[8-15],[16-27], not [0-7],[8-15],[16-23],[24-27])."""
        m = re.search(r'\.layers\.(\d+)\.', name)
        if not m:
            return -1
        gid = int(m.group(1)) // block_size
        num_layers = getattr(self, '_num_decoder_layers', None)
        if num_layers:
            last_gid = (num_layers // block_size) - 1
            if last_gid >= 0:
                gid = min(gid, last_gid)
        return gid

    @torch.no_grad()
    def _block_candidate_masks(self, fisher: 'FisherAccumulator', sparsity: float, block_size: int) -> dict:
        """Non-FSDP only (see candidate_masks). Each block group independently
        hits `sparsity` via its own binary-search threshold -- same math as
        the 'global' branch in candidate_masks, just scoped per group instead
        of pooling every param together."""
        local_imps = {n: fisher.importance(n, p) for n, p in self.named_params.items()}
        groups = {}
        for name in self.named_params:
            groups.setdefault(self._param_block_group(name, block_size), []).append(name)

        new_masks = {}
        for _gid, names in groups.items():
            imp_tensors = [local_imps[n] for n in names]
            if any(torch.isnan(v).any() or torch.isinf(v).any() for v in imp_tensors):
                for n in names:
                    new_masks[n] = self.masks[n].clone()
                continue
            n_total = sum(v.numel() for v in imp_tensors)
            k = int(n_total * sparsity)
            if k == 0:
                for n in names:
                    new_masks[n] = torch.ones_like(local_imps[n], dtype=torch.bool)
                continue
            if k >= n_total:
                for n in names:
                    new_masks[n] = torch.zeros_like(local_imps[n], dtype=torch.bool)
                continue
            lo = min(v.min().item() for v in imp_tensors)
            hi = max(v.max().item() for v in imp_tensors)
            for _ in range(48):
                mid = (lo + hi) / 2.0
                cnt = sum((v <= mid).sum().item() for v in imp_tensors)
                if cnt < k:
                    lo = mid
                else:
                    hi = mid
            for n in names:
                new_masks[n] = local_imps[n] > hi
        return new_masks

    @torch.no_grad()
    def update(self, fisher: 'FisherAccumulator', sparsity: float, fsdp_model=None, block_size=None):
        """Recompute global mask at target sparsity using Fisher importance.

        FSDP note: importance scoring and mask application run on local shards directly.
        For global unstructured pruning, scores are all-gathered across ranks to compute
        a consistent threshold. summon_full_params is NOT used — it causes shape mismatch
        because optimizer exp_avg_sq retains local shard shape while param.data becomes
        full shape inside the context manager.
        """
        if sparsity <= 0.0:
            return
        new_masks = self.candidate_masks(fisher, sparsity, fsdp_model, block_size=block_size)
        self.masks = new_masks
        for name, param in self.named_params.items():
            _apply_mask(param, self.masks[name], ste=self.ste)

    def apply(self, fsdp_model=None):
        """Zero out masked weights (call after every optimizer step).

        Applies masks to local shards directly — FSDP all-gathers params before each
        forward pass, so zeroing local shards is sufficient to enforce sparsity globally.
        In STE mode (self.ste=True) this is a no-op: sparsity is enforced only in the
        forward pass via install_ste_forward_hooks, so param.data is intentionally left
        untouched here.
        """
        for name, param in self.named_params.items():
            _apply_mask(param, self.masks[name], ste=self.ste)

    def current_sparsity(self):
        total = sum(m.numel() for m in self.masks.values())
        zeros = sum((~m).sum().item() for m in self.masks.values())
        try:
            import torch.distributed as _dist
            # world_size > 1 (not just is_initialized()) -- vLLM's in-process
            # LLM() engine initializes the DEFAULT process group as a side
            # effect of its own TP=1 setup even for a single-GPU job with no
            # real multi-rank training, making is_initialized() true when
            # there is nothing to actually reduce across. Worse, that shared
            # default group is also touched by vLLM's own internal (size-1,
            # still real) NCCL usage, which its sleep-mode CuMemAllocator can
            # leave in a bad state -- an all_reduce here was observed to fail
            # ("NCCL WARN Cuda failure 'out of memory'") on every call from
            # step 1 onward in single-GPU runs, silently swallowed by this
            # try/except, until an unwrapped all_reduce elsewhere hit the same
            # corrupted state and crashed the process. A world_size==1
            # all_reduce is a mathematical no-op anyway, so skipping it here
            # changes nothing for real multi-GPU runs and removes an entirely
            # unnecessary (and apparently unsafe) NCCL call for single-GPU ones.
            if _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
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


def _kl_loss(s_logits, t_logits, labels, temperature, topk, reverse=False, chunk_size=0,
             prune_opd=False, prune_opd_drop=0.01, prune_opd_wbase=0.5,
             prune_opd_topk=256, prune_opd_threshold=0.7):
    """Token-level KL divergence on CoT positions (labels != -100).

    reverse=False: forward KL D(T||S) over teacher top-K tokens (default)
    reverse=True:  reverse KL D(S||T) full vocab, always >= 0
    topk used for forward KL and for diag metrics in both modes.

    prune_opd: ports Prune-OPD's (github.com/yangzhch6/Prune-OPD)
    token-reliability weighting from long-horizon on-policy RL distillation
    to our much shorter on-policy KD rollouts. Motivation here is different
    from that paper's (rollout length): our model's own quality visibly
    wobbles right after a mask-growth event (self-KL spikes for 1-3 PGD
    calls before settling, see gmp_pgd_debug_importance_hist / the PGD
    self-KL analysis this session), so a rollout SAMPLED during that
    unstable window is a less trustworthy on-policy KD target than one
    sampled once the mask has settled -- but gmp_onpolicy_kd_lambda
    currently weights every rollout identically regardless of when in the
    mask-growth cycle it was drawn. Once a token's student-top-K stops
    overlapping the teacher's top-K enough ("bad_event": overlap_ratio <
    prune_opd_threshold), every later token in that SAME rollout has its
    loss weight decay by prune_opd_drop per additional violation (monotone
    non-increasing, floored at prune_opd_wbase) -- an unreliable rollout's
    later (compounding-error) tokens count for less, without having to
    label which mask-growth phase produced it. The reliability check uses
    its OWN top-K (prune_opd_topk, default 256) independent of `topk`
    (which gates the forward-KL loss itself and may be 0 = full vocab) --
    matches Prune-OPD's own decoupling of its overlap metric's
    log_prob_top_k=256 from the RL objective, and its published
    threshold=0.7 / w_drop=0.01 / w_base=0.5 defaults (from
    experiments_scripts/prune-opd-*.sh, not the library fallback in
    prune_opd.py). Applied regardless of `reverse` or `topk`.

    chunk_size > 0 (--gmp_kl_chunk_size): process the sequence dimension in
    chunks of this many tokens instead of materializing the full (B,T,V)
    log-softmax tensors at once -- at seqlen=8192 x ~152k vocab each such
    tensor is already ~5GB in fp32 regardless of GPU count (this computation
    isn't FSDP-sharded), which fits fine on an 80GB card but OOMs a 40GB one.
    Chunking trades a Python-level loop (same total FLOPs) for a bounded
    peak. diag metrics are skipped when chunked (informational only, and
    skipping them is itself part of what keeps the peak down).
    """
    # align: logit at position t predicts token at t+1
    s_logits = s_logits[:, :-1, :]       # (B, T-1, V) (batch size, seq len-1, vocab size)
    t_logits = t_logits[:, :-1, :]
    labels   = labels[:, 1:]             # (B, T-1)
    mask = (labels != -100).float()
    denom = mask.sum().clamp(min=1)
    if mask.sum() == 0:
        return s_logits.new_tensor(0.0), {}

    if chunk_size > 0 and topk == 0 and s_logits.shape[1] > chunk_size:
        kl_chunks = []
        for start in range(0, s_logits.shape[1], chunk_size):
            end = start + chunk_size
            s_c = F.log_softmax(s_logits[:, start:end] / temperature, dim=-1)
            t_c = F.log_softmax(t_logits[:, start:end] / temperature, dim=-1)
            if reverse:
                kl_chunks.append(F.kl_div(t_c, s_c, log_target=True, reduction='none').sum(dim=-1))
            else:
                kl_chunks.append(F.kl_div(s_c, t_c, log_target=True, reduction='none').sum(dim=-1))
            del s_c, t_c
        kl = torch.cat(kl_chunks, dim=1)
        loss = (kl * mask).sum() / denom
        return loss, {}

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
    weight = None
    if topk > 0:
        with torch.no_grad():
            t_topk_idx = t_logits.topk(topk, dim=-1).indices
            s_topk_idx = s_logits.topk(topk, dim=-1).indices
            overlap = (s_topk_idx.unsqueeze(-1) == t_topk_idx.unsqueeze(-2)).any(dim=-1)
            overlap_ratio = overlap.float().mean(dim=-1)  # (B, T-1), per-token, in [0,1]
            diag["kd/overlap_ratio"] = ((overlap_ratio * mask).sum() / denom).item()
            s_logp_s = s_logp_full.gather(-1, s_topk_idx)
            t_logp_t = t_logp_full.gather(-1, t_topk_idx)
            s_ent = -(s_logp_s.exp() * s_logp_s).sum(dim=-1)
            t_ent = -(t_logp_t.exp() * t_logp_t).sum(dim=-1)
            diag["kd/entropy_gap"] = (((s_ent - t_ent).abs() * mask).sum() / denom).item()

    if prune_opd:
        # Prune-OPD-style monotone-decay reliability weight -- uses its OWN
        # top-K (prune_opd_topk), independent of `topk`/the diagnostic block
        # above, so this fires even when the loss itself is full-vocab
        # (topk=0). See docstring for the threshold/drop/wbase defaults.
        with torch.no_grad():
            pk = min(prune_opd_topk, t_logits.shape[-1])
            t_topk_idx_p = t_logits.topk(pk, dim=-1).indices
            s_topk_idx_p = s_logits.topk(pk, dim=-1).indices
            overlap_p = (s_topk_idx_p.unsqueeze(-1) == t_topk_idx_p.unsqueeze(-2)).any(dim=-1)
            overlap_ratio_p = overlap_p.float().mean(dim=-1)  # (B, T-1)
            bad_event = (overlap_ratio_p < prune_opd_threshold) & mask.bool()
            cum_bad = bad_event.float().cumsum(dim=-1)
            weight = (1.0 - prune_opd_drop * cum_bad).clamp(min=0.0, max=1.0) + prune_opd_wbase
            weight = torch.where(mask.bool(), weight, torch.zeros_like(weight))
            diag["kd/prune_opd_overlap_ratio"] = ((overlap_ratio_p * mask).sum() / denom).item()
            diag["kd/prune_opd_weight_mean"] = ((weight * mask).sum() / denom).item()
            diag["kd/prune_opd_bad_frac"] = (bad_event.sum(dim=-1) > 0).float().mean().item()

            # WHERE reliability first breaks down within a rollout, as a
            # fraction of that rollout's valid length -- tracks whether the
            # "safe" prefix (before the model starts drifting from the
            # teacher) shrinks/grows across training/mask-growth stages.
            # 1.0 = never broke down (or empty rollout) this micro-batch.
            has_bad = bad_event.any(dim=-1)
            first_bad_idx = torch.where(has_bad, bad_event.float().argmax(dim=-1).float(),
                                         mask.sum(dim=-1).float())
            seq_valid_len = mask.sum(dim=-1).clamp(min=1).float()
            first_bad_frac = (first_bad_idx / seq_valid_len).clamp(max=1.0)
            diag["kd/prune_opd_first_bad_frac"] = first_bad_frac.mean().item()

    if weight is not None:
        wdenom = weight.sum().clamp(min=1e-6)
        loss = (kl * weight).sum() / wdenom
    else:
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


def _opkd_flatten_pool_batches(pool_batches: list) -> tuple:
    """Flatten dataloader batches (each holding up to gmp_batch_size prompts,
    all sharing one prompt_len) into one vLLM input per individual prompt,
    plus (batch_idx, row_idx, prompt_len) metadata to pair each vLLM output
    back to its row of origin. Needed because indexing row 0 only (the old
    behavior) silently dropped every other row in the batch when
    gmp_batch_size > 1, then mismatched shapes when re-assembling full_seq
    against the whole batch's input_ids.
    """
    from vllm.inputs import TokensPrompt as _TokensPrompt
    vllm_inputs = []
    flat_meta = []
    for _bi, _pb in enumerate(pool_batches):
        _plen = int(_pb['prompt_len'].item())
        _bsz = _pb['input_ids'].shape[0]
        for _ri in range(_bsz):
            vllm_inputs.append(_TokensPrompt(prompt_token_ids=_pb['input_ids'][_ri][:_plen].tolist()))
            flat_meta.append((_bi, _ri, _plen))
    return vllm_inputs, flat_meta


def _opkd_build_pool_from_outputs(pool_batches: list, flat_meta: list, vllm_outs: list) -> list:
    """Pair flattened vLLM outputs (see _opkd_flatten_pool_batches) back to
    their originating (batch, row), producing one {"full_seq", "prompt_len"}
    entry per individual prompt regardless of gmp_batch_size."""
    pool = []
    for (_bi, _ri, _plen), _vo in zip(flat_meta, vllm_outs):
        _p_ids = pool_batches[_bi]['input_ids'][_ri:_ri + 1, :_plen].cpu()
        _gen_ids = torch.tensor([_vo.outputs[0].token_ids], dtype=torch.long)
        _full_seq = torch.cat([_p_ids, _gen_ids], dim=1)
        pool.append({"full_seq": _full_seq, "prompt_len": _plen})
    return pool


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


def _pgd_kl_calib_batch(prompt_iter, n: int, seqlen: int, device: str) -> dict:
    """Build a small, short calibration batch for --gmp_pgd_kl_budget's
    per-step self-KL check -- deliberately cheap (few short sequences), NOT
    the real training batch, since this only needs to be a fast proxy signal
    re-measured every PGD step, not a faithful behavior reproduction."""
    seqs = []
    for _ in range(n):
        b = next(prompt_iter)
        seqs.append(b['input_ids'][:, :seqlen])
    max_len = max(s.shape[1] for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    attn   = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.shape[1]
        padded[i, :L] = s[0]
        attn[i, :L]   = 1
    return {'input_ids': padded.to(device), 'attention_mask': attn.to(device)}


@torch.no_grad()
def _compute_tr_kl(model: nn.Module, cal_batch: dict, cand_masks: dict,
                   maskmgr: 'GradualMaskManager', device: str,
                   kl_reduce: str = 'mean', kl_quantile: float = 0.95,
                   ref_cache: dict = None) -> float:
    """Compute KL(old || cand) over valid token positions.

    kl_reduce: 'mean' (default) or 'quantile' (uses kl_quantile percentile).
    cal_batch may come from prompt_iter (has 'labels') or from OPKD pool
    (no 'labels' — all non-padding positions are valid).
    Temporarily applies candidate masks, runs two forward passes, then restores.
    @torch.no_grad() -- callers only ever read the (already-detached) logits/KL
    values, never backward() through them; without it these two full-model
    forward passes built a full autograd graph for nothing, roughly doubling
    the peak activation memory this function needs on top of the training
    step's own forward/backward.

    ref_cache: optional dict for callers that invoke this repeatedly in a
    bisection loop over `cand_masks` with the SAME model/maskmgr.masks/
    cal_batch each time (TR-GMP's own delta search, PGD's kl_budget search).
    `old_lp`/`valid` depend only on maskmgr.masks (the CURRENT mask, unchanged
    across such a loop) and cal_batch, never on cand_masks -- so recomputing
    the "old" forward pass on every iteration is pure waste. Pass the same
    dict across all iterations of one search to compute it once and reuse it;
    pass None (default) for a one-off call.
    """
    input_ids = cal_batch['input_ids'].to(device)
    attn_mask = cal_batch['attention_mask'].to(device)

    if ref_cache is not None and 'old_lp' in ref_cache:
        old_lp = ref_cache['old_lp']
        valid  = ref_cache['valid']
    else:
        # labels may be absent (prompt-only / OPKD rollout batches)
        if 'labels' in cal_batch:
            labels = cal_batch['labels'].to(device)
            valid  = (labels[:, 1:] != -100)  # [B, T-1]
        else:
            valid  = (attn_mask[:, 1:] == 1)  # [B, T-1]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            old_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits.detach()
        old_lp = F.log_softmax(old_logits[:, :-1, :], dim=-1)  # [B, T-1, V] bf16
        del old_logits
        if ref_cache is not None:
            ref_cache['old_lp'] = old_lp
            ref_cache['valid']  = valid

    # Temporarily zero newly-pruned weights (old_mask=True & cand_mask=False).
    # No `.any()` skip-if-empty check here on purpose: boolean-mask indexing
    # with an all-False mask is already a correct no-op (selects/writes zero
    # elements), and `.any()` forces a GPU->CPU sync to read the Python bool
    # -- with ~200 Linear-weight tensors in a 1.7B model, that's ~200 sync
    # round-trips *per call*, measured at ~1.6s total (dominating this
    # function's wall-clock far more than the actual forward pass, which is
    # why truncating the calibration seqlen barely moved the per-call cost).
    # Skipping the check removes those syncs entirely for the same result.
    saved = {}
    for name, param in maskmgr.named_params.items():
        newly_pruned = maskmgr.masks[name] & ~cand_masks[name]
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
    # materializing multiple [B,T,V] float32 tensors simultaneously.
    # (old_lp was already computed above -- fresh or from ref_cache.)
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


def _mc_fisher_named_params(model, named_params, cal_batch, device, nsamples=3):
    """MC Fisher diagonal over `named_params` (the same Linear-weight tensors
    maskmgr/PGD prune): F_ii = mean_n(grad[-log Q(y|h)]_i^2), y ~ Q_theta(.|h).

    Per-SEQUENCE gradients (looping over the batch dim), not a single
    batch-mean-loss backward: F.cross_entropy's default reduction='mean'
    over a whole (B*T) flattened batch computes grad(mean_n loss_n), and
    squaring THAT gives (mean_n g_n)^2, not the target mean_n(g_n^2) --
    opposite-sign per-token/per-sequence gradients cancel in the mean before
    squaring, which can make a genuinely important (high-curvature)
    coordinate look ~0 just because different sequences pull it different
    directions. This under-corrected version was ported faithfully from the
    CPU toy's own mc_fisher_head, which has the identical bug (confirmed:
    toy's "before" correlation 0.432 vs "after" per-sequence-fix 0.735-0.867
    on the same rollout). Looping per-sequence (not per-token, which would
    be B*T backward passes) is the practical middle ground -- still avoids
    the cross-example cancellation, which is where the toy saw the large
    effect. nsamples fresh y~Q_theta draws x B sequences = nsamples*B
    backward passes total.

    Accumulator lives on CPU, not GPU: keeping a persistent fp32 buffer the
    size of every prunable Linear weight resident on-device for the whole
    N-sample loop was itself a meaningful chunk of the OOMs this pilot hit
    (on top of the already-tight co-located-vLLM + multi-loss training
    memory). Moving each sample's g^2 to CPU immediately after computing it
    keeps peak VRAM close to a single batch-size-1 backward pass, at the
    cost of a small per-sample CPU transfer -- matches the reference
    scout_stage_local_fisher bundle's design.
    """
    accum = {name: torch.zeros(p.shape, dtype=torch.float32, device='cpu') for name, p in named_params.items()}
    input_ids = cal_batch['input_ids'].to(device)
    attn_mask = cal_batch.get('attention_mask')
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    B = input_ids.shape[0]
    was_training = model.training
    model.train()
    n_total = 0
    for _ in range(nsamples):
        for b in range(B):
            ids_b = input_ids[b:b + 1]
            mask_b = attn_mask[b:b + 1] if attn_mask is not None else None
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids_b, attention_mask=mask_b).logits[:, :-1, :]
            with torch.no_grad():
                probs = logits.float().softmax(-1)
                y = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
            loss.backward()
            for name, p in named_params.items():
                if p.grad is not None:
                    accum[name] += (p.grad.detach().float() ** 2).cpu()
            del logits, probs, y, loss
            model.zero_grad(set_to_none=True)
            n_total += 1
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    for name in accum:
        accum[name] /= max(n_total, 1)
    return accum


def saliency_snapshot_diagnostic(model, maskmgr, fisher, cal_batch_on, cal_batch_ref, device,
                                  k=4096, mc_nsamples=3, use_wandb=False, global_step=0):
    """One-shot, single-timepoint comparison of candidate saliency estimators
    (see scout_cpu_saliency_poc.zip / the CPU toy this mirrors): each
    candidate proposes the same k lowest-scored (globally, across all
    maskmgr.named_params) coordinates for pruning, and we measure the ACTUAL
    self-KL each proposal would cause via _compute_tr_kl (same primitive PGD's
    own kl-budget bisection uses). No EMA-over-stages here (needs persisted
    state across multiple mask_interval boundaries) -- this is the cheap
    single-snapshot pass meant to decide whether a full multi-stage run is
    worth it at all.

    cal_batch_on: on-policy rollout batch (student's own generations)
    cal_batch_ref: fixed-reference batch (from the standard training data)
    """
    named_params = maskmgr.named_params
    logging.info(f"[saliency_diag] computing candidate scores (k={k}, mc_nsamples={mc_nsamples}) at step={global_step}")

    def _measure(score, cname):
        # global top-k LOWEST score across all named_params -> proposed prune set.
        # Mask out ALREADY-PRUNED positions (maskmgr.masks==False) first: at
        # 70% sparsity most positions are already zero, and every candidate
        # score (magnitude, Fisher*w^2) is trivially near-0 there too, so an
        # unmasked global bottom-k is dominated by the huge already-dead pool
        # and "prunes" things that are already pruned -- a near no-op that
        # measured actual_kl~=0.000000 for every candidate before this fix.
        # Only ALIVE positions can be meaningfully proposed for pruning.
        masked_score = {name: torch.where(maskmgr.masks[name], score[name], score[name].new_full((), float('inf')))
                         for name in named_params}
        flat_scores = torch.cat([masked_score[name].reshape(-1) for name in named_params])
        del masked_score
        flat_sizes = [score[name].numel() for name in named_params]
        offsets = [0]
        for sz in flat_sizes:
            offsets.append(offsets[-1] + sz)
        k_eff = min(k, flat_scores.numel())
        lowest_idx = torch.topk(flat_scores, k=k_eff, largest=False).indices
        del flat_scores
        cand_masks = {name: maskmgr.masks[name].clone() for name in named_params}
        for name, off0, off1 in zip(named_params, offsets[:-1], offsets[1:]):
            in_range = (lowest_idx >= off0) & (lowest_idx < off1)
            sel = lowest_idx[in_range] - off0
            if sel.numel() == 0:
                continue
            flat_mask = cand_masks[name].reshape(-1)
            flat_mask[sel] = False  # candidate proposes pruning these (mask=False -> pruned)
            cand_masks[name] = flat_mask.reshape(cand_masks[name].shape)
        del lowest_idx
        kl, _ = _compute_tr_kl(model, cal_batch_on, cand_masks, maskmgr, str(device))
        del cand_masks
        logging.info(f"[saliency_diag]   {cname:<28s} actual_kl={kl:.6f}")
        return kl, k_eff

    # Candidates are built and measured ONE AT A TIME, freed immediately after
    # (torch.cuda.empty_cache() between each) -- holding all 5 full-model-sized
    # (per prunable Linear weight) score dicts simultaneously was itself the
    # cause of an OOM on an 80GB card after the (already memory-heavy) 32-sample
    # MC-Fisher pass left the allocator with little headroom.
    results = {}
    k_eff = k

    baseline = {name: fisher.importance(name, p) for name, p in named_params.items()}
    results['baseline_composite_adam'], k_eff = _measure(baseline, 'baseline_composite_adam')
    del baseline; torch.cuda.empty_cache()

    magnitude = {name: p.data.float() ** 2 for name, p in named_params.items()}
    results['magnitude'], _ = _measure(magnitude, 'magnitude')
    del magnitude; torch.cuda.empty_cache()

    # _mc_fisher_named_params accumulates on CPU (see its docstring -- avoids
    # keeping a persistent full-model-sized fp32 buffer on GPU for the whole
    # N-sample loop); move each back to `device` here, once, at combination
    # time, not inside the sampling loop.
    f_ref = {n: t.to(device) for n, t in _mc_fisher_named_params(model, named_params, cal_batch_ref, device, nsamples=mc_nsamples).items()}

    fixed_fisher_score = {name: p.data.float() ** 2 * f_ref[name] for name, p in named_params.items()}
    results['fixed_fisher'], _ = _measure(fixed_fisher_score, 'fixed_fisher')
    del fixed_fisher_score; torch.cuda.empty_cache()

    f_on = {n: t.to(device) for n, t in _mc_fisher_named_params(model, named_params, cal_batch_on, device, nsamples=mc_nsamples).items()}

    on_fisher_score = {name: p.data.float() ** 2 * f_on[name] for name, p in named_params.items()}
    results['on_fisher'], _ = _measure(on_fisher_score, 'on_fisher')
    del on_fisher_score; torch.cuda.empty_cache()

    mix_score = {name: p.data.float() ** 2 * (0.75 * f_ref[name] + 0.25 * f_on[name]) for name, p in named_params.items()}
    del f_ref, f_on; torch.cuda.empty_cache()
    results['mix_fixed75_on25'], _ = _measure(mix_score, 'mix_fixed75_on25')
    del mix_score; torch.cuda.empty_cache()

    if use_wandb:
        import wandb as _wandb
        _wandb.log({f"saliency_diag/{k}": v for k, v in results.items()}, step=global_step)
        _wandb.log({"saliency_diag/k": k_eff}, step=global_step)
    return results


def _spearman_corr(a, b):
    """Spearman rank correlation, no scipy dependency (matches the CPU toy's
    pandas .rank(method='average') closely enough for continuous KL values
    where exact ties are rare -- plain argsort-based ranks, not tie-aware)."""
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.pow(2).sum().sqrt() * rb.pow(2).sum().sqrt()).item()
    return (ra * rb).sum().item() / denom if denom > 0 else float('nan')


def update_onpolicy_fisher_ema_continuous(model, maskmgr, cal_batch_on, device, ema_state, gammas=(0.999,), nsamples=1):
    """Cheap, CONTINUOUS on-policy Fisher EMA update -- call this EVERY
    training step (gmp_saliency_ema_every_step=true) so it has as many
    updates baked in as Adam's own exp_avg_sq by any given step. Mirrors
    Adam's design directly: a single (nsamples=1) MC-Fisher sample per call
    -- cost = batch_size backward passes (e.g. 1-4), NOT the nsamples=8 (32
    backward passes) used for a one-shot diagnostic snapshot -- with the EMA
    itself, accumulated over every step across the whole run, doing the
    variance reduction instead of paying for many MC samples within a
    single call. Default gamma=0.999 matches --gmp_fisher_beta (Adam's own
    beta2) exactly -- only meaningful once this runs every step; at
    mask_interval-boundary-only cadence (32-step gaps) 0.999 barely moves
    and 0.5/0.75 were the sensible choices instead (see the earlier
    mask_interval-boundary version of this idea, superseded by this one
    since 32-step spacing turned out barely deeper than the sparse
    diagnostic-only blend it was meant to replace).

    Deliberately does its OWN separate forward+backward rather than reusing
    the training step's on-policy forward (which would need
    retain_graph=True on the real loss's backward -- extra activation
    memory kept alive, reopening the exact OOM fights this whole recipe
    already went through today). Full cost accepted for now; only worth
    optimizing later if this run proves the comparison is worth it.

    ema_state: dict, mutated in place. Populates/updates
    ema_state['f_on_ema999'] etc (CPU tensors, one per named_param, keyed by
    round(gamma*1000)) directly -- saliency_random_group_correlation_diagnostic
    reads these straight out when present instead of re-deriving its own
    shallower blend from a single previous snapshot.
    """
    named_params = maskmgr.named_params
    f_on = _mc_fisher_named_params(model, named_params, cal_batch_on, device, nsamples=nsamples)  # CPU tensors
    _ema_blend_update(ema_state, f_on, named_params, gammas)


def update_ntp_only_fisher_ema(model, maskmgr, ntp_batch, device, ema_state, gammas=(0.999,)):
    """Decoupled-saliency building block: track an NTP-ONLY gradient-squared
    EMA (real observed labels, NOT a resampled/self-sampled y -- this is the
    genuine empirical Fisher estimator our early derivation showed NTP alone
    gives cleanly, unlike KD/OPD's soft-target loss whose gradient already
    marginalizes out label noise). Meant to run EVERY step regardless of
    whether NTP is part of the actual training objective this run -- the
    point is to let training drop NTP (avoids the forgetting NTP+KD+OPD
    together showed in dense) while mask selection still gets NTP's
    higher-quality curvature signal, by computing NTP's gradient here WITHOUT
    ever handing it to the optimizer.

    Same safe-timing contract as update_onpolicy_fisher_ema_continuous: call
    this only where .grad is guaranteed empty (right after
    optimizer.step()/optimizer.zero_grad()), since it does its own
    independent forward+backward+zero_grad and does not attempt the
    snapshot/restore dance needed to run mid-accumulation safely.

    ntp_batch: a real NTP training batch (input_ids/attention_mask/labels
    with true dataset labels, e.g. next(data_iter)) -- NOT a cal_batch built
    for on-policy/self-sampled Fisher use.
    """
    named_params = maskmgr.named_params
    input_ids = ntp_batch['input_ids'].to(device)
    attn_mask = ntp_batch.get('attention_mask')
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    labels = ntp_batch['labels'].to(device)
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        ntp_loss = out.loss
    ntp_loss.backward()
    f_ntp = {}
    for name, p in named_params.items():
        if p.grad is not None:
            f_ntp[name] = p.grad.detach().float().cpu() ** 2
        else:
            f_ntp[name] = torch.zeros(p.shape, dtype=torch.float32, device='cpu')
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    for gamma in gammas:
        key = f'f_ntp_ema{round(gamma * 1000)}'
        prev = ema_state.get(key)
        ema_state[key] = f_ntp if prev is None else {n: gamma * prev[n] + (1 - gamma) * f_ntp[n] for n in named_params}


def _ema_blend_update(ema_state, f_on, named_params, gammas):
    """Pure EMA-blend step, factored out so both the separate-forward path
    (update_onpolicy_fisher_ema_continuous) and the reused-forward path
    (fisher_grad_from_reused_onpolicy_forward, called inline from the real
    on-policy KD step so it shares that step's own forward pass instead of
    paying for a second one) update the same ema_state the same way."""
    for gamma in gammas:
        key = f'f_on_ema{round(gamma * 1000)}'
        prev = ema_state.get(key)
        if prev is None:
            ema_state[key] = f_on
        else:
            ema_state[key] = {n: gamma * prev[n] + (1 - gamma) * f_on[n] for n in named_params}


def saliency_random_group_correlation_diagnostic(model, maskmgr, fisher, cal_batch_on, cal_batch_ref, device,
                                                   group_size=4096, n_groups=20, mc_nsamples=8,
                                                   use_wandb=False, global_step=0, seed=0, ema_cache=None):
    """Random-group Spearman correlation between each candidate saliency's
    predicted score and ACTUAL pruning-induced KL -- the statistically sound
    counterpart to saliency_snapshot_diagnostic's single bottom-k point
    estimate. That single-point comparison (baseline/magnitude/fixed_fisher/
    on_fisher/mix all landing within ~7% of each other on one k, one step)
    can't distinguish a real ranking difference from noise; this instead
    samples `n_groups` INDEPENDENT random alive-weight groups (same groups
    for every candidate, so the comparison is paired), measures each group's
    real KL once, and correlates that against every candidate's own
    score-sum over that same group. One snapshot now yields a correlation
    coefficient with actual samples behind it, not one number.

    Groups are sampled from ALIVE positions only (same reasoning as
    saliency_snapshot_diagnostic's fix -- an unrestricted pool is dominated
    by already-pruned dead weight at high sparsity, which is a free "always
    zero KL" group for every candidate and would inflate correlation as a
    trivial artifact, not a real ranking result).

    ema_cache: optional dict, reused across repeated calls within the SAME
    training run (e.g. one call per entry in a multi-step
    --gmp_saliency_corr_step="40,80,120" list). If given, also scores
    on_fisher_ema50/on_fisher_ema75 -- the CPU-toy's own best-performing
    candidates (temporal smoothing of the on-policy Fisher across
    mask-growth stages, cutting single-snapshot sampling noise) -- which the
    single-snapshot version of this diagnostic could never test since it has
    no persisted state across calls. First call in a run has no previous
    on-policy Fisher to smooth against, so EMA candidates are skipped that
    time (falls back silently, not an error). Stored on CPU between calls
    for the same reason _mc_fisher_named_params accumulates on CPU.
    """
    named_params = maskmgr.named_params
    logging.info(f"[saliency_corr] sampling {n_groups} random alive-weight groups "
                 f"(size={group_size}) at step={global_step}")

    offsets = [0]
    alive_parts = []
    for name in named_params:
        m = maskmgr.masks[name].reshape(-1)
        alive_parts.append(m)
        offsets.append(offsets[-1] + m.numel())
    alive_flat = torch.cat(alive_parts)
    del alive_parts
    alive_idx = alive_flat.nonzero(as_tuple=True)[0]
    del alive_flat
    n_alive = alive_idx.numel()
    g_eff = min(group_size, n_alive)

    rng = torch.Generator(device='cpu').manual_seed(seed)
    groups = [alive_idx[torch.randperm(n_alive, generator=rng)[:g_eff]] for _ in range(n_groups)]

    # Ground truth: actual KL for each random group, measured ONCE and shared
    # across every candidate below (paired comparison).
    actual_kls = []
    for gi, grp_idx in enumerate(groups):
        cand_masks = {name: maskmgr.masks[name].clone() for name in named_params}
        for name, off0, off1 in zip(named_params, offsets[:-1], offsets[1:]):
            in_range = (grp_idx >= off0) & (grp_idx < off1)
            sel = grp_idx[in_range] - off0
            if sel.numel() == 0:
                continue
            flat_mask = cand_masks[name].reshape(-1)
            flat_mask[sel] = False
            cand_masks[name] = flat_mask.reshape(cand_masks[name].shape)
        kl, _ = _compute_tr_kl(model, cal_batch_on, cand_masks, maskmgr, str(device))
        actual_kls.append(kl)
        del cand_masks
        if (gi + 1) % 5 == 0:
            logging.info(f"[saliency_corr]   ground-truth KL {gi + 1}/{n_groups} measured")
    torch.cuda.empty_cache()

    def _group_sums(score):
        flat = torch.cat([score[name].reshape(-1) for name in named_params])
        sums = [flat[grp_idx].sum().item() for grp_idx in groups]
        del flat
        return sums

    correlations = {}

    baseline = {name: fisher.importance(name, p) for name, p in named_params.items()}
    correlations['baseline_composite_adam'] = _spearman_corr(_group_sums(baseline), actual_kls)
    del baseline; torch.cuda.empty_cache()

    # sqrt(v)*w^2 -- the Sparse Projected Adam (SPA) lr->0 limit ('sqrt_fisher'
    # in AdamFisherTracker.importance): same Adam exp_avg_sq state and same
    # w^2 term as baseline_composite_adam above, differing only in whether the
    # metric weight is v (raw Fisher/Hessian-diagonal approx) or sqrt(v) (the
    # metric Adam's own update actually preconditions by, i.e. its 1/(sqrt(v)+eps)
    # denominator). Reuses the SAME exp_avg_sq state as baseline -- no extra
    # forward/backward, isolates just the v-vs-sqrt(v) exponent choice.
    f_sqrt = {name: fisher.fisher_factor(p) for name, p in named_params.items()}
    sqrt_fisher_score = {name: (f_sqrt[name].clamp(min=0).sqrt() * p.data.float() ** 2
                                 if f_sqrt[name] is not None else p.data.float() ** 2)
                          for name, p in named_params.items()}
    correlations['sqrt_fisher_baseline'] = _spearman_corr(_group_sums(sqrt_fisher_score), actual_kls)
    del f_sqrt, sqrt_fisher_score; torch.cuda.empty_cache()

    magnitude = {name: p.data.float() ** 2 for name, p in named_params.items()}
    correlations['magnitude'] = _spearman_corr(_group_sums(magnitude), actual_kls)
    del magnitude; torch.cuda.empty_cache()

    f_ref = {n: t.to(device) for n, t in _mc_fisher_named_params(model, named_params, cal_batch_ref, device, nsamples=mc_nsamples).items()}
    fixed_fisher_score = {name: p.data.float() ** 2 * f_ref[name] for name, p in named_params.items()}
    correlations['fixed_fisher'] = _spearman_corr(_group_sums(fixed_fisher_score), actual_kls)
    del fixed_fisher_score; torch.cuda.empty_cache()

    f_on_raw = _mc_fisher_named_params(model, named_params, cal_batch_on, device, nsamples=mc_nsamples)  # CPU tensors
    f_on = {n: t.to(device) for n, t in f_on_raw.items()}
    on_fisher_score = {name: p.data.float() ** 2 * f_on[name] for name, p in named_params.items()}
    correlations['on_fisher'] = _spearman_corr(_group_sums(on_fisher_score), actual_kls)
    del on_fisher_score; torch.cuda.empty_cache()

    mix_score = {name: p.data.float() ** 2 * (0.75 * f_ref[name] + 0.25 * f_on[name]) for name, p in named_params.items()}
    del f_ref; torch.cuda.empty_cache()
    correlations['mix_fixed75_on25'] = _spearman_corr(_group_sums(mix_score), actual_kls)
    del mix_score; torch.cuda.empty_cache()

    if ema_cache is not None:
        # Prefer the CONTINUOUSLY-tracked EMA (update_onpolicy_fisher_ema_continuous,
        # called every step throughout training -- as many updates by this
        # step as Adam's own exp_avg_sq) over this function's own shallow
        # single-previous-snapshot blend, which only has as many updates as
        # there have been diagnostic calls (e.g. 2-3 total).
        continuous_keys = sorted(k for k in ema_cache if k.startswith('f_on_ema'))
        if continuous_keys:
            for key in continuous_keys:
                gamma_x1000 = int(key[len('f_on_ema'):])
                cname = f'on_fisher_ema{gamma_x1000}'
                f_ema = {n: t.to(device) for n, t in ema_cache[key].items()}
                ema_score = {name: p.data.float() ** 2 * f_ema[name] for name, p in named_params.items()}
                correlations[cname] = _spearman_corr(_group_sums(ema_score), actual_kls)
                del f_ema, ema_score; torch.cuda.empty_cache()
        else:
            f_on_prev = ema_cache.get('f_on_prev')  # CPU tensors from the previous diag call, or None on the first call
            if f_on_prev is not None:
                logging.info("[saliency_corr]   (no continuous EMA tracker found -- falling back to shallow "
                             "single-previous-snapshot blend, only as deep as the number of diag calls so far)")
                for gamma, cname in ((0.5, 'on_fisher_ema50'), (0.75, 'on_fisher_ema75')):
                    f_ema = {n: (gamma * f_on_prev[n] + (1 - gamma) * f_on_raw[n]).to(device) for n in named_params}
                    ema_score = {name: p.data.float() ** 2 * f_ema[name] for name, p in named_params.items()}
                    correlations[cname] = _spearman_corr(_group_sums(ema_score), actual_kls)
                    del f_ema, ema_score; torch.cuda.empty_cache()
            else:
                logging.info("[saliency_corr]   (first diag call in this run -- no previous on-policy Fisher yet, "
                             "skipping on_fisher_ema50/75 this time)")
            ema_cache['f_on_prev'] = f_on_raw  # already CPU tensors, cheap to keep between calls

        # NTP-ONLY tracked signal (update_ntp_only_fisher_ema) -- the
        # decoupled-saliency test: real-label NTP gradient EMA, tracked
        # independently of whatever the actual training objective is this
        # run (may or may not include NTP itself).
        ntp_keys = sorted(k for k in ema_cache if k.startswith('f_ntp_ema'))
        for key in ntp_keys:
            gamma_x1000 = int(key[len('f_ntp_ema'):])
            cname = f'ntp_only_ema{gamma_x1000}'
            f_ema = {n: t.to(device) for n, t in ema_cache[key].items()}
            ema_score = {name: p.data.float() ** 2 * f_ema[name] for name, p in named_params.items()}
            correlations[cname] = _spearman_corr(_group_sums(ema_score), actual_kls)
            del f_ema, ema_score; torch.cuda.empty_cache()
    del f_on

    for cname, rho in correlations.items():
        logging.info(f"[saliency_corr]   {cname:<28s} spearman_rho={rho:.4f}")

    if use_wandb:
        import wandb as _wandb
        _wandb.log({f"saliency_corr/{k}": v for k, v in correlations.items()}, step=global_step)
        _wandb.log({"saliency_corr/n_groups": n_groups, "saliency_corr/group_size": g_eff}, step=global_step)
    return correlations


def _squarehead_anchor_layers(num_hidden_states, block_size):
    """Anchor-layer indices into a HF `output_hidden_states=True` tuple
    (index 0 = embedding output, skipped; indices 1..num_hidden_states-1 =
    each decoder layer's output) for a given block size.

    block_size=1 anchors EVERY layer (full per-layer distillation, matching
    IST-DASLab/SparseFinetuning's SquareHead exactly -- see
    scripts/train/train_sparse.py's KnowledgeDistillation.apply). Larger
    block_size skips intermediate anchors, giving the layers BETWEEN
    surviving anchors joint freedom to compensate for each other instead of
    each being forced to independently match the dense reference at every
    single point -- the anchor set always keeps the final layer, so the
    weakest possible constraint (block_size >= num_hidden_states-1) still
    checks the model's actual output, degenerating exactly to output-only
    KD (equivalent to plain TR-GMP's behavior)."""
    if block_size <= 1:
        return list(range(1, num_hidden_states))
    anchors = list(range(block_size, num_hidden_states, block_size))
    if not anchors or anchors[-1] != num_hidden_states - 1:
        anchors.append(num_hidden_states - 1)
    return anchors


def _squarehead_loss(student_hidden_states, teacher_hidden_states, anchor_layers, attention_mask):
    """Per-layer normalized-MSE distillation loss at `anchor_layers` only.

    Formula (Kurtic et al., "Sparse Fine-tuning for Inference Acceleration
    of Large Language Models", https://arxiv.org/abs/2310.06927, code at
    github.com/IST-DASLab/SparseFinetuning/blob/main/scripts/train/train_sparse.py):
    each layer's term is (student-teacher)^2 mean, divided by the teacher's
    own squared-activation mean -- normalizes away the fact that different
    layers' activations can have wildly different natural scale, so no
    single large-norm layer dominates the gradient. The paper SUMS this
    over every layer (block_size=1 always); we MEAN over just the current
    anchor set instead, so the loss magnitude stays comparable as the
    anchor count changes with block size (summing would make the loss look
    smaller purely because there are fewer terms, not because anything
    actually improved)."""
    eps = torch.finfo(torch.bfloat16).eps
    valid = attention_mask == 1
    losses = []
    for i in anchor_layers:
        s = student_hidden_states[i][valid].float()
        t = teacher_hidden_states[i][valid].detach().float()
        losses.append((s - t).pow(2).mean() / (t.pow(2).mean() + eps))
    return sum(losses) / len(losses)


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
                   "train/pcg_max_relative_shift": worst_resid_ratio}, step=global_step)


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
                   "train/pcg_seq_max_relative_shift": worst_resid_ratio}, step=global_step)


def _tr_mask_update(maskmgr: 'GradualMaskManager', fisher: 'FisherAccumulator',
                    fsdp_model, model: nn.Module, cal_batch: dict,
                    final_sparsity: float, tr_delta: float,
                    kl_threshold: float, delta_min: float,
                    device: str, max_iters: int = 16,
                    kl_reduce: str = 'mean', kl_quantile: float = 0.95,
                    use_wandb: bool = False, global_step: int = 0,
                    block_size=None) -> tuple:
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
        # world_size > 1, not just is_initialized() -- see the matching note
        # in GradualMaskManager.current_sparsity() above for why is_initialized()
        # alone is unsafe here (vLLM's in-process engine can leave a default
        # process group initialized even for a real single-GPU job).
        if _td.is_available() and _td.is_initialized() and _td.get_world_size() > 1:
            _tr_dist = _td
    except Exception:
        pass
    _tr_rank = _tr_dist.get_rank() if _tr_dist else 0

    current_sp = maskmgr.current_sparsity()

    # For N:M structured sparsity, "reached" must mean EVERY group has
    # actually closed to exactly prune_n alive (true sparsity == exactly
    # prune_n/prune_m, e.g. 0.5 for 2:4) -- unlike unstructured pruning,
    # where being within a fraction of a percent of the target is a fine
    # final state (a few thousand weights either way, spread randomly, is
    # noise), a 2:4 pattern that's still short of target by even 0.5% means
    # a meaningful number of groups have only 0 or 1 (not 2) weights
    # pruned -- not a valid 2:4 pattern at all. Declaring "reached" there
    # freezes the mask for the rest of training under that invalid
    # intermediate state, wasting most of the step budget fine-tuning
    # against a mask that gets silently replaced by the unconditional final
    # hard-cut (`maskmgr.update(..., final_sparsity, ...)` at the very end
    # of training) anyway. So for N:M, check the actual mask structure
    # directly (_nm_fully_closed: every group at exactly prune_n alive) —
    # not the aggregate sparsity fraction, even against a tight/zero
    # tolerance, since a scalar ratio matching is only necessary, not
    # sufficient, for "the pattern is actually done."
    #
    # Unstructured pruning used a 5e-3 (0.5%) undershoot allowance here
    # (pre-existing, since 2026-07-21) -- changed to a hard >= (0% tolerance,
    # never declare "reached" while still short) at the user's request: even
    # though undershoot is far less consequential for unstructured than for
    # N:M (no invalid-pattern concept), there's no reason to allow it either
    # when TR growth can just keep trying instead of quietly falling back on
    # the same unconditional end-of-training hard-cut.
    _is_nm = getattr(maskmgr, 'prune_n', 0) > 0 and getattr(maskmgr, 'prune_m', 0) > 0
    _reach_tol = 0.0

    # Early return: all_reduce so all ranks agree (local shard sparsity can differ).
    _early = int(_nm_fully_closed(maskmgr.masks, maskmgr.prune_n, maskmgr.prune_m,
                                   shapes=(maskmgr.named_shapes if _tr_dist else None))
                 if _is_nm else current_sp >= final_sparsity - _reach_tol)
    if _tr_dist:
        _et = torch.tensor([_early], dtype=torch.int32, device=device)
        _tr_dist.all_reduce(_et, op=_tr_dist.ReduceOp.MAX)
        _early = _et.item()
    if _early:
        maskmgr.apply(fsdp_model)
        return current_sp, tr_delta, True, {}, 0.0

    delta               = tr_delta
    last_accepted_masks = None
    last_accepted_sp    = current_sp
    last_accepted_delta = 0.0
    last_kl             = float('inf')
    prev_accepted       = False  # True if the previous iter was accepted
    # maskmgr.masks (the "old"/reference side of every _compute_tr_kl call
    # below) is never mutated during this loop -- only after it ends -- so
    # the reference forward pass is identical across all `i`. Cache it once
    # instead of recomputing on every iteration (same math, fewer forwards).
    _tr_kl_ref_cache = {}

    for i in range(max_iters):
        if _tr_dist:
            _tr_dist.barrier()
            logging.info(f"  [BARRIER] TR-GMP iter {i} start (rank={_tr_rank})")
        try_sp   = min(current_sp + delta, final_sparsity)
        cand     = maskmgr.candidate_masks(fisher, try_sp, fsdp_model, block_size=block_size)
        if _tr_dist:
            _tr_dist.barrier()
            logging.info(f"  [BARRIER] after candidate_masks iter {i} (rank={_tr_rank})")
        kl, kl_vals = _compute_tr_kl(model, cal_batch, cand, maskmgr, device,
                                      kl_reduce=kl_reduce, kl_quantile=kl_quantile,
                                      ref_cache=_tr_kl_ref_cache)
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
            if (_nm_fully_closed(cand, maskmgr.prune_n, maskmgr.prune_m,
                                  shapes=(maskmgr.named_shapes if _tr_dist else None)) if _is_nm
                    else try_sp >= final_sparsity - _reach_tol):
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
        if _is_nm:
            reached = _nm_fully_closed(maskmgr.masks, maskmgr.prune_n, maskmgr.prune_m,
                                        shapes=(maskmgr.named_shapes if _tr_dist else None))
        else:
            # Also check last_accepted_sp: subsampling threshold can make actual
            # sparsity land slightly below target even when accepted at target.
            reached = new_sp >= final_sparsity - _reach_tol or last_accepted_sp >= final_sparsity - _reach_tol
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

    # last_kl: the accepted iteration's measured KL (0.0 if nothing was ever
    # accepted this call -- no growth happened, so no KL was "spent"). Used
    # by --gmp_pgd_kl_share to derive this window's PGD swap budget from
    # TR-GMP's own already-measured headroom instead of a separate per-step
    # forward-pass-based measurement (see globalprune_gmp).
    _tr_kl_spent = 0.0 if last_accepted_masks is None else last_kl
    return new_sp, new_delta, reached, _mask_delta, _tr_kl_spent


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
    named_shapes = _find_linear_shapes(model)

    # ── FSDP detection ─────────────────────────────────────────────────────────
    fsdp_model = None
    is_fsdp = False
    if _FSDP_AVAILABLE:
        _root = next((m for m in model.modules() if isinstance(m, FSDP)), None)
        if _root is not None:
            fsdp_model = _root
            is_fsdp = True
            logging.info("FSDP detected — enabling summon_full_params for mask updates")

    # Distributed state (DDP or FSDP). world_size > 1, not just
    # is_initialized() -- see the note in GradualMaskManager.current_sparsity()
    # for why is_initialized() alone is unsafe (vLLM's in-process engine can
    # leave a default process group initialized even for a real single-GPU job).
    import torch.distributed as _dist
    is_distributed = _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1
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
    _saliency_corr_steps = sorted({int(s) for s in getattr(FLAGS, 'gmp_saliency_corr_step', '').split(',') if s.strip()})
    _saliency_ema_cache = {}  # persists across repeated saliency_random_group_correlation_diagnostic calls in this run
    _saliency_ema_every_step = getattr(FLAGS, 'gmp_saliency_ema_every_step', False)
    _saliency_ema_nsamples = getattr(FLAGS, 'gmp_saliency_ema_nsamples', 1)
    _saliency_ntp_ema_every_step = getattr(FLAGS, 'gmp_saliency_ntp_ema_every_step', False)
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
    cubic_log_kl    = getattr(FLAGS, 'gmp_cubic_log_kl', False)  # diagnostic-only KL before/after each cubic mask update
    growth_schedule = getattr(FLAGS, 'gmp_growth_schedule', 'cubic')  # non-TR fixed schedule: 'cubic' or 'cosine'
    _schedule_fn = _cosine_sparsity if growth_schedule == 'cosine' else _cubic_sparsity
    use_wandb      = getattr(FLAGS, 'wandb', False) and is_main_process
    ntp_lambda     = getattr(FLAGS, 'gmp_ntp_lambda', 1.0)
    kd_lambda      = getattr(FLAGS, 'gmp_kd_lambda', 0.0)
    kd_temperature = getattr(FLAGS, 'gmp_kd_temperature', 2.0)
    kd_topk        = getattr(FLAGS, 'gmp_kd_topk', 0)
    kl_chunk_size  = getattr(FLAGS, 'gmp_kl_chunk_size', 0)  # 0=disabled, see _kl_loss
    kd_only        = getattr(FLAGS, 'gmp_kd_only', False)
    hidden_lambda  = getattr(FLAGS, 'gmp_hidden_lambda', 0.0)
    hidden_only    = getattr(FLAGS, 'gmp_hidden_only', False)
    hidden_mode    = getattr(FLAGS, 'gmp_hidden_mode', 'cosine')
    hidden_mask    = getattr(FLAGS, 'gmp_hidden_mask', 'cot')
    hidden_layers  = getattr(FLAGS, 'gmp_hidden_layers', 'final')  # 'final' or 'anneal_all_to_final'
    onpolicy_lambda     = getattr(FLAGS, 'gmp_onpolicy_kd_lambda', 0.0)
    onpolicy_interval   = getattr(FLAGS, 'gmp_onpolicy_kd_interval', -1)
    if onpolicy_interval < 0:
        onpolicy_interval = mask_interval  # default: tie rollout-refresh cadence to mask growth cadence (old behavior)
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
    opkd_prune_opd = getattr(FLAGS, 'gmp_opkd_prune_opd', False)
    opkd_prune_opd_drop = getattr(FLAGS, 'gmp_opkd_prune_opd_drop', 0.01)
    opkd_prune_opd_wbase = getattr(FLAGS, 'gmp_opkd_prune_opd_wbase', 0.5)
    opkd_prune_opd_topk = getattr(FLAGS, 'gmp_opkd_prune_opd_topk', 256)
    opkd_prune_opd_threshold = getattr(FLAGS, 'gmp_opkd_prune_opd_threshold', 0.7)
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
    pgd_max_swap_frac = getattr(FLAGS, 'gmp_pgd_max_swap_frac', 0.0)  # trust-region cap on PGD mask churn, 0=unlimited (see below)
    pgd_kl_budget = getattr(FLAGS, 'gmp_pgd_kl_budget', 0.0)  # alternative to pgd_max_swap_frac: self-KL-gated instead of fixed-count (see below)
    pgd_kl_share = getattr(FLAGS, 'gmp_pgd_kl_share', False)  # cheaper alternative to pgd_kl_budget: derive this window's swap_frac from TR-GMP's own measured KL headroom (no extra forward passes) instead of a fresh per-step self-KL measurement
    _pgd_dynamic_swap_frac = 0.0  # set at each mask_interval boundary when pgd_kl_share=true (see near _tr_mask_update call)
    pgd_kl_calib_size = max(1, getattr(FLAGS, 'gmp_pgd_kl_calib_size', 4))
    pgd_kl_calib_seqlen = max(1, getattr(FLAGS, 'gmp_pgd_kl_calib_seqlen', 512))
    pgd_kl_bisect_iters = max(1, getattr(FLAGS, 'gmp_pgd_kl_bisect_iters', 6))
    pgd_skip_growth_step = getattr(FLAGS, 'gmp_pgd_skip_growth_step', False)  # skip PGD on the exact step growth just fired, so the model trains at least one step under the mask growth decided before PGD can touch it again
    pgd_post_target_only = getattr(FLAGS, 'gmp_pgd_post_target_only', False)  # isolate PGD's post-target-maintenance role from its during-growth-ramp role -- see gate check at the main PGD condition below
    pgd_interval = max(1, getattr(FLAGS, 'gmp_pgd_interval', 1))  # only run PGD's reprojection every Nth step (default 1 = every step, prior behavior) -- decouples PGD's own cadence from mask_interval's growth cadence
    pgd_topk_impl = getattr(FLAGS, 'gmp_pgd_topk_impl', 'bisect')  # DIAGNOSTIC/A-B only (analysis phase, not a default): 'bisect' (default, unchanged 64-iter value-threshold search) or 'kthvalue' (torch.kthvalue on a one-time concat of the candidate pool instead) -- only wired into the unstructured gmp_pgd_kl_budget branch (where gmp_pgd_grow_to_target lives), non-FSDP only.
    pgd_grow_to_target = getattr(FLAGS, 'gmp_pgd_grow_to_target', False)  # PGD-driven growth (no separate TR-GMP growth): _pgd_desired targets final_sparsity directly (instead of matching current keep-count), and revive is no longer forced equal to prune -- revive saturates at min(k, revive_cand) while prune keeps going up to k, so whenever prune_cand > revive_cand (current sparsity < target) the self-KL-gated bisection alone drives net sparsity growth, at whatever pace the budget allows. Once current sparsity reaches target, prune_cand/revive_cand naturally converge and it degrades to pure polish/maintenance -- no separate at-target branch needed (unlike the old N:M pre/post-target split). Intended for --gmp_tr_enabled=false. Also supports sparsity_type=N:M now: _pgd_desired is replaced with a per-group top-prune_n projection of current importance (_pgd_nm_post_target) instead of a global threshold, and N:M bypasses the old pre/post-target split entirely, falling through to this same unstructured-shaped bisection -- intermediate masks are free to violate the N:M pattern group-by-group (a group's dead-count can sit anywhere from 0 to prune_m-prune_n mid-training, not just 0-or-exactly-N), converging to exactly N:M only once _pgd_desired is fully matched. Only affects the gmp_pgd_kl_budget branch below.
    if pgd_grow_to_target:
        if tr_enabled:
            logging.warning("  --gmp_pgd_grow_to_target=true with --gmp_tr_enabled=true: TR-GMP growth and "
                             "PGD-driven growth will both be moving the mask independently -- almost certainly "
                             "not intended together. Pass --gmp_tr_enabled=false.")
        logging.info(f"  PGD-driven growth ENABLED (--gmp_pgd_grow_to_target): _pgd_desired targets "
                     f"final_sparsity directly, revive saturates at min(k, revive_cand) instead of forcing "
                     f"revive==prune -- self-KL budget ({pgd_kl_budget}) alone paces growth toward target.")
    _pgd_kl_cal_batch = None  # small/short batch, refreshed every mask_interval steps (see below), reused every PGD step in between
    _pgd_scratch   = {}  # name -> preallocated fp32 buffer, reused in-place every PGD step (see below)
    pgd_debug_repeat_swap = getattr(FLAGS, 'gmp_pgd_debug_repeat_swap', False)  # diagnostic: track what fraction of each step's flips are positions that ALSO flipped within the last gmp_pgd_debug_repeat_window steps (are the same weights repeatedly swapping back and forth, or is a growing set of distinct weights each swapping once)
    pgd_debug_importance_hist = getattr(FLAGS, 'gmp_pgd_debug_importance_hist', False)  # diagnostic: dump the importance distribution's quantile/density every 5 steps (off by default -- ~0.6s/step amortized cost for a purely informational value)
    pgd_debug_repeat_window = getattr(FLAGS, 'gmp_pgd_debug_repeat_window', 5)
    _pgd_last_flip_step = {}  # name -> int64 tensor, step index each position last flipped (revive or prune), -1e9 sentinel = never
    _pgd_last_k_actual = 0  # warm-start anchor for gmp_pgd_kl_budget's bisection (see below) -- last step's accepted k, so a persistently-collapsed (k=0) or persistently-generous regime converges in 1-2 forward passes instead of always spending the full bisect_iters budget
    # ── STE mode (opt-in, --gmp_ste=true) ───────────────────────────────────
    # Replaces the earlier shadow-weight-replay approach entirely (removed --
    # mathematically the two give the same accumulated trajectory for a plain
    # linear layer with weight_decay=0, since dL/dW at a masked position
    # doesn't depend on W's own stored value, only on activation * upstream
    # grad -- so the gradient sequence feeding Adam's m/v is identical either
    # way). STE is strictly better: maskmgr.apply()/update() become a no-op
    # when GradualMaskManager.ste=True (see _apply_mask), so param.data itself
    # is never hard-reset -- Adam accumulates the true, multi-step trajectory
    # directly in the real parameter (not a separate replayed buffer), and a
    # revived weight's value is the mature accumulated one, not a cold-started
    # near-zero one. Sparsity is enforced only in the forward pass via
    # install_ste_forward_hooks (weight*mask, gradient passes straight
    # through) -- see GradualMaskManager construction below, gated by the
    # same flag.
    ste_enabled = getattr(FLAGS, 'gmp_ste', False)

    use_kd         = (teacher_model is not None) and (kd_lambda > 0.0)
    use_hidden     = (teacher_model is not None) and (hidden_lambda > 0.0)
    # ── Blockwise SquareHead (opt-in, --gmp_blockwise_squarehead=true) ──────
    # Adaptive-anchor-spacing per-layer distillation (see
    # _squarehead_anchor_layers/_squarehead_loss docstrings). Block size
    # starts tight (every layer anchored) and widens (fewer anchors) only
    # when TR-GMP's own trust-region growth stalls completely (no delta
    # accepted down to gmp_tr_delta_min) -- see the widening check right
    # after the _tr_mask_update call below. Requires tr_enabled=true and a
    # teacher model; otherwise there's no "stall" signal to widen on and no
    # dense reference to distill from.
    blockwise_enabled = (getattr(FLAGS, 'gmp_blockwise_squarehead', False)
                          and teacher_model is not None and tr_enabled)
    blockwise_hardness = getattr(FLAGS, 'gmp_blockwise_hardness', 1.0)
    blockwise_widen_factor = max(2, getattr(FLAGS, 'gmp_blockwise_widen_factor', 2))
    _block_size = max(1, getattr(FLAGS, 'gmp_blockwise_init_block', 1))
    # Sparsity at the last widening event -- lets us detect when widening has
    # stopped earning its keep (next stall happens at the SAME sparsity, i.e.
    # zero growth was made at the newly-widened block_size before stalling
    # again) vs. genuinely helping (real growth happened first). See the
    # zero-growth short-circuit right after the widening block below.
    _sp_at_last_widen = None
    _num_decoder_layers = len(_get_decoder_layers(model))
    # --gmp_blockwise_delay_global_signal: hold NTP/KD/OPKD at 0 (SquareHead
    # loss alone drives training + Fisher importance) until block_size has
    # widened ALL THE WAY to _num_decoder_layers (no more widening possible),
    # THEN switch them on. Tests whether the local per-layer anchor signal is
    # sufficient on its own to keep growth safe, instead of always having the
    # global losses available to trivially keep TR-GMP's KL check satisfied
    # (which was observed to make widening basically never fire in practice).
    _delay_global_signal = blockwise_enabled and getattr(FLAGS, 'gmp_blockwise_delay_global_signal', False)
    _global_signal_active = not _delay_global_signal
    _ntp_lambda_cfg, _kd_lambda_cfg, _onpolicy_lambda_cfg = ntp_lambda, kd_lambda, onpolicy_lambda
    if _delay_global_signal:
        ntp_lambda, kd_lambda, onpolicy_lambda = 0.0, 0.0, 0.0
        logging.info("  [blockwise] gmp_blockwise_delay_global_signal=true: NTP/KD/OPKD held at 0 "
                     "until block_size reaches its max (SquareHead-only training + Fisher importance until then)")
    use_teacher_gen_kd_flag = getattr(FLAGS, 'gmp_teacher_gen_kd', False)
    # Teacher-gen KD (forward KL, prompts pre-generated once from data_path)
    # and on-policy/OPD (reverse KL, live student rollouts from gmp_prompt_path)
    # draw from independently-configurable prompt sources and don't share any
    # generation state, so they can run together -- both are weighted by
    # gmp_onpolicy_kd_lambda (same knob, applied to each loss term separately).
    # NOTE: gated on _onpolicy_lambda_cfg (the pre-delay configured value), not
    # the live onpolicy_lambda -- this is a one-time infrastructure-setup gate
    # (vLLM engine, rollout pool, etc.) that must stay True even while
    # gmp_blockwise_delay_global_signal has temporarily zeroed the live
    # lambda, or OPKD could never actually turn back on later.
    use_onpolicy   = (teacher_model is not None) and (_onpolicy_lambda_cfg > 0.0)
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
    if use_onpolicy or use_teacher_seqkd or tr_enabled or (pgd_enabled and pgd_kl_budget > 0):
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
        # `interval` printed here is gmp_onpolicy_kd_interval. The vLLM pool is
        # always refilled at every gmp_mask_interval boundary regardless of this
        # value (that refill is tied to mask growth's own calibration needs and
        # can't be decoupled). When onpolicy_interval < mask_interval, the pool
        # is ALSO refilled at every extra onpolicy_interval boundary in between
        # (see the "OPKD vLLM pool refilled (mid-window, ...)" log line below),
        # giving rollouts a real refresh cadence independent of mask growth.
        # When onpolicy_interval >= mask_interval (the default in every launcher
        # script -- ROLLOUT_INTERVAL defaults to MASK_INTERVAL), this is a no-op
        # and behavior is unchanged from before this flag did anything. It also
        # still gates the grad-conflict-filter snapshot and the no-pool
        # fallback path, as before.
        logging.info(f"  On-policy KD: lambda={onpolicy_lambda}, interval={onpolicy_interval} "
                     f"(mask_interval={mask_interval}; pool refresh fires every min(interval, mask_interval) steps), "
                     f"max_new_tokens={onpolicy_max_new}, topk={onpolicy_topk}")

    _opkd_vllm_engine = None
    _opkd_vllm_params = None
    _opkd_standalone_pool: list = []
    _opkd_standalone_pool_ptr: int = 0
    _opkd_refilled_pre_mask = False  # set True once the OPKD pool has been refilled at least once (see below) -- initialized here so the gmp_pgd_kl_budget calibration-batch bootstrap (which reads this before step 1's own mask_interval block ever runs) doesn't hit an UnboundLocalError
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
                # This engine is always tensor_parallel_size=1 (single GPU) --
                # vLLM's custom-all-reduce plugin exists purely to coordinate
                # multiple GPUs and is a no-op for TP=1, but it still sets up
                # its own NCCL/IPC machinery by default (disable_custom_all_
                # reduce=False). Repeated "NCCL WARN Cuda failure 'out of
                # memory'" messages were observed starting from step 1 (long
                # before any real memory pressure), most from this engine
                # co-existing with the training process's own NCCL usage on
                # the same device -- disabling it removes an entirely
                # unnecessary source of GPU-side NCCL/IPC state for a
                # single-GPU engine.
                disable_custom_all_reduce=True,
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
            _vllm_inputs, _flat_meta = _opkd_flatten_pool_batches(_pool_batches)
            _vllm_outs = _opkd_vllm_engine.generate(_vllm_inputs, _opkd_vllm_params)
            _opkd_vllm_sleep(_opkd_vllm_engine)
            _opkd_standalone_pool.extend(_opkd_build_pool_from_outputs(_pool_batches, _flat_meta, _vllm_outs))
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
                                  pruning_scope=getattr(FLAGS, 'gmp_pruning_scope', 'global'),
                                  ste=ste_enabled, named_shapes=named_shapes)
    # Used by _param_block_group (gmp_pruning_scope='block') to fold a
    # leftover/remainder tail of decoder layers into the last FULL group
    # instead of letting it form its own small orphan group -- e.g. 28
    # layers / block_size=8 becomes groups [0-7],[8-15],[16-27] (last group
    # absorbs the remaining 4), not [0-7],[8-15],[16-23],[24-27].
    maskmgr._num_decoder_layers = _num_decoder_layers
    if fixed_mask:
        maskmgr.init_from_weights()
        maskmgr.apply(fsdp_model)
    if ste_enabled:
        install_ste_forward_hooks(model, maskmgr)
        logging.info("  STE masking ENABLED (--gmp_ste): forward masks weight*mask, "
                     "gradient passes straight through; param.data is never hard-reset.")
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
    accum_blockwise = 0.0
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
                                   kd_temperature, kd_topk, reverse=False, chunk_size=kl_chunk_size)

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

            # Structured-L1 (lasso) preconditioning only makes sense while the
            # mask is still growing toward final_sparsity -- its entire
            # purpose is shrinking soon-to-be-pruned weights toward zero
            # BEFORE the cut that closes each group, so the eventual hard cut
            # (or the exact-2:4 completion check, see _tr_mask_update) causes
            # minimal disruption. Once tr_reached (mask frozen, into the
            # sparse-training tail), there's nothing left to precondition for
            # -- continuing to apply it there would just be an unmotivated
            # extra regularizer actively fighting the fine-tuning objective.
            _l1_active = use_l1 and not (tr_enabled and tr_reached)
            _l1_fsdp_hooks, _l1_fsdp_terms = (
                _register_structured_l1_hooks(fsdp_model, named_params, maskmgr.masks, prune_n, prune_m)
                if (_l1_active and use_structured_l1 and is_fsdp) else ([], [])
            )
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                fwd_inputs = {k: v for k, v in batch.items()}
                out = model(**fwd_inputs, output_hidden_states=(use_hidden or blockwise_enabled))
                for _h in _l1_fsdp_hooks:
                    _h.remove()
                ntp_loss = out.loss

                if use_kd or use_hidden or blockwise_enabled:
                    t_inputs = {k: v for k, v in batch.items() if k != 'labels'}
                    with torch.no_grad():
                        t_out = teacher_model(
                            **t_inputs,
                            output_hidden_states=(use_hidden or blockwise_enabled),
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
                                               kd_temperature, kd_topk, chunk_size=kl_chunk_size)
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

                    if blockwise_enabled:
                        _sq_anchors = _squarehead_anchor_layers(len(out.hidden_states), _block_size)
                        _sq_loss = _squarehead_loss(out.hidden_states, t_out.hidden_states,
                                                     _sq_anchors, batch['attention_mask'])
                        loss = loss + blockwise_hardness * _sq_loss / grad_accum
                        accum_blockwise += _sq_loss.item() / grad_accum
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

            if _l1_active:
                if use_structured_l1:
                    if is_fsdp:
                        # _l1_fsdp_terms was populated by forward pre-hooks
                        # registered right before this micro-step's model(...)
                        # call (see above) -- each hook computed its layer's
                        # contribution while FSDP had that layer's params
                        # fully gathered, as part of the SAME autograd graph
                        # this micro-step's loss.backward() below already
                        # walks. This rides FSDP's own, already-correct
                        # forward/backward machinery instead of a separate
                        # summon_full_params + backward() pass, which torch's
                        # exit-time grad-resharding doesn't support for a
                        # freshly computed (not pre-existing) gradient --
                        # tried, failed two different ways (shape mismatch,
                        # then a 2x-oversized grad from a skipped reshard).
                        if _l1_fsdp_terms:
                            total = sum(t for t, _ in _l1_fsdp_terms)
                            count = sum(n for _, n in _l1_fsdp_terms)
                            l1 = total / count if count > 0 else None
                        else:
                            l1 = None
                    else:
                        l1 = _structured_l1_loss(named_params, maskmgr.masks, prune_n, prune_m)
                else:
                    l1 = _gmp_l1_regularizer(named_params, maskmgr, fisher,
                                             mode=l1_mode,
                                             clip_min=l1_fisher_cmin,
                                             clip_max=l1_fisher_cmax,
                                             open_groups_only=l1_open_only,
                                             prune_n=prune_n, prune_m=prune_m)
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
            import os as _os_dbg
            if _os_dbg.environ.get('GMP_ANOMALY_DEBUG'):
                with _bwd_ctx, torch.autograd.detect_anomaly():
                    loss.backward()
            else:
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
                                     kd_temperature, onpolicy_topk, chunk_size=kl_chunk_size)
                anc_loss = anchor_lambda * anc_kl / grad_accum
            anc_loss.backward()
            accum_onpolicy_diag.update({"anchor/kl_loss": anc_kl.item()})

        step += 1

        def _refresh_pgd_kl_cal_batch():
            # Prefer real OPKD rollouts, matching _tr_mask_update's own
            # calibration-source choice (see the "Use OPKD rollouts as
            # calibration if available, else fall back to prompt_iter" branch
            # below) -- until this fix, PGD's kl_budget calibration always
            # used raw prompt-dataset text regardless of rollout availability,
            # measuring self-KL against a different data distribution than
            # the TR growth check it's meant to share a trust-region budget
            # with.
            if is_main_process:
                if _opkd_refilled_pre_mask and _opkd_standalone_pool:
                    _n_cal = min(pgd_kl_calib_size, len(_opkd_standalone_pool))
                    _pool_items = [
                        {'full_seq': item['full_seq'][:, :pgd_kl_calib_seqlen], 'prompt_len': item['prompt_len']}
                        for item in _opkd_standalone_pool[:_n_cal]
                    ]
                    _batch = _opkd_pool_to_batch(_pool_items, str(device))
                else:
                    _batch = _pgd_kl_calib_batch(
                        prompt_iter, pgd_kl_calib_size, pgd_kl_calib_seqlen, str(device))
            else:
                _batch = None
            if fsdp_model is not None and _FSDP_AVAILABLE:
                import torch.distributed as _dist
                # All ranks must share the IDENTICAL calibration batch (and later,
                # the identical bisected K) since the self-KL forward passes are
                # collective under FSDP -- broadcast rank 0's batch to the rest.
                _obj = [_batch]
                _dist.broadcast_object_list(_obj, src=0)
                _b = _obj[0]
                _batch = {'input_ids': _b['input_ids'].to(device),
                          'attention_mask': _b['attention_mask'].to(device)}
            return _batch

        # BUGFIX: _pgd_kl_cal_batch starts as None and used to be filled in ONLY
        # inside the `step % mask_interval == 0` block below -- for
        # mask_interval=32 that means the entire first window (steps 1-31) ran
        # with _pgd_kl_cal_batch still None, so the `elif pgd_kl_budget > 0 and
        # _pgd_kl_cal_batch is not None` check further down was false the whole
        # time and PGD silently fell through to the fully uncapped branch for
        # ~1.5% of training every single run (verified empirically: step 1
        # alone reprojected ~71M positions in one shot). Bootstrap it here,
        # unconditionally on the very first step it's needed, so the self-KL
        # gate is live from step 1 instead of only from the first mask_interval
        # boundary onward.
        if pgd_enabled and pgd_kl_budget > 0 and _pgd_kl_cal_batch is None:
            _pgd_kl_cal_batch = _refresh_pgd_kl_cal_batch()

        _saliency_diag_step = getattr(FLAGS, 'gmp_saliency_diag_step', 0)
        if _saliency_diag_step > 0 and step == _saliency_diag_step and is_main_process:
            if _pgd_kl_cal_batch is None:
                _pgd_kl_cal_batch = _refresh_pgd_kl_cal_batch()
            _diag_ref_batch = _pgd_kl_calib_batch(prompt_iter, pgd_kl_calib_size, pgd_kl_calib_seqlen, str(device))
            saliency_snapshot_diagnostic(
                model, maskmgr, fisher, _pgd_kl_cal_batch, _diag_ref_batch, str(device),
                k=getattr(FLAGS, 'gmp_saliency_diag_k', 4096),
                mc_nsamples=getattr(FLAGS, 'gmp_saliency_diag_mc_nsamples', 3),
                use_wandb=use_wandb, global_step=step,
            )
            logging.info(f"[saliency_diag] done at step={step}, exiting (diagnostic-only run).")
            import sys as _sys
            _sys.exit(0)

        if _saliency_corr_steps and step in _saliency_corr_steps and is_main_process:
            if _pgd_kl_cal_batch is None:
                _pgd_kl_cal_batch = _refresh_pgd_kl_cal_batch()
            _corr_ref_batch = _pgd_kl_calib_batch(prompt_iter, pgd_kl_calib_size, pgd_kl_calib_seqlen, str(device))
            saliency_random_group_correlation_diagnostic(
                model, maskmgr, fisher, _pgd_kl_cal_batch, _corr_ref_batch, str(device),
                group_size=getattr(FLAGS, 'gmp_saliency_corr_group_size', 4096),
                n_groups=getattr(FLAGS, 'gmp_saliency_corr_groups', 20),
                mc_nsamples=getattr(FLAGS, 'gmp_saliency_diag_mc_nsamples', 8),
                use_wandb=use_wandb, global_step=step,
                seed=getattr(FLAGS, 'gmp_saliency_corr_seed', 0),
                ema_cache=_saliency_ema_cache,
            )
            if step == _saliency_corr_steps[-1]:
                logging.info(f"[saliency_corr] done at step={step} (last of {_saliency_corr_steps}), exiting (diagnostic-only run).")
                import sys as _sys
                _sys.exit(0)
            else:
                logging.info(f"[saliency_corr] done at step={step}, continuing training toward next diag step "
                             f"in {_saliency_corr_steps} (skip_growth_step's own PGD/mask logic below still runs normally).")

        # Mid-window OPKD vLLM pool refresh: makes gmp_onpolicy_kd_interval
        # actually control rollout freshness (previously dead code under the
        # pool path -- pool refill was hardcoded to mask_interval only, see the
        # "On-policy KD:" log above and job 819502's post-mortem). When
        # onpolicy_interval < mask_interval, refresh the pool at every extra
        # onpolicy_interval boundary that falls strictly between mask_interval
        # boundaries (which already get their own refill below, tied to mask
        # growth's calibration). No-op whenever onpolicy_interval >=
        # mask_interval -- every existing launcher script defaults
        # ROLLOUT_INTERVAL to MASK_INTERVAL, so this leaves all prior runs'
        # behavior unchanged.
        if (use_onpolicy and 0 < onpolicy_interval < mask_interval
                and step % onpolicy_interval == 0 and step % mask_interval != 0):
            _in_fsdp_refill_mid = fsdp_model is not None and _FSDP_AVAILABLE
            _fsdp_sync_ctx_mid = (FSDP.summon_full_params(fsdp_model, writeback=False, offload_to_cpu=True, rank0_only=True)
                                  if _in_fsdp_refill_mid else nullcontext())
            with _fsdp_sync_ctx_mid:
                if is_main_process and _opkd_vllm_engine is not None:
                    if fsdp_model is None:
                        _offload_optimizer_state(optimizer)
                    _opkd_vllm_wake(_opkd_vllm_engine)
                    if _in_fsdp_refill_mid and hasattr(_opkd_vllm_engine, 'sync_weights'):
                        _sd = {n: p.data.cpu() for n, p in model.named_parameters()}
                        _opkd_vllm_engine.sync_weights(_sd)
                        del _sd
                    elif not _in_fsdp_refill_mid:
                        _sync_opkd_weights_to_vllm(model, _opkd_vllm_engine)
            if is_main_process and _opkd_vllm_engine is not None:
                _n_pool_mid = onpolicy_interval * grad_accum
                _pool_batches = [next(prompt_iter) for _ in range(_n_pool_mid)]
                _vllm_inputs, _flat_meta = _opkd_flatten_pool_batches(_pool_batches)
                _vllm_outs = _opkd_vllm_engine.generate(_vllm_inputs, _opkd_vllm_params)
                _opkd_vllm_sleep(_opkd_vllm_engine)
                if fsdp_model is None:
                    _reload_optimizer_state(optimizer, device)
                _opkd_standalone_pool = _opkd_build_pool_from_outputs(_pool_batches, _flat_meta, _vllm_outs)
                logging.info(f"  OPKD vLLM pool refilled (mid-window, onpolicy_interval={onpolicy_interval}): "
                             f"{len(_opkd_standalone_pool)} rollouts (step={step})")
            _opkd_standalone_pool = _opkd_broadcast_pool(_opkd_standalone_pool, is_distributed, device)
            _opkd_standalone_pool_ptr = 0
            torch.cuda.empty_cache()

        # periodic mask update (freeze mask after pruning_end_steps)
        if step % mask_interval == 0:
            # NOTE: _refresh_pgd_kl_cal_batch() used to be called right here,
            # BEFORE the vLLM wake below -- this was the one extra GPU
            # allocation (.to(device) on the calib batch) that plain TR-GMP
            # (no PGD) never does at this point in the step. That extra
            # allocation immediately before vLLM's sleep-mode wake/remap is
            # the likely trigger for a reproducible NCCL "CUDA out of memory"
            # at every mask_interval boundary when cudagraphs are enabled
            # (cudagraph capture holds fixed pointers that can conflict with
            # CuMemAllocator's remap on wake if the allocator layout shifts).
            # Moved to AFTER the OPKD pool refill (below) as a debug/fix
            # attempt -- also incidentally more correct, since it now sees
            # THIS window's freshly-refilled _opkd_standalone_pool instead of
            # the previous window's stale one.

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
                    # DEBUG: device-wide (not just this process's PyTorch
                    # allocator view) free/total memory at each stage of the
                    # vLLM wake/generate/sleep cycle -- diagnosing a
                    # reproducible NCCL "CUDA failure: out of memory" right
                    # after this cycle at every mask_interval boundary.
                    _dbg_f0, _dbg_t0 = torch.cuda.mem_get_info()
                    logging.info(f"  [DBG mem] pre-wake: free={_dbg_f0/1e9:.2f}GB / total={_dbg_t0/1e9:.2f}GB (step={step})")
                    _n_pool = mask_interval * grad_accum
                    _pool_batches = [next(prompt_iter) for _ in range(_n_pool)]
                    _vllm_inputs, _flat_meta = _opkd_flatten_pool_batches(_pool_batches)
                    _dbg_f1, _ = torch.cuda.mem_get_info()
                    logging.info(f"  [DBG mem] post-wake: free={_dbg_f1/1e9:.2f}GB (step={step})")
                    _vllm_outs = _opkd_vllm_engine.generate(_vllm_inputs, _opkd_vllm_params)
                    _dbg_f2, _ = torch.cuda.mem_get_info()
                    logging.info(f"  [DBG mem] post-generate: free={_dbg_f2/1e9:.2f}GB (step={step})")
                    _opkd_vllm_sleep(_opkd_vllm_engine)
                    _dbg_f3, _ = torch.cuda.mem_get_info()
                    logging.info(f"  [DBG mem] post-sleep: free={_dbg_f3/1e9:.2f}GB (step={step})")
                    if fsdp_model is None:
                        _reload_optimizer_state(optimizer, device)
                    _opkd_standalone_pool = _opkd_build_pool_from_outputs(_pool_batches, _flat_meta, _vllm_outs)
                    logging.info(f"  OPKD vLLM pool refilled (pre-mask): {len(_opkd_standalone_pool)} rollouts (step={step})")
                _opkd_standalone_pool = _opkd_broadcast_pool(_opkd_standalone_pool, is_distributed, device)
                _opkd_standalone_pool_ptr = 0
                _opkd_refilled_pre_mask = True
                # Defrag before _tr_mask_update's own big allocations (candidate
                # masks, KL forward passes) start -- the vLLM wake/generate/sleep
                # cycle just above leaves the caching allocator fragmented on
                # this single co-located GPU, and growth's own peak was
                # observed to hit a genuine CUDA OOM (NCCL all_reduce inside
                # _tr_mask_update) right at this boundary otherwise. Same
                # end-of-block empty_cache() already used further down, just
                # moved earlier so it covers the actual peak instead of only
                # cleaning up after it.
                torch.cuda.empty_cache()

            # Refresh the small/short calibration batch used by PGD's
            # per-step self-KL gate (--gmp_pgd_kl_budget), reused unchanged
            # for every PGD step until the next mask_interval boundary --
            # amortizes data-loading cost across the whole window instead of
            # re-sampling every single step.
            if pgd_enabled and pgd_kl_budget > 0:
                _pgd_kl_cal_batch = _refresh_pgd_kl_cal_batch()

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
                _sp_before_tr = maskmgr.current_sparsity()
                _dbg_f4, _ = torch.cuda.mem_get_info()
                logging.info(f"  [DBG mem] pre-tr_mask_update: free={_dbg_f4/1e9:.2f}GB (step={step})")
                current_sparsity, tr_delta, tr_reached, _tr_mask_delta, _tr_kl_spent = _tr_mask_update(
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
                    block_size=(_block_size if (blockwise_enabled and maskmgr.pruning_scope == 'block') else None),
                )
                # --gmp_pgd_kl_share: derive THIS window's per-step PGD swap
                # fraction from TR-GMP's own just-measured KL headroom instead
                # of a separate per-step forward-pass-based measurement.
                # headroom_ratio=1 (TR used none of its budget) -> PGD may
                # swap up to ~1/mask_interval of all masked params per step
                # (i.e. the whole window's worth of movement spread evenly);
                # headroom_ratio=0 (TR used its full budget) -> PGD gets no
                # extra churn room this window. No new hyperparameter needed
                # beyond what TR-GMP and mask_interval already define.
                if pgd_enabled and pgd_kl_share:
                    _headroom_ratio = max(0.0, 1.0 - (_tr_kl_spent / tr_kl_threshold)) if tr_kl_threshold > 0 else 0.0
                    _pgd_dynamic_swap_frac = _headroom_ratio / mask_interval
                    logging.info(f"  [pgd_kl_share] TR KL spent={_tr_kl_spent:.5f}/{tr_kl_threshold:.5f} "
                                 f"(headroom={_headroom_ratio:.3f}) -> PGD swap_frac={_pgd_dynamic_swap_frac:.6f} for this window")
                if is_distributed:
                    import torch.distributed as _td2
                    _td2.barrier()
                    logging.info(f"  [BARRIER] after _tr_mask_update step={step} (rank={_td2.get_rank()})")
                # ── Blockwise SquareHead: widen anchor spacing on a full stall ──
                # _tr_mask_update accepted NOTHING this call (current_sparsity
                # unchanged, not reached) -- the current block's per-layer
                # anchors are too tight a constraint for the model to find a
                # KL-acceptable growth step within. Widen (fewer anchors, more
                # inter-layer compensation freedom) and reset delta so the
                # newly-widened scope gets a fresh full search next call,
                # instead of continuing to retry at the same collapsed delta
                # forever. Never fires once block_size already covers every
                # layer (global/output-only, same ceiling as plain TR-GMP).
                if (blockwise_enabled and not tr_reached
                        and current_sparsity <= _sp_before_tr + 1e-9
                        and _block_size < _num_decoder_layers):
                    _old_block_size = _block_size
                    # If the LAST widening produced zero growth before this new
                    # stall (current_sparsity hasn't moved since _sp_at_last_widen),
                    # widening again is very unlikely to help either -- observed
                    # empirically (746292/747457): once widening stops earning its
                    # keep, it keeps failing all the way to block_size's ceiling
                    # anyway, just burning mask_interval steps on zero-growth
                    # cascades. Skip straight to the ceiling (triggers global-signal
                    # reactivation below in the same iteration) instead of retrying
                    # every intermediate block_size one stall at a time.
                    _widen_was_futile = (
                        _delay_global_signal and _sp_at_last_widen is not None
                        and current_sparsity <= _sp_at_last_widen + 1e-9
                    )
                    if _widen_was_futile:
                        _block_size = _num_decoder_layers
                        tr_delta = tr_delta_init
                        logging.info(f"  [blockwise] widening to {_old_block_size} produced zero growth "
                                     f"(sparsity still {current_sparsity:.4f}) -- skipping remaining "
                                     f"widening stages, jumping straight to block_size={_block_size} "
                                     f"(step={step})")
                    else:
                        _block_size = min(_block_size * blockwise_widen_factor, _num_decoder_layers)
                        tr_delta = tr_delta_init
                        logging.info(f"  [blockwise] TR-GMP stalled at sparsity={current_sparsity:.4f} "
                                     f"(no delta accepted down to {tr_delta_min}) -- widening block size "
                                     f"{_old_block_size} -> {_block_size} anchors, resetting delta to "
                                     f"{tr_delta_init} (step={step})")
                    _sp_at_last_widen = current_sparsity
                    if use_wandb and is_main_process:
                        wandb.log({"train/block_size": _block_size}, step=step)
                # Once block_size can't widen any further (fully expanded --
                # degenerates to output-only anchoring, same ceiling as plain
                # TR-GMP), switch the global losses back on for good. One-way
                # flip: never re-disables even if block_size were somehow
                # reduced (it never is).
                if _delay_global_signal and not _global_signal_active and _block_size >= _num_decoder_layers:
                    _global_signal_active = True
                    ntp_lambda, kd_lambda, onpolicy_lambda = _ntp_lambda_cfg, _kd_lambda_cfg, _onpolicy_lambda_cfg
                    logging.info(f"  [blockwise] block_size reached max ({_block_size}) -- "
                                 f"switching NTP/KD/OPKD back on (lambda={_ntp_lambda_cfg}/{_kd_lambda_cfg}/{_onpolicy_lambda_cfg}) (step={step})")
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
                               "train/tr_delta": tr_delta}, step=step)
                if tr_reached:
                    logging.info(f"TR-GMP: target sparsity {final_sparsity} reached at step {step}, "
                                 f"switching to sparse training (mask frozen) for remaining steps.")
                    if _tr_reached_step is None:
                        _tr_reached_step = step
            else:
                current_sparsity = 0.0 if step <= dense_warmup_steps else _schedule_fn(
                    min(step, pruning_end_steps), pruning_end_steps, final_sparsity, dense_warmup_steps)
                if step <= pruning_end_steps:
                    if cubic_log_kl:
                        # Diagnostic only (--gmp_cubic_log_kl): unlike TR-GMP,
                        # the cubic schedule never checks whether a mask update
                        # keeps the model's behavior within any KL budget --
                        # it just follows the fixed curve regardless. Measure
                        # KL(old||candidate) at every mask-update boundary
                        # anyway (same _compute_tr_kl the TR path uses to
                        # accept/reject growth) purely to see how far outside
                        # a trust-region budget the cubic schedule's forced
                        # steps actually land, for a fair cubic-vs-TR
                        # comparison at matched reach-step/step-budget.
                        if _opkd_refilled_pre_mask and _opkd_standalone_pool:
                            _n_cal = min(8, len(_opkd_standalone_pool))
                            _cubic_cal_batch = _opkd_pool_to_batch(_opkd_standalone_pool[:_n_cal], str(device))
                        else:
                            _cubic_cal_batch = next(prompt_iter)
                        _cand_masks = maskmgr.candidate_masks(fisher, current_sparsity, fsdp_model)
                        _cubic_kl, _ = _compute_tr_kl(model, _cubic_cal_batch, _cand_masks, maskmgr,
                                                       str(device), kl_reduce=tr_kl_reduce,
                                                       kl_quantile=tr_kl_quantile)
                        if use_wandb and is_main_process:
                            wandb.log({"cubic/kl_before_after": _cubic_kl,
                                       "cubic/sparsity": current_sparsity}, step=step)
                        maskmgr.masks = _cand_masks
                        maskmgr.apply(fsdp_model)
                    else:
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
                                    "train/delta_T_pos_rate": _delta_pos}, step=step)

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
                    _vllm_inputs, _flat_meta = _opkd_flatten_pool_batches(_pool_batches)
                    _vllm_outs = _opkd_vllm_engine.generate(_vllm_inputs, _opkd_vllm_params)
                    _opkd_vllm_sleep(_opkd_vllm_engine)
                    if fsdp_model is None:
                        _reload_optimizer_state(optimizer, device)
                    _opkd_standalone_pool = _opkd_build_pool_from_outputs(_pool_batches, _flat_meta, _vllm_outs)
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

            # Reclaim the caching allocator's fragmented free blocks. Every
            # mask_interval boundary above churns through several differently-
            # shaped full-model-forward allocations (OPKD pool refill batches
            # padded to whatever the longest rollout happened to be this round,
            # trust-region candidate-mask KL forwards) that PYTORCH_CUDA_ALLOC_CONF=
            # expandable_segments:True would normally defragment -- can't set
            # that flag here since it's incompatible with vLLM's CuMemAllocator
            # sleep mode (co-located on this single GPU). empty_cache() is the
            # next-best manual defrag: only touches PyTorch's free-block cache
            # (nothing live gets freed), so purely a memory-fragmentation
            # mitigation, no effect on the mask-search/KL logic or its outputs.
            torch.cuda.empty_cache()

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
        # Gated on the LIVE onpolicy_lambda (not use_onpolicy, which stays true from
        # _onpolicy_lambda_cfg for infra-setup purposes even while delay_global_signal
        # has zeroed it) -- otherwise gmp_blockwise_delay_global_signal's delay phase
        # still burns real vLLM rollout compute for a loss term that gets multiplied
        # by 0 anyway.
        _opkd_fires = use_onpolicy and onpolicy_lambda > 0.0 and (
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
                                    chunk_size=kl_chunk_size,
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
                                                  reverse=onpolicy_reverse_kl, chunk_size=kl_chunk_size,
                                                  prune_opd=opkd_prune_opd,
                                                  prune_opd_drop=opkd_prune_opd_drop,
                                                  prune_opd_wbase=opkd_prune_opd_wbase,
                                                  prune_opd_topk=opkd_prune_opd_topk,
                                                  prune_opd_threshold=opkd_prune_opd_threshold)
                        if t_prev_out is not None:
                            op_kl_prev, _ = _kl_loss(s_out.logits, t_prev_out.logits, gen_labels,
                                                      kd_temperature, onpolicy_topk,
                                                      reverse=onpolicy_reverse_kl, chunk_size=kl_chunk_size,
                                                      prune_opd=opkd_prune_opd,
                                                      prune_opd_drop=opkd_prune_opd_drop,
                                                      prune_opd_wbase=opkd_prune_opd_wbase,
                                                      prune_opd_topk=opkd_prune_opd_topk,
                                                      prune_opd_threshold=opkd_prune_opd_threshold)
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

        if _saliency_ema_every_step and is_main_process and _pgd_kl_cal_batch is not None:
            # Safe placement: .grad is guaranteed empty right here (just
            # zeroed above, nothing pending until the next step's backward
            # calls start accumulating) -- a separate forward+backward
            # (own zero_grad inside _mc_fisher_named_params) can't clobber
            # anything. The retain_graph-based reuse-the-on-policy-forward
            # version (fisher_grad_from_reused_onpolicy_forward) was tried
            # and reverted: avoiding a second forward pass there meant
            # cloning ALL of .grad as a snapshot (full-model-sized, several
            # GB) to safely undo its own backward's contribution mid-step,
            # which is a worse memory trade than just paying for the extra
            # forward here where no snapshot is needed at all.
            update_onpolicy_fisher_ema_continuous(model, maskmgr, _pgd_kl_cal_batch, str(device), _saliency_ema_cache,
                                                   nsamples=_saliency_ema_nsamples)

        if _saliency_ntp_ema_every_step and is_main_process:
            # Decoupled-saliency test: NTP's real-label gradient tracked
            # here EVERY step regardless of whether ntp_lambda>0 actually
            # feeds the optimizer this run -- lets a KD+OPD-only training
            # objective (dropping NTP to reduce forgetting) still keep a
            # good empirical-Fisher signal for mask selection. Same safe
            # post-optimizer.step() timing as the on-policy tracker above.
            #
            # Prefer the on-policy rollout ALREADY generated for OPD this
            # step (`generated`/`gen_labels`, still in scope -- Python has
            # no block scoping, so the on-policy loop's last values survive
            # to here) over a fixed-dataset batch: teacher-forcing NTP loss
            # against the rollout's own already-sampled tokens is still a
            # genuine observed-label gradient (same empirical-Fisher
            # validity as fixed-dataset NTP), but the CONTEXT distribution
            # is on-policy -- matches this session's original motivation
            # (sample h from the current model's own occupancy, not a fixed
            # reference corpus) with no extra generation cost, since OPD
            # already paid for this rollout. Falls back to a fixed-dataset
            # batch only if on-policy is off or this step's rollout somehow
            # isn't available.
            if use_onpolicy and 'generated' in dir() and 'gen_labels' in dir():
                _ntp_batch = {'input_ids': generated, 'attention_mask': (generated != _pad_id).long(),
                              'labels': gen_labels}
            else:
                _ntp_batch = next(data_iter)
            # gamma=0.999 matches Adam's own beta2 exactly (~1000-step
            # effective window); gamma=0.95 is a much shallower EMA
            # (~20-step window) tried alongside out of curiosity -- both
            # reuse this SAME single Fisher sample, no extra backward cost.
            update_ntp_only_fisher_ema(model, maskmgr, _ntp_batch, str(device), _saliency_ema_cache, gammas=(0.999, 0.95))

        # ── PGD projection (fisher-saliency, FSDP-aware) ─────────────────────
        # Runs every step by design (classic PGD: project back onto the
        # sparsity constraint after every unconstrained gradient step, not
        # just at mask_interval boundaries). fisher.importance() allocates
        # 3-4 fresh full-precision tensors PER PARAMETER, model-wide, every
        # call -- doing that every step (not just every mask_interval steps
        # like the base training path's own mask updates) churned enough
        # large short-lived allocations to fragment CUDA memory badly enough
        # to crash 3/3 first real runs of this never-before-exercised flag
        # (a segfault in an unrelated later kl_div call once fragmentation
        # left no contiguous block big enough, rather than a clean
        # OutOfMemoryError -- at unpredictable step counts consistent with
        # cumulative fragmentation, not a fixed peak-memory ceiling).
        # Fix: for the common gmp_saliency='fisher' case, compute v_t*w^2
        # in-place into a persistent per-parameter fp32 buffer allocated
        # ONCE (_pgd_scratch) and reused every step -- no new allocation at
        # all on the hot path, so nothing to fragment and no empty_cache()
        # needed. Falls back to fisher.importance() unchanged for any other
        # saliency mode (spa/wanda/magnitude/sqrt_fisher), which this
        # in-place path does not special-case.
        if (pgd_enabled and step > dense_warmup_steps and not math.isnan(grad_norm) and not math.isinf(grad_norm)
                and step % pgd_interval == 0
                and not (pgd_skip_growth_step and step % mask_interval == 0)
                and (not pgd_post_target_only or tr_reached or fixed_mask or not tr_enabled)):
            _pgd_revivals = 0
            _pgd_prunings = 0
            _pgd_use_fsdp = _FSDP_AVAILABLE and fsdp_model is not None
            if _pgd_use_fsdp:
                import torch.distributed as _dist

            # importance scores (v_t * w^2), skip empty FSDP shards
            _pgd_imps = {}
            _pgd_fast_path = (fisher.saliency == 'fisher')
            for _n, _p in maskmgr.named_params.items():
                # _p.data itself is the source: in STE mode (--gmp_ste) it is
                # never hard-reset by maskmgr.apply()/update() (see
                # _apply_mask), so it already holds the true, continuously-
                # compounding Adam trajectory -- no separate shadow replay
                # needed (the earlier shadow-buffer approach was removed;
                # mathematically equivalent to STE for weight_decay=0, but
                # STE is strictly better since it also fixes "cold revival"
                # -- see install_ste_forward_hooks). In hard-mask mode,
                # _p.data is the usual one-step-from-zero snapshot, as before.
                if _pgd_fast_path:
                    _f = fisher.fisher_factor(_p)
                    if _f is None:
                        _t = _p.data.float() ** 2  # pre-first-optimizer-step fallback (rare, one-off)
                    else:
                        _buf = _pgd_scratch.get(_n)
                        if _buf is None or _buf.shape != _p.shape:
                            _buf = torch.empty_like(_p.data, dtype=torch.float32)
                            _pgd_scratch[_n] = _buf
                        _buf.copy_(_p.data)
                        _buf.pow_(2)
                        _buf.mul_(_f)
                        _t = _buf
                else:
                    _t = fisher.importance(_n, _p)
                if _t.numel() > 0:
                    _pgd_imps[_n] = _t

            if _pgd_imps:
                _pgd_dev = next(iter(_pgd_imps.values())).device

                # DEBUG (every 5 steps): dump the real fisher-weighted
                # importance distribution's quantiles + local density near
                # each quantile, to check whether the pruning threshold for a
                # LOW target sparsity sits in a denser region of the
                # importance distribution than for a HIGH target sparsity
                # (hypothesis for why low-sparsity PGD churn never settles),
                # and to track how that density evolves over training instead
                # of a single snapshot.
                if pgd_debug_importance_hist and step % 5 == 0 and is_main_process:
                    _dbg_cap_per_t = 200000
                    _dbg_parts = []
                    for _v in _pgd_imps.values():
                        _flat = _v.flatten().float()
                        if _flat.numel() > _dbg_cap_per_t:
                            _idx = torch.randperm(_flat.numel(), device=_flat.device)[:_dbg_cap_per_t]
                            _flat = _flat[_idx]
                        _dbg_parts.append(_flat.detach().cpu())
                    _dbg_all = torch.cat(_dbg_parts)
                    if _dbg_all.numel() > 10_000_000:
                        _idx = torch.randperm(_dbg_all.numel())[:10_000_000]
                        _dbg_all = _dbg_all[_idx]
                    _dbg_qs = [0.10, 0.30, 0.50, 0.70, 0.90]
                    _dbg_vals = torch.quantile(_dbg_all, torch.tensor(_dbg_qs))
                    _dbg_msg = f"  [DBG importance_hist] sampled_n={_dbg_all.numel()} target_sparsity={final_sparsity:.3f}"
                    for _q, _v in zip(_dbg_qs, _dbg_vals.tolist()):
                        _lo, _hi = _v * 0.99, _v * 1.01
                        _cnt = int(((_dbg_all >= _lo) & (_dbg_all <= _hi)).sum().item())
                        _dbg_msg += f" | q={_q:.2f} thr={_v:.3e} band_cnt={_cnt}"
                    logging.info(_dbg_msg)
                    if use_wandb:
                        import wandb as _wandb
                        _wandb.log({f"dbg/imp_q{int(_q*100)}_band_cnt": int(((_dbg_all >= _v*0.99) & (_dbg_all <= _v*1.01)).sum().item())
                                    for _q, _v in zip(_dbg_qs, _dbg_vals.tolist())}, step=step)
                # n_keep / n_total — one all_reduce for FSDP
                _pgd_stats = torch.tensor(
                    [sum(maskmgr.masks[n].sum().item() for n in _pgd_imps),
                     sum(v.numel() for v in _pgd_imps.values())],
                    dtype=torch.long, device=_pgd_dev)
                if _pgd_use_fsdp:
                    _dist.all_reduce(_pgd_stats, op=_dist.ReduceOp.SUM)
                if pgd_grow_to_target:
                    # _pgd_desired targets final_sparsity directly instead of
                    # the current keep-count -- see gmp_pgd_grow_to_target's
                    # docstring. Without this, _pgd_desired's target always
                    # matches current sparsity by construction, so prune_cand
                    # and revive_cand are forced equal (up to threshold-tie
                    # noise) and there is no real asymmetry to drive growth.
                    _pgd_k_prune = round(final_sparsity * _pgd_stats[1].item())
                else:
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

                # binary search — _pgd_cnt_t reused in-place, one all_reduce/iter for FSDP.
                # BUGFIX: this loop used to mutate _pgd_lo/_pgd_hi directly as its
                # own search bracket -- by the time the loop finished, those
                # names no longer held the true global min/max, they'd been
                # permanently narrowed down to converge on _pgd_thr. Later
                # calls to _pgd_topk_mask(..., _pgd_lo, _pgd_hi) for the N:M
                # pre-target combination logic then received this corrupted,
                # much-too-narrow bracket instead of the real global bounds --
                # verified empirically: _pgd_topk_mask's internal search
                # bracket ended up at [true_min, log(~1e-18)] instead of
                # [true_min, log(true_max~3.68e-5)], causing its own bisection
                # to never converge properly and select ~987M positions
                # instead of a requested k=285K (the returned threshold landed
                # exactly at the corrupted sentinel/bracket edge, sweeping in
                # every non-candidate position too). Use separate local
                # variables for THIS search so _pgd_lo/_pgd_hi stay intact as
                # the true global bounds for later reuse.
                _pgd_thr_lo, _pgd_thr_hi = _pgd_lo, _pgd_hi
                _pgd_cnt_t = torch.zeros(1, dtype=torch.long, device=_pgd_dev)
                for _ in range(48):
                    _pgd_mid = (_pgd_thr_lo + _pgd_thr_hi) / 2.0
                    _pgd_cnt_t.zero_()
                    for _v in _pgd_imps.values():
                        _pgd_cnt_t += (_v <= _pgd_mid).sum(dtype=torch.long)
                    if _pgd_use_fsdp:
                        _dist.all_reduce(_pgd_cnt_t, op=_dist.ReduceOp.SUM)
                    if _pgd_cnt_t.item() < _pgd_k_prune:
                        _pgd_thr_lo = _pgd_mid
                    else:
                        _pgd_thr_hi = _pgd_mid
                _pgd_thr = _pgd_thr_hi

                # desired mask (uncapped) -- same as before pgd_max_swap_frac existed
                _pgd_desired = {
                    _n: (_pgd_imps[_n] > _pgd_thr) if _n in _pgd_imps else maskmgr.masks[_n].clone()
                    for _n in maskmgr.named_params
                }

                if pgd_grow_to_target and getattr(maskmgr, 'prune_n', 0) > 0 and getattr(maskmgr, 'prune_m', 0) > 0:
                    # N:M-aware PGD-driven growth: a global-threshold _pgd_desired
                    # (above) has no group structure, so it can't serve as an N:M
                    # target directly -- an N:M target must itself always be
                    # exactly prune_n-of-prune_m per group. Replace _pgd_desired
                    # with an independent per-group top-prune_n projection of
                    # CURRENT importance instead (_pgd_nm_post_target -- already
                    # used unchanged for N:M's post-target maintenance phase;
                    # reused here as-is, since "recompute the ideal N:M mask from
                    # scratch off current importance" is exactly what a moving
                    # target needs, regardless of whether the CURRENT mask is
                    # itself N:M-valid yet). Everything downstream (prune_cand/
                    # revive_cand, the self-KL bisection, tie-breaking) is then
                    # IDENTICAL to the unstructured grow_to_target path below --
                    # intermediate masks are free to violate the N:M pattern
                    # group-by-group (a group can sit at 0, 1, 2, ... dead
                    # coordinates mid-training, not just 0 or prune_n); only
                    # _pgd_desired is constrained, so once m_t converges to it
                    # (prune_cand/revive_cand both empty) every group is
                    # automatically exactly N:M again -- no separate structural
                    # eligibility check or at-target branch needed, unlike the
                    # old pre/post-target split this bypasses (see the
                    # `_pgd_is_nm and not pgd_grow_to_target` gate below).
                    _pgd_desired = _pgd_nm_post_target(
                        _pgd_imps, maskmgr.masks, maskmgr.prune_n, maskmgr.prune_m,
                        shapes=(maskmgr.named_shapes if _pgd_use_fsdp else None))

                # N:M-aware PGD (sparsity_type=2:4/4:8): plain global-threshold
                # reprojection has no group awareness and silently breaks the
                # N:M pattern (verified empirically: ~1.5-2.4% of groups ended
                # up with the wrong dead-count on a 2:4 checkpoint pruned this
                # way). Two regimes, since the correctness requirement differs:
                # before TR-GMP growth reaches final_sparsity, a group only
                # needs to never be over-pruned (a cap suffices, free to grow
                # asymmetrically across groups); once at final_sparsity, every
                # group must stay at EXACTLY (prune_m-prune_n) dead forever
                # after (a cap alone would freeze the mask completely, since
                # every group already at its cap has zero spare prune budget)
                # -- so post-target uses an independent per-group top-prune_n
                # recompute instead. See _pgd_nm_pre_target/_pgd_nm_post_target.
                _pgd_is_nm = getattr(maskmgr, 'prune_n', 0) > 0 and getattr(maskmgr, 'prune_m', 0) > 0
                if _pgd_is_nm and pgd_grow_to_target:
                    # N:M-aware PGD-driven growth, ONE joint self-KL-gated
                    # bisection per step (not three independently-budgeted
                    # phases -- an earlier version of this code did that and
                    # was caught by user review: three separate <=delta
                    # checks do NOT imply the combined transition is <=delta,
                    # since KL has no triangle inequality
                    # (D(Q_t,Q_A)<=delta and D(Q_A,Q_B)<=delta does not bound
                    # D(Q_t,Q_B)). The ORIGINAL spec was always a single
                    # check on the FULL candidate (theta_{t+1}, m_{t+1})
                    # against the true pre-PGD state (theta_t, m_t) -- that
                    # is what this now does).
                    #
                    # Per-group classification by current alive-count a_g vs
                    # prune_n still splits candidates into three POOLS (this
                    # part of the design was independently verified correct
                    # by user review and is unchanged):
                    #   - a_g > prune_n (OVERSHOOT): prune-only pool, capped
                    #     per-group at (a_g - prune_n).
                    #   - a_g < prune_n (UNDERSHOOT): revive-only pool,
                    #     capped per-group at (prune_n - a_g).
                    #   - a_g == prune_n (FINISHED): atomic per-group swap
                    #     pool (group is the candidate unit, not a coordinate
                    #     -- see _pgd_nm_finished_swap_2d).
                    #
                    # What changed: instead of bisecting each pool's own k
                    # independently and applying phases sequentially (so
                    # each phase's KL was measured against the PREVIOUS
                    # phase's already-mutated state), a single shared
                    # fraction alpha in [0,1] scales all three pools'
                    # accepted counts together (k_x = round(alpha * n_x)),
                    # and EVERY bisection trial's KL is measured by building
                    # the FULL combined candidate straight from
                    # _pgd_nm_masks_before (the true snapshot taken before
                    # any of this step's changes) -- maskmgr.masks is never
                    # mutated until the bisection is done and exactly one
                    # alpha is applied, once, atomically. This guarantees
                    # D_KL(Q_before || Q_after) <= pgd_kl_budget for the
                    # WHOLE step by construction, not just per-phase.
                    _pgd_nm_masks_before = {_n: maskmgr.masks[_n].clone() for _n in maskmgr.named_params}

                    _pgd_elig_prune_dir, _pgd_elig_revive_dir = _pgd_nm_directional(
                        _pgd_imps, _pgd_nm_masks_before, _pgd_desired, maskmgr.prune_n, maskmgr.prune_m,
                        shapes=(maskmgr.named_shapes if _pgd_use_fsdp else None))
                    _fin_scores, _fin_pg, _fin_rg, _fin_meta = _pgd_nm_finished_swap_build(
                        _pgd_imps, _pgd_nm_masks_before, _pgd_desired, maskmgr.prune_n, maskmgr.prune_m,
                        shapes=(maskmgr.named_shapes if _pgd_use_fsdp else None))
                    _pgd_nm_rank = _dist.get_rank() if _pgd_use_fsdp else None

                    _n_a_t = torch.tensor(sum(v.sum().item() for v in _pgd_elig_prune_dir.values()), dtype=torch.long, device=_pgd_dev)
                    _n_b_t = torch.tensor(sum(v.sum().item() for v in _pgd_elig_revive_dir.values()), dtype=torch.long, device=_pgd_dev)
                    _n_c_t = torch.tensor(sum(torch.isfinite(v).sum().item() for v in _fin_scores.values()), dtype=torch.long, device=_pgd_dev)
                    if _pgd_use_fsdp:
                        _dist.all_reduce(_n_a_t, op=_dist.ReduceOp.SUM)
                        _dist.all_reduce(_n_b_t, op=_dist.ReduceOp.SUM)
                        _dist.all_reduce(_n_c_t, op=_dist.ReduceOp.SUM)
                    _n_a, _n_b, _n_c = int(_n_a_t.item()), int(_n_b_t.item()), int(_n_c_t.item())

                    _k_a_final = _k_b_final = _k_c_final = 0
                    _kl_final = 0.0
                    if (_n_a > 0 or _n_b > 0 or _n_c > 0) and pgd_kl_budget > 0 and _pgd_kl_cal_batch is not None:
                        _pv_a, _plo_a, _phi_a = _pgd_build_topk_vals(_pgd_imps, _pgd_elig_prune_dir, False, _pgd_lo, _pgd_hi) if _n_a > 0 else (None, None, None)
                        _pv_b, _plo_b, _phi_b = _pgd_build_topk_vals(_pgd_imps, _pgd_elig_revive_dir, True, _pgd_lo, _pgd_hi) if _n_b > 0 else (None, None, None)
                        _cache_joint = {}

                        def _nm_combined_candidate(alpha):
                            _k_a = round(alpha * _n_a)
                            _k_b = round(alpha * _n_b)
                            _k_c = round(alpha * _n_c)
                            _sel_p_a = (_pgd_topk_mask_from_vals(_pv_a, _plo_a, _phi_a, _k_a, _pgd_dev, _pgd_use_fsdp, False)
                                        if _k_a > 0 else {_n: torch.zeros_like(_m) for _n, _m in _pgd_nm_masks_before.items()})
                            _sel_r_b = (_pgd_topk_mask_from_vals(_pv_b, _plo_b, _phi_b, _k_b, _pgd_dev, _pgd_use_fsdp, True)
                                        if _k_b > 0 else {_n: torch.zeros_like(_m) for _n, _m in _pgd_nm_masks_before.items()})
                            if _k_c > 0:
                                _sel_g_c = _pgd_topk_groups_from_scores(_fin_scores, _k_c, _pgd_dev, _pgd_use_fsdp)
                                _sel_p_c, _sel_r_c = _pgd_nm_expand_group_sel(_sel_g_c, _fin_pg, _fin_rg, _fin_meta, rank=_pgd_nm_rank)
                            else:
                                _sel_p_c = {_n: torch.zeros_like(_m) for _n, _m in _pgd_nm_masks_before.items()}
                                _sel_r_c = {_n: torch.zeros_like(_m) for _n, _m in _pgd_nm_masks_before.items()}
                            _cand = {}
                            for _n in maskmgr.named_params:
                                _base = _pgd_nm_masks_before[_n]
                                if _n in _pgd_imps:
                                    _pa = _sel_p_a.get(_n, torch.zeros_like(_base))
                                    _rb = _sel_r_b.get(_n, torch.zeros_like(_base))
                                    _pc = _sel_p_c.get(_n, torch.zeros_like(_base))
                                    _rc = _sel_r_c.get(_n, torch.zeros_like(_base))
                                    _cand[_n] = (_base | _rb | _rc) & ~_pa & ~_pc
                                else:
                                    _cand[_n] = _base
                            return _cand, _k_a, _k_b, _k_c

                        def _nm_kl_at(alpha):
                            if alpha <= 0.0:
                                return 0.0, 0, 0, 0, None
                            _cand, _k_a, _k_b, _k_c = _nm_combined_candidate(alpha)
                            _kl, _ = _compute_tr_kl(fsdp_model if fsdp_model is not None else model,
                                                     _pgd_kl_cal_batch, _cand, maskmgr, str(device),
                                                     kl_reduce=tr_kl_reduce, kl_quantile=tr_kl_quantile,
                                                     ref_cache=_cache_joint)
                            if _pgd_use_fsdp:
                                _kl_t = torch.tensor([_kl], dtype=torch.float64, device=_pgd_dev)
                                _dist.broadcast(_kl_t, src=0)
                                _kl = _kl_t.item()
                            return _kl, _k_a, _k_b, _k_c, _cand

                        # _cand_final is saved from whichever bisection trial
                        # actually won, NOT rebuilt afterward -- _pgd_topk_
                        # mask_from_vals/_pgd_topk_groups_from_scores both use
                        # torch.rand_like for tie-breaking (see their
                        # docstrings), so calling _nm_combined_candidate(alpha)
                        # a SECOND time at the exact same alpha draws a
                        # DIFFERENT random tie-break and can silently produce a
                        # different mask than the one the search actually
                        # verified. Caught empirically (2:4 smoke test, step 1):
                        # bisection accepted alpha=0.0312 at kl=0.0199, but a
                        # naive re-call at the same alpha for the "independent"
                        # apply-time check measured kl=0.020121 > 0.02 on the
                        # freshly-redrawn (different) candidate and crashed.
                        # Reusing the SAME saved candidate for both the search
                        # and the apply-time re-check removes the randomness as
                        # a confound while keeping the re-check honest: it's
                        # still a genuinely fresh _compute_tr_kl call, just not
                        # a fresh MASK.
                        _alpha_lo, _alpha_hi = 0.0, 1.0
                        _kl_at_lo = 0.0
                        _cand_final = None
                        for _ in range(pgd_kl_bisect_iters):
                            _alpha_mid = (_alpha_lo + _alpha_hi) / 2.0
                            _kl_mid, _ka_mid, _kb_mid, _kc_mid, _cand_mid = _nm_kl_at(_alpha_mid)
                            if _kl_mid <= pgd_kl_budget:
                                _alpha_lo, _kl_at_lo = _alpha_mid, _kl_mid
                                _k_a_final, _k_b_final, _k_c_final = _ka_mid, _kb_mid, _kc_mid
                                _cand_final = _cand_mid
                            else:
                                _alpha_hi = _alpha_mid
                        _kl_final = _kl_at_lo
                        if _alpha_lo > 0.0 and _cand_final is not None:
                            # Re-measure the SAME (already-fixed) candidate
                            # that won the search -- a fresh _compute_tr_kl
                            # call, so this still catches a genuine bug in KL
                            # computation itself, just not a re-drawn mask.
                            _kl_final, _ = _compute_tr_kl(fsdp_model if fsdp_model is not None else model,
                                                           _pgd_kl_cal_batch, _cand_final, maskmgr, str(device),
                                                           kl_reduce=tr_kl_reduce, kl_quantile=tr_kl_quantile,
                                                           ref_cache=_cache_joint)
                            if _pgd_use_fsdp:
                                _kl_t = torch.tensor([_kl_final], dtype=torch.float64, device=_pgd_dev)
                                _dist.broadcast(_kl_t, src=0)
                                _kl_final = _kl_t.item()
                            if _kl_final > pgd_kl_budget + 1e-6:
                                raise RuntimeError(
                                    f"[pgd_nm_invariant] step={step}: whole-step self-KL check failed at apply time -- "
                                    f"D_KL(before||after)={_kl_final:.6f} > budget={pgd_kl_budget} "
                                    f"(alpha={_alpha_lo:.4f}, k_a={_k_a_final} k_b={_k_b_final} k_c={_k_c_final}).")
                            for _n in maskmgr.named_params:
                                _old = _pgd_nm_masks_before[_n]
                                _new = _cand_final[_n]
                                _pgd_revivals += int((_new & ~_old).sum().item())
                                _pgd_prunings += int((~_new & _old).sum().item())
                                maskmgr.masks[_n] = _new
                    logging.info(f"  [pgd_kl_budget][nm][grow] n_overshoot_cand={_n_a} n_undershoot_cand={_n_b} n_finished_cand={_n_c} "
                                 f"k_overshoot={_k_a_final} k_undershoot={_k_b_final} k_finished={_k_c_final} "
                                 f"kl_final={_kl_final:.6f} budget={pgd_kl_budget} "
                                 f"post_sparsity={maskmgr.current_sparsity():.4f} (step={step})")
                    _pgd_nm_check_invariant(_pgd_nm_masks_before, maskmgr.masks, maskmgr.prune_n, maskmgr.prune_m,
                                             step, shapes=(maskmgr.named_shapes if _pgd_use_fsdp else None))
                    if use_wandb and is_main_process:
                        wandb.log({"pgd/nm_overshoot_k": _k_a_final, "pgd/nm_overshoot_n_cand": _n_a,
                                   "pgd/nm_undershoot_k": _k_b_final, "pgd/nm_undershoot_n_cand": _n_b,
                                   "pgd/nm_finished_k": _k_c_final, "pgd/nm_finished_n_cand": _n_c,
                                   "pgd/nm_kl_final": _kl_final}, step=step)
                elif _pgd_is_nm:
                    _pgd_at_target = maskmgr.current_sparsity() >= final_sparsity
                    if _pgd_at_target:
                        _new_masks = _pgd_nm_post_target(_pgd_imps, maskmgr.masks, maskmgr.prune_n, maskmgr.prune_m,
                                                          shapes=(maskmgr.named_shapes if _pgd_use_fsdp else None))
                        if pgd_kl_budget > 0 and _pgd_kl_cal_batch is not None:
                            # Self-KL-gated swap instead of applying _new_masks
                            # unconditionally. _new_masks and maskmgr.masks both
                            # have EXACTLY (prune_m-prune_n) dead per group by
                            # construction (_pgd_nm_post_target_2d's keep_g always
                            # keeps exactly prune_n per group) -- so their diff's
                            # revive count equals its prune count PER GROUP, and
                            # therefore in total, automatically. No min()
                            # reconciliation needed (unlike pre-target/unstructured,
                            # whose candidates come from an unconstrained global
                            # comparison) -- simpler than that case, not more
                            # complex, since candidates are already same-group
                            # swap pairs by construction. Apply only the k
                            # most-confident of these swaps within the self-KL
                            # budget, same warm-started bisection as elsewhere,
                            # instead of the full diff every time.
                            _revive_cand = {_n: _new_masks[_n] & ~maskmgr.masks[_n] for _n in maskmgr.named_params}
                            _prune_cand = {_n: (~_new_masks[_n]) & maskmgr.masks[_n] for _n in maskmgr.named_params}
                            _n_swap_cand_t = torch.tensor(
                                sum(v.sum().item() for v in _prune_cand.values()), dtype=torch.long, device=_pgd_dev)
                            if _pgd_use_fsdp:
                                _dist.all_reduce(_n_swap_cand_t, op=_dist.ReduceOp.SUM)
                            _n_swap_cand = int(_n_swap_cand_t.item())
                            _prune_vals, _prune_vlo, _prune_vhi = _pgd_build_topk_vals(
                                _pgd_imps, _prune_cand, False, _pgd_lo, _pgd_hi)
                            _pgd_kl_ref_cache = {}

                            def _pgd_kl_at_nm_post(k):
                                """Self-KL if the k most-confident (lowest prune-side importance) diff-swaps were applied on top of the CURRENT mask."""
                                if k <= 0:
                                    return 0.0
                                _sel_p = _pgd_topk_mask_from_vals(_prune_vals, _prune_vlo, _prune_vhi, k, _pgd_dev, _pgd_use_fsdp, False)
                                _sel_r = _pgd_topk_mask(_pgd_imps, _revive_cand, k, True, _pgd_dev, _pgd_use_fsdp, _pgd_lo, _pgd_hi)
                                _cand_masks = {_n: (maskmgr.masks[_n] | _sel_r[_n]) & ~_sel_p[_n] if _n in _pgd_imps else maskmgr.masks[_n]
                                               for _n in maskmgr.named_params}
                                _kl, _ = _compute_tr_kl(fsdp_model if fsdp_model is not None else model,
                                                         _pgd_kl_cal_batch, _cand_masks, maskmgr, str(device),
                                                         kl_reduce=tr_kl_reduce, kl_quantile=tr_kl_quantile,
                                                         ref_cache=_pgd_kl_ref_cache)
                                if _pgd_use_fsdp:
                                    import torch.distributed as _dist
                                    _kl_t = torch.tensor([_kl], dtype=torch.float64, device=_pgd_dev)
                                    _dist.broadcast(_kl_t, src=0)
                                    _kl = _kl_t.item()
                                return _kl

                            # Warm-started bisection -- identical shape to
                            # pre-target's _pgd_kl_at_nm search above.
                            _pgd_k_lo, _pgd_kl_at_k_lo = 0, 0.0
                            _pgd_k_hi = _n_swap_cand
                            _pgd_iters_left = pgd_kl_bisect_iters
                            _pgd_probe = min(max(_pgd_last_k_actual, 1), _n_swap_cand) if _n_swap_cand > 0 else 0
                            if _pgd_probe > 0 and _pgd_iters_left > 0:
                                _pgd_kl_probe = _pgd_kl_at_nm_post(_pgd_probe)
                                _pgd_iters_left -= 1
                                if _pgd_kl_probe <= pgd_kl_budget:
                                    _pgd_k_lo, _pgd_kl_at_k_lo = _pgd_probe, _pgd_kl_probe
                                    _pgd_step = _pgd_probe
                                    while _pgd_k_lo < _n_swap_cand and _pgd_iters_left > 0:
                                        _pgd_step = min(_pgd_step * 2, _n_swap_cand - _pgd_k_lo)
                                        _pgd_cand = _pgd_k_lo + _pgd_step
                                        _pgd_kl_cand = _pgd_kl_at_nm_post(_pgd_cand)
                                        _pgd_iters_left -= 1
                                        if _pgd_kl_cand <= pgd_kl_budget:
                                            _pgd_k_lo, _pgd_kl_at_k_lo = _pgd_cand, _pgd_kl_cand
                                        else:
                                            _pgd_k_hi = _pgd_cand - 1
                                            break
                                    else:
                                        _pgd_k_hi = _pgd_k_lo
                                else:
                                    _pgd_k_hi = _pgd_probe - 1
                            for _ in range(_pgd_iters_left):
                                if _pgd_k_hi <= _pgd_k_lo:
                                    break
                                _pgd_k_mid = (_pgd_k_lo + _pgd_k_hi + 1) // 2
                                _pgd_kl_mid = _pgd_kl_at_nm_post(_pgd_k_mid)
                                if _pgd_kl_mid <= pgd_kl_budget:
                                    _pgd_k_lo = _pgd_k_mid
                                    _pgd_kl_at_k_lo = _pgd_kl_mid
                                else:
                                    _pgd_k_hi = _pgd_k_mid - 1
                            _k_actual = _pgd_k_lo
                            _pgd_last_k_actual = _k_actual
                            logging.info(f"  [pgd_kl_budget][nm][post_target] n_swap_cand={_n_swap_cand} "
                                         f"k_actual={_k_actual} kl_at(k_actual)={_pgd_kl_at_k_lo:.6f} "
                                         f"budget={pgd_kl_budget} pre_sparsity={maskmgr.current_sparsity():.4f} (step={step})")
                            if use_wandb and is_main_process:
                                wandb.log({"pgd/kl_at_k_actual": _pgd_kl_at_k_lo, "pgd/kl_budget": pgd_kl_budget,
                                           "pgd/k_actual": _k_actual, "pgd/n_prune_cand": _n_swap_cand}, step=step)
                            _sel_prune = _pgd_topk_mask_from_vals(_prune_vals, _prune_vlo, _prune_vhi, _k_actual, _pgd_dev, _pgd_use_fsdp, False)
                            del _prune_vals
                            _sel_revive = (_pgd_topk_mask(_pgd_imps, _revive_cand, _k_actual, True, _pgd_dev, _pgd_use_fsdp, _pgd_lo, _pgd_hi)
                                           if _k_actual > 0 else {n: torch.zeros_like(m) for n, m in maskmgr.masks.items()})
                            for _n in maskmgr.named_params:
                                _old = maskmgr.masks[_n]
                                _new = _old.clone()
                                if _n in _pgd_imps:
                                    _new = (_new | _sel_revive[_n]) & ~_sel_prune[_n]
                                _pgd_revivals += int((_new & ~_old).sum().item())
                                _pgd_prunings += int((~_new & _old).sum().item())
                                maskmgr.masks[_n] = _new
                        else:
                            # No self-KL budget configured (or calib batch not
                            # ready yet) -- original behavior: apply the
                            # freshly-recomputed ideal mask unconditionally.
                            for _n in maskmgr.named_params:
                                _old = maskmgr.masks[_n]
                                _new = _new_masks.get(_n, _old)
                                _pgd_revivals += int((_new & ~_old).sum().item())
                                _pgd_prunings += int((~_new & _old).sum().item())
                                maskmgr.masks[_n] = _new
                    else:
                        _revive_cand = {_n: _pgd_desired[_n] & ~maskmgr.masks[_n] for _n in maskmgr.named_params}
                        _eligible_prune = _pgd_nm_pre_target(_pgd_imps, maskmgr.masks, _pgd_desired, maskmgr.prune_n,
                                                              maskmgr.prune_m, _pgd_k_prune, _pgd_dev, _pgd_use_fsdp,
                                                              shapes=(maskmgr.named_shapes if _pgd_use_fsdp else None))
                        _n_elig_t = torch.tensor(
                            sum(v.sum().item() for v in _eligible_prune.values()), dtype=torch.long, device=_pgd_dev)
                        if _pgd_use_fsdp:
                            _dist.all_reduce(_n_elig_t, op=_dist.ReduceOp.SUM)
                        # BUGFIX: _k_actual (eligible-to-prune, budget-based) and
                        # n_revive_cand (genuine PGD-disagreement count) are
                        # UNRELATED quantities -- capping only revive to
                        # min(_k_actual, n_revive_cand) while pruning the full
                        # (uncapped) _k_actual, as an earlier version of this
                        # code did, breaks the "revive/prune counts stay equal"
                        # invariant this whole cap-based design depends on:
                        # whenever genuine revive candidates (n_revive_cand) are
                        # scarcer than the per-group prune budget (_n_elig_t) --
                        # the common case right after a growth event, when most
                        # groups still have spare budget -- prune count >>
                        # revive count, and PGD (running every step, not just
                        # at mask_interval) silently drives sparsity toward the
                        # structural cap within a handful of steps instead of
                        # preserving it. Verified empirically: a real 4B 2:4
                        # PGD run jumped from 20% (set by TR-GMP growth at step
                        # 32) to 50% -- the exact prune_m-prune_n structural
                        # ceiling -- by step 33, one step later. Fix: compute
                        # n_revive_cand FIRST and use the SAME final count for
                        # both prune and revive selection.
                        _n_revive_cand_t = torch.tensor(
                            sum(v.sum().item() for v in _revive_cand.values()), dtype=torch.long, device=_pgd_dev)
                        if _pgd_use_fsdp:
                            _dist.all_reduce(_n_revive_cand_t, op=_dist.ReduceOp.SUM)
                        _n_elig = int(_n_elig_t.item())
                        _n_revive_cand = int(_n_revive_cand_t.item())
                        # N:M-aware self-KL bisection (--gmp_pgd_kl_budget):
                        # same bisection machinery as the unstructured
                        # --gmp_pgd_kl_budget branch below, but the candidate
                        # set to bisect over is _eligible_prune (N:M
                        # group-capped -- see _pgd_nm_pre_target) instead of a
                        # raw unstructured prune_cand set, so the accepted
                        # swap can never exceed a group's remaining prune
                        # budget. Revive count == accepted prune count, same
                        # invariant as everywhere else in this file.
                        if pgd_kl_budget > 0 and _pgd_kl_cal_batch is not None:
                            _n_prune_cand_nm = min(_n_elig, _n_revive_cand)
                            _prune_vals, _prune_vlo, _prune_vhi = _pgd_build_topk_vals(
                                _pgd_imps, _eligible_prune, False, _pgd_lo, _pgd_hi)
                            _pgd_kl_ref_cache = {}

                            def _pgd_kl_at_nm(k):
                                """Self-KL if the k lowest-importance N:M-eligible prune candidates were applied on top of the CURRENT mask."""
                                if k <= 0:
                                    return 0.0
                                _sel = _pgd_topk_mask_from_vals(_prune_vals, _prune_vlo, _prune_vhi, k, _pgd_dev, _pgd_use_fsdp, False)
                                _cand_masks = {_n: (maskmgr.masks[_n] & ~_sel[_n]) if _n in _pgd_imps else maskmgr.masks[_n]
                                               for _n in maskmgr.named_params}
                                _kl, _ = _compute_tr_kl(fsdp_model if fsdp_model is not None else model,
                                                         _pgd_kl_cal_batch, _cand_masks, maskmgr, str(device),
                                                         kl_reduce=tr_kl_reduce, kl_quantile=tr_kl_quantile,
                                                         ref_cache=_pgd_kl_ref_cache)
                                if _pgd_use_fsdp:
                                    # _compute_tr_kl's forward pass is an FSDP collective (all_gather
                                    # per unit), but `_kl` itself is computed locally per rank -- any
                                    # floating-point discrepancy between ranks near the budget boundary
                                    # (all_gather concatenation order, etc.) would make the bisection
                                    # loop below take a DIFFERENT NUMBER of iterations (and therefore a
                                    # different number of these collective forward calls) per rank,
                                    # deadlocking every rank on the resulting mismatched call sequence.
                                    # Broadcast rank 0's value so every rank's loop makes byte-identical
                                    # decisions -- verified this was the actual cause of a real hang
                                    # (both FSDP jobs stuck forever at the same PGD step, CPU-spinning).
                                    import torch.distributed as _dist
                                    _kl_t = torch.tensor([_kl], dtype=torch.float64, device=_pgd_dev)
                                    _dist.broadcast(_kl_t, src=0)
                                    _kl = _kl_t.item()
                                return _kl

                            # Warm-started bisection -- identical shape to the
                            # unstructured branch's search below (probe near
                            # last step's accepted k, expand or fall back to
                            # plain bisection), just reusing _pgd_kl_at_nm.
                            _pgd_k_lo, _pgd_kl_at_k_lo = 0, 0.0
                            _pgd_k_hi = _n_prune_cand_nm
                            _pgd_iters_left = pgd_kl_bisect_iters
                            _pgd_probe = min(max(_pgd_last_k_actual, 1), _n_prune_cand_nm) if _n_prune_cand_nm > 0 else 0
                            if _pgd_probe > 0 and _pgd_iters_left > 0:
                                _pgd_kl_probe = _pgd_kl_at_nm(_pgd_probe)
                                _pgd_iters_left -= 1
                                if _pgd_kl_probe <= pgd_kl_budget:
                                    _pgd_k_lo, _pgd_kl_at_k_lo = _pgd_probe, _pgd_kl_probe
                                    _pgd_step = _pgd_probe
                                    while _pgd_k_lo < _n_prune_cand_nm and _pgd_iters_left > 0:
                                        _pgd_step = min(_pgd_step * 2, _n_prune_cand_nm - _pgd_k_lo)
                                        _pgd_cand = _pgd_k_lo + _pgd_step
                                        _pgd_kl_cand = _pgd_kl_at_nm(_pgd_cand)
                                        _pgd_iters_left -= 1
                                        if _pgd_kl_cand <= pgd_kl_budget:
                                            _pgd_k_lo, _pgd_kl_at_k_lo = _pgd_cand, _pgd_kl_cand
                                        else:
                                            _pgd_k_hi = _pgd_cand - 1
                                            break
                                    else:
                                        _pgd_k_hi = _pgd_k_lo
                                else:
                                    _pgd_k_hi = _pgd_probe - 1
                            for _ in range(_pgd_iters_left):
                                if _pgd_k_hi <= _pgd_k_lo:
                                    break
                                _pgd_k_mid = (_pgd_k_lo + _pgd_k_hi + 1) // 2
                                _pgd_kl_mid = _pgd_kl_at_nm(_pgd_k_mid)
                                if _pgd_kl_mid <= pgd_kl_budget:
                                    _pgd_k_lo = _pgd_k_mid
                                    _pgd_kl_at_k_lo = _pgd_kl_mid
                                else:
                                    _pgd_k_hi = _pgd_k_mid - 1
                            _k_actual = _pgd_k_lo
                            _pgd_last_k_actual = _k_actual
                            logging.info(f"  [pgd_kl_budget][nm] n_elig={_n_elig} n_revive_cand={_n_revive_cand} "
                                         f"k_actual={_k_actual} kl_at(k_actual)={_pgd_kl_at_k_lo:.6f} "
                                         f"budget={pgd_kl_budget} pre_sparsity={maskmgr.current_sparsity():.4f} (step={step})")
                            if use_wandb and is_main_process:
                                wandb.log({"pgd/kl_at_k_actual": _pgd_kl_at_k_lo, "pgd/kl_budget": pgd_kl_budget,
                                           "pgd/k_actual": _k_actual, "pgd/n_prune_cand": _n_prune_cand_nm}, step=step)
                            _sel_prune = _pgd_topk_mask_from_vals(_prune_vals, _prune_vlo, _prune_vhi, _k_actual, _pgd_dev, _pgd_use_fsdp, False)
                            del _prune_vals
                            _sel_revive = (_pgd_topk_mask(_pgd_imps, _revive_cand, _k_actual, True, _pgd_dev, _pgd_use_fsdp, _pgd_lo, _pgd_hi)
                                           if _k_actual > 0 else {n: torch.zeros_like(m) for n, m in maskmgr.masks.items()})
                        else:
                            _k_actual = min(_pgd_k_prune, _n_elig, _n_revive_cand)
                            # Trust-region cap for N:M PGD (--gmp_pgd_kl_share /
                            # --gmp_pgd_max_swap_frac), used only when
                            # --gmp_pgd_kl_budget is unset -- same cap formula
                            # as the unstructured swap_frac branch below.
                            if pgd_kl_share or pgd_max_swap_frac > 0:
                                _effective_swap_frac = _pgd_dynamic_swap_frac if pgd_kl_share else pgd_max_swap_frac
                                _pgd_cap = max(1, round(_effective_swap_frac * _pgd_stats[1].item()))
                                _k_actual = min(_k_actual, _pgd_cap)
                            _sel_prune = (_pgd_topk_mask(_pgd_imps, _eligible_prune, _k_actual, False, _pgd_dev, _pgd_use_fsdp, _pgd_lo, _pgd_hi)
                                          if _k_actual > 0 else {n: torch.zeros_like(m) for n, m in maskmgr.masks.items()})
                            _sel_revive = (_pgd_topk_mask(_pgd_imps, _revive_cand, _k_actual, True, _pgd_dev, _pgd_use_fsdp, _pgd_lo, _pgd_hi)
                                           if _k_actual > 0 else {n: torch.zeros_like(m) for n, m in maskmgr.masks.items()})
                        for _n in maskmgr.named_params:
                            _old = maskmgr.masks[_n]
                            _new = _old.clone()
                            if _n in _pgd_imps:
                                _new = (_new | _sel_revive[_n]) & ~_sel_prune[_n]
                            _pgd_revivals += int((_new & ~_old).sum().item())
                            _pgd_prunings += int((~_new & _old).sum().item())
                            maskmgr.masks[_n] = _new
                # KL-gated trust region (--gmp_pgd_kl_budget): instead of a
                # fixed swap-count cap, bisect the number of accepted PRUNE
                # candidates (lowest-importance first) so that self-KL
                # (pre-prune model || post-prune model), measured on the
                # small cached calibration batch, stays within budget.
                # Revive count is always set equal to the accepted prune
                # count (same invariant as the swap_frac branch below) --
                # revival is never separately KL-checked because a masked
                # weight is architecturally zero until later gradient steps
                # regrow it, so an instantaneous revival swap has no
                # measurable forward-pass effect; bounding revive volume to
                # the same K the prune-side KL search already found is
                # sufficient to keep it inside the same trust region (and
                # bounds how fast newly-revived capacity can drift the
                # policy between OPKD rollout-pool refreshes).
                elif pgd_kl_budget > 0 and _pgd_kl_cal_batch is not None:
                    import time as _time_dbg
                    torch.cuda.synchronize(); _t_klb0 = _time_dbg.time()
                    _revive_cand = {_n: _pgd_desired[_n] & ~maskmgr.masks[_n] for _n in maskmgr.named_params}
                    _prune_cand  = {_n: (~_pgd_desired[_n]) & maskmgr.masks[_n] for _n in maskmgr.named_params}
                    _n_prune_cand_t = torch.tensor(
                        sum(v.sum().item() for v in _prune_cand.values()), dtype=torch.long, device=_pgd_dev)
                    _n_revive_cand_t = torch.tensor(
                        sum(v.sum().item() for v in _revive_cand.values()), dtype=torch.long, device=_pgd_dev)
                    if _pgd_use_fsdp:
                        _dist.all_reduce(_n_prune_cand_t, op=_dist.ReduceOp.SUM)
                        _dist.all_reduce(_n_revive_cand_t, op=_dist.ReduceOp.SUM)
                    _n_prune_cand = int(_n_prune_cand_t.item())
                    _n_revive_cand = int(_n_revive_cand_t.item())
                    # BUGFIX: the bisection below used to search k in
                    # [0, _n_prune_cand] and then apply that SAME k to both
                    # prune and revive selection, assuming revive_cand is at
                    # least as large as whatever k the KL search picks. That
                    # assumption fails hard right after a growth event (or at
                    # step 1, before any growth has run at all): revive_cand
                    # can be far smaller than prune_cand (e.g. step 1, mask
                    # still all-kept, revive_cand==0 while prune_cand is
                    # huge from fisher*weight^2==0 ties -- see the "already
                    # keep-count-conserving by construction" comment on the
                    # fully-uncapped branch above for the same root cause).
                    # _pgd_topk_mask can't select more than exist, so revive
                    # silently comes up short of k while prune hits it
                    # exactly -- observed empirically as sparsity oscillating
                    # between ~1% and ~25% every single step. Bound the
                    # bisection's own search range by min(prune_cand,
                    # revive_cand) so k_actual can never exceed what's
                    # actually revivable, making the applied swap
                    # keep-count-conserving by construction again.
                    #
                    # EXCEPT under gmp_pgd_grow_to_target: there, prune_cand
                    # exceeding revive_cand is the whole point (that's what
                    # drives net sparsity growth -- see its docstring), so
                    # the search range is left at the FULL prune_cand instead
                    # of being capped down to revive_cand. Revive selection
                    # below saturates at min(k, revive_cand) instead, so it
                    # can never ask for more than actually exists either --
                    # same safety property, without forcing prune down to
                    # match revive.
                    if not pgd_grow_to_target:
                        _n_prune_cand = min(_n_prune_cand, _n_revive_cand)

                    # Build the expensive log-space state ONCE (same candidate
                    # set for every bisection iteration -- only k changes), and
                    # reuse it via _pgd_topk_mask_from_vals. Calling the full
                    # _pgd_topk_mask() from scratch per iteration was observed to
                    # OOM (each call re-allocates ~2x the whole model's importance
                    # tensors just for setup, x pgd_kl_bisect_iters times, on top
                    # of everything else already live at a mask_interval boundary).
                    _prune_vals, _prune_vlo, _prune_vhi = _pgd_build_topk_vals(
                        _pgd_imps, _prune_cand, False, _pgd_lo, _pgd_hi)

                    # maskmgr.masks doesn't change across this whole bisection
                    # search (only after it's done, when the winning k is
                    # applied) -- same invariant as TR's own search above --
                    # so the reference forward pass is cacheable across every
                    # _pgd_kl_at() call, including the extra kl_at_full call.
                    _pgd_kl_ref_cache = {}
                    _pgd_kl_at_calls = [0]
                    _pgd_kl_at_time = [0.0]
                    # DIAGNOSTIC (analysis-phase only, no behavior change): split
                    # _pgd_kl_at_time into "inner" (the 64-iter value-threshold
                    # search in _pgd_topk_mask_from_vals, scans the full candidate
                    # pool -- up to ~986M elements under gmp_pgd_grow_to_target)
                    # vs "fwd" (the actual _compute_tr_kl model forward pass, on
                    # the small pgd_kl_calib_size-sequence calibration batch) --
                    # to find out which one actually dominates PGD's per-call cost
                    # before deciding where (if anywhere) to optimize.
                    _pgd_topk_time = [0.0]
                    _pgd_fwd_time = [0.0]

                    def _pgd_kl_at(k):
                        """Self-KL if the k lowest-importance prune candidates were applied on top of the CURRENT mask."""
                        if k <= 0:
                            return 0.0
                        torch.cuda.synchronize(); _t_call0 = _time_dbg.time()
                        if pgd_topk_impl == 'kthvalue' and not _pgd_use_fsdp:
                            _sel = _pgd_topk_mask_from_vals_kthvalue(_prune_vals, k, _pgd_dev, False)
                        else:
                            _sel = _pgd_topk_mask_from_vals(_prune_vals, _prune_vlo, _prune_vhi, k, _pgd_dev, _pgd_use_fsdp, False)
                        torch.cuda.synchronize(); _t_topk1 = _time_dbg.time()
                        _pgd_topk_time[0] += _t_topk1 - _t_call0
                        _cand_masks = {_n: (maskmgr.masks[_n] & ~_sel[_n]) if _n in _pgd_imps else maskmgr.masks[_n]
                                       for _n in maskmgr.named_params}
                        _kl, _ = _compute_tr_kl(fsdp_model if fsdp_model is not None else model,
                                                 _pgd_kl_cal_batch, _cand_masks, maskmgr, str(device),
                                                 kl_reduce=tr_kl_reduce, kl_quantile=tr_kl_quantile,
                                                 ref_cache=_pgd_kl_ref_cache)
                        if _pgd_use_fsdp:
                            # See the matching comment in _pgd_kl_at_nm: _kl is computed
                            # locally per rank but _compute_tr_kl's forward pass is an FSDP
                            # collective, so any rank-to-rank float discrepancy near the
                            # budget boundary would desync the bisection loop's iteration
                            # count across ranks and deadlock on the mismatched collective
                            # call sequence. Broadcast rank 0's value so every rank decides
                            # identically.
                            import torch.distributed as _dist
                            _kl_t = torch.tensor([_kl], dtype=torch.float64, device=_pgd_dev)
                            _dist.broadcast(_kl_t, src=0)
                            _kl = _kl_t.item()
                        torch.cuda.synchronize()
                        _t_call2 = _time_dbg.time()
                        _pgd_fwd_time[0] += _t_call2 - _t_topk1
                        _pgd_kl_at_calls[0] += 1
                        _pgd_kl_at_time[0] += _t_call2 - _t_call0
                        return _kl

                    # Warm-started search: instead of always bisecting the full
                    # [0, n_prune_cand] range from scratch (wasting forward
                    # passes re-discovering "still 0" or "still huge" every
                    # step when consecutive steps' true answers are close),
                    # probe near last step's accepted k first, then expand
                    # (doubling) or fall back to plain bisection as needed.
                    # Same worst-case cost as the old unconditional bisection
                    # (bounded by pgd_kl_bisect_iters), strictly cheaper
                    # whenever the answer is persistently near last step's --
                    # e.g. a persistently-collapsed (k=0) regime converges in
                    # 1 forward pass instead of spending the full budget
                    # rediscovering 0 every single step.
                    _pgd_k_lo, _pgd_kl_at_k_lo = 0, 0.0  # k=0 -> KL trivially 0
                    _pgd_k_hi = _n_prune_cand
                    _pgd_iters_left = pgd_kl_bisect_iters
                    _pgd_probe = min(max(_pgd_last_k_actual, 1), _n_prune_cand) if _n_prune_cand > 0 else 0
                    if _pgd_probe > 0 and _pgd_iters_left > 0:
                        _pgd_kl_probe = _pgd_kl_at(_pgd_probe)
                        _pgd_iters_left -= 1
                        if _pgd_kl_probe <= pgd_kl_budget:
                            _pgd_k_lo, _pgd_kl_at_k_lo = _pgd_probe, _pgd_kl_probe
                            _pgd_step = _pgd_probe
                            while _pgd_k_lo < _n_prune_cand and _pgd_iters_left > 0:
                                _pgd_step = min(_pgd_step * 2, _n_prune_cand - _pgd_k_lo)
                                _pgd_cand = _pgd_k_lo + _pgd_step
                                _pgd_kl_cand = _pgd_kl_at(_pgd_cand)
                                _pgd_iters_left -= 1
                                if _pgd_kl_cand <= pgd_kl_budget:
                                    _pgd_k_lo, _pgd_kl_at_k_lo = _pgd_cand, _pgd_kl_cand
                                else:
                                    _pgd_k_hi = _pgd_cand - 1
                                    break
                            else:
                                _pgd_k_hi = _pgd_k_lo  # ran out of iters while still expanding successfully
                        else:
                            _pgd_k_hi = _pgd_probe - 1
                    for _ in range(_pgd_iters_left):
                        if _pgd_k_hi <= _pgd_k_lo:
                            break
                        _pgd_k_mid = (_pgd_k_lo + _pgd_k_hi + 1) // 2
                        _pgd_kl_mid = _pgd_kl_at(_pgd_k_mid)
                        if _pgd_kl_mid <= pgd_kl_budget:
                            _pgd_k_lo = _pgd_k_mid
                            _pgd_kl_at_k_lo = _pgd_kl_mid
                        else:
                            _pgd_k_hi = _pgd_k_mid - 1
                    _k_actual = _pgd_k_lo
                    _pgd_last_k_actual = _k_actual
                    _pgd_kl_at_full = _pgd_kl_at(_n_prune_cand)  # KL if pure/uncapped PGD had applied ALL prune_cand -- not visited by the search itself, so this is one extra forward pass purely for this reference number.
                    logging.info(f"  [DBG kl_at_timing] calls={_pgd_kl_at_calls[0]} total_forward_time={_pgd_kl_at_time[0]:.3f}s "
                                 f"avg_per_call={(_pgd_kl_at_time[0]/max(1,_pgd_kl_at_calls[0])):.3f}s "
                                 f"topk_time={_pgd_topk_time[0]:.3f}s fwd_time={_pgd_fwd_time[0]:.3f}s (step={step})")
                    logging.info(f"  [pgd_kl_budget] n_prune_cand={_n_prune_cand} n_revive_cand={_n_revive_cand} "
                                 f"k_actual={_k_actual} kl_at(k_actual)={_pgd_kl_at_k_lo:.6f} "
                                 f"kl_at(n_prune_cand)={_pgd_kl_at_full:.6f} budget={pgd_kl_budget} "
                                 f"pre_sparsity={maskmgr.current_sparsity():.4f} (step={step})")
                    if use_wandb and is_main_process:
                        # net_prune_predicted = max(0, k*-V): the theoretical
                        # net sparsity increase this call SHOULD produce, from
                        # the search result alone, before the mask is actually
                        # touched below -- should match pgd/net_growth (the
                        # ground-truth post-hoc mask diff, logged after
                        # application) almost exactly when things are working
                        # as intended; a persistent gap between the two would
                        # flag a bug.
                        wandb.log({"pgd/kl_at_k_actual": _pgd_kl_at_k_lo, "pgd/kl_at_full_pgd": _pgd_kl_at_full,
                                   "pgd/kl_budget": pgd_kl_budget,
                                   "pgd/k_actual": _k_actual, "pgd/n_prune_cand": _n_prune_cand,
                                   "pgd/n_revive_cand": _n_revive_cand,
                                   "pgd/net_prune_predicted": max(0, _k_actual - _n_revive_cand)}, step=step)
                    if pgd_topk_impl == 'kthvalue' and not _pgd_use_fsdp:
                        _sel_prune = _pgd_topk_mask_from_vals_kthvalue(_prune_vals, _k_actual, _pgd_dev, False)
                    else:
                        _sel_prune = _pgd_topk_mask_from_vals(_prune_vals, _prune_vlo, _prune_vhi, _k_actual, _pgd_dev, _pgd_use_fsdp, False)
                    del _prune_vals
                    # min(_k_actual, _n_revive_cand): under gmp_pgd_grow_to_target
                    # _k_actual can exceed _n_revive_cand by design (that excess
                    # is what nets out as growth, unmatched by any revive) --
                    # saturate instead of asking _pgd_topk_mask for more revive
                    # candidates than exist. No-op when not growing (there
                    # _k_actual <= _n_revive_cand always, per the min() above).
                    _k_revive = min(_k_actual, _n_revive_cand)
                    _sel_revive = (_pgd_topk_mask(_pgd_imps, _revive_cand, _k_revive, True, _pgd_dev, _pgd_use_fsdp, _pgd_lo, _pgd_hi)
                                   if _k_revive > 0 else {n: torch.zeros_like(m) for n, m in maskmgr.masks.items()})
                    for _n in maskmgr.named_params:
                        _old = maskmgr.masks[_n]
                        _new = _old.clone()
                        if _n in _pgd_imps:
                            _new = (_new | _sel_revive[_n]) & ~_sel_prune[_n]
                        _pgd_revivals += int((_new & ~_old).sum().item())
                        _pgd_prunings += int((~_new & _old).sum().item())
                        maskmgr.masks[_n] = _new
                    # BUGFIX: current_sparsity() does a dist.all_reduce() under FSDP
                    # (world_size>1) -- calling it a second time from inside the
                    # is_main_process-only wandb block below (as this used to)
                    # issues that collective on rank0 only, with no matching call
                    # on other ranks, desyncing the NCCL collective count across
                    # ranks and deadlocking (reproduced live: rank0 stuck here
                    # forever while other ranks race ahead). Call it once, here,
                    # unconditionally on every rank, and reuse the value below.
                    _pgd_post_sparsity = maskmgr.current_sparsity()
                    logging.info(f"  [pgd_kl_budget] applied revivals={_pgd_revivals} prunings={_pgd_prunings} "
                                 f"post_sparsity={_pgd_post_sparsity:.4f} (step={step})")
                    if use_wandb and is_main_process:
                        # Ground-truth applied counts (real mask diff, not the
                        # search's own k_actual/k_revive) -- the metric that
                        # actually matters for gmp_pgd_grow_to_target: net
                        # growth this call = prunings - revivals, should be
                        # persistently > 0 while current sparsity < target if
                        # the asymmetric revive/prune design is working.
                        wandb.log({"pgd/revivals": _pgd_revivals, "pgd/prunings": _pgd_prunings,
                                   "pgd/net_growth": _pgd_prunings - _pgd_revivals,
                                   "pgd/turnover": _pgd_revivals + _pgd_prunings,
                                   "pgd/post_sparsity": _pgd_post_sparsity}, step=step)
                # trust-region cap: limit how many positions actually flip this
                # step (--gmp_pgd_max_swap_frac, fraction of total masked
                # params). Uncapped PGD projects straight onto the full
                # top-k-by-importance set every step regardless of how many
                # positions that moves -- observed to be enormous under STE
                # (hundreds of thousands to millions/step, since param.data
                # is never hard-reset there and masked weights can grow
                # without bound, see install_ste_forward_hooks) vs ~tens/step
                # under hard-masking. When capped, only the most-confident
                # revivals (highest importance among revival candidates) and
                # most-confident prunings (lowest importance among pruning
                # candidates) are applied -- capped symmetrically since
                # revival-candidate count == pruning-candidate count exactly
                # (old and desired masks both have the same target keep-count).
                elif pgd_kl_share or pgd_max_swap_frac > 0:
                    _effective_swap_frac = _pgd_dynamic_swap_frac if pgd_kl_share else pgd_max_swap_frac
                    _revive_cand = {_n: _pgd_desired[_n] & ~maskmgr.masks[_n] for _n in maskmgr.named_params}
                    _prune_cand  = {_n: (~_pgd_desired[_n]) & maskmgr.masks[_n] for _n in maskmgr.named_params}
                    # BUGFIX: this used to gate capping on _n_revive_cand alone, on the
                    # assumption (stated below) that revive-candidate count ==
                    # prune-candidate count. That assumption only holds once TR-GMP's
                    # mask has actually diverged from all-kept -- at the very first PGD
                    # step (step 1, before any _tr_mask_update has run), maskmgr.masks
                    # is still ~all-True while _pgd_desired already reflects a nonzero
                    # target sparsity, so revive-candidates are ~0 (nothing masked out
                    # yet to revive) while prune-candidates are already huge. The old
                    # `if _n_revive_cand > _pgd_cap` was false in that case, so it fell
                    # into the uncapped `else` below and jumped the mask straight to
                    # _pgd_desired in one shot -- observed empirically as a ~71M-weight
                    # one-step pruning spike (vs. tens-to-hundreds per step normally) on
                    # every pgd_kl_share/pgd_max_swap_frac run. Gate on whichever side is
                    # larger instead, so a lopsided candidate count can't bypass the cap.
                    _n_prune_cand_t = torch.tensor(
                        sum(v.sum().item() for v in _prune_cand.values()), dtype=torch.long, device=_pgd_dev)
                    _n_revive_cand_t = torch.tensor(
                        sum(v.sum().item() for v in _revive_cand.values()), dtype=torch.long, device=_pgd_dev)
                    if _pgd_use_fsdp:
                        _dist.all_reduce(_n_revive_cand_t, op=_dist.ReduceOp.SUM)
                        _dist.all_reduce(_n_prune_cand_t, op=_dist.ReduceOp.SUM)
                    _n_revive_cand = int(_n_revive_cand_t.item())
                    _n_prune_cand = int(_n_prune_cand_t.item())
                    _pgd_cap = max(1, round(_effective_swap_frac * _pgd_stats[1].item()))
                    if max(_n_revive_cand, _n_prune_cand) > _pgd_cap:
                        _sel_revive = _pgd_topk_mask(_pgd_imps, _revive_cand, _pgd_cap, True, _pgd_dev, _pgd_use_fsdp, _pgd_lo, _pgd_hi)
                        _sel_prune  = _pgd_topk_mask(_pgd_imps, _prune_cand, _pgd_cap, False, _pgd_dev, _pgd_use_fsdp, _pgd_lo, _pgd_hi)
                        for _n in maskmgr.named_params:
                            _old = maskmgr.masks[_n]
                            _new = _old.clone()
                            if _n in _pgd_imps:
                                _new = (_new | _sel_revive[_n]) & ~_sel_prune[_n]
                            _pgd_revivals += int((_new & ~_old).sum().item())
                            _pgd_prunings += int((~_new & _old).sum().item())
                            maskmgr.masks[_n] = _new
                    else:
                        for _n in maskmgr.named_params:
                            _old = maskmgr.masks[_n]
                            _new = _pgd_desired[_n]
                            _pgd_revivals += int((_new & ~_old).sum().item())
                            _pgd_prunings += int((~_new & _old).sum().item())
                            maskmgr.masks[_n] = _new
                else:
                    # BUGFIX: snapping directly to _pgd_desired assumes it has
                    # EXACTLY the same total keep-count as the current mask
                    # (desired and current built from equal-size threshold
                    # cuts) -- true in exact arithmetic, but the 48-iter
                    # numeric bisection that builds _pgd_desired can land its
                    # threshold on a value shared by many tied importances
                    # (fisher*weight^2 == 0 for every currently-masked AND
                    # every just-revived-but-not-yet-gradient-updated weight
                    # alike), so the resulting keep-count can be off by
                    # however many candidates sit exactly at that tie. Applied
                    # directly, this desyncs revive/prune counts every step
                    # and the imbalance compounds into a slow one-directional
                    # sparsity drift even with zero external growth events
                    # (observed: 0.119 -> 0.120 over 45 steps at a nominal
                    # fixed 10% target). Same fix already used for N:M PGD
                    # above: compute both candidate sets, then cap BOTH sides
                    # to k=min(count) so the applied mask change is
                    # keep-count-conserving by construction, not by
                    # assumption.
                    _revive_cand = {_n: _pgd_desired[_n] & ~maskmgr.masks[_n] for _n in maskmgr.named_params}
                    _prune_cand  = {_n: (~_pgd_desired[_n]) & maskmgr.masks[_n] for _n in maskmgr.named_params}
                    _n_revive_cand_t = torch.tensor(
                        sum(v.sum().item() for v in _revive_cand.values()), dtype=torch.long, device=_pgd_dev)
                    _n_prune_cand_t = torch.tensor(
                        sum(v.sum().item() for v in _prune_cand.values()), dtype=torch.long, device=_pgd_dev)
                    if _pgd_use_fsdp:
                        _dist.all_reduce(_n_revive_cand_t, op=_dist.ReduceOp.SUM)
                        _dist.all_reduce(_n_prune_cand_t, op=_dist.ReduceOp.SUM)
                    _k_actual = min(int(_n_revive_cand_t.item()), int(_n_prune_cand_t.item()))
                    _sel_revive = (_pgd_topk_mask(_pgd_imps, _revive_cand, _k_actual, True, _pgd_dev, _pgd_use_fsdp, _pgd_lo, _pgd_hi)
                                   if _k_actual > 0 else {n: torch.zeros_like(m) for n, m in maskmgr.masks.items()})
                    _sel_prune = (_pgd_topk_mask(_pgd_imps, _prune_cand, _k_actual, False, _pgd_dev, _pgd_use_fsdp, _pgd_lo, _pgd_hi)
                                  if _k_actual > 0 else {n: torch.zeros_like(m) for n, m in maskmgr.masks.items()})
                    if pgd_debug_repeat_swap:
                        _pgd_repeat_flips = 0
                        _pgd_total_flips = 0
                    for _n in maskmgr.named_params:
                        _old = maskmgr.masks[_n]
                        _new = _old.clone()
                        if _n in _pgd_imps:
                            _new = (_new | _sel_revive[_n]) & ~_sel_prune[_n]
                        _pgd_revivals += int((_new & ~_old).sum().item())
                        _pgd_prunings += int((~_new & _old).sum().item())
                        if pgd_debug_repeat_swap:
                            # int8 countdown-TTL instead of an int64 absolute-step
                            # tensor -- storing a full-model-sized int64 per param
                            # (~1.4B params x 8 bytes = ~11GB, held permanently
                            # once any growth event first populates it) was
                            # observed to OOM the very next KD forward pass on a
                            # tight 80GB budget. TTL only ever needs values in
                            # [0, pgd_debug_repeat_window], so int8 is 8x smaller.
                            _flipped = _new != _old
                            if _n not in _pgd_last_flip_step:
                                _pgd_last_flip_step[_n] = torch.zeros_like(_old, dtype=torch.int8)
                            else:
                                _pgd_last_flip_step[_n] = torch.clamp(_pgd_last_flip_step[_n] - 1, min=0)
                            _recent = _pgd_last_flip_step[_n] > 0
                            _pgd_repeat_flips += int((_flipped & _recent).sum().item())
                            _pgd_total_flips += int(_flipped.sum().item())
                            _pgd_last_flip_step[_n] = torch.where(
                                _flipped, torch.full_like(_pgd_last_flip_step[_n], pgd_debug_repeat_window), _pgd_last_flip_step[_n])
                        maskmgr.masks[_n] = _new
                    # This branch (pgd_kl_budget<=0, not N:M) previously applied
                    # this swap with zero logging -- silent to the point that a
                    # 128-step pilot run's complete absence of any "[pgd*]" log
                    # line was first (wrongly) read as "PGD never fired" instead
                    # of "this is the one branch that doesn't log". Mirroring the
                    # elif branch's log line here so uncapped PGD is as
                    # observable as the KL-gated path.
                    # BUGFIX: same is_main_process-only current_sparsity() collective
                    # mismatch as the kl_budget branch above -- call unconditionally.
                    _pgd_uncapped_post_sparsity = maskmgr.current_sparsity()
                    if is_main_process:
                        logging.info(f"  [pgd_uncapped] applied revivals={_pgd_revivals} prunings={_pgd_prunings} "
                                     f"post_sparsity={_pgd_uncapped_post_sparsity:.4f} (step={step})")
                    if pgd_debug_repeat_swap and use_wandb and is_main_process:
                        wandb.log({"pgd/repeat_swap_frac": (_pgd_repeat_flips / _pgd_total_flips) if _pgd_total_flips > 0 else 0.0,
                                   "pgd/total_flips": _pgd_total_flips}, step=step)
                maskmgr.apply(fsdp_model)
                del _pgd_imps, _pgd_desired
                torch.cuda.empty_cache()

                # sum revival/pruning counts across ranks (FSDP only)
                if _pgd_use_fsdp:
                    _pgd_rv_t = torch.tensor([_pgd_revivals, _pgd_prunings],
                                             dtype=torch.long, device=_pgd_dev)
                    _dist.all_reduce(_pgd_rv_t, op=_dist.ReduceOp.SUM)
                    _pgd_revivals, _pgd_prunings = int(_pgd_rv_t[0].item()), int(_pgd_rv_t[1].item())

            if use_wandb and is_main_process:
                wandb.log({"pgd/revivals": _pgd_revivals, "pgd/prunings": _pgd_prunings}, step=step)

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
                                        kd_temperature, onpolicy_topk, reverse=onpolicy_reverse_kl,
                                        chunk_size=kl_chunk_size)
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
            current_sparsity = 0.0 if step <= dense_warmup_steps else _schedule_fn(
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
            if blockwise_enabled:
                log_dict["train/blockwise_loss"] = accum_blockwise
                log_dict["train/block_size"] = _block_size
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
            accum_blockwise      = 0.0
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
    maskmgr.update(fisher, final_sparsity, fsdp_model,
                   block_size=(_block_size if (blockwise_enabled and maskmgr.pruning_scope == 'block') else None))
    logging.info(f"Final sparsity: {maskmgr.current_sparsity():.4f}")

    if maskmgr.ste:
        # STE mode never hard-resets param.data during training (that's the
        # whole point -- see _apply_mask) -- so param.data is still dense at
        # this point even though maskmgr.masks / current_sparsity() already
        # reflect the final target sparsity. Hard-apply the final mask into
        # the real weights once, here, before anything gets saved or
        # evaluated -- otherwise the checkpoint on disk (and any in-run PPL/
        # sparsity sanity check) would see a dense model.
        with torch.no_grad():
            for name, param in maskmgr.named_params.items():
                param.data.mul_(maskmgr.masks[name])
        logging.info("STE finalize: hard-applied final mask into param.data before save/eval.")

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
            pad_val = -100 if k == 'labels' else (0 if k == 'attention_mask' else pad_token_id)
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
