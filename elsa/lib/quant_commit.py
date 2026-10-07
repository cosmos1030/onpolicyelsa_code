"""Gradual weight quantization with ASQ's Fisher-weighted saliency.

The pruning side of this codebase already is a "gradual commitment engine":
GradualMaskManager keeps one boolean per coordinate, a global top-k selector
ranks coordinates by a Fisher-weighted quadratic cost, a self-KL trust region
paces how many decisions get applied per event, and grow_to_target lets a
decision reverse while forcing the NET count toward the target. Nothing in
that machinery cares that the boolean means "alive". This module reuses the
same shape with the boolean meaning "already snapped onto the n-bit grid".

Saliency is ASQ's (Song et al., POSTECH, ICLR 2027 submission). For weight v_i
with Fisher F_ii (Adam's bias-corrected exp_avg_sq, exactly what
FisherAccumulator.fisher_factor returns):

    G_i = F_ii * ( v_i^2  -  (v_i - qhat_i)^2 )      qhat = nearest NON-ZERO grid value
          \\____/   \\__________/
       error if pruned   error if retained

G ranks retain-vs-prune under a sparsity budget. Its second term alone,

    C_i = F_ii * (v_i - Q(v_i))^2                    Q = plain nearest grid value

is the loss increase from snapping coordinate i onto the grid, which is what
ranks commitment order when there is no sparsity budget. Both halves come from
the same expression; which one is in play depends on whether a sparsity target
is set.

Note G -> F_ii * v_i^2 as bits -> inf (qhat -> v), i.e. exactly the existing
`fisher` saliency in FisherAccumulator.importance. That degeneracy is the
correctness test in test_quant_commit.py: at 16 bits this path must reproduce
the unquantized scores.

qhat is deliberately the nearest non-zero level. Allowing qhat = 0 would make
"retain" silently able to choose zero, collapsing the retain-vs-prune contrast
that G exists to measure.
"""

import logging
import torch
import torch.nn.functional as F
import types


def grid_params(w: torch.Tensor, bits: int, group_size: int = 0, sym: bool = False,
                fisher: torch.Tensor = None, mse_grid: int = 0,
                max_shrink: float = 0.8):
    """Per-row (or per-group) quantization grid: (scale, zero), both broadcast to w's shape.

    sym=False (default) is ASYMMETRIC: the grid carries a zero-point as well as
    a scale, so it slides to cover the group's actual [min, max] instead of
    being pinned around 0, and it spans the full 2**bits levels. sym=True pins
    it at 0 and therefore only reaches 2*(2**(bits-1)-1)+1 levels.

    That difference is not cosmetic at low bit-width. Measured on an
    off-centre group (mean 0.15): 2-bit symmetric uses 2 of its 3 nominal
    levels and lands at MSE 0.0237, while 2-bit asymmetric uses all 4 and
    lands at 0.00343 -- 6.9x better. 3-bit is 3.6x, 4-bit 3.0x. GPTQ's
    Quantizer is asymmetric by default (sym=False in qwen3_main.py), so
    running our side symmetric would have handicapped every head-to-head.

    mse_grid > 0 runs ASQ's scale re-optimisation: shrink the range by p in
    [1-max_shrink, 1] and keep the p minimising the Fisher-weighted rounding
    error over each row/group (plain MSE when fisher is None, which is what
    GPTQ's Quantizer does).
    """
    maxq = (2 ** bits - 1) if not sym else (2 ** (bits - 1) - 1)
    if maxq < 1:
        raise ValueError(f"bits={bits} leaves no usable level")
    w2 = w.reshape(w.shape[0], -1) if group_size <= 0 else w.reshape(-1, group_size)
    f2 = fisher.reshape(w2.shape) if fisher is not None else None

    def _params(p):
        if sym:
            s_ = (w2.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) * p) / maxq
            return s_, torch.zeros_like(s_)
        lo = torch.minimum(w2.amin(dim=1, keepdim=True), torch.zeros(1, device=w.device, dtype=w2.dtype)) * p
        hi = torch.maximum(w2.amax(dim=1, keepdim=True), torch.zeros(1, device=w.device, dtype=w2.dtype)) * p
        s_ = ((hi - lo) / maxq).clamp(min=1e-8)
        return s_, torch.round(-lo / s_)

    if mse_grid <= 0:
        s_, z_ = _params(1.0)
        return s_.expand_as(w2).reshape(w.shape), z_.expand_as(w2).reshape(w.shape)
    best_s, best_z = _params(1.0)
    best_err = torch.full((w2.shape[0], 1), float('inf'), device=w.device, dtype=torch.float32)
    for i in range(int(mse_grid)):
        p = 1.0 - max_shrink * (i / max(1, mse_grid - 1))
        if p <= 0:
            continue
        s_, z_ = _params(p)
        q = (torch.clamp(torch.round(w2 / s_) + z_, 0 if not sym else -maxq, maxq) - z_) * s_
        err = (w2.float() - q.float()) ** 2
        if f2 is not None:
            err = err * f2.float()
        err = err.sum(dim=1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s_, best_s)
        best_z = torch.where(better, z_, best_z)
    return best_s.expand_as(w2).reshape(w.shape), best_z.expand_as(w2).reshape(w.shape)


def grid_params_from_levels(w, bits, group_size=0):
    """Recover the asymmetric grid (scale, zero) of weights that are ALREADY on one,
    e.g. a GPTQ checkpoint, for the "GPTQ -> training" baseline.

    grid_params() re-derives the grid from each group's min/max, which only
    matches GPTQ's when the group happens to use both extreme levels. Measured
    on qwen3_1.7b_gptq_w3_g128: re-deriving moved 51% of the weights (3% rel.
    error), i.e. it silently re-quantised the init the baseline is meant to
    start from. Here the step is read off the levels themselves (smallest gap,
    then refined to (max-min)/n_steps since the stored levels carry fp16
    rounding noise that biases the smallest gap low), and the zero-point is any
    integer that maps every present level into [0, maxq] (centred when the
    group leaves room on both sides). Same check: no weight moves by more than
    0.05 of a step at 3-bit, 2-bit moves 0.0002%.
    """
    maxq = 2 ** bits - 1
    w2 = (w.reshape(w.shape[0], -1) if group_size <= 0 else w.reshape(-1, group_size)).float()
    v, _ = torch.sort(w2, dim=1)
    d = v[:, 1:] - v[:, :-1]
    tol = 1e-6 * v.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    d = torch.where(d > tol, d, torch.full_like(d, float('inf')))
    s = d.amin(dim=1, keepdim=True)
    lo, hi = v[:, :1], v[:, -1:]
    bad = ~torch.isfinite(s)                       # single distinct value
    s = torch.where(bad, (hi.abs().clamp(min=1e-8) / maxq), s)
    span = torch.round((hi - lo) / s)
    # min gap is biased low by the stored levels' rounding noise; the end-to-end
    # span averages it out over every step between the extreme levels
    s = torch.where(span > 0, (hi - lo) / span.clamp(min=1), s)
    a = torch.round(-lo / s)                       # = zero - q(lo)
    z_lo = a.clamp(min=0); z_hi = torch.minimum(a + maxq - span, torch.full_like(a, maxq))
    z = torch.round((z_lo + z_hi) / 2)
    return s.expand_as(w2).reshape(w.shape), z.expand_as(w2).reshape(w.shape)


def fake_quant(w, scale, zero, bits: int, sym: bool = False):
    """Nearest grid value (the zero level is allowed)."""
    maxq = (2 ** bits - 1) if not sym else (2 ** (bits - 1) - 1)
    lo = 0 if not sym else -maxq
    return (torch.clamp(torch.round(w / scale) + zero, lo, maxq) - zero) * scale


def nearest_nonzero(w, scale, zero, bits: int, sym: bool = False):
    """Nearest grid value whose VALUE is not 0 -- ASQ's qhat.

    On an asymmetric grid the zero value sits at level index `zero`, so the
    excluded level is that one, not index 0. A coordinate that would land on
    it is pushed one level along its own sign.
    """
    maxq = (2 ** bits - 1) if not sym else (2 ** (bits - 1) - 1)
    lo = 0 if not sym else -maxq
    k = torch.clamp(torch.round(w / scale) + zero, lo, maxq)
    sgn = torch.sign(w)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    k = torch.where(k == zero, torch.clamp(zero + sgn, lo, maxq), k)
    # if clamping pushed it back onto the zero level (zero at a grid edge), go the other way
    k = torch.where(k == zero, torch.clamp(zero - sgn, lo, maxq), k)
    return (k - zero) * scale


def asq_gain(w, fisher, scale, zero, bits, sym=False):
    """G_i = F_ii (v_i^2 - (v_i - qhat_i)^2): retain-vs-prune gain."""
    qhat = nearest_nonzero(w, scale, zero, bits, sym)
    return fisher * (w.float() ** 2 - (w.float() - qhat.float()) ** 2)


def commit_cost(w, fisher, scale, zero, bits, sym=False):
    """C_i = F_ii (v_i - Q(v_i))^2: loss increase from snapping i onto the grid."""
    q = fake_quant(w, scale, zero, bits, sym)
    return fisher * (w.float() - q.float()) ** 2


class _STEQuantFn(torch.autograd.Function):
    """Straight-through fake-quant on the committed coordinates only.

    forward: w_eff = where(committed, Q(w), w). backward: gradient passes to w
    unchanged, so committed coordinates keep accumulating a real trajectory in
    the optimizer (standard QAT STE) and an un-commit later resumes from a
    mature value rather than from the snapped one.
    """

    @staticmethod
    def forward(ctx, weight, committed, scale, zero, bits, sym):
        maxq = (2 ** bits - 1) if not sym else (2 ** (bits - 1) - 1)
        lo = 0 if not sym else -maxq
        q = (torch.clamp(torch.round(weight / scale) + zero, lo, maxq) - zero) * scale
        return torch.where(committed, q, weight)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None


# ---- FSDP (classic FSDP1, use_orig_params=True) --------------------------
# Under FSDP each named_params entry is this rank's LOCAL FLAT SHARD of the
# weight (possibly empty), not the [out, in] matrix -- the same situation the
# pruning side handles with named_shapes + _fsdp_gather_flat/_fsdp_scatter_flat
# in gmp_trainer.py. The quant state follows the same layout:
#   * committed[n] is shard-shaped, like maskmgr.masks[n];
#   * the grid (scale, zero) is computed on the gathered full [out, in] weight,
#     so its groups of group_size are the real ones, and kept COMPACT and
#     replicated (one value per group, ~1/group_size of the weight);
#   * the forward hook sees the unsharded weight, so it gathers committed[n]
#     to full shape there.
# Every collective below is driven off `shapes` (rank-identical order), never
# off which shards happen to be non-empty on this rank.

def _dist_world():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return dist
    return None


def _gather_var(local_flat, sizes):
    """All-gather a flat tensor whose length differs per rank (sizes known)."""
    dist = _dist_world()
    mx = max(sizes)
    pad = local_flat.new_zeros(mx)
    if local_flat.numel() > 0:
        pad[:local_flat.numel()] = local_flat
    parts = [torch.empty_like(pad) for _ in sizes]
    dist.all_gather(parts, pad)
    return torch.cat([parts[r][:sizes[r]] for r in range(len(sizes))])


def _gather_sizes(numel, device):
    dist = _dist_world()
    t = torch.tensor([numel], device=device, dtype=torch.long)
    out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t)
    return [int(x.item()) for x in out]


class QuantCommitManager:
    """Boolean 'committed' state per coordinate + ASQ-ranked gradual commitment."""

    def __init__(self, named_params: dict, bits: int, group_size: int = 0,
                 mse_grid: int = 0, sym: bool = False, grid_from_levels: bool = False,
                 shapes: dict = None):
        self.named_params = named_params
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.mse_grid = int(mse_grid)
        self.sym = bool(sym)
        self.committed = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in named_params.items()}
        self.scales = {}
        self.zeros = {}
        if grid_from_levels and self.sym:
            raise ValueError("grid_from_levels recovers an asymmetric grid; sym=True is not supported")
        # FSDP mode: shapes = {name: (out_features, in_features)} (the pruning
        # side's named_shapes). See the FSDP block above.
        self.shapes = dict(shapes) if (shapes and _dist_world() is not None) else None
        if self.shapes is not None:
            if self.group_size <= 0:
                raise ValueError("FSDP quantization needs --gmp_quant_group_size > 0")
            for n, (o, i) in self.shapes.items():
                if i % self.group_size:
                    raise ValueError(f"{n}: in_features {i} not divisible by group_size {self.group_size}")
            self._sizes, self._off = {}, {}
            _rank = _dist_world().get_rank()
            for n in self.shapes:
                p = self.named_params[n]
                self._sizes[n] = _gather_sizes(p.numel(), p.device)
                self._off[n] = sum(self._sizes[n][:_rank])
        if grid_from_levels:
            if self.shapes is not None:
                self._refresh_scales_fsdp(from_levels=True)
            else:
                for n, p in named_params.items():
                    _s, _z = grid_params_from_levels(p.data.float(), self.bits, self.group_size)
                    self.scales[n] = _s.to(p.dtype)
                    self.zeros[n] = _z.to(p.dtype)
        else:
            self.refresh_scales()

    @torch.no_grad()
    def refresh_scales(self, fisher=None):
        if self.shapes is not None:
            return self._refresh_scales_fsdp(fisher)
        for n, p in self.named_params.items():
            f = None
            if fisher is not None:
                f = fisher.fisher_factor(p)
            _s, _z = grid_params(p.data.float(), self.bits, self.group_size, self.sym,
                                 fisher=f, mse_grid=self.mse_grid)
            self.scales[n] = _s.to(p.dtype)
            self.zeros[n] = _z.to(p.dtype)

    @torch.no_grad()
    def _refresh_scales_fsdp(self, fisher=None, from_levels=False):
        dist = _dist_world()
        gs = self.group_size
        self.scales, self.zeros = {}, {}  # compact, replicated: one value per group
        for n, (o, i) in self.shapes.items():
            p = self.named_params[n]
            sizes = self._sizes[n]
            w = _gather_var(p.data.reshape(-1), sizes).float().view(o, i)
            f_full = None
            if fisher is not None and not from_levels:
                f = fisher.fisher_factor(p)
                # an empty local shard has no grad, hence no Adam state: that
                # rank contributes nothing, it does not mean "no Fisher yet"
                ok = torch.tensor([1 if (f is not None or p.numel() == 0) else 0],
                                  device=p.device, dtype=torch.int32)
                dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                if int(ok.item()) == 1:
                    f_loc = f.reshape(-1).float() if f is not None else p.new_zeros(0, dtype=torch.float32)
                    f_full = _gather_var(f_loc, sizes).view(o, i)
            if from_levels:
                s_, z_ = grid_params_from_levels(w, self.bits, gs)
            else:
                s_, z_ = grid_params(w, self.bits, gs, self.sym, fisher=f_full, mse_grid=self.mse_grid)
            self.scales[n] = s_.reshape(-1, gs)[:, 0].contiguous().to(p.dtype)
            self.zeros[n] = z_.reshape(-1, gs)[:, 0].contiguous().to(p.dtype)
            del w, f_full, s_, z_

    def _local_grid(self, name, param):
        """(scale, zero) per coordinate of THIS rank's shard, from the compact grid."""
        if self.shapes is None:
            return self.scales[name], self.zeros[name]
        gidx = (self._off[name] + torch.arange(param.numel(), device=param.device)) // self.group_size
        return (self.scales[name][gidx].view(param.shape),
                self.zeros[name][gidx].view(param.shape))

    def committed_frac(self) -> float:
        tot = sum(m.numel() for m in self.committed.values())
        com = sum(int(m.sum().item()) for m in self.committed.values())
        dist = _dist_world() if self.shapes is not None else None
        if dist is not None:
            t = torch.tensor([com, tot], dtype=torch.float64,
                             device=next(iter(self.committed.values())).device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            com, tot = float(t[0].item()), float(t[1].item())
        return com / max(1, tot)

    @torch.no_grad()
    def _cost_one(self, name, param, fisher):
        f = fisher.fisher_factor(param)
        if f is None:
            f = torch.ones_like(param, dtype=torch.float32)
        s_, z_ = self._local_grid(name, param)
        return commit_cost(param.data.float(), f.float().view(param.shape), s_.float(),
                           z_.float(), self.bits, self.sym)

    @torch.no_grad()
    def sample_costs(self, fisher, max_samples: int = 2_000_000) -> torch.Tensor:
        """Stratified sample of commit costs, for estimating a global threshold.

        NEVER torch.cat the per-parameter cost tensors: one fp32 score tensor
        over every linear weight is ~6.8GB for Qwen3-1.7B and ~13.5GB for 4B,
        on top of the model, optimizer and activations -- the pruning side
        documents exactly this OOM at the 'global pruning: single threshold'
        comment in gmp_trainer.py, and ignoring it is what killed job 1080026
        (tried to allocate 10.47GiB). Each tensor's costs are computed, sampled
        and freed one at a time, so peak extra memory is one layer's worth.

        A fixed per-tensor sampling RATE (not a fixed per-tensor count) keeps
        the sample unbiased across differently-sized layers, which matters
        because the threshold is global.
        """
        if self.shapes is not None:
            # one GLOBAL threshold: every rank samples its shards at the same
            # rate, the samples are pooled, so all ranks read the same quantile
            tot = sum(o * i for o, i in self.shapes.values())
        else:
            tot = sum(p.numel() for p in self.named_params.values())
        rate = min(1.0, max_samples / max(1, tot))
        out = []
        for n, p in self.named_params.items():
            if p.numel() == 0:
                continue
            c = self._cost_one(n, p, fisher).reshape(-1)
            k = max(1, int(c.numel() * rate))
            idx = torch.randint(0, c.numel(), (k,), device=c.device)
            out.append(c[idx].float() if self.shapes is not None else c[idx].float().cpu())
            del c
        if self.shapes is None:
            return torch.cat(out)
        dev = next(iter(self.named_params.values())).device
        loc = torch.cat(out) if out else torch.zeros(0, device=dev)
        return _gather_var(loc, _gather_sizes(loc.numel(), dev)).cpu()

    # ---- per-event cost cache -------------------------------------------
    # Within one PGD event the weights, scales and Fisher are all frozen, so
    # every coordinate's snapping cost is constant. The first version
    # recomputed the full-model cost pass inside threshold_for AND again inside
    # preview_commitment, for each of up to 12 bisection probes: 24 full passes
    # over every linear weight per event, for 24 identical results. Sampling
    # once per event and reading every probe's threshold off that one sample
    # removes all of it -- a quantile is just a different index into the same
    # sorted sample.
    _event_sample = None

    @torch.no_grad()
    def begin_event(self, fisher, max_samples: int = 2_000_000):
        self._event_sample = self.sample_costs(fisher, max_samples)

    @torch.no_grad()
    def end_event(self):
        self._event_sample = None

    @torch.no_grad()
    def threshold_for(self, fisher, frac: float) -> float:
        frac = float(min(max(frac, 0.0), 1.0))
        if frac <= 0.0:
            return float('-inf')
        if frac >= 1.0:
            return float('inf')
        smp = self._event_sample if self._event_sample is not None else self.sample_costs(fisher)
        return float(torch.quantile(smp, frac).item())

    @torch.no_grad()
    def set_commitment(self, fisher, frac: float) -> float:
        """Commit exactly the globally cheapest `frac` of coordinates.

        The knob is the FRACTION, not a swap count. Because the schedule only
        ever asks for frac >= the current committed fraction, the net committed
        count can only move toward the target -- while an individual coordinate
        IS free to leave the set when the weights move and its snapping cost
        rises. That is the pruning side's "reversal allowed, net motion toward
        the target only" rule (grow_to_target's revive-saturates-at-prune),
        enforced directly on the quantity it is a rule about instead of
        indirectly through a per-event swap budget.
        """
        thr = self.threshold_for(fisher, frac)
        for n, p in self.named_params.items():
            c = self._cost_one(n, p, fisher)
            self.committed[n] = (c <= thr)
            del c
        return thr

    @torch.no_grad()
    def stage_commitment(self, fisher, frac: float) -> dict:
        """Write the candidate set for `frac` straight into self.committed.

        No scratch buffer and no restore. Both are unnecessary because the
        incumbent is already captured OUTSIDE this dict -- the trust-region
        screen caches the incumbent's logits once per event -- and because the
        bisection's last action is always a set_commitment() that rewrites the
        dict from the accepted fraction. So the candidate can simply overwrite
        in place.

        That matters for VRAM, not elegance: the dict is a bool per linear
        weight, 1.40GB for Qwen3-1.7B, and the earlier versions either
        allocated a fresh one per probe (~17GB of alloc/free churn per event,
        which is what fragments the allocator under max_split_size_mb:256 and
        feeds the random SIGSEGVs) or held a second one as scratch. Measured,
        the scratch was the single largest item in the event's +3.43GB peak.
        """
        thr = self.threshold_for(fisher, frac)
        for n, p in self.named_params.items():
            c = self._cost_one(n, p, fisher)
            torch.le(c, thr, out=self.committed[n])
            del c
        return self.committed

    @torch.no_grad()
    def finalize(self):
        """Hard-write the grid values into param.data for every committed
        coordinate, before anything is saved or evaluated -- the forward hook
        alone never touches param.data (that is the point of STE), so without
        this the checkpoint on disk would be full precision."""
        for n, p in self.named_params.items():
            if self.shapes is not None:
                # FSDP: p is a view into the flat param -- write in place,
                # never rebind .data
                if p.numel() == 0:
                    continue
                s_, z_ = self._local_grid(n, p)
                q = fake_quant(p.data.float(), s_.float(), z_.float(), self.bits, self.sym).to(p.dtype)
                p.data.copy_(torch.where(self.committed[n], q, p.data))
                continue
            q = fake_quant(p.data.float(), self.scales[n].float(), self.zeros[n].float(),
                           self.bits, self.sym).to(p.dtype)
            p.data = torch.where(self.committed[n], q, p.data)


def install_quant_forward_hooks(model, qmgr: QuantCommitManager, modules: dict = None):
    """Route each managed nn.Linear's forward through _STEQuantFn, reading
    qmgr.committed/qmgr.scales fresh every call so later commitment updates --
    including a whole-dict swap, which is how the trust-region screen evaluates
    a candidate without mutating any weight -- are picked up with no
    re-registration. Mirrors install_ste_forward_hooks."""
    # modules: {param name: nn.Linear}, from the same traversal that built
    # named_params -- under FSDP model.named_modules() carries the wrapper
    # prefixes and would not match.
    name_to_module = dict(model.named_modules())
    for full_name in qmgr.named_params:
        assert full_name.endswith('.weight')
        module = modules[full_name] if modules is not None else name_to_module[full_name[:-len('.weight')]]
        if qmgr.shapes is not None:
            module.forward = types.MethodType(_make_fsdp_quant_forward(qmgr, full_name), module)
            continue

        def _make_forward(pname):
            def _quant_forward(self, x):
                w = _STEQuantFn.apply(self.weight, qmgr.committed[pname],
                                      qmgr.scales[pname], qmgr.zeros[pname],
                                      qmgr.bits, qmgr.sym)
                return F.linear(x, w, self.bias)
            return _quant_forward

        module.forward = types.MethodType(_make_forward(full_name), module)


def _make_fsdp_quant_forward(qmgr: QuantCommitManager, pname: str):
    """FSDP forward: self.weight is the unsharded [out, in] weight here, while
    qmgr.committed[pname] is this rank's flat shard -- gather it (every rank
    runs the same layers in the same order, so the collective lines up) and
    apply the compact grid per group."""
    o, i = qmgr.shapes[pname]
    gs = qmgr.group_size
    sizes = qmgr._sizes[pname]

    def _quant_forward(self, x):
        c = _gather_var(qmgr.committed[pname].reshape(-1).to(torch.uint8), sizes).bool()
        w3 = self.weight.view(o, i // gs, gs)
        w = _STEQuantFn.apply(w3, c.view(o, i // gs, gs),
                              qmgr.scales[pname].view(o, i // gs, 1),
                              qmgr.zeros[pname].view(o, i // gs, 1),
                              qmgr.bits, qmgr.sym).view(o, i)
        return F.linear(x, w, self.bias)
    return _quant_forward


def remove_quant_forward_hooks(qmgr: QuantCommitManager, modules: dict):
    """Drop the per-instance forward override (back to nn.Linear.forward).
    Called after finalize() under FSDP: the grid is in param.data by then, and
    a gathering hook must not survive into a rank-0-only eval/save forward."""
    for full_name in qmgr.named_params:
        m = modules[full_name]
        if 'forward' in m.__dict__:
            del m.forward
