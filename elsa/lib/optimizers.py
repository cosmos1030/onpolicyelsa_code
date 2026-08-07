import torch
from typing import Union, Dict
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate
import math

# Import base optimizers
from torch.optim import Adam, AdamW, SGD
from torch.optim.optimizer import _get_scalar_dtype, _device_dtype_check_for_fused
from torchao.optim import Adam8bit,Adam4bit
from .utils import FP8Config, FP8State, ScalingType, _as_dense_a

def _is_dtensor(x): 
    return hasattr(x, "to_local")

def _loc(x):
    # Return local shard if DTensor, otherwise the tensor itself
    return x.to_local() if _is_dtensor(x) else x

def get_admm_optimizer(base_optimizer_cls):
    """
    Factory function to create an ADMM optimizer class that inherits from a base optimizer.
    This preserves the single-class structure required for FSDP compatibility.
    """
    base_optimizer_cls = base_optimizer_cls.lower()
    if base_optimizer_cls not in ['adam', 'adamw', 'adam8bit', 'adam4bit', 'sgd']:
        raise ValueError("base_optimizer_cls must be one of 'adam', 'adamw', 'adam8bit', 'adam4bit', or 'sgd'.")
    if base_optimizer_cls == 'adam':
        base_optimizer_cls = Adam
    elif base_optimizer_cls == 'adamw':
        base_optimizer_cls = AdamW
    elif base_optimizer_cls == 'adam8bit':
        base_optimizer_cls = Adam8bit
    elif base_optimizer_cls == 'adam4bit':
        base_optimizer_cls = Adam4bit
    elif base_optimizer_cls == 'sgd':
        base_optimizer_cls = SGD
    class ADMMOptimizer(base_optimizer_cls):
        """
        ADMM optimizer built by subclassing a base optimizer (e.g., Adam).
        - Proximal term is added AFTER gradient clipping and BEFORE the actual step.
        - Compatible with FSDP2/DTensor: all state kept per-shard, reductions in fp32.
        """
        def __init__(
            self,
            param_groups,
            projection_fn,
            sparsity: float,
            interval: int,
            # ADMM specific arguments
            lmda: float = 1e-3, # For constant schedule
            init_lmda: float = 0.0, # For scheduling
            final_lmda: float = 0.01, # For scheduling
            lmda_schedule_mode: str = 'constant', # 'constant', 'linear', 'cosine', 'exponential'
            total_steps: int = 1, # Total steps for fixed lmda schedules
            lasso_lmda: float = 0.0, # L1 penalty on prunable weights; 0 = disabled
            prune_n: int = 0,
            prune_m: int = 0,
            projection_mode: str = "identity",   # 'identity' | 'momentum'
            projection_bias_correction: bool = False, # use bias correction in projection (for momentum)
            dual_dtype: str = 'fp32',
            split_dtype: str = 'fp32',
            accelerator=None,                    # optional: to get world_size and device
            init_lambda_from_inv_resid: bool = False,
            dynamic_barrier: bool = False,        # replace fixed-lmda proximal term with an adaptive coefficient
            barrier_alpha: float = 0.5,           # how aggressively to close the residual-vs-target gap per step
            barrier_beta: float = 0.8,            # per-interval target = beta * residual at interval start
            barrier_eps: float = 1e-12,
            barrier_lambda_max: float = 100.0,    # safety clamp against small-||r||^2 blowup
            **base_optimizer_kwargs
        ):
            super().__init__(param_groups, **base_optimizer_kwargs)

            # --- ADMM config ---
            self.projection      = projection_fn
            self.sparsity        = float(sparsity)
            self.interval        = int(interval)
            self.init_lmda = float(init_lmda)
            self.final_lmda = float(final_lmda)
            self.lmda_schedule_mode = lmda_schedule_mode.lower()
            self.init_lambda_from_inv_resid = init_lambda_from_inv_resid

            if self.lmda_schedule_mode == 'constant':
                self.lmda_default = float(lmda)
            else:
                self.lmda_default = float(init_lmda)
            self.lasso_lmda = float(lasso_lmda)

            self.total_steps     = int(total_steps)
            self.prune_n         = int(prune_n)
            self.prune_m         = int(prune_m)
            self.projection_mode  = projection_mode.lower()
            self.projection_bias_correction = bool(projection_bias_correction)

            if self.lmda_schedule_mode != 'constant' and self.init_lmda is None:
                raise ValueError("For lambda scheduling, init_lmda must be provided.")

            if dual_dtype == 'bf16':
                self.dual_dtype = torch.bfloat16
            elif dual_dtype == 'fp32':
                self.dual_dtype = torch.float32
            elif dual_dtype == 'float8_e5m2':
                self.dual_dtype = torch.float8_e5m2
            elif dual_dtype == 'float8_e4m3fn':
                self.dual_dtype = torch.float8_e4m3fn
            else:
                raise ValueError(f"Unsupported dual_dtype: {dual_dtype}")

            if split_dtype == 'bf16':
                self.split_dtype = torch.bfloat16
            elif split_dtype == 'fp32':
                self.split_dtype = torch.float32
            elif split_dtype == 'float8_e5m2':
                self.split_dtype = torch.float8_e5m2
            elif split_dtype == 'float8_e4m3fn':
                self.split_dtype = torch.float8_e4m3fn
            else:
                raise ValueError(f"Unsupported split_dtype: {split_dtype}")

            if self.projection_mode not in ("identity", "momentum"):
                raise ValueError(f"projection_mode must be 'identity' or 'momentum', got {self.projection_mode}")
            if self.lmda_schedule_mode not in ('constant', 'linear', 'cosine', 'exponential'):
                raise ValueError(f"lmda_schedule_mode must be 'constant', 'linear', 'cosine', or 'exponential', got {self.lmda_schedule_mode}")

            # Runtime helpers
            self.accelerator = accelerator
            self.process_group = getattr(accelerator, "process_group", None) if accelerator is not None else None
            self.current_step = 0
            self.mask_metrics = {'step_hamming': 0.0, 'initial_hamming': 0.0, 'step_iou': 0.0, 'initial_iou': 0.0}
            # TR global z-projection: callable() -> {id(w): z_tensor}, set by ADMMTrainer
            self._z_override_fn = None
            # Diagnostic: split gradient norm into task (NTP) vs ADMM-proximal components,
            # recomputed every _proximal_update() call (i.e. every optimizer step).
            self._last_ntp_grad_norm_sq = 0.0
            self._last_admm_grad_norm_sq = 0.0

            # --- Dynamic Barrier x-update (replaces fixed lmda*(w-z+u) with an
            # adaptively-computed coefficient -- see lib/trainer.py callers for the
            # derivation). c_t (the per-interval residual target) is set at the end
            # of each _dual_update() call, using the freshly-updated z/u; None until
            # the first _dual_update() runs, at which point dynamic_barrier falls
            # back to a plain KD-only step (lambda=0) for that first interval.
            self.dynamic_barrier = bool(dynamic_barrier)
            self.barrier_alpha = float(barrier_alpha)
            self.barrier_beta = float(barrier_beta)
            self.barrier_eps = float(barrier_eps)
            self.barrier_lambda_max = float(barrier_lambda_max)
            self._barrier_c = None
            self._last_barrier_lambda = 0.0
            self._last_barrier_residual = 0.0
            self._last_barrier_dot_qr = 0.0

        def _lazy_init_admm_state(self, p: torch.nn.Parameter, group: Dict):
            """
            Lazily initialize all required states for a parameter for both ADMM and the base optimizer.
            This must be called before the base optimizer's step if ADMM state is used before it,
            as it ensures the base optimizer's state is created before we add our own ADMM state.
            For Adam8bit support, make sure to pass group, gindx, pindx to initialize Adam8bit state properly.
            """
            st = self.state[p]
            if len(st) == 0: ## optimizer states for base optimizers 
                if isinstance(self, Adam): ## lazy init of Adam state, official implementation.
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
                elif isinstance(self, (Adam4bit, Adam8bit)):
                    st["step"] = torch.tensor(0.0)
                    st["exp_avg"] = self._new_buffer(p, True)
                    st["exp_avg_sq"] = self._new_buffer(p, False)
                    if group["amsgrad"]:
                        st["max_exp_avg_sq"] = self._new_buffer(p, False)
                elif isinstance(self, SGD): ## sgd is stateless
                    pass
                else: 
                    raise NotImplementedError("Base optimizer state initialization not implemented for this optimizer.")
            if 'dual' in st: ## return if ADMM state is already initialized
                return

            # --- Initialize ADMM's state ---
            if self.dual_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                st["dual"] = FP8State.from_tensor(
                    torch.zeros_like(p), fp8_dtype=self.dual_dtype, granularity="tensorwise",
                    scaling_type=ScalingType.DYNAMIC, safety_margin=1.05,
                    sync_scales=True, process_group=self.process_group
                )
            else:
                st["dual"] = torch.zeros_like(p, dtype=self.dual_dtype, memory_format=torch.preserve_format)
            st["sparsity"] = self.sparsity


            init_importance = None
            # Initial split z: under TR z-projection, sparsity grows gradually from 0
            # (trainer._tr_z_sp starts at 0.0), so z0 must start unpruned too — otherwise
            # the first TR-z comparison pits "already pruned to final sparsity" against
            # "candidate at ~5%", producing a huge KL that can never pass the threshold
            # and permanently stalls mask growth at 0%. st["sparsity"] itself is left at
            # the final target since the non-TR fallback projection path (_dual_update,
            # final_projection) still needs it.
            _z0_sparsity = 0.0 if getattr(self, '_z_override_fn', None) is not None else st["sparsity"]
            z0 = self.projection([p.detach()], _z0_sparsity, self.prune_n, self.prune_m,
                                 [init_importance], comparison_group="layer")[0]
            if self.init_lambda_from_inv_resid:
                initial_residual = torch.norm(p.detach() - z0.detach())
                st["lmda"] = self.lmda_default / (initial_residual.item() + 1e-8)
                st['prev_lmda'] = st["lmda"]
            else:
                st["lmda"] = self.lmda_default
                st['prev_lmda'] = self.lmda_default

            if self.split_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                st["split"] = FP8State.from_tensor(
                    z0, fp8_dtype=self.split_dtype, granularity="tensorwise",
                    scaling_type=ScalingType.DYNAMIC, safety_margin=1.05,
                    sync_scales=True, process_group=self.process_group
                )
            else:
                st["split"] = z0.detach().clone().to(device=p.device, dtype=self.split_dtype)
            st["initial_split"] = z0.detach().ne(0).clone().to(device=p.device)

        @torch.no_grad()
        def _compute_barrier_lambda(self):
            """
            Dynamic Barrier coefficient: the minimum lambda_k such that
            v = q + lambda_k * r satisfies r^T v >= alpha*(g(x) - c_t), where
            q is the (pre-proximal) task gradient, r = w - z + u, and
            g(x) = 0.5*||r||^2. Closed form for a single linear constraint:

                lambda_k = max((phi_k - q^T r) / (||r||^2 + eps), 0)
                phi_k    = alpha * (g(x) - c_t)

            All dot products are summed GLOBALLY across every ADMM param (and
            all-reduced across ranks under FSDP, since each rank only holds a
            shard) -- this is one scalar lambda_k shared by every ADMM param
            this step, not a per-parameter value. Returns 0.0 (falls back to
            pure KD gradient, no ADMM pull) until the first _dual_update() has
            set self._barrier_c.
            """
            if self._barrier_c is None:
                return 0.0

            dot_qr = torch.zeros((), device='cpu', dtype=torch.float64)
            norm_r_sq = torch.zeros((), device='cpu', dtype=torch.float64)

            for g in self.param_groups:
                if not g.get("admm", False):
                    continue
                for w in g["params"]:
                    if w.grad is None:
                        continue
                    self._lazy_init_admm_state(w, g)
                    st = self.state[w]
                    dual = st["dual"].dequant() if isinstance(st["dual"], FP8State) else st["dual"]
                    split = st["split"].dequant() if isinstance(st["split"], FP8State) else st["split"]
                    r = (w.detach() - split.detach() + dual.detach())
                    r_local = _loc(r).float()
                    q_local = (w.grad.to_local() if hasattr(w.grad, "to_local") else w.grad).detach().float()
                    dot_qr += (q_local * r_local).sum().double().cpu()
                    norm_r_sq += (r_local * r_local).sum().double().cpu()

            if dist.is_initialized():
                _t = torch.tensor([dot_qr.item(), norm_r_sq.item()], dtype=torch.float64,
                                  device=next(iter(self.state.keys())).device if self.state else 'cuda')
                dist.all_reduce(_t, op=dist.ReduceOp.SUM)
                dot_qr, norm_r_sq = _t[0].cpu(), _t[1].cpu()

            g_val = 0.5 * norm_r_sq.item()
            phi = self.barrier_alpha * (g_val - self._barrier_c)
            lam = max((phi - dot_qr.item()) / (norm_r_sq.item() + self.barrier_eps), 0.0)
            lam = min(lam, self.barrier_lambda_max)

            self._last_barrier_lambda = lam
            self._last_barrier_residual = g_val
            self._last_barrier_dot_qr = dot_qr.item()
            return lam

        @torch.no_grad()
        def _proximal_update(self):
            """
            Add proximal term to gradients AFTER global gradient clipping and
            BEFORE the actual optimizer step. This ensures proximal is not clipped.
            We also scale proximal to match distributed gradient averaging.
            """
            # Determine world size for average scaling (DDP/FSDP usually average grads across ranks)
            if self.accelerator is not None and getattr(self.accelerator, "num_processes", None):
                world = int(self.accelerator.num_processes)
            elif dist.is_initialized():
                world = dist.get_world_size()
            else:
                world = 1
            avg_div = world if world > 0 else 1

            ntp_sq = torch.zeros((), device='cpu')
            admm_sq = torch.zeros((), device='cpu')

            _barrier_lam = self._compute_barrier_lambda() if self.dynamic_barrier else None

            for g in self.param_groups:
                if not g.get("admm", False):
                    continue
                for w in g["params"]:
                    if w.grad is None:
                        continue
                    self._lazy_init_admm_state(w, g)
                    st = self.state[w]
                    dual, split = st["dual"], st["split"]
                    lmda = _barrier_lam if _barrier_lam is not None else st["lmda"]
                    ## for fp8 states, upcast to fp32 for computation
                    dual = dual.dequant() if isinstance(dual, FP8State) else dual
                    split = split.dequant() if isinstance(split, FP8State) else split

                    # Proximal term: λ (w - z + u), add to gradient before optimizer step
                    # (λ is either the fixed schedule value st["lmda"], or -- under
                    # dynamic_barrier -- the single global lambda_k computed above,
                    # shared by every ADMM param this step).
                    penalty = w.detach() - split.detach() + dual.detach()
                    prox = lmda * penalty
                    # Lasso term: lasso_lmda * sign(w), applied to all weights
                    if self.lasso_lmda > 0.0:
                        prox = prox + self.lasso_lmda * w.detach().sign()
                    prox_local = _loc(prox)
                    prox_local = prox_local.to(w.grad.dtype)
                    if avg_div > 1:
                        prox_local = prox_local / avg_div

                    gl = w.grad.to_local() if hasattr(w.grad, "to_local") else w.grad
                    # Diagnostic: task-gradient norm (pre-proximal) vs proximal-term norm,
                    # accumulated before prox is added so the two components are separable.
                    ntp_sq += gl.detach().float().pow(2).sum().cpu()
                    admm_sq += prox_local.detach().float().pow(2).sum().cpu()
                    gl.add_(prox_local)

            if dist.is_initialized():
                _t = torch.tensor([ntp_sq.item(), admm_sq.item()], device=next(iter(self.state.keys())).device if self.state else 'cuda')
                dist.all_reduce(_t, op=dist.ReduceOp.SUM)
                ntp_sq, admm_sq = _t[0].cpu(), _t[1].cpu()
            self._last_ntp_grad_norm_sq = ntp_sq.item()
            self._last_admm_grad_norm_sq = admm_sq.item()

        def get_grad_norm_breakdown(self) -> Dict[str, float]:
            """Return (ntp_grad_norm, admm_grad_norm) from the most recent _proximal_update()."""
            return {
                'ntp_grad_norm': self._last_ntp_grad_norm_sq ** 0.5,
                'admm_grad_norm': self._last_admm_grad_norm_sq ** 0.5,
            }

        def get_barrier_diagnostics(self) -> Dict[str, float]:
            """Dynamic Barrier diagnostics from the most recent _proximal_update()/_dual_update()."""
            return {
                'barrier/lambda': self._last_barrier_lambda,
                'barrier/residual': self._last_barrier_residual,
                'barrier/dot_qr': self._last_barrier_dot_qr,
                'barrier/target_c': self._barrier_c if self._barrier_c is not None else float('nan'),
            }

        @torch.no_grad()
        def _dual_update(self):
            """
            Every 'interval' steps, update split (z) and dual (u), and compute mask_diff.
            - z^{k+1} = Proj(w + u)
            - u^{k+1} = u + α (w - z^{k+1})
            Also compute global mask flip ratio between old z and new z.
            """
            if (self.current_step % self.interval) != 0:
                return

            # TR global z-projection: if callback set, use it instead of layerwise projection
            _z_override = {}
            if self._z_override_fn is not None:
                try:
                    _z_override = self._z_override_fn() or {}
                except Exception as _e:
                    import logging as _logging
                    _logging.warning(f"TR z_override_fn failed: {_e}")

            self.mask_metrics = {'step_hamming': 0.0, 'initial_hamming': 0.0, 'step_iou': 0.0, 'initial_iou': 0.0,
                                  'revived_frac': 0.0, 'newly_pruned_frac': 0.0,
                                  'revived_cycle2_frac': 0.0, 'pruned_cycle2_frac': 0.0,
                                  'revived_w2_ratio': 0.0, 'pruned_w2_ratio': 0.0}
            admm_groups = 0
            # Dynamic Barrier: global residual right after this interval's z/u
            # refresh, summed across every admm group/param (not averaged like the
            # mask-churn diagnostics above -- g = 0.5*||r||^2 is a single global
            # quantity). Used below to set self._barrier_c = beta * g_start, the
            # shrinking feasibility target for the NEXT interval's x-updates.
            _barrier_r_sq_total = torch.zeros((), dtype=torch.float64)

            for g in self.param_groups:
                if not g.get("admm", False):
                    continue
                admm_groups += 1
                weights = list(g["params"])
                if not weights:
                    continue

                device = weights[0].device
                flip_sum_step = torch.tensor(0, device=device, dtype=torch.int64)
                flip_sum_initial = torch.tensor(0, device=device, dtype=torch.int64)
                revived_sum = torch.tensor(0, device=device, dtype=torch.int64)
                newly_pruned_sum = torch.tensor(0, device=device, dtype=torch.int64)
                revived_cycle2_sum = torch.tensor(0, device=device, dtype=torch.int64)
                pruned_cycle2_sum = torch.tensor(0, device=device, dtype=torch.int64)
                revived_w2_sum = torch.zeros((), device=device, dtype=torch.float64)
                pruned_w2_sum = torch.zeros((), device=device, dtype=torch.float64)
                stable_kept_w2_sum = torch.zeros((), device=device, dtype=torch.float64)
                stable_kept_count = torch.tensor(0, device=device, dtype=torch.int64)
                barrier_r_sq_sum = torch.zeros((), device=device, dtype=torch.float64)
                per_param_log = []
                intersection_step = torch.tensor(0, device=device, dtype=torch.int64)
                union_step = torch.tensor(0, device=device, dtype=torch.int64)
                intersection_initial = torch.tensor(0, device=device, dtype=torch.int64)
                union_initial = torch.tensor(0, device=device, dtype=torch.int64)
                numel_sum = torch.tensor(0, device=device, dtype=torch.int64)

                for w in weights:
                    if w.numel() == 0:
                        continue
                    st = self.state[w]
                    if "initial_split" not in st:
                        self._lazy_init_admm_state(w, g)
                    initial_split = st["initial_split"]
                    spars = st["sparsity"]
                    current_lmda = st["lmda"]
                    previous_lmda = st["prev_lmda"]

                    ## for fp8 states, upcast to fp32 for computation
                    dual = st["dual"].dequant() if isinstance(st["dual"], FP8State) else st["dual"]
                    split = st["split"].dequant() if isinstance(st["split"], FP8State) else st["split"]

                    if current_lmda != previous_lmda:
                        dual.mul_(previous_lmda / current_lmda)

                    
                    importance_i = st.get("importance", None)
                    ## objective-aware projection
                    if self.projection_mode == "momentum":
                        v_t = st.get("exp_avg_sq",None)
                        if v_t is None:
                            raise ValueError("For momentum projection mode, optimizer must have 'exp_avg_sq' state (e.g., Adam).")
                        if self.projection_bias_correction:
                            beta2 = g.get('betas', (0.9, 0.95))[1]
                            importance_i = v_t / (1.0 - beta2**(st.get("step", 1)))
                        else:
                            importance_i = v_t
                        

                    if id(w) in _z_override:
                        z_new = _z_override[id(w)].to(w.device).detach().clone()
                    else:
                        z_in  = (w.detach() + dual.detach())
                        z_new = self.projection([z_in], spars, self.prune_n, self.prune_m,
                                                [importance_i], comparison_group="layer")[0]
                        z_new = z_new.detach().clone().to(w.device)

                    u_new = dual.detach() + (w.detach() - z_new)

                    w_l = _loc(w)
                    s_l = _loc(split)
                    d_l = _loc(dual)
                    z_new_l = _loc(z_new)

                    if self.dynamic_barrier:
                        # Fresh residual r = w - z_new + u_new, i.e. exactly the
                        # proximal-term residual the NEXT _compute_barrier_lambda()
                        # call will see once split/dual are committed below.
                        u_new_l = _loc(u_new)
                        r_new_l = w_l.float() - z_new_l.float() + u_new_l.float()
                        barrier_r_sq_sum += (r_new_l * r_new_l).sum().double()

                    new_lmda_for_param = current_lmda
                    t = self.current_step
                    T = self.total_steps
                    s0 = self.init_lmda
                    s1 = self.final_lmda

                    if self.lmda_schedule_mode == 'constant':
                        new_lmda_for_param = current_lmda
                    elif self.lmda_schedule_mode == 'linear':
                        new_lmda_for_param = s0 + (s1 - s0) * (t / T)
                    elif self.lmda_schedule_mode == 'cosine':
                        new_lmda_for_param = s0 + (s1 - s0) * 0.5 * (1 - math.cos(math.pi * t / T))
                    elif self.lmda_schedule_mode == 'exponential':
                        if s1 <= 0:
                            raise ValueError("For exponential lambda schedule, final_lmda must be positive.")
                        if s0 < 0:
                            raise ValueError("For exponential lambda schedule, init_lmda must be non-negative.")

                        if s0 == 0:
                            s0_eff = 1e-12
                            new_lmda_for_param = s0_eff * (s1 / s0_eff)**(t / T)
                        else:
                            new_lmda_for_param = s0 * (s1/s0)**(t/T)

                    old_local = _loc(split)
                    new_local = _loc(z_new)
                    initial_local = _loc(initial_split)

                    old_mask = (old_local != 0)
                    new_mask = (new_local != 0)
                    initial_mask = initial_local

                    flip_local_step = (old_mask ^ new_mask).sum().to(device=device)
                    flip_local_initial = (initial_mask ^ new_mask).sum().to(device=device)
                    numel_local = torch.tensor(old_local.numel(), device=device)

                    intersection_step += (old_mask & new_mask).sum().to(device=device)
                    union_step += (old_mask | new_mask).sum().to(device=device)
                    intersection_initial += (initial_mask & new_mask).sum().to(device=device)
                    union_initial += (initial_mask | new_mask).sum().to(device=device)

                    flip_sum_step += flip_local_step
                    flip_sum_initial += flip_local_initial
                    numel_sum += numel_local
                    # Per-layer churn breakdown: which module is oscillating, not just
                    # how much overall. Only meaningful on rank 0 / non-distributed
                    # (per_param_log is never all-reduced, just logged for inspection).
                    _pname = getattr(self, '_param_id_to_name', {}).get(id(w), None)
                    if _pname is not None:
                        per_param_log.append((_pname, float(flip_local_step) / (numel_local.item() + 1e-12), numel_local.item()))
                    # Split the step flip into direction: revived (pruned -> kept) vs
                    # newly pruned (kept -> pruned) — these sum to flip_local_step.
                    revived_mask = (~old_mask & new_mask)
                    newly_pruned_mask = (old_mask & ~new_mask)
                    revived_sum += revived_mask.sum().to(device=device)
                    newly_pruned_sum += newly_pruned_mask.sum().to(device=device)

                    # Does churn hit "important" (large-magnitude) weights disproportionately?
                    # Compare avg w^2 of flipped params against avg w^2 of the stably-kept
                    # population (kept before AND after this interval) — same churn %
                    # could have wildly different loss impact depending on which weights
                    # it touches.
                    w_sq = _loc(w).detach().float().pow(2)
                    stable_kept_mask = old_mask & new_mask
                    revived_w2_sum += w_sq[revived_mask].sum().double().to(device=device)
                    pruned_w2_sum += w_sq[newly_pruned_mask].sum().double().to(device=device)
                    stable_kept_w2_sum += w_sq[stable_kept_mask].sum().double().to(device=device)
                    stable_kept_count += stable_kept_mask.sum().to(device=device)

                    # Period-2 oscillation check: of the params revived this interval
                    # (pruned last interval, kept now), how many were ALSO kept 2
                    # intervals ago (kept -> pruned -> kept, a clean flip-flop on the
                    # same coordinates)? Symmetric check for newly-pruned params that
                    # were also pruned 2 intervals ago (kept -> pruned -> ... no wait,
                    # kept briefly then pruned again). Requires caching the mask from
                    # 2 intervals back per-param.
                    mask_2ago = st.get("_mask_2ago", None)
                    if mask_2ago is not None:
                        revived_cycle2_sum += (revived_mask & mask_2ago).sum().to(device=device)
                        pruned_cycle2_sum += (newly_pruned_mask & ~mask_2ago).sum().to(device=device)
                    st["_mask_2ago"] = old_mask.clone()

                    ## for fp8 states, requantize to save fp8 states
                    st['dual'].requant(u_new) if isinstance(st['dual'], FP8State) else dual.copy_(u_new)
                    st['split'].requant(z_new) if isinstance(st['split'], FP8State) else split.copy_(z_new)
                    
                    st["lmda"] = new_lmda_for_param
                    st["prev_lmda"] = current_lmda

                if per_param_log and (not dist.is_initialized() or dist.get_rank() == 0):
                    import logging as _logging
                    per_param_log.sort(key=lambda x: x[1], reverse=True)
                    top = ", ".join(f"{n}={h:.3f}" for n, h, _ in per_param_log[:8])
                    bot = ", ".join(f"{n}={h:.3f}" for n, h, _ in per_param_log[-5:])
                    _logging.info(f"  [layer churn] step {self.current_step} highest: {top}")
                    _logging.info(f"  [layer churn] step {self.current_step} lowest:  {bot}")

                if dist.is_initialized():
                    dist.all_reduce(flip_sum_step,  op=dist.ReduceOp.SUM)
                    dist.all_reduce(flip_sum_initial, op=dist.ReduceOp.SUM)
                    dist.all_reduce(revived_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(newly_pruned_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(revived_cycle2_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(pruned_cycle2_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(revived_w2_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(pruned_w2_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(stable_kept_w2_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(stable_kept_count, op=dist.ReduceOp.SUM)
                    dist.all_reduce(intersection_step, op=dist.ReduceOp.SUM)
                    dist.all_reduce(union_step, op=dist.ReduceOp.SUM)
                    dist.all_reduce(intersection_initial, op=dist.ReduceOp.SUM)
                    dist.all_reduce(union_initial, op=dist.ReduceOp.SUM)
                    dist.all_reduce(numel_sum, op=dist.ReduceOp.SUM)
                    if self.dynamic_barrier:
                        dist.all_reduce(barrier_r_sq_sum, op=dist.ReduceOp.SUM)

                if self.dynamic_barrier:
                    _barrier_r_sq_total += barrier_r_sq_sum.double().cpu()

                eps = 1e-12
                self.mask_metrics['step_hamming'] += float(flip_sum_step.float() / (numel_sum.float() + eps))
                self.mask_metrics['initial_hamming'] += float(flip_sum_initial.float() / (numel_sum.float() + eps))
                self.mask_metrics['step_iou'] += float(intersection_step.float() / (union_step.float() + eps))
                self.mask_metrics['initial_iou'] += float(intersection_initial.float() / (union_initial.float() + eps))
                # Fraction of all prunable params that flipped, split by direction —
                # of the total, how many were revived (pruned->kept) vs newly pruned
                # (kept->pruned) at this interval. revived_frac + newly_pruned_frac == step_hamming.
                self.mask_metrics['revived_frac'] += float(revived_sum.float() / (numel_sum.float() + eps))
                self.mask_metrics['newly_pruned_frac'] += float(newly_pruned_sum.float() / (numel_sum.float() + eps))
                # Of the params revived/newly-pruned THIS interval, what fraction were
                # also kept/pruned (respectively) 2 intervals ago — i.e. literally the
                # same coordinates flip-flopping (kept -> pruned -> kept), rather than a
                # continually-different pool churning through. Denominator is the
                # revived/pruned count itself, not numel_sum, since this is a
                # conditional fraction ("of the ones that flipped, how many are repeat
                # offenders").
                self.mask_metrics['revived_cycle2_frac'] += float(revived_cycle2_sum.float() / (revived_sum.float() + eps))
                self.mask_metrics['pruned_cycle2_frac'] += float(pruned_cycle2_sum.float() / (newly_pruned_sum.float() + eps))
                # Are the flipped params "bigger" (more important) than the params that
                # stayed kept the whole time? Ratio of avg w^2 among revived/newly-pruned
                # params to avg w^2 among stably-kept params — same churn % could have
                # very different loss impact depending on which weights it touches.
                revived_avg_w2 = revived_w2_sum / (revived_sum.double() + eps)
                pruned_avg_w2 = pruned_w2_sum / (newly_pruned_sum.double() + eps)
                stable_avg_w2 = stable_kept_w2_sum / (stable_kept_count.double() + eps)
                self.mask_metrics['revived_w2_ratio'] += float(revived_avg_w2 / (stable_avg_w2 + eps))
                self.mask_metrics['pruned_w2_ratio'] += float(pruned_avg_w2 / (stable_avg_w2 + eps))

            if admm_groups > 0:
                self.mask_metrics['step_hamming'] /= admm_groups
                self.mask_metrics['initial_hamming'] /= admm_groups
                self.mask_metrics['step_iou'] /= admm_groups
                self.mask_metrics['initial_iou'] /= admm_groups
                self.mask_metrics['revived_frac'] /= admm_groups
                self.mask_metrics['newly_pruned_frac'] /= admm_groups
                self.mask_metrics['revived_cycle2_frac'] /= admm_groups
                self.mask_metrics['pruned_cycle2_frac'] /= admm_groups
                self.mask_metrics['revived_w2_ratio'] /= admm_groups
                self.mask_metrics['pruned_w2_ratio'] /= admm_groups

            if self.dynamic_barrier:
                g_start = 0.5 * _barrier_r_sq_total.item()
                self._barrier_c = self.barrier_beta * g_start

        @torch.no_grad()
        def step(self, closure=None):
            """
            1) (Trainer did backward and clipping)
            2) _proximal_update() adds proximal term to grad
            3) super().step() uses combined grad
            4) _dual_update() for z/u
            """
            self._proximal_update()
            out = super().step(closure)
            self._dual_update()
            self.current_step += 1
            return out

        @torch.no_grad()
        def final_projection(self):
            """
            Apply the final projection to ADMM-tagged parameter groups (in-place).
            Called after training AND at every intermediate sparse-eval (see
            trainer._evaluate_sparse_model), so it must always reflect the CURRENT
            w, not a stale snapshot.

            Under TR z-projection, sparsity is grown gradually and validated via a KL
            trust region (or the cubic schedule), so the achieved level may be below
            st["sparsity"] — the fixed final target. Jumping straight to st["sparsity"]
            here would discard all of that gradual validation. Previously this branch
            just copied st["split"] verbatim, which is the z from the LAST admm_interval
            dual update (up to admm_interval-1 steps stale) — between updates, every
            intermediate eval kept showing that same frozen snapshot, then jumped
            discretely the instant split refreshed, producing a staircase in
            eval/sparse_loss even though the underlying dense w was training smoothly.
            Fixed to re-derive the mask fresh from the CURRENT w+dual each call, same
            as the non-TR path, just targeting the sparsity level already achieved
            (read off st["split"]'s current zero-fraction) instead of the fixed final
            target.
            """
            _tr_z_active = getattr(self, '_z_override_fn', None) is not None
            for g in self.param_groups:
                if not g.get("admm", False):
                    continue
                for w in g["params"]:
                    if w.numel() == 0:
                        continue
                    st = self.state[w]
                    importance = None
                    if self.projection_mode == "momentum":
                        v_t = st.get("exp_avg_sq")
                        if self.projection_bias_correction:
                            beta2 = g.get('betas', (0.9, 0.95))[1]
                            importance = v_t / (1.0 - beta2**(st.get("step", 1)))
                        else:
                            importance = v_t
                        if isinstance(importance, DTensor):
                            importance = importance.redistribute(placements=[Replicate()]).to_local()

                    if _tr_z_active:
                        z_prev = st["split"].dequant() if hasattr(st["split"], 'dequant') else st["split"]
                        achieved_sparsity = float((_as_dense_a(z_prev) == 0).float().mean().item())
                        dual = st["dual"].dequant() if isinstance(st["dual"], FP8State) else st["dual"]
                        z_in = w.detach() + dual.detach()
                        wnew = self.projection([z_in], achieved_sparsity, self.prune_n, self.prune_m,
                                               [importance], comparison_group="layer")[0]
                        w.data.copy_(wnew.to(w.dtype))
                        continue

                    wnew = self.projection([w.detach()], st["sparsity"], self.prune_n, self.prune_m,
                                           [importance], comparison_group="layer")[0]
                    w.data.copy_(wnew)

        def get_mask_metrics(self) -> Dict[str, float]:
            """
            Return the averaged mask metrics computed at the last interval update.
            """
            return self.mask_metrics

        def get_lmda_stats(self) -> Dict[str, float]:
            """
            Calculates and returns statistics (average, min, max) of per-parameter lmda values.
            """
            total_lmda = 0.0
            count = 0
            min_lmda = float('inf')
            max_lmda = float('-inf')

            for g in self.param_groups:
                if not g.get("admm", False):
                    continue
                for w in g["params"]:
                    if w in self.state:
                        lmda_val = self.state[w].get("lmda")
                        if lmda_val is not None:
                            total_lmda += lmda_val
                            count += 1
                            min_lmda = min(min_lmda, lmda_val)
                            max_lmda = max(max_lmda, lmda_val)
            
            if count == 0:
                return {"avg_lmda": 0.0, "min_lmda": 0.0, "max_lmda": 0.0}
            else:
                return {"avg_lmda": total_lmda / count, "min_lmda": min_lmda, "max_lmda": max_lmda}

    return ADMMOptimizer


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


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            grad_norm = self._grad_norm()
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAFE(torch.optim.Optimizer):
    """ADMM optimizer with SAM (Sharpness-Aware Minimization) as base optimizer."""

    def __init__(self, param_groups, projection_fn, sparsity, interval,
                 base_optimizer=torch.optim.Adam, lmda=1e-3, lr=2e-4,
                 rho=0.05, prune_n=0, prune_m=0, importance_matrix=None,
                 comparison_group='layer', betas=(0.9, 0.999), **kwargs):
        if not callable(projection_fn):
            raise TypeError("projection_fn must be callable")
        self.projection = projection_fn
        self.comparison_group = comparison_group.lower()
        self.importance_matrix = importance_matrix
        self.sparsity = sparsity
        self.interval = interval
        self.current_step = 0
        self.prune_n = prune_n
        self.prune_m = prune_m
        self.alpha = 1.0

        processed = []
        for i, group in enumerate(param_groups):
            if group.get('admm', False) and group['params']:
                admm_params = group['params']
                # Keep duals/splits in float32 regardless of weight dtype (bfloat16 has insufficient
                # precision for ADMM residual accumulation — small (w-z) values round to 0)
                group['duals'] = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in admm_params]
                group['splits'] = [s.float() for s in self.projection(admm_params, sparsity, prune_n, prune_m,
                                                                       importance_matrix, comparison_group=self.comparison_group)]
                if 'lmda' not in group:
                    group['lmda'] = lmda
            processed.append(group)

        defaults = dict(lr=lr, rho=rho, betas=betas, **kwargs)
        super(SAFE, self).__init__(processed, defaults)

        sam_pgs = [{k: v for k, v in pg.items() if k not in ['duals', 'splits', 'admm', 'lmda']}
                   for pg in self.param_groups]
        self.base_optimizer = SAM(sam_pgs, base_optimizer, rho=rho, betas=betas, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.base_optimizer.first_step(zero_grad)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            if group.get('admm', False):
                weights, lmda = group['params'], group['lmda']
                duals, splits = group['duals'], group['splits']
                for i in range(len(weights)):
                    # Compute proximal term in float32; duals/splits are already float32
                    proximal = lmda * (weights[i].detach().float() - splits[i] + duals[i])
                    weights[i].grad.add_(proximal.to(weights[i].dtype))
        self.base_optimizer.second_step(zero_grad)

        if (self.current_step + 1) % self.interval == 0:
            with torch.no_grad():
                for group in self.param_groups:
                    if group.get('admm', False):
                        weights, duals, splits = group['params'], group['duals'], group['splits']
                        for i in range(len(duals)):
                            # All arithmetic in float32 — duals/splits are float32
                            z_in = weights[i].detach().float() + duals[i]
                            z_new = self.projection([z_in], self.sparsity, prune_n=self.prune_n,
                                                    prune_m=self.prune_m,
                                                    importance_matrix=self.importance_matrix,
                                                    comparison_group=self.comparison_group)[0].float()
                            u_new = duals[i] + self.alpha * (weights[i].detach().float() - z_new)
                            duals[i].copy_(u_new)
                            splits[i].copy_(z_new)
        self.current_step += 1

    def final_projection(self):
        for group in self.param_groups:
            if group.get('admm', False):
                weights = group['params']
                final_weights = self.projection(weights, self.sparsity, prune_n=self.prune_n,
                                                prune_m=self.prune_m,
                                                importance_matrix=self.importance_matrix,
                                                comparison_group=self.comparison_group)
                for w, fw in zip(weights, final_weights):
                    w.data.copy_(fw)
