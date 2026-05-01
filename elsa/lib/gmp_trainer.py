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
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_linear_weights(model):
    """Return {name: param} for all Linear weight tensors, excluding lm_head and embeddings."""
    skip = {"lm_head", "embed_tokens", "embed_out"}
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf not in skip:
                result[name + ".weight"] = module.weight
    return result


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
    """Accumulates empirical Fisher diagonal (running avg of g_i^2) for Linear weights."""

    def __init__(self, named_params, beta=0.999):
        self.beta = beta
        self.named_params = named_params          # {name: param}
        self.F = {n: torch.zeros_like(p.data) for n, p in named_params.items()}
        self._step = 0

    def update(self):
        """Call after loss.backward() but before optimizer.step()."""
        self._step += 1
        b = self.beta
        for name, param in self.named_params.items():
            if param.grad is not None:
                self.F[name].mul_(b).addcmul_(param.grad.data, param.grad.data, value=1 - b)

    def importance(self, name, param):
        """Fisher-weighted magnitude: F_hat_ii * w_i^2."""
        f = self.F[name]
        if self._step > 0:
            # bias correction
            f = f / (1.0 - self.beta ** self._step)
        return f * param.data ** 2


# ---------------------------------------------------------------------------
# Mask manager
# ---------------------------------------------------------------------------

class GradualMaskManager:
    """Maintains binary masks and updates them on a schedule."""

    def __init__(self, named_params):
        self.named_params = named_params
        # start fully dense
        self.masks = {n: torch.ones_like(p.data, dtype=torch.bool)
                      for n, p in named_params.items()}

    @torch.no_grad()
    def update(self, fisher: FisherAccumulator, sparsity: float):
        """Recompute global mask at target sparsity using Fisher importance."""
        if sparsity <= 0.0:
            return

        # collect all importance scores globally
        scores = []
        for name, param in self.named_params.items():
            scores.append(fisher.importance(name, param).flatten())
        all_scores = torch.cat(scores)

        k = int(all_scores.numel() * sparsity)
        if k == 0:
            return
        threshold = torch.kthvalue(all_scores, k).values

        for name, param in self.named_params.items():
            imp = fisher.importance(name, param)
            self.masks[name] = imp > threshold
            _apply_mask(param, self.masks[name])

    def apply(self):
        """Zero out masked weights (call after every optimizer step)."""
        for name, param in self.named_params.items():
            _apply_mask(param, self.masks[name])

    def current_sparsity(self):
        total = sum(m.numel() for m in self.masks.values())
        zeros = sum((~m).sum().item() for m in self.masks.values())
        return zeros / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def _kl_loss(s_logits, t_logits, labels, temperature, topk):
    """Token-level forward KL D(teacher||student) on CoT positions (labels != -100).

    Applies the same shift as HuggingFace NTP: logits[:-1] predicts labels[1:].
    Returns (loss, diag_dict) where diag_dict has overlap/entropy metrics for topk > 0.
    """
    # align: logit at position t predicts token at t+1
    s_logits = s_logits[:, :-1, :]       # (B, T-1, V)
    t_logits = t_logits[:, :-1, :]
    labels   = labels[:, 1:]             # (B, T-1)
    mask = (labels != -100).float()
    denom = mask.sum().clamp(min=1)
    if mask.sum() == 0:
        return s_logits.new_tensor(0.0), {}

    s_logp_full = F.log_softmax(s_logits / temperature, dim=-1)
    t_logp_full = F.log_softmax(t_logits / temperature, dim=-1)
    diag = {}

    if topk > 0:
        t_topk = t_logits.topk(topk, dim=-1)
        t_topk_idx = t_topk.indices                          # (B, T-1, K)
        s_topk_idx = s_logits.topk(topk, dim=-1).indices

        t_logp = t_logp_full.gather(-1, t_topk_idx)
        s_logp = s_logp_full.gather(-1, t_topk_idx)

        kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)

        with torch.no_grad():
            # Overlap ratio: |S_topK ∩ T_topK| / K
            overlap = (s_topk_idx.unsqueeze(-1) == t_topk_idx.unsqueeze(-2)).any(dim=-1)
            diag["kd/overlap_ratio"] = ((overlap.float().mean(dim=-1) * mask).sum() / denom).item()

            # Entropy gap: H(T_topK) - H(S_topK)
            s_logp_s = s_logp_full.gather(-1, s_topk_idx)
            s_ent = -(s_logp_s.exp() * s_logp_s).sum(dim=-1)
            t_ent = -(t_logp.exp() * t_logp).sum(dim=-1)
            diag["kd/entropy_gap"] = (((s_ent - t_ent).abs() * mask).sum() / denom).item()
    else:
        kl = (t_logp_full.exp() * (t_logp_full - s_logp_full)).sum(dim=-1)

    loss = (kl * mask).sum() / denom
    return loss, diag


def globalprune_gmp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    FLAGS,
    teacher_model: AutoModelForCausalLM = None,
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

    total_steps    = FLAGS.gmp_steps
    batch_size     = getattr(FLAGS, 'gmp_batch_size', 1)
    grad_accum     = getattr(FLAGS, 'gmp_grad_accum', 8)
    lr             = getattr(FLAGS, 'gmp_lr', 1e-5)
    warmup_ratio   = getattr(FLAGS, 'gmp_warmup_ratio', 0.05)
    mask_interval  = getattr(FLAGS, 'gmp_mask_interval', 32)
    fisher_beta    = getattr(FLAGS, 'gmp_fisher_beta', 0.999)
    final_sparsity = FLAGS.sparsity_ratio
    warmup_steps   = int(total_steps * warmup_ratio)
    use_wandb      = getattr(FLAGS, 'wandb', False)
    kd_lambda      = getattr(FLAGS, 'gmp_kd_lambda', 0.0)
    kd_temperature = getattr(FLAGS, 'gmp_kd_temperature', 2.0)
    kd_topk        = getattr(FLAGS, 'gmp_kd_topk', 0)
    use_kd         = (teacher_model is not None) and (kd_lambda > 0.0)

    if use_kd:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    fisher  = FisherAccumulator(named_params, beta=fisher_beta)
    maskmgr = GradualMaskManager(named_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    data_iter = _infinite(loader)

    model.train()
    optimizer.zero_grad()

    start_time = time.time()
    step = 0
    accum_loss = 0.0
    accum_ntp  = 0.0
    accum_kd   = 0.0
    accum_diag: dict = {}
    accum_diag_n = 0

    logging.info("***** Running GMP Training *****")
    logging.info(f"  Total steps = {total_steps}")
    logging.info(f"  Batch size  = {batch_size}, grad_accum = {grad_accum}")
    logging.info(f"  LR = {lr}, warmup = {warmup_steps} steps")
    logging.info(f"  Target sparsity = {final_sparsity}, mask_interval = {mask_interval}")
    if use_kd:
        logging.info(f"  KD: lambda={kd_lambda}, temperature={kd_temperature}, topk={kd_topk}")

    while step < total_steps:
        for micro_step in range(grad_accum):
            batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(**batch)
                ntp_loss = out.loss

                if use_kd:
                    with torch.no_grad():
                        t_out = teacher_model(**{k: v for k, v in batch.items() if k != 'labels'})
                    kl, kd_diag = _kl_loss(out.logits, t_out.logits, batch['labels'],
                                           kd_temperature, kd_topk)
                    loss = (ntp_loss + kd_lambda * kl) / grad_accum
                    accum_ntp += ntp_loss.item() / grad_accum
                    accum_kd  += kl.item() / grad_accum
                    for k, v in kd_diag.items():
                        accum_diag[k] = accum_diag.get(k, 0.0) + v
                    accum_diag_n += 1
                else:
                    loss = ntp_loss / grad_accum

            loss.backward()
            fisher.update()
            accum_loss += loss.item()

        # gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # enforce mask after every optimizer step
        maskmgr.apply()

        step += 1

        # periodic mask update
        if step % mask_interval == 0:
            current_sparsity = _cubic_sparsity(step, total_steps, final_sparsity, warmup_steps)
            maskmgr.update(fisher, current_sparsity)
            real_sparsity = maskmgr.current_sparsity()

            log_dict = {
                "train/loss": accum_loss,
                "train/sparsity": real_sparsity,
                "train/target_sparsity": current_sparsity,
                "train/lr": scheduler.get_last_lr()[0],
                "step": step,
            }
            if use_kd:
                log_dict["train/ntp_loss"] = accum_ntp
                log_dict["train/kd_loss"]  = accum_kd
                if accum_diag_n > 0:
                    log_dict.update({k: v / accum_diag_n for k, v in accum_diag.items()})
            logging.info(f"Step {step}/{total_steps} | loss={accum_loss:.4f} | "
                         f"sparsity={real_sparsity:.3f} | lr={scheduler.get_last_lr()[0]:.2e}")
            if use_wandb and wandb.run is not None:
                wandb.log(log_dict, step=step)
            accum_loss = 0.0
            accum_ntp  = 0.0
            accum_kd   = 0.0
            accum_diag = {}
            accum_diag_n = 0

    # final mask at full sparsity
    maskmgr.update(fisher, final_sparsity)
    logging.info(f"Final sparsity: {maskmgr.current_sparsity():.4f}")

    # save
    saved_path = None
    if getattr(FLAGS, 'save_model', False) and getattr(FLAGS, 'gmp_save_path', None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = f"{FLAGS.gmp_save_path}/{_run_tag(FLAGS)}_{ts}"
        model.save_pretrained(saved_path)
        tokenizer.save_pretrained(saved_path)
        logging.info(f"Saved pruned model to {saved_path}")

    # optional downstream eval
    if eval_fn is not None:
        metrics = eval_fn(model)
        if use_wandb and wandb.run is not None:
            wandb.log(metrics, step=step)

    total_time = time.time() - start_time
    logging.info(f"GMP training done in {total_time/3600:.2f}h")
    return saved_path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _collate(batch):
    # Only use fields needed for NTP forward pass
    ntp_keys = [k for k in batch[0].keys() if k in ('input_ids', 'attention_mask', 'labels')]
    max_len = max(b['input_ids'].shape[0] for b in batch)
    pad_id = 0
    result = {}
    for k in ntp_keys:
        tensors = []
        for b in batch:
            t = b[k]
            pad_val = -100 if k == 'labels' else pad_id
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
    return f"gmp_s{int(sp*100)}pct_lr{lr}"
