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
        self.F = {n: torch.zeros_like(p.data, dtype=torch.float32) for n, p in named_params.items()}
        self._step = 0

    def update(self):
        """Call after loss.backward() but before optimizer.step()."""
        self._step += 1
        b = self.beta
        for name, param in self.named_params.items():
            if param.grad is not None:
                g = param.grad.data.float()
                self.F[name].mul_(b).addcmul_(g, g, value=1 - b)

    def importance(self, name, param):
        """Fisher-weighted magnitude: F_hat_ii * w_i^2.
        Falls back to plain magnitude when Fisher is all-zero (e.g. zero-gradient loss at init).
        """
        f = self.F[name]
        if self._step > 0:
            f = f / (1.0 - self.beta ** self._step)  # bias correction
        imp = f * param.data ** 2
        # If all-zero (no gradient signal yet), use magnitude-based pruning as fallback
        if imp.sum() == 0:
            imp = param.data ** 2
        return imp


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

        if torch.isnan(all_scores).any() or torch.isinf(all_scores).any():
            logging.warning("NaN/Inf in Fisher importance scores, skipping mask update")
            return

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
    warmup_ratio        = getattr(FLAGS, 'gmp_warmup_ratio', 0.05)
    mask_interval       = getattr(FLAGS, 'gmp_mask_interval', 32)
    log_interval        = getattr(FLAGS, 'gmp_log_interval', 1)
    fisher_beta         = getattr(FLAGS, 'gmp_fisher_beta', 0.999)
    final_sparsity      = FLAGS.sparsity_ratio
    warmup_steps        = int(total_steps * warmup_ratio)
    pruning_end_ratio   = getattr(FLAGS, 'gmp_pruning_end_ratio', 1.0)
    pruning_end_steps   = int(total_steps * pruning_end_ratio)
    use_wandb      = getattr(FLAGS, 'wandb', False)
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
    use_kd         = (teacher_model is not None) and (kd_lambda > 0.0)
    use_hidden     = (teacher_model is not None) and (hidden_lambda > 0.0)
    use_onpolicy   = (teacher_model is not None) and (onpolicy_lambda > 0.0)
    use_anchor     = (teacher_model is not None) and (anchor_lambda > 0.0)

    if use_kd or use_hidden or use_onpolicy:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    # Prompt dataset for on-policy generation
    prompt_iter = None
    if use_onpolicy:
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

    rollout_buffer = RolloutBuffer() if use_rollout else None

    fisher  = FisherAccumulator(named_params, beta=fisher_beta)
    maskmgr = GradualMaskManager(named_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    _pad_tok = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    _collate_fn = lambda b: _collate(b, pad_token_id=_pad_tok)

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    model.train()
    optimizer.zero_grad()

    start_time = time.time()
    step = 0
    accum_loss      = 0.0
    accum_ntp       = 0.0
    accum_kd        = 0.0
    accum_grad_norm = 0.0
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

        # ── NTP + offline KD micro-steps ──────────────────────────────────────
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
                        loss = (ntp_loss + aux) / grad_accum
                    if not skip_ntp:
                        accum_ntp += ntp_loss.item() / grad_accum
                else:
                    loss = ntp_loss / grad_accum
                    accum_ntp += ntp_loss.item() / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss at micro_step {micro_step}, skipping")
                continue
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
                maskmgr.apply()

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
            current_sparsity = _cubic_sparsity(min(step, pruning_end_steps), pruning_end_steps, final_sparsity, warmup_steps)
            if step <= pruning_end_steps:
                maskmgr.update(fisher, current_sparsity)
            else:
                maskmgr.apply()

        # ── On-policy: rollout collection + RL grad accumulation (combined step fires below) ──
        if use_onpolicy and step % onpolicy_interval == 0:
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
                maskmgr.apply()

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
                _diag_s_ent = 0.0
                _diag_t_ent = 0.0
                _diag_overlap = 0.0

                for _op_i in range(onpolicy_grad_accum):
                    p_batch = next(prompt_iter)
                    prompt_ids  = p_batch['input_ids'].to(device)
                    prompt_mask = p_batch['attention_mask'].to(device)
                    prompt_len  = int(p_batch['prompt_len'].item())

                    model.config.use_cache = True
                    model.eval()
                    _t_gen = time.time()
                    if use_mixed:
                        generated = _mixed_sample(
                            model, teacher_model, prompt_ids, prompt_mask,
                            onpolicy_max_new, onpolicy_mixed_alpha, onpolicy_temp,
                            _pad_id, _eos_id,
                        )
                    else:
                        with torch.no_grad():
                            generated = model.generate(
                                input_ids=prompt_ids,
                                attention_mask=prompt_mask,
                                max_new_tokens=onpolicy_max_new,
                                do_sample=True,
                                temperature=onpolicy_temp,
                                pad_token_id=_pad_id,
                            )
                    _total_gen_time += time.time() - _t_gen
                    _total_gen_tokens += generated.shape[1] - prompt_len
                    model.train()
                    model.config.use_cache = False
                    maskmgr.apply()

                    gen_labels = generated.clone()
                    gen_labels[:, :prompt_len] = -100
                    gen_labels[generated == _pad_id] = -100

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        s_out = model(input_ids=generated)
                        with torch.no_grad():
                            t_out = teacher_model(input_ids=generated)

                        op_kl, op_diag = _kl_loss(s_out.logits, t_out.logits, gen_labels,
                                                  kd_temperature, onpolicy_topk,
                                                  reverse=onpolicy_reverse_kl)

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
                            op_loss = (onpolicy_lambda * op_kl + onpolicy_lambda * pg) / (grad_accum * onpolicy_grad_accum)
                        else:
                            op_loss = onpolicy_lambda * op_kl / (grad_accum * onpolicy_grad_accum)

                    if torch.isnan(op_loss) or torch.isinf(op_loss):
                        logging.warning(f"NaN/Inf on-policy loss at step {step} micro {_op_i}, skipping")
                    else:
                        op_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), onpolicy_grad_clip)
                        fisher.update()
                        optimizer.step()
                        optimizer.zero_grad()
                        maskmgr.apply()
                        accum_onpolicy += op_kl.item()

                    _diag_kl      += op_kl.item()
                    _diag_s_ent   += _s_ent_mean.item()
                    _diag_t_ent   += _t_ent_mean.item()
                    _diag_overlap += _overlap_mean.item()

                accum_onpolicy /= onpolicy_grad_accum
                _n = onpolicy_grad_accum
                accum_onpolicy_diag.update({
                    "onpolicy/kl_loss":              _diag_kl / _n,
                    "onpolicy/gen_tokens":           _total_gen_tokens / _n,
                    "onpolicy/gen_time_sec":         _total_gen_time,
                    "onpolicy/student_entropy":      _diag_s_ent / _n,
                    "onpolicy/teacher_entropy":      _diag_t_ent / _n,
                    "onpolicy/entropy_gap":          (_diag_s_ent - _diag_t_ent) / _n,
                    "onpolicy/overlap_ratio_top100": _diag_overlap / _n,
                })
                accum_onpolicy_diag.update({f"onpolicy/{k.split('/')[-1]}": v
                                            for k, v in op_diag.items()})

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
        maskmgr.apply()

        # periodic logging
        if step % log_interval == 0:
            real_sparsity = maskmgr.current_sparsity()
            current_sparsity = _cubic_sparsity(min(step, pruning_end_steps), pruning_end_steps, final_sparsity, warmup_steps)
            log_dict = {
                "train/loss": accum_loss,
                "train/ntp_loss": accum_ntp,
                "train/sparsity": real_sparsity,
                "train/target_sparsity": current_sparsity,
                "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": accum_grad_norm / log_interval,
                "step": step,
            }
            if use_kd or use_hidden:
                log_dict["train/aux_loss"] = accum_kd
                if accum_diag_n > 0:
                    log_dict.update({k: v / accum_diag_n for k, v in accum_diag.items()})
            if use_onpolicy:
                if accum_onpolicy > 0 or not use_rollout:
                    log_dict["train/onpolicy_kd_loss"] = accum_onpolicy
                log_dict.update(accum_onpolicy_diag)
            logging.info(f"Step {step}/{total_steps} | loss={accum_loss:.4f} | "
                         f"sparsity={real_sparsity:.3f} | lr={scheduler.get_last_lr()[0]:.2e}")
            if use_wandb and wandb.run is not None:
                wandb.log(log_dict, step=step)
            accum_loss           = 0.0
            accum_ntp            = 0.0
            accum_kd             = 0.0
            accum_grad_norm      = 0.0
            accum_diag           = {}
            accum_diag_n         = 0
            accum_onpolicy_diag  = {}

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

def _collate(batch, pad_token_id=0):
    # Only use fields needed for NTP forward pass
    ntp_keys = [k for k in batch[0].keys() if k in ('input_ids', 'attention_mask', 'labels')]
    max_len = max(b['input_ids'].shape[0] for b in batch)
    result = {}
    for k in ntp_keys:
        tensors = []
        for b in batch:
            t = b[k]
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
