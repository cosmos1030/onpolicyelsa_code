"""
lib/grpo_opkd.py

GRPO-style On-Policy KD (GRPO-OPKD) for GMP sparse training.

Version 1: sequence-level advantage.

Loss structure:
    L = L_NTP + grpo_lambda * L_GRPO_OPKD

GRPO-OPKD:
    r_{i,t}  = log p_T(y_t|s_t) - log q_old(y_t|s_t)
    R_i      = (1/T_i) * sum_t r_{i,t}          (sequence-level reward)
    A_i      = (R_i - mu_G) / (std_G + eps)     (group-relative advantage)
    rho_{i,t}= exp(log q_theta - log q_old)
    L_GRPO   = -mean[ min(rho*A, clip(rho, 1-eps, 1+eps)*A) ]

Optionally adds on-policy reverse KL on rollouts:
    L_KD     = KL(p_T || q_theta) on student-generated tokens
    L_total  = L_GRPO + kd_lambda * L_KD
"""

import re
import time
import torch
import torch.nn.functional as F
from absl import logging
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from lib.gmp_trainer import (
    _find_linear_weights,
    _infinite,
    _collate,
    _cubic_sparsity,
    _kl_loss,
    _mixed_sample,
    _run_tag,
    FisherAccumulator,
    GradualMaskManager,
)
from lib.gkd_admm_trainer import (
    collate_prompts,
    MathPromptWithAnswerDataset,
    collate_prompts_with_answers,
)


# ── Correctness reward helpers ────────────────────────────────────────────────

def _extract_boxed(text: str):
    """Extract last \\boxed{} content with brace-balanced matching."""
    idx = text.rfind(r'\boxed{')
    if idx == -1:
        return None
    depth = 0
    start = idx + len(r'\boxed{')
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None  # unclosed brace = truncated


def _math_verify_correct(pred: str, gold: str) -> bool:
    """lm_eval/lighteval과 동일한 math_verify 기반 정답 비교.

    gold: HF dataset answer 필드 (raw string, \boxed 없음)
    pred: 모델 생성에서 추출한 \boxed{} 내용 (raw string)
    """
    try:
        from math_verify import parse, verify, LatexExtractionConfig
        return verify(
            parse(f'\\boxed{{{gold}}}', extraction_config=[LatexExtractionConfig()]),
            parse(f'\\boxed{{{pred}}}', extraction_config=[LatexExtractionConfig()]),
        )
    except Exception:
        # fallback: 정규화 exact match
        pred_n = pred.strip().replace(' ', '').replace('\n', '')
        gold_n = gold.strip().replace(' ', '').replace('\n', '')
        if pred_n == gold_n:
            return True
        try:
            return abs(float(pred_n) - float(gold_n)) < 1e-6
        except (ValueError, TypeError):
            return False


def compute_correctness_reward(
    generated: str,
    gold: str,
    format_reward: float = 0.1,
) -> tuple:
    """
    Correctness + format reward for one rollout.

    Returns (reward: float, diag: dict) where diag tracks:
      truncated  : no </think> found
      has_boxed  : \boxed{} present in answer part
      correct    : answer matches gold

    Reward schedule:
      truncated                          → 0.0
      </think> present, no \boxed{}     → format_reward * 0.5
      </think> + \boxed{}, wrong answer → format_reward
      correct answer                    → 1.0

    정답 비교: math_verify (lighteval과 동일) 사용.
    """
    has_think_close = '</think>' in generated
    if not has_think_close:
        return 0.0, {'truncated': 1, 'has_boxed': 0, 'correct': 0}

    answer_part = generated.split('</think>')[-1]
    pred = _extract_boxed(answer_part)

    if pred is None:
        return format_reward * 0.5, {'truncated': 0, 'has_boxed': 0, 'correct': 0}

    if _math_verify_correct(pred, gold):
        return 1.0, {'truncated': 0, 'has_boxed': 1, 'correct': 1}

    return format_reward, {'truncated': 0, 'has_boxed': 1, 'correct': 0}


# ── GRPO-OPKD loss ────────────────────────────────────────────────────────────

def grpo_opkd_loss(
    student,
    teacher,
    prompt_ids,         # (B, Lp)
    prompt_mask,        # (B, Lp)
    num_rollouts,       # G
    max_new_tokens,
    mixed_alpha,
    temperature,
    eps_clip,
    kd_lambda,          # weight for on-policy reverse KL on rollouts
    kd_topk,
    pad_id,
    eos_id,
    tokenizer=None,     # required when gold_answers is provided
    gold_answers=None,  # list[str] of length B; if set, use correctness reward
    format_reward=0.1,  # partial reward for correct format but wrong answer
    grpo_lambda=1.0,    # passed in so per-seq backward uses correct scale
):
    """
    Compute GRPO-OPKD loss for one batch of prompts.

    Two reward modes:
      - gold_answers is None  → log-ratio reward (teacher required)
      - gold_answers provided → correctness + format reward (teacher optional)

    Returns:
        loss  : scalar tensor (with grad)
        diag  : dict of loggable scalars
    """
    B, Lp = prompt_ids.shape
    G = num_rollouts
    device = prompt_ids.device

    # 1. Repeat each prompt G times → (B*G, Lp)
    rep_ids  = prompt_ids.repeat_interleave(G, dim=0)
    rep_mask = prompt_mask.repeat_interleave(G, dim=0)

    # 2. Generate G rollouts per prompt
    student.eval()
    student.config.use_cache = True
    with torch.no_grad():
        if mixed_alpha > 0.0 and teacher is not None:
            gen_ids = _mixed_sample(
                student, teacher, rep_ids, rep_mask,
                max_new_tokens, mixed_alpha, temperature, pad_id, eos_id,
            )
        else:
            gen_ids = student.generate(
                input_ids=rep_ids,
                attention_mask=rep_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=pad_id,
            )
    student.train()
    student.config.use_cache = False

    # gen_ids: (B*G, Lp + T_gen)
    T_gen = gen_ids.shape[1] - Lp
    if T_gen <= 0:
        zero = prompt_ids.new_zeros(1, dtype=torch.float).requires_grad_(True).sum()
        return zero, {}

    gen_part = gen_ids[:, Lp:]                              # (B*G, T_gen)
    gen_mask = (gen_part != pad_id).float()                 # (B*G, T_gen)

    # full attention mask: prompt (left-padded) + completion (ones up to pad)
    full_mask = torch.cat([rep_mask, gen_mask.long()], dim=1)  # (B*G, Lp+T_gen)

    def _get_logps(model, scale=1.0):
        """Forward pass → log-probs for generated tokens only.
        Processes one sequence at a time to avoid OOM from (B*G, T_gen, vocab) float32.
        """
        BG = gen_ids.shape[0]
        lp_tok_list = []
        out_ref = None
        for i in range(BG):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out_i = model(
                    input_ids=gen_ids[i:i+1],
                    attention_mask=full_mask[i:i+1],
                )
            # (T_gen, vocab) in bf16 — only generated positions
            logits_i  = out_i.logits[0, Lp - 1:-1]
            tgt_i     = gen_part[i].clamp(min=0)                               # (T_gen,)
            tgt_logit = logits_i.gather(-1, tgt_i.unsqueeze(-1)).squeeze(-1)   # (T_gen,)
            # Use bf16 logsumexp to avoid materialising full float32 vocab tensor
            log_z     = torch.logsumexp(logits_i / scale, dim=-1)              # (T_gen,)
            lp_i      = tgt_logit.float() / scale - log_z.float()              # (T_gen,) fp32
            lp_tok_list.append(lp_i)
            del logits_i, tgt_logit, log_z
            out_ref = out_i          # keep last output for optional KL reuse
        lp_tok = torch.stack(lp_tok_list, dim=0)   # (B*G, T_gen)
        return lp_tok, out_ref

    # 3. Old student logprobs — no_grad
    with torch.no_grad():
        old_lp_tok, _ = _get_logps(student, scale=temperature)

    # 4. Teacher logprobs — no_grad (keep out for KL reuse)
    if teacher is not None and gold_answers is None:
        with torch.no_grad():
            t_lp_tok, t_out = _get_logps(teacher, scale=temperature)
    else:
        t_lp_tok = torch.zeros_like(old_lp_tok)
        t_out = None

    # 5-6. Reward
    seq_len = gen_mask.sum(dim=-1).clamp_min(1)             # (B*G,)

    if gold_answers is not None:
        # ── Correctness + format reward ──────────────────────────────────────
        assert tokenizer is not None, "tokenizer required for correctness reward"
        # Decode only the generated part (strip prompt)
        decoded = tokenizer.batch_decode(gen_part, skip_special_tokens=False)
        # gold_answers has length B; repeat G times to match B*G
        golds_rep = [g for g in gold_answers for _ in range(G)]  # length B*G

        reward_list, diag_truncated, diag_boxed, diag_correct = [], 0, 0, 0
        for dec, gold in zip(decoded, golds_rep):
            r, rd = compute_correctness_reward(dec, gold, format_reward=format_reward)
            reward_list.append(r)
            diag_truncated += rd['truncated']
            diag_boxed     += rd['has_boxed']
            diag_correct   += rd['correct']

        seq_reward = torch.tensor(reward_list, dtype=torch.float, device=device)  # (B*G,)
        reward_tok = None  # not used in correctness mode
        n_rollouts = float(B * G)
        extra_diag = {
            "grpo/truncated_rate": diag_truncated / n_rollouts,
            "grpo/boxed_rate":     diag_boxed     / n_rollouts,
            "grpo/correct_rate":   diag_correct   / n_rollouts,
        }
    else:
        # ── Log-ratio reward (original) ───────────────────────────────────────
        reward_tok = (t_lp_tok - old_lp_tok) * gen_mask    # (B*G, T_gen)
        seq_reward = reward_tok.sum(dim=-1) / seq_len       # (B*G,)
        extra_diag = {}

    # 7. Group-relative advantage
    seq_reward_g = seq_reward.view(B, G)                    # (B, G)
    mean_g = seq_reward_g.mean(dim=1, keepdim=True)         # (B, 1)
    std_g  = seq_reward_g.std(dim=1, keepdim=True, unbiased=False)
    adv_g  = (seq_reward_g - mean_g) / (std_g + 1e-6)      # (B, G)
    adv    = adv_g.view(B * G, 1).expand_as(gen_mask).detach()  # (B*G, T_gen)

    # 8. Current student logprobs — per-sequence forward+backward to avoid OOM.
    # If we run all B*G forward passes before calling backward(), PyTorch keeps
    # all B*G activation graphs in memory simultaneously (~10 GB each → 80 GB total).
    # Fix: immediately call .backward() after each sequence so only ONE backward
    # graph lives in memory at a time.
    BG = gen_ids.shape[0]
    pg_loss_accum = 0.0
    cur_lp_list  = []   # detached, for diagnostics
    ratio_list   = []   # detached, for diagnostics

    for i in range(BG):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out_i = student(
                input_ids=gen_ids[i:i+1],
                attention_mask=full_mask[i:i+1],
            )
        logits_i  = out_i.logits[0, Lp - 1:-1]                                   # (T_gen, V) bf16
        tgt_i     = gen_part[i].clamp(min=0)                                      # (T_gen,)
        tgt_logit = logits_i.gather(-1, tgt_i.unsqueeze(-1)).squeeze(-1)          # (T_gen,) bf16
        log_z     = torch.logsumexp(logits_i / temperature, dim=-1)               # (T_gen,) bf16
        cur_lp_i  = tgt_logit.float() / temperature - log_z.float()               # (T_gen,) fp32
        del logits_i, tgt_logit, log_z, out_i

        # PPO ratio + clipped objective for this sequence
        log_ratio_i = (cur_lp_i - old_lp_tok[i].detach()) * gen_mask[i]
        ratio_i     = log_ratio_i.exp()
        adv_i       = adv[i]
        obj1_i      = ratio_i * adv_i
        obj2_i      = torch.clamp(ratio_i, 1.0 - eps_clip, 1.0 + eps_clip) * adv_i
        n_valid_i   = gen_mask[i].sum().clamp_min(1)
        pg_i        = -(torch.minimum(obj1_i, obj2_i) * gen_mask[i]).sum() / n_valid_i

        # Backward for this sequence immediately (frees its activation graph)
        (grpo_lambda * pg_i / BG).backward()
        pg_loss_accum += pg_i.item()

        # Collect detached tensors for diagnostics
        cur_lp_list.append(cur_lp_i.detach())
        ratio_list.append(ratio_i.detach())

        del cur_lp_i, log_ratio_i, ratio_i, adv_i, obj1_i, obj2_i, pg_i
        torch.cuda.empty_cache()

    pg_loss     = torch.tensor(pg_loss_accum / BG, device=device)  # detached scalar
    cur_lp_tok  = torch.stack(cur_lp_list, dim=0)                  # (B*G, T_gen) detached
    ratio       = torch.stack(ratio_list,  dim=0)                  # (B*G, T_gen) detached

    # 9. Optional: on-policy reverse KL on rollouts — only when kd_lambda > 0
    kl_loss_val = torch.tensor(0.0, device=device)
    # (kd_lambda > 0 with per-seq backward not yet implemented; grads already done above)

    loss = pg_loss  # detached; grpo_lambda scaling already applied in per-seq backward

    # 10. Diagnostics
    with torch.no_grad():
        valid = gen_mask.bool()
        clip_frac = ((ratio - 1.0).abs() > eps_clip)[valid].float().mean()
        approx_kl = (old_lp_tok - cur_lp_tok)[valid].mean()
        diag = {
            "grpo/seq_reward":  seq_reward.mean().item(),
            "grpo/adv_std":     adv[valid].std().item(),
            "grpo/ratio_mean":  ratio[valid].mean().item(),
            "grpo/clip_frac":   clip_frac.item(),
            "grpo/approx_kl":   approx_kl.item(),
            "grpo/pg_loss":     pg_loss.item(),
            "grpo/kl_loss":     kl_loss_val.item(),
        }
        if reward_tok is not None:
            diag["grpo/reward_mean"] = reward_tok[valid].mean().item()
            diag["grpo/reward_std"]  = reward_tok[valid].std().item()
        diag.update(extra_diag)

    return loss, diag


# ── Chunk-GRPO-OPKD helpers ──────────────────────────────────────────────────

def _make_eos_mask(chunk_ids, eos_id):
    """
    chunk_ids: (N, K) int
    Returns float mask (N, K) where 1 = token is valid (before or at first EOS).
    """
    is_eos = (chunk_ids == eos_id)
    # cumsum > 0 after first EOS; we want 1 up to and including first EOS
    # shift: mark positions *including* the EOS token
    eos_cumsum = is_eos.cumsum(dim=-1)
    # valid if we haven't seen an EOS yet, or this IS the EOS
    mask = (eos_cumsum <= 1).float()
    return mask


@torch.no_grad()
def _get_chunk_logps_no_grad(model, full_ids, full_mask, prefix_len, chunk_size):
    """
    full_ids:   (N, Lp + K) where K = chunk_size
    full_mask:  (N, Lp + K)
    Returns:    (N, K) log-probs for the K generated tokens
    """
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(input_ids=full_ids, attention_mask=full_mask)
    chunk_ids = full_ids[:, prefix_len:]                                             # (N, K)
    # Slice before float() to avoid materializing full (N, seq_len, V) float32 tensor
    logits_chunk = out.logits[:, prefix_len - 1: prefix_len - 1 + chunk_size].float()  # (N, K, V)
    lp_tok = F.log_softmax(logits_chunk, dim=-1).gather(
        -1, chunk_ids.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)  # (N, K)
    return lp_tok


def _get_chunk_logps_with_grad(model, full_ids, full_mask, prefix_len, chunk_size):
    """Same as above but without no_grad — retains computation graph."""
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(input_ids=full_ids, attention_mask=full_mask)
    chunk_ids = full_ids[:, prefix_len:]
    # Slice before float() — (N, K, V) instead of (N, seq_len, V), saves ~34x memory
    logits_chunk = out.logits[:, prefix_len - 1: prefix_len - 1 + chunk_size].float()  # (N, K, V)
    del out
    lp_tok = F.log_softmax(logits_chunk, dim=-1).gather(
        -1, chunk_ids.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)  # (N, K)
    return lp_tok


def chunk_grpo_opkd_loss(
    student,
    teacher,
    prompt_ids,         # (B, Lp)
    prompt_mask,        # (B, Lp)
    num_rollouts,       # G
    max_new_tokens,     # total generation budget
    chunk_size,         # K: tokens per chunk
    temperature,
    eps_clip,
    adv_clip,           # clamp advantage to [-adv_clip, adv_clip]
    pad_id,
    eos_id,
    grpo_lambda=1.0,    # loss scale; backward called per chunk step to free graphs
    reward_logratio=True,  # True: reward = log p_T - log q_old, False: reward = log p_T
    kd_lambda=0.0,     # weight for on-policy reverse KL on full generated sequence after loop
):
    """
    Chunk-wise GRPO-OPKD loss.

    Selection score : mean log p_T  over chunk (teacher log-prob)
    Reward          : mean (log p_T - log q_old) [reward_logratio=True]
                   or mean log p_T               [reward_logratio=False]

    For each chunk step:
      1. Generate G candidate chunks (batch) from current prefix.
      2. Score by log p_T (teacher) → select best chunk → extend prefix.
      3. Compute GRPO advantage from log-ratio reward across G candidates.
      4. PPO-clipped loss over ALL G candidates (with grad).
      5. Backward immediately to free the computation graph.

    Returns:
        loss_val : float (detached, for logging — backward already called internally)
        diag     : dict of loggable scalars
    """
    B, Lp = prompt_ids.shape
    G = num_rollouts
    N = B * G
    device = prompt_ids.device
    K = chunk_size
    num_steps = max_new_tokens // K

    # Track per-sample prefix (extended chunk by chunk)
    # shape: (B, current_len)  — starts as (B, Lp)
    prefix = prompt_ids.clone()
    prefix_mask = prompt_mask.clone()

    # Track which samples are fully finished (generated EOS)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    loss_val = 0.0
    # Diagnostics accumulators
    diag_reward_sum = 0.0
    diag_adv_std_sum = 0.0
    diag_clip_sum = 0.0
    diag_kl_sum = 0.0
    num_active_steps = 0
    num_steps = max_new_tokens // K

    student.config.use_cache = True
    student.eval()

    for _ in range(num_steps):
        if finished.all():
            break

        cur_prefix_len = prefix.shape[1]

        # ── 1. Generate G chunks per sample (batched) ─────────────────────────
        rep_ids  = prefix.repeat_interleave(G, dim=0)   # (N, cur_prefix_len)
        rep_mask = prefix_mask.repeat_interleave(G, dim=0)

        with torch.no_grad():
            gen_out = student.generate(
                input_ids=rep_ids,
                attention_mask=rep_mask,
                max_new_tokens=K,
                do_sample=True,
                temperature=temperature,
                pad_token_id=pad_id,
            )  # (N, cur_prefix_len + up_to_K)

        # Pad/trim so generated part is exactly K tokens
        gen_part = gen_out[:, cur_prefix_len:]          # (N, actual_gen)
        actual_K = gen_part.shape[1]
        if actual_K < K:
            pad_cols = torch.full((N, K - actual_K), pad_id, device=device, dtype=gen_part.dtype)
            gen_part = torch.cat([gen_part, pad_cols], dim=1)
        elif actual_K > K:
            gen_part = gen_part[:, :K]

        chunk_mask = _make_eos_mask(gen_part, eos_id)   # (N, K) float

        # Build full sequences for logprob computation
        full_ids  = torch.cat([rep_ids,  gen_part], dim=1)      # (N, cur_prefix_len+K)
        full_mask = torch.cat([rep_mask, chunk_mask.long()], dim=1)

        # ── 2. Old student logprobs (no grad) ─────────────────────────────────
        with torch.no_grad():
            old_lp = _get_chunk_logps_no_grad(student, full_ids, full_mask, cur_prefix_len, K)  # (N, K)

        # ── 3. Teacher logprobs (no grad) ─────────────────────────────────────
        with torch.no_grad():
            t_lp = _get_chunk_logps_no_grad(teacher, full_ids, full_mask, cur_prefix_len, K)    # (N, K)

        chunk_len = chunk_mask.sum(-1).clamp_min(1)  # (N,)

        # ── 4. Selection score: mean log p_T → pick best chunk per sample ─────
        select_score = (t_lp * chunk_mask).sum(-1) / chunk_len   # (N,)
        select_score_g = select_score.view(B, G)                 # (B, G)
        best_idx = select_score_g.argmax(dim=1)                  # (B,)

        # ── 5. Reward: log p_T - log q_old  or  log p_T ──────────────────────
        if reward_logratio:
            reward_tok = (t_lp - old_lp) * chunk_mask           # (N, K)
        else:
            reward_tok = t_lp * chunk_mask                       # (N, K)
        reward = reward_tok.sum(-1) / chunk_len                  # (N,)
        reward_g = reward.view(B, G)                             # (B, G)
        mean_g = reward_g.mean(dim=1, keepdim=True)
        std_g  = reward_g.std(dim=1, keepdim=True, unbiased=False)
        adv_g  = ((reward_g - mean_g) / (std_g + 1e-6)).clamp(-adv_clip, adv_clip)  # (B, G)
        adv    = adv_g.view(N, 1).expand_as(chunk_mask).detach()  # (N, K)

        # ── 6. Current student logprobs WITH grad ──────────────────────────────
        student.train()
        student.config.use_cache = False
        cur_lp = _get_chunk_logps_with_grad(student, full_ids, full_mask, cur_prefix_len, K)  # (N, K)

        # ── 7. PPO clipped loss over all G candidates ─────────────────────────
        ratio = (cur_lp - old_lp.detach()).exp()
        obj1 = ratio * adv
        obj2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * adv
        pg = -(
            (torch.minimum(obj1, obj2) * chunk_mask).sum(-1)
            / chunk_mask.sum(-1).clamp_min(1)
        ).mean()

        # Backward immediately — frees this chunk's computation graph
        # Divide by num_steps for mean across chunks (consistent scale)
        scaled_pg = pg * grpo_lambda / num_steps
        if not (torch.isnan(scaled_pg) or torch.isinf(scaled_pg)):
            scaled_pg.backward()
        loss_val += pg.detach().item()

        # ── 8. Update prefix with best chunk ─────────────────────────────────
        student.eval()
        student.config.use_cache = True

        with torch.no_grad():
            # Gather best chunk for each sample: (B, K)
            best_flat = best_idx + torch.arange(B, device=device) * G
            best_chunk = gen_part[best_flat]          # (B, K)
            best_cmask = chunk_mask[best_flat]        # (B, K)

            prefix      = torch.cat([prefix,      best_chunk],        dim=1)
            prefix_mask = torch.cat([prefix_mask, best_cmask.long()], dim=1)

            # A sample is finished if its best chunk contained EOS
            has_eos = (best_chunk == eos_id).any(dim=1)
            finished = finished | has_eos

        # ── Diagnostics ───────────────────────────────────────────────────────
        with torch.no_grad():
            valid = chunk_mask.bool()
            clip_frac = ((ratio.detach() - 1.0).abs() > eps_clip)[valid].float().mean()
            approx_kl = (old_lp - cur_lp.detach())[valid].mean()
            diag_reward_sum += reward.mean().item()
            diag_adv_std_sum += adv[valid].std().item()
            diag_clip_sum += clip_frac.item()
            diag_kl_sum += approx_kl.item()
        num_active_steps += 1

    student.train()
    student.config.use_cache = False

    # ── On-policy reverse KL on full generated sequence ───────────────────────
    kd_loss_val = 0.0
    if kd_lambda > 0.0 and teacher is not None and prefix.shape[1] > Lp:
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                t_out = teacher(input_ids=prefix, attention_mask=prefix_mask)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_out = student(input_ids=prefix, attention_mask=prefix_mask)
        # Mask: only supervise generated tokens (after prompt)
        labels = prefix.clone()
        labels[:, :Lp] = -100
        labels[prefix_mask == 0] = -100
        kl_loss_t, _ = _kl_loss(
            s_out.logits, t_out.logits, labels,
            temperature=1.0, topk=0, reverse=True,
        )
        if not (torch.isnan(kl_loss_t) or torch.isinf(kl_loss_t)):
            (kl_loss_t * kd_lambda).backward()
        kd_loss_val = kl_loss_t.item()

    ns = max(num_active_steps, 1)
    diag = {
        "cgrpo/reward_mean":  diag_reward_sum / ns,
        "cgrpo/adv_std":      diag_adv_std_sum / ns,
        "cgrpo/clip_frac":    diag_clip_sum / ns,
        "cgrpo/approx_kl":    diag_kl_sum / ns,
        "cgrpo/num_chunks":   float(num_active_steps),
        "cgrpo/pg_loss":      loss_val / ns,
        "cgrpo/kd_loss":      kd_loss_val,
    }
    return loss_val / ns, diag


# ── Main training loop ────────────────────────────────────────────────────────

def run_grpo_opkd(model, teacher_model, tokenizer, train_dataset, prompt_dataset, FLAGS):
    """
    GMP training loop with GRPO-OPKD on-policy loss.

    FLAGS (new, on top of standard GMP flags):
        gmp_grpo_num_rollouts       int   rollouts per prompt G (default 4)
        gmp_grpo_interval           int   steps between GRPO updates (default 8)
        gmp_grpo_lambda             float weight for GRPO loss (default 1.0)
        gmp_grpo_eps_clip           float PPO clip range (default 0.2)
        gmp_onpolicy_kd_lambda      float weight for on-policy reverse KL (default 0.0)
        gmp_onpolicy_mixed_alpha    float teacher mix ratio during sampling (default 0.2)
        gmp_onpolicy_max_new_tokens int   max generated tokens (default 512)
        gmp_onpolicy_temperature    float sampling temperature (default 1.0)
        gmp_onpolicy_kd_topk        int   top-k for KL (default 100)
        gmp_correctness_reward      bool  use correctness+format reward instead of log-ratio
        gmp_format_reward           float partial reward for correct format, wrong answer (default 0.1)
    """
    device = next(model.parameters()).device
    named_params = _find_linear_weights(model)

    # ── hyperparams ───────────────────────────────────────────────────────────
    lr              = FLAGS.lr
    total_steps     = FLAGS.steps
    batch_size      = getattr(FLAGS, 'gmp_batch_size', 1)
    grad_accum      = getattr(FLAGS, 'gmp_grad_accum', 8)
    warmup_steps    = int(total_steps * getattr(FLAGS, 'gmp_warmup_ratio', 0.05))
    mask_interval   = getattr(FLAGS, 'gmp_mask_interval', 32)
    final_sparsity  = FLAGS.sparsity_ratio

    num_rollouts        = getattr(FLAGS, 'gmp_grpo_num_rollouts', 4)
    grpo_interval       = getattr(FLAGS, 'gmp_grpo_interval', 8)
    grpo_lambda         = getattr(FLAGS, 'gmp_grpo_lambda', 1.0)
    eps_clip            = getattr(FLAGS, 'gmp_grpo_eps_clip', 0.2)
    kd_lambda           = getattr(FLAGS, 'gmp_onpolicy_kd_lambda', 0.0)
    mixed_alpha         = getattr(FLAGS, 'gmp_onpolicy_mixed_alpha', 0.2)
    max_new_tokens      = getattr(FLAGS, 'gmp_onpolicy_max_new_tokens', 512)
    temperature         = getattr(FLAGS, 'gmp_onpolicy_temperature', 1.0)
    kd_topk             = getattr(FLAGS, 'gmp_onpolicy_kd_topk', 100)
    correctness_reward  = getattr(FLAGS, 'gmp_correctness_reward', False)
    format_reward       = getattr(FLAGS, 'gmp_format_reward', 0.1)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id or pad_id

    # ── prompt dataset: with or without gold answers ──────────────────────────
    if correctness_reward:
        prompt_collate_fn = collate_prompts_with_answers(pad_id)
    else:
        prompt_collate_fn = collate_prompts(pad_id)

    # ── optimizer / scheduler / mask ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    fisher    = FisherAccumulator(named_params, optimizer)
    maskmgr   = GradualMaskManager(named_params)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    _collate_fn = lambda b: _collate(b, pad_token_id=pad_id)

    loader      = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=_collate_fn)
    data_iter   = _infinite(loader)

    prompt_loader = DataLoader(prompt_dataset, batch_size=batch_size,
                               shuffle=True, collate_fn=prompt_collate_fn)
    prompt_iter   = _infinite(prompt_loader)

    model.train()
    optimizer.zero_grad()

    logging.info("***** Running GMP Training with GRPO-OPKD *****")
    logging.info(f"  Steps={total_steps}, G={num_rollouts}, grpo_interval={grpo_interval}")
    logging.info(f"  LR={lr}, sparsity={final_sparsity}, mask_interval={mask_interval}")
    logging.info(f"  grpo_lambda={grpo_lambda}, eps_clip={eps_clip}, kd_lambda={kd_lambda}")
    logging.info(f"  correctness_reward={correctness_reward}, format_reward={format_reward}")

    step = 0
    accum_ntp  = 0.0
    accum_grpo_diag: dict = {}
    t_start = time.time()

    while step < total_steps:

        # ── NTP micro-steps ───────────────────────────────────────────────────
        for _ in range(grad_accum):
            batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(**batch)
                ntp_loss = out.loss / grad_accum
            ntp_loss.backward()
            accum_ntp += ntp_loss.item()

        step += 1

        # ── Mask update ───────────────────────────────────────────────────────
        if step % mask_interval == 0 and step <= total_steps:
            curr_sp = _cubic_sparsity(step, total_steps, final_sparsity, warmup_steps)
            maskmgr.update(fisher, curr_sp)

        # ── GRPO-OPKD ─────────────────────────────────────────────────────────
        grpo_active = (step % grpo_interval == 0 and
                       (correctness_reward or teacher_model is not None))
        if grpo_active:
            p_batch     = next(prompt_iter)
            p_ids       = p_batch['input_ids'].to(device)
            p_mask      = p_batch['attention_mask'].to(device)
            gold_answers = p_batch.get('gold_answers', None)  # list[str] or None

            grpo_loss, grpo_diag = grpo_opkd_loss(
                student=model,
                teacher=teacher_model,
                prompt_ids=p_ids,
                prompt_mask=p_mask,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                mixed_alpha=mixed_alpha,
                temperature=temperature,
                eps_clip=eps_clip,
                kd_lambda=kd_lambda,
                kd_topk=kd_topk,
                pad_id=pad_id,
                eos_id=eos_id,
                tokenizer=tokenizer if correctness_reward else None,
                gold_answers=gold_answers,
                format_reward=format_reward,
                grpo_lambda=grpo_lambda,
            )
            # grpo_opkd_loss already called .backward() per-sequence internally.
            # grpo_loss is a detached scalar (requires_grad=False) — skip outer backward.
            scaled = grpo_lambda * grpo_loss
            if scaled.requires_grad and not (torch.isnan(scaled) or torch.isinf(scaled)):
                scaled.backward()
            accum_grpo_diag = grpo_diag

        # ── Optimizer step ────────────────────────────────────────────────────
        maskmgr.apply()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % 10 == 0 or step == total_steps:
            curr_sp = maskmgr.current_sparsity()
            lr_now  = scheduler.get_last_lr()[0]
            logging.info(
                f"Step {step}/{total_steps} | ntp={accum_ntp:.4f} | "
                f"sparsity={curr_sp:.3f} | lr={lr_now:.2e} | "
                f"grpo_r={accum_grpo_diag.get('grpo/seq_reward', 0):.3f} | "
                f"clip={accum_grpo_diag.get('grpo/clip_frac', 0):.3f} | "
                f"kl={accum_grpo_diag.get('grpo/approx_kl', 0):.3f}"
            )
            log_dict = {
                "train/step": step,
                "train/ntp_loss": accum_ntp,
                "train/sparsity": curr_sp,
                "train/lr": lr_now,
                **accum_grpo_diag,
            }
            if getattr(FLAGS, 'wandb', False):
                import wandb
                wandb.log(log_dict, step=step)
            accum_ntp = 0.0
            accum_grpo_diag = {}

    # ── Final mask + save ─────────────────────────────────────────────────────
    maskmgr.update(fisher, final_sparsity)
    logging.info(f"Final sparsity: {maskmgr.current_sparsity():.4f}")
    logging.info(f"Training done in {(time.time()-t_start)/3600:.2f}h")

    saved_path = None
    if getattr(FLAGS, 'save_model', False) and getattr(FLAGS, 'gmp_save_path', None):
        import time as _t
        ts = _t.strftime("%m%d_%H%M")
        tag = _run_tag(FLAGS)
        saved_path = f"{FLAGS.gmp_save_path}/{tag}_grpo_{ts}"
        model.save_pretrained(saved_path)
        tokenizer.save_pretrained(saved_path)
        logging.info(f"Saved model to {saved_path}")

    return saved_path


# ── Chunk-GRPO training loop ──────────────────────────────────────────────────

def run_chunk_grpo_opkd(model, teacher_model, tokenizer, train_dataset, prompt_dataset, FLAGS):
    """
    GMP training loop with chunk-wise GRPO-OPKD on-policy loss.

    FLAGS (in addition to standard GMP flags):
        do_chunk_grpo_opkd      bool  enable this loop
        gmp_chunk_size          int   tokens per chunk K (default 32)
        gmp_chunk_adv_clip      float advantage clamp value (default 2.0)
        gmp_grpo_num_rollouts   int   G (default 4)
        gmp_grpo_interval       int   steps between chunk-GRPO updates (default 8)
        gmp_grpo_lambda         float weight for chunk-GRPO loss (default 1.0)
        gmp_grpo_eps_clip       float PPO clip epsilon (default 0.2)
        gmp_onpolicy_max_new_tokens int total generation budget (default 512)
        gmp_onpolicy_temperature    float sampling temperature (default 1.0)
    """
    device = next(model.parameters()).device
    named_params = _find_linear_weights(model)

    lr              = FLAGS.lr
    total_steps     = FLAGS.steps
    batch_size      = getattr(FLAGS, 'gmp_batch_size', 1)
    grad_accum      = getattr(FLAGS, 'gmp_grad_accum', 8)
    warmup_steps    = int(total_steps * getattr(FLAGS, 'gmp_warmup_ratio', 0.05))
    mask_interval   = getattr(FLAGS, 'gmp_mask_interval', 32)
    final_sparsity  = FLAGS.sparsity_ratio

    num_rollouts    = getattr(FLAGS, 'gmp_grpo_num_rollouts', 4)
    grpo_interval   = getattr(FLAGS, 'gmp_grpo_interval', 8)
    grpo_lambda     = getattr(FLAGS, 'gmp_grpo_lambda', 1.0)
    eps_clip        = getattr(FLAGS, 'gmp_grpo_eps_clip', 0.2)
    chunk_size      = getattr(FLAGS, 'gmp_chunk_size', 32)
    adv_clip        = getattr(FLAGS, 'gmp_chunk_adv_clip', 2.0)
    max_new_tokens  = getattr(FLAGS, 'gmp_onpolicy_max_new_tokens', 512)
    temperature     = getattr(FLAGS, 'gmp_onpolicy_temperature', 1.0)
    reward_logratio = getattr(FLAGS, 'gmp_chunk_reward_logratio', True)
    kd_lambda       = getattr(FLAGS, 'gmp_chunk_kd_lambda', 0.0)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id or pad_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    fisher    = FisherAccumulator(named_params, optimizer)
    maskmgr   = GradualMaskManager(named_params)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    _collate_fn = lambda b: _collate(b, pad_token_id=pad_id)
    loader      = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=_collate_fn)
    data_iter   = _infinite(loader)

    prompt_collate_fn = collate_prompts(pad_id)
    prompt_loader = DataLoader(prompt_dataset, batch_size=batch_size,
                               shuffle=True, collate_fn=prompt_collate_fn)
    prompt_iter   = _infinite(prompt_loader)

    model.train()
    optimizer.zero_grad()

    logging.info("***** Running GMP Training with Chunk-GRPO-OPKD *****")
    logging.info(f"  Steps={total_steps}, G={num_rollouts}, K={chunk_size}, grpo_interval={grpo_interval}")
    logging.info(f"  LR={lr}, sparsity={final_sparsity}, mask_interval={mask_interval}")
    logging.info(f"  grpo_lambda={grpo_lambda}, eps_clip={eps_clip}, adv_clip={adv_clip}")

    step = 0
    accum_ntp = 0.0
    accum_cgrpo_diag: dict = {}
    t_start = time.time()

    while step < total_steps:

        # ── NTP micro-steps ───────────────────────────────────────────────────
        for _ in range(grad_accum):
            batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(**batch)
                ntp_loss = out.loss / grad_accum
            ntp_loss.backward()
            accum_ntp += ntp_loss.item()

        step += 1

        # ── Mask update ───────────────────────────────────────────────────────
        if step % mask_interval == 0 and step <= total_steps:
            curr_sp = _cubic_sparsity(step, total_steps, final_sparsity, warmup_steps)
            maskmgr.update(fisher, curr_sp)

        # ── Chunk-GRPO-OPKD ───────────────────────────────────────────────────
        if step % grpo_interval == 0 and teacher_model is not None:
            p_batch = next(prompt_iter)
            p_ids   = p_batch['input_ids'].to(device)
            p_mask  = p_batch['attention_mask'].to(device)

            cgrpo_loss, cgrpo_diag = chunk_grpo_opkd_loss(
                student=model,
                teacher=teacher_model,
                prompt_ids=p_ids,
                prompt_mask=p_mask,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                chunk_size=chunk_size,
                temperature=temperature,
                eps_clip=eps_clip,
                adv_clip=adv_clip,
                pad_id=pad_id,
                eos_id=eos_id,
                grpo_lambda=grpo_lambda,
                reward_logratio=reward_logratio,
                kd_lambda=kd_lambda,
            )
            # No .backward() here — already called inside chunk_grpo_opkd_loss
            accum_cgrpo_diag = cgrpo_diag

        # ── Optimizer step ────────────────────────────────────────────────────
        maskmgr.apply()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % 10 == 0 or step == total_steps:
            curr_sp = maskmgr.current_sparsity()
            lr_now  = scheduler.get_last_lr()[0]
            logging.info(
                f"Step {step}/{total_steps} | ntp={accum_ntp:.4f} | "
                f"sparsity={curr_sp:.3f} | lr={lr_now:.2e} | "
                f"cgrpo_r={accum_cgrpo_diag.get('cgrpo/reward_mean', 0):.3f} | "
                f"clip={accum_cgrpo_diag.get('cgrpo/clip_frac', 0):.3f} | "
                f"kl={accum_cgrpo_diag.get('cgrpo/approx_kl', 0):.3f} | "
                f"chunks={accum_cgrpo_diag.get('cgrpo/num_chunks', 0):.0f}"
            )
            log_dict = {
                "train/step": step,
                "train/ntp_loss": accum_ntp,
                "train/sparsity": curr_sp,
                "train/lr": lr_now,
                **accum_cgrpo_diag,
            }
            if getattr(FLAGS, 'wandb', False):
                import wandb
                wandb.log(log_dict, step=step)
            accum_ntp = 0.0
            accum_cgrpo_diag = {}

    # ── Final mask + save ─────────────────────────────────────────────────────
    maskmgr.update(fisher, final_sparsity)
    logging.info(f"Final sparsity: {maskmgr.current_sparsity():.4f}")
    logging.info(f"Training done in {(time.time()-t_start)/3600:.2f}h")

    saved_path = None
    if getattr(FLAGS, 'save_model', False) and getattr(FLAGS, 'gmp_save_path', None):
        import time as _t
        ts = _t.strftime("%m%d_%H%M")
        tag = _run_tag(FLAGS)
        saved_path = f"{FLAGS.gmp_save_path}/{tag}_cgrpo_{ts}"
        model.save_pretrained(saved_path)
        tokenizer.save_pretrained(saved_path)
        logging.info(f"Saved model to {saved_path}")

    return saved_path
