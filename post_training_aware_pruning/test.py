from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

def compute_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_config: Any,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:

    metrics = {}

    loss_mask = response_mask
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)

    if self_distillation_config.full_logit_distillation:
        use_topk = self_distillation_config.distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError("top-k distillation requires student_topk_log_probs and teacher_topk_log_probs.")

            def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                # Compute tail log-probability using logsumexp for numerical stability
                # log(1 - sum(p_i)) = log(1 - exp(log_sum_exp(log(p_i))))
                log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)  # Clamp to avoid log_s >= 0 (which implies sum(probs) >= 1)
                tail_log = torch.log(-torch.expm1(log_s))  # We use the identity: 1 - exp(x) = -(exp(x) - 1); torch.expm1(x) computes (e^x - 1) with high precision for small x.
                return torch.cat([log_probs, tail_log], dim=-1)

            def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                return logp - logZ

            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs
            if self_distillation_config.distillation_add_tail:
                student_distill_log_probs = add_tail(student_distill_log_probs)
                teacher_distill_log_probs = add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = renorm_topk_log_probs(teacher_distill_log_probs)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError("full_logit_distillation requires student_all_log_probs and teacher_all_log_probs.")
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs

        if self_distillation_config.alpha == 0.0:
            kl_loss = F.kl_div(
                student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True
            )
        elif self_distillation_config.alpha == 1.0:
            kl_loss = F.kl_div(
                teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True
            )
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            alpha = torch.tensor(
                self_distillation_config.alpha,
                dtype=student_distill_log_probs.dtype,
                device=student_distill_log_probs.device,
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_distill_log_probs + torch.log(1 - alpha), teacher_distill_log_probs + torch.log(alpha)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha)  # Compute the Generalized Jensen-Shannon Divergence

        per_token_loss = kl_loss.sum(-1)
    else:
        assert self_distillation_config.alpha == 1.0, "Only reverse KL is supported for non-full-logit distillation"
        log_ratio = student_log_probs - teacher_log_probs
        per_token_loss = log_ratio.detach() * student_log_probs

    is_clip = self_distillation_config.is_clip
    if is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for distillation IS ratio.")

        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    loss = agg_loss(
        loss_mat=per_token_loss,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        batch_num_tokens=loss_mask.sum().clamp(min=1.0),
    )
    return loss, metrics


teacher_inputs = {
    "responses": model_inputs["responses"],
    "input_ids": model_inputs["teacher_input_ids"],
    "attention_mask": model_inputs["teacher_attention_mask"],
    "position_ids": model_inputs["teacher_position_ids"],
}


with torch.no_grad():
    teacher_outputs = self._forward_micro_batch(
        teacher_inputs,
        temperature=temperature,
        calculate_entropy=False,
        return_all_logps=return_all_logps,
        distill_topk=distill_topk,
        topk_indices=student_topk_indices,
        module=teacher_model,
    )
teacher_log_prob = teacher_outputs["log_probs"]
teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
pg_loss, pg_metrics = compute_self_distillation_loss(
    student_log_probs=log_prob,
    teacher_log_probs=teacher_log_prob,
    response_mask=response_mask,
    self_distillation_config=self_distillation_cfg,
    old_log_probs=old_log_prob,
    student_all_log_probs=student_all_logps,
    teacher_all_log_probs=teacher_all_logps,
    student_topk_log_probs=student_topk_logps,
    teacher_topk_log_probs=teacher_topk_logps,
    self_distillation_mask=self_distillation_mask,
    loss_agg_mode=loss_agg_mode,
    rollout_is_weights=rollout_is_weights,
)

pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
micro_batch_metrics.update(pg_metrics)


from transformers import GenerationConfig

generation_config = GenerationConfig(do_sample=False)
actor_model.cuda()
output = actor_model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=32,
    # max_length=max_length,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    generation_config=generation_config,
    # renormalize_logits=True,
    output_scores=False,  # this is potentially very large
    return_dict_in_generate=True,
    use_cache=False,
)  # may OOM when use_cache = True
seq = output.sequences
response = seq[:, max_prompt_length:]

print(f"hf response: {tokenizer.batch_decode(response)}")




def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        if loss_scale_factor is None:
            loss_scale_factor = loss_mask.shape[-1]
        loss = torch.sum(seq_losses) / loss_scale_factor
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss
