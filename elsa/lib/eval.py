from absl import logging
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import fnmatch
from .data import get_loaders 

# Code adopted from https://github.com/locuslab/wanda

def eval_ppl(
    args,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device = torch.device("cuda:0"),
    data_path: str = None
) -> dict:
    """
    Evaluate the model on the wikitext2 and c4 datasets.
    Args:
        args: Namespace, command line arguments.
        model (AutoModelForCausalLM): The model to evaluate.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the data.
        device (torch.device): The device to use for evaluation.
        data_path: The path to the dataset.
    Returns:
        dict: A dictionary containing the perplexity (ppl) for each dataset.
    """
    dataset = ["wikitext2", "c4"]
    ppls = defaultdict(float)
    for d in dataset:
        # Print status
        logging.info(f"evaluating on {d}")

        # Get the test loader
        _, testloader = get_loaders(
            d, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer, data_path=data_path 
        )
        # Evaluate ppl in no grad context to avoid updating the model
        with torch.no_grad():
            ppl_test = calculate_ppl(model, testloader,tokenizer, 1)
            ppls[d] = ppl_test
    return ppls 

@torch.no_grad()
def calculate_ppl(
    model: AutoModelForCausalLM,
    testenc,
    tokenizer: AutoTokenizer,
    bs: int = 1
) -> float:

    seqlen = model.seqlen
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    nlls = []
    
    for i in range(0, nsamples, bs):
        j = min(i + bs, nsamples)
        batch_size = j - i

        # First cut the input to seqlen length for all models
        model_inputs = testenc[:, (i * seqlen):(j * seqlen)].to(model.device)
        model_inputs = model_inputs.reshape(batch_size, seqlen)

        # Special handling for Gemma model architecture
        is_gemma = 'Gemma' in model.config.architectures[0]
        
        # === Modified part: Gemma handling logic ===
        if is_gemma:
            # 1. Replace the first token of existing sequence with BOS token
            model_inputs[:, 0] = tokenizer.bos_token_id

        # 2. Label generation logic applies equally to both Gemma and other models
        # [BOS, t2, t3, ...] -> labels: [t2, t3, ...]
        # [t1, t2, t3, ...] -> labels: [t2, t3, ...]
        shift_labels = model_inputs[:, 1:].contiguous()
        # ==================================

        # Model forward pass
        with torch.no_grad():
            lm_logits = model(model_inputs).logits

        # Reorder logits for loss calculation
        # Excluding the last logit makes the length equal to prediction targets (labels)
        shift_logits = lm_logits[:, :-1, :].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32), shift_labels.view(-1))

        neg_log_likelihood = loss.float() * (shift_labels.numel() / batch_size) # Calculate NLL based on actual label length
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    torch.cuda.empty_cache()

    return ppl.item()




@torch.no_grad()
def eval_math500(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    max_samples: int = None,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    batch_size: int = 8,
    system_prompt: str = None,
    seed: int = 0,
) -> dict:
    """MATH-500 pass@1 eval using math_verify.

    Returns: {"math500_pass@1": float, "math500_correct": int, "math500_total": int}
    """
    from datasets import load_dataset
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    def _score(completion, gold_answer):
        gold_parsed = parse(f"\\boxed{{{gold_answer}}}", extraction_mode="first_match")
        if len(gold_parsed) == 0:
            gold_parsed = parse(gold_answer, extraction_mode="first_match")
        if len(gold_parsed) == 0:
            return None
        answer_parsed = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False, malformed_operators=False, basic_latex=True,
                        equations=True, boxed="all", units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        try:
            return float(verify(gold_parsed, answer_parsed))
        except Exception:
            return 0.0

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    problems = [ex["problem"] for ex in ds]
    golds = [ex["answer"] for ex in ds]
    logging.info(f"[eval_math500] loaded {len(problems)} problems")

    # Set up model state for generation
    prev_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    prompts = []
    for p in problems:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": p})
        try:
            prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            ))
        except Exception:
            # Fallback for tokenizers without chat template
            prompts.append(f"Problem: {p}\n\nSolution:")

    torch.manual_seed(seed)
    scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        gld = golds[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=2048, padding_side="left",
        ).to(device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=pad_id, use_cache=True,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for text, gold in zip(texts, gld):
            s = _score(text, gold)
            scores.append(s if s is not None else 0.0)
        logging.info(
            f"[eval_math500] {i + len(batch)}/{len(prompts)} done, "
            f"running pass@1={sum(scores)/len(scores):.3f}"
        )

    # Restore model state
    model.config.use_cache = prev_use_cache
    if was_training:
        model.train()

    pass_at_1 = sum(scores) / len(scores) if scores else 0.0
    correct = int(sum(scores))
    total = len(scores)
    logging.info(f"[eval_math500] FINAL: pass@1={pass_at_1:.4f} ({correct}/{total})")
    return {"math500_pass@1": pass_at_1, "math500_correct": correct, "math500_total": total}


def eval_zero_shot(args, model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"],
        num_fewshot=0, use_accelerate=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    tm = tasks.TaskManager()
    task_names = pattern_match(task_list, tm.all_tasks)
    limit = None
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    from lm_eval.models.huggingface import HFLM
    if "70b" in model_name or "65b" in model_name:
        model = HFLM(model,parallelize=True,max_memory_per_gpu="40GB")
    else:
        model = HFLM(model)
    results = evaluator.simple_evaluate(
        model=model,
        # model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        cache_requests=False,
        batch_size="auto",
        device=model.device,
        use_cache=None,
        limit=limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        check_integrity=False
    )
    results = results['results'] ## return only the results
    return results 
