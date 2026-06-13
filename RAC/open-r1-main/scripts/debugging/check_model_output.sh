#!/bin/bash
#SBATCH --job-name=check_output
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH -t 0-00:10:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

MODEL_ORIG="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_50="/home1/doyoonkim/projects/RAC/open-r1-main/models/DeepSeek-R1-Distill-Qwen-1_pruned_50_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k"
MODEL_GRPO="/home1/doyoonkim/projects/RAC/open-r1-main/models/grpo"

python - "$MODEL_ORIG" "$MODEL_50" "$MODEL_GRPO" <<'PYEOF'
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

models = [("original (dense)", sys.argv[1]), ("50% pruned (before GRPO)", sys.argv[2]), ("after GRPO", sys.argv[3])]
for label, path in models:
    print(f"\n{'='*60}")
    print(f"MODEL: {label}")
    print('='*60)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    prompt = "What is 2 + 2?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    print(generated[:1500])
    del model
    torch.cuda.empty_cache()
PYEOF