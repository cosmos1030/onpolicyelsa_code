"""Quick check: verify dense (sparsity=0) training actually updates weights."""
import torch
from absl import flags, app
from lib.gmp_trainer import globalprune_gmp
from lib.utils import get_llm
from lib.gkd_admm_trainer import MathCotKDDataset
from transformers import AutoTokenizer

FLAGS = flags.FLAGS
flags.DEFINE_string('model', '/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562', '')
flags.DEFINE_integer('seqlen', 2048, '')
flags.DEFINE_float('sparsity_ratio', 0.0, '')
flags.DEFINE_float('gmp_lr', 1e-4, '')
flags.DEFINE_integer('gmp_steps', 10, '')
flags.DEFINE_integer('gmp_batch_size', 1, '')
flags.DEFINE_integer('gmp_grad_accum', 2, '')
flags.DEFINE_float('gmp_warmup_ratio', 0.05, '')
flags.DEFINE_integer('gmp_mask_interval', 4, '')
flags.DEFINE_float('gmp_fisher_beta', 0.999, '')
flags.DEFINE_float('gmp_kd_lambda', 0.0, '')
flags.DEFINE_string('gmp_save_path', None, '')
flags.DEFINE_integer('gmp_max_prompt_len', 128, '')
flags.DEFINE_integer('gmp_max_seq_len', 256, '')
flags.DEFINE_boolean('save_model', False, '')
flags.DEFINE_boolean('wandb', False, '')
flags.DEFINE_integer('seed', 42, '')
flags.DEFINE_string('data_path', '/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl', '')

def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = get_llm(FLAGS.model, FLAGS.seqlen)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model, use_fast=False)
    model.to(device)

    # snapshot weights before training
    param = next(p for p in model.parameters() if p.requires_grad)
    before = param.data.clone()
    print(f'param before: {before.flatten()[:5]}')

    dataset = MathCotKDDataset(
        jsonl_path=FLAGS.data_path,
        tokenizer=tokenizer,
        max_prompt_len=FLAGS.gmp_max_prompt_len,
        max_len=FLAGS.gmp_max_seq_len,
    )

    globalprune_gmp(model, tokenizer, dataset, FLAGS)

    after = param.data.clone()
    print(f'param after:  {after.flatten()[:5]}')
    print(f'weights changed: {not torch.allclose(before, after)}')
    print('DONE')

if __name__ == '__main__':
    app.run(main)
