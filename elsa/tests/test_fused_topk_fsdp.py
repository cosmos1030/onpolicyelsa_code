"""FSDP check for the fused top-k loss (lib.gmp_trainer._fused_topk_distill).

Run with torchrun --nproc_per_node=2. Every rank gets the SAME batch, so the
FSDP-averaged gradient must equal the single-process gradient. Rank 0 also
builds an unwrapped copy and computes
  (1) the fused loss without FSDP, and
  (2) a plain-autograd reference on materialized logits (CE + verl
      forward_kl_topk at the teacher's top-k),
then compares loss and per-parameter grads of a few parameters, including the
(tied) lm_head/embedding and a decoder layer, against the FSDP run.
"""
import os, sys, functools, torch, torch.nn.functional as F, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # elsa/
from transformers import AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from lib.gmp_trainer import _fused_topk_distill

MODEL = os.environ.get('TEST_MODEL', 'Qwen/Qwen3-0.6B')
K, TAU, NTP_W, KD_W = 64, 2.0, 0.33, 0.33
dist.init_process_group('nccl')
rank = dist.get_rank(); torch.cuda.set_device(rank); dev = torch.device('cuda', rank)

def load():
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    return m.to(dev)

torch.manual_seed(0)
teacher = load().eval()
for p in teacher.parameters(): p.requires_grad_(False)
# student = teacher with noise so KL != 0
student = load()
with torch.no_grad():
    g = torch.Generator(device='cpu').manual_seed(1)
    for p in student.parameters():
        p.add_((torch.randn(p.shape, generator=g) * 0.002).to(p.dtype).to(dev))
student.train(); student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
ref_student = None
if rank == 0:
    import copy
    ref_student = copy.deepcopy(student)

B, T = 2, 384
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib', 'gmp_trainer.py')).read()[200000:260000]
ids = tok(txt, return_tensors='pt').input_ids[:, :B * T].view(B, T).to(dev)
attn = torch.ones_like(ids); attn[1, -50:] = 0
labels = ids.clone(); labels[:, :100] = -100; labels[attn == 0] = -100

names = ['model.embed_tokens.weight', 'model.layers.3.self_attn.q_proj.weight',
         'model.layers.20.mlp.down_proj.weight', 'model.norm.weight']

def ref_loss(m):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        z = m(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
        with torch.no_grad():
            zt = teacher(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
    sel = labels[:, 1:] != -100
    zs, zts, ys = z[sel], zt[sel], labels[:, 1:][sel]
    ce = F.cross_entropy(zs, ys)
    tlp, idx = F.log_softmax(zts / TAU, -1).topk(K, -1)
    slp = F.log_softmax(zs / TAU, -1).gather(-1, idx)
    kl = (tlp.exp() * (tlp - slp)).sum(-1).mean()
    return NTP_W * ce + KD_W * kl, ce, kl

def grads_of(fs):
    out = {}
    with FSDP.summon_full_params(fs, with_grads=True, writeback=False):
        for n, p in fs.named_parameters():
            n = n.replace('_fsdp_wrapped_module.', '').replace('_checkpoint_wrapped_module.', '')
            if n in names:
                out[n] = p.grad.detach().float().clone()
    return out

# ---- FSDP run
blk = type(student.model.layers[0])
fs = FSDP(student, auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={blk}),
          mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
          sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True, device_id=dev)
torch.cuda.reset_peak_memory_stats()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    L, ntp, kd, mass = _fused_topk_distill(fs, teacher, ids, attn, labels, K, NTP_W, KD_W, TAU, 128)
L.backward()
fsdp_grads = grads_of(fs)
fs.zero_grad(set_to_none=True)
# same fused loss with an FSDP-sharded teacher (--gmp_teacher_fsdp layout)
import copy as _copy
t_fs = FSDP(_copy.deepcopy(teacher), auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={blk}),
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
            sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True, device_id=dev)
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    Lt, _, kdt, masst = _fused_topk_distill(fs, t_fs, ids, attn, labels, K, NTP_W, KD_W, TAU, 128)
Lt.backward()
fsdp_tgrads = grads_of(fs)
fs.zero_grad(set_to_none=True)
print(f'[rank{rank}] FSDP-teacher fused loss {Lt.item():.6f} kd {kdt:.5f} mass {masst:.4f} | '
      + ' '.join(f"{n.split('.')[-2]}:rel={((fsdp_tgrads[n]-fsdp_grads[n]).norm()/fsdp_grads[n].norm()).item():.2e}" for n in names), flush=True)
del t_fs
Lp, _, _ = ref_loss(fs)
Lp.backward()
fsdp_plain_grads = grads_of(fs)
print(f'[rank{rank}] FSDP plain loss {Lp.item():.6f} | FSDP fused loss {L.item():.6f} ntp {ntp:.5f} kd {kd:.5f} topk_mass {mass:.4f} '
      f'peak {torch.cuda.max_memory_allocated()/2**30:.2f}GiB', flush=True)

if rank == 0:
    m = ref_student
    # (1) fused, no FSDP
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        L1, ntp1, kd1, _ = _fused_topk_distill(m, teacher, ids, attn, labels, K, NTP_W, KD_W, TAU, 128)
    L1.backward()
    g1 = {n: p.grad.detach().float().clone() for n, p in m.named_parameters() if n in names}
    m.zero_grad(set_to_none=True)
    # (2) plain autograd reference
    L2, ce, kl = ref_loss(m)
    L2.backward()
    g2 = {n: p.grad.detach().float().clone() for n, p in m.named_parameters() if n in names}
    print(f'no-FSDP fused loss {L1.item():.6f} (ntp {ntp1:.5f} kd {kd1:.5f}) | reference {L2.item():.6f} (ce {ce.item():.5f} kl {kl.item():.5f})')
    cos = lambda a, b: F.cosine_similarity(a.flatten(), b.flatten(), 0).item()
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    for n in names:
        print(f'{n:45s} FSDP vs nofsdp cos {cos(fsdp_grads[n], g1[n]):.5f} rel {rel(fsdp_grads[n], g1[n]):.2e} | '
              f'FSDP vs ref cos {cos(fsdp_grads[n], g2[n]):.5f} rel {rel(fsdp_grads[n], g2[n]):.2e} | nofsdp vs ref rel {rel(g1[n], g2[n]):.2e} | '
              f'FLOOR FSDP-plain vs ref rel {rel(fsdp_plain_grads[n], g2[n]):.2e} | FSDP-fused vs FSDP-plain rel {rel(fsdp_grads[n], fsdp_plain_grads[n]):.2e}')
dist.barrier(); dist.destroy_process_group()
