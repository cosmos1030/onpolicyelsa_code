"""Check _FusedLMHeadTopKKL against plain autograd on materialized logits."""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # elsa/
from lib.gmp_trainer import _FusedLMHeadTopKKL, _teacher_topk_logp
torch.manual_seed(0)
dev = 'cuda'
N, D, V, tau = 700, 256, 151936, 2.0
for dtype in (torch.float32, torch.bfloat16):
    h = (torch.randn(N, D, device=dev) * 0.5).to(dtype).requires_grad_()
    W = (torch.randn(V, D, device=dev) * 0.05).to(dtype).requires_grad_()
    th = (torch.randn(N, D, device=dev) * 0.5).to(dtype)
    tW = (W.detach() + 0.02 * torch.randn(V, D, device=dev).to(dtype))
    y = torch.randint(0, V, (N,), device=dev)
    for k in (V, 64):
        idx, tlp = _teacher_topk_logp(th, tW, k, tau, 256)
        parts = {}
        L = _FusedLMHeadTopKKL.apply(h, W, y, idx, tlp, 0.33, 0.33, tau, 256, parts)
        (L * 0.7).backward()
        gh, gW = h.grad.clone(), W.grad.clone(); h.grad = None; W.grad = None
        # reference
        z = (h @ W.t()).float()
        ce = F.cross_entropy(z, y)
        slp = F.log_softmax(z / tau, -1).gather(-1, idx)
        kl = (tlp.exp() * (tlp - slp)).sum(-1).mean()
        Lr = 0.33 * ce + 0.33 * kl
        (Lr * 0.7).backward()
        rh, rW = h.grad.clone(), W.grad.clone(); h.grad = None; W.grad = None
        cos = lambda a, b: F.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()
        rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
        if k == V:
            full = F.kl_div(F.log_softmax(z / tau, -1), F.log_softmax((th @ tW.t()).float() / tau, -1), log_target=True, reduction='none').sum(-1).mean()
            print(f'  full-vocab KL ref {full.item():.6f} vs topk(k=V) {kl.item():.6f}')
        print(f'{dtype} k={k}: loss fused {L.item():.6f} ref {Lr.item():.6f} | ntp {parts["ntp"]:.5f}/{ce.item():.5f} kd {parts["kd"]:.5f}/{kl.item():.5f} | '
              f'grad_h cos {cos(gh, rh):.6f} rel {rel(gh, rh):.2e} | grad_W cos {cos(gW, rW):.6f} rel {rel(gW, rW):.2e}')
print('peak MB', torch.cuda.max_memory_allocated() / 2**20)
