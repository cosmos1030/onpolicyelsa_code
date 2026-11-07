"""_ChunkedNTPKDLoss must match HF-style shifted CE + _kl_loss(topk=0) in value and gradient."""
import torch
import torch.nn.functional as F
from absl import flags
from lib.gmp_trainer import _ChunkedNTPKDLoss, _kl_loss


def _ref(s, t, labels, ntp_w, kd_w, T):
    ce = F.cross_entropy(s[:, :-1].float().reshape(-1, s.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
    kl, _ = _kl_loss(s.float(), t.float(), labels, T, 0)
    return ntp_w * ce + kd_w * kl, ce, kl


def test_matches_reference():
    torch.manual_seed(0)
    B, L, V = 2, 37, 50
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        s = torch.randn(B, L, V, dtype=dtype, requires_grad=True)
        t = torch.randn(B, L, V, dtype=dtype)
        labels = torch.randint(0, V, (B, L))
        labels[0, :10] = -100
        labels[1, 25:] = -100
        for ntp_w, kd_w, T in ((0.33, 0.33, 1.0), (0.0, 1.0, 2.0), (1.0, 0.0, 1.0)):
            ref, ce, kl = _ref(s, t, labels, ntp_w, kd_w, T)
            g_ref, = torch.autograd.grad(ref, s)
            parts = {}
            out = _ChunkedNTPKDLoss.apply(s, t, labels, ntp_w, kd_w, T, 7, parts)
            g, = torch.autograd.grad(out, s)
            assert torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol), (dtype, ntp_w, out, ref)
            assert torch.allclose(g.float(), g_ref.float(), atol=tol, rtol=tol), (dtype, ntp_w, (g - g_ref).abs().max())
            # a term with weight 0 is skipped (logged as 0)
            assert abs(parts['ntp'] - (ce.item() if ntp_w else 0.0)) < tol * 10
            assert abs(parts['kd'] - (kl.item() if kd_w else 0.0)) < tol * 10


if __name__ == '__main__':
    flags.FLAGS(['x'])
    test_matches_reference()
    print('OK')
