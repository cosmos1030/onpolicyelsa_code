import torch


class ProjectedMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.9, nesterov=True, weight_decay=0.0, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def set_masks(self, _masks):
        pass  # lazy init in step()

    def _lazy_init_mask(self, p):
        st = self.state[p]
        if 'momentum_buffer' not in st:
            if p.data.dim() >= 2:
                mask = (p.data.to(torch.float32) != 0.0).to(p.dtype)
                if not mask.all():
                    st['mask'] = mask
            st['momentum_buffer'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                self._lazy_init_mask(p)
                state = self.state[p]

                # MaskedAdam과 동일: 매 step p.data, grad, momentum_buffer 마스킹
                if 'mask' in state:
                    mask = state['mask']
                    p.data.mul_(mask)
                    p.grad.data.mul_(mask)
                    state['momentum_buffer'].mul_(mask)

                g = p.grad

                # 1. Weight Decay
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

                # 2. Momentum
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                # 3. Projected Newton-Schulz (Interleaved)
                if 'mask' in state:
                    mask = state['mask']

                    # (A) Input Reshape
                    X = g.reshape(g.size(0), -1)
                    flat_mask = mask.reshape(mask.size(0), -1) if g.dim() == 4 else mask

                    # (B) Initialize on Manifold
                    X = X * flat_mask
                    X = X.bfloat16() / (X.norm() + 1e-7)

                    transposed = False
                    if X.size(0) > X.size(1):
                        X = X.T
                        flat_mask = flat_mask.T
                        transposed = True

                    # (C) Interleaved Loop
                    for _ in range(ns_steps):
                        A = X @ X.T
                        B = 3.0 * torch.eye(A.shape[0], device=A.device, dtype=X.dtype) - A
                        X = 0.5 * B @ X
                        X = X * flat_mask

                    if transposed:
                        X = X.T

                    update = X.to(g.dtype).reshape(g.shape)

                    # (D) Energy Compensation
                    expected_norm = (min(g.size(0), g.numel() // g.size(0))) ** 0.5
                    current_norm = update.norm()
                    if current_norm > 0:
                        update.mul_(expected_norm / current_norm)

                else:
                    # Dense Case
                    if g.size(0) > 1:
                        X = g.reshape(g.size(0), -1).bfloat16()
                        X = X / (X.norm() + 1e-7)
                        transposed = False
                        if X.size(0) > X.size(1):
                            X = X.T
                            transposed = True
                        for _ in range(ns_steps):
                            A = X @ X.T
                            B = 3.0 * torch.eye(A.shape[0], device=A.device, dtype=X.dtype) - A
                            X = 0.5 * B @ X
                        if transposed:
                            X = X.T
                        update = X.to(g.dtype).reshape(g.shape)
                    else:
                        update = g

                # 4. Final Update
                p.data.add_(update, alpha=-lr)
