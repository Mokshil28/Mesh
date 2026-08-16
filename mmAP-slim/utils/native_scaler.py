# --------------------------------------------------------
# Based on timm and MAE-priv code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------

import math

import numpy as np
import torch
# from torch._six import inf
from torch import inf


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True, device_type: str = "cuda"):
        # MPS/CPU: keep GradScaler disabled; CUDA keeps AMP enabled.
        self.device_type = device_type if device_type in {"cuda", "cpu"} else "cpu"
        self._scaler = torch.amp.GradScaler(self.device_type, enabled=bool(enabled) and self.device_type == "cuda")

    def __call__(self, loss, optimizer, clip_grad=None, skip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            elif skip_grad is not None:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
                if norm >= skip_grad:
                    self._scaler.update()
                    return norm
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    total_iters = int(epochs * niter_per_ep)
    if total_iters <= 0:
        return np.array([], dtype=np.float64)

    warmup_iters = int(warmup_epochs * niter_per_ep)
    if warmup_steps > 0:
        warmup_iters = int(warmup_steps)
    warmup_iters = max(0, min(warmup_iters, total_iters))
    print("Set warmup steps = %d" % warmup_iters)

    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters, endpoint=False)
    else:
        warmup_schedule = np.array([], dtype=np.float64)

    cosine_iters = total_iters - warmup_iters
    if cosine_iters > 0:
        iters = np.arange(cosine_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / cosine_iters))
    else:
        schedule = np.array([], dtype=np.float64)

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule
