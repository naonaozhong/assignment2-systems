from typing import Any, Iterable

import torch
from torch.optim import Optimizer
import math

class AdamW(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]],
                 lr: float, betas: tuple[float, float], weight_decay: float, eps: float) -> None:
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
        
    @torch.no_grad()    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr =group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 0
                    
                m = state["m"]
                v = state["v"]
                t = state["t"] + 1
                state["t"] = t
                
                g = p.grad
                alpha_t = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p -= lr * weight_decay * p
                state["m"] = beta1 * m + (1 - beta1) * g
                state["v"] = beta2 * v + (1 - beta2) * (g**2)
                p -= alpha_t * (state["m"] / (torch.sqrt(state["v"]) + eps))
                
        return loss
        