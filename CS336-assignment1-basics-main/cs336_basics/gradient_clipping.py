import torch
import torch.nn as nn
import math

def gradient_clipping(param_list: list[nn.Parameter], M: float, eps: float = 1e-6):
    
    total = 0.0
    
    for param in param_list:
        if param.grad is not None:
            total += param.grad.norm(2).pow(2).item()
    
    total = math.sqrt(total)
            
    if total > M:
        scale = M / (total + eps)
        
        for param in param_list:
            if param.grad is not None:
                param.grad.mul_(scale)