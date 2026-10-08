import torch
import torch.nn as nn
import math
from einops import einsum

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.std = math.sqrt(2 / (self.d_model + self.d_ff))
        
        self.w1 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.d_ff, self.d_model),
                0,
                self.std,
                -3 * self.std,
                3 * self.std
            )
        )
        self.w2 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.d_model, self.d_ff),
                0,
                self.std,
                -3 * self.std,
                3 * self.std
            )
        )
        self.w3 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.d_ff, self.d_model),
                0,
                self.std,
                -3 * self.std,
                3 * self.std
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        y = y * torch.sigmoid(y)
        y = y * einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        y = einsum(self.w2, y, "d_model d_ff, ... d_ff -> ... d_model")
        
        return y