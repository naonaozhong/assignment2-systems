import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(torch.nn.Module):     
    
    def __init__(self, in_features: int,
                 out_features: int, 
                 device: torch.device | None=None, 
                 dtype: torch.dtype | None=None):
        super().__init__()
        self.in_fearures = in_features # final dimension of the input
        self.out_features = out_features # final dimension of the output
        self.device = device
        self.dtype = dtype
        self.std = math.sqrt(2/(in_features + out_features))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, dtype=self.dtype, device=self.device),
                            0, self.std, -3 * self.std, 3 * self.std)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return einsum(x, self.weight, "... in_feature, out_feature in_feature -> ... out_feature")
        