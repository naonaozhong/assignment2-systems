import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, 
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        
        super().__init__()
        self.d_model = d_model # Hidden dimension of the model
        self.eps = eps # For numerical stabability
        self.device = device
        self.dtype = dtype
        
        self.gi = nn.Parameter(
            torch.ones(d_model, dtype=self.dtype, device=self.device)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape"""
        x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result =  x / rms * self.gi
        result.to(self.dtype)
        return result