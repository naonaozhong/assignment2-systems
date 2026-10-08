import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int,
                 device: torch.device | None = None):
        super().__init__()
        self.theta = theta # \Theta value for the RoPE
        self.d_k = d_k # dimension of query and key vectors
        self.max_seq_len = max_seq_len # maximum sequence length that will be input
        self.device = device # Device to store the buffer on
        
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        
        i = torch.arange(self.max_seq_len, device=self.device, dtype=torch.float32)
        k = torch.arange(self.d_k // 2, device=self.device, dtype=torch.float32)
        denom = self.theta ** ((2 * k) / self.d_k) # 0-indexed
        angles = i[:, None] / denom[None, :]
        self.para = torch.stack([torch.cos(angles), torch.sin(angles)], dim = -1)
        self.register_buffer("rotate", self.para, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """ Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. 
        token_positions are a tensor of shape(..., seq_len) specifying the token positions of x along the sequence dimension"""
        
        x_reshape = rearrange(x, "... seq_len (d_k_half b) -> ... seq_len d_k_half b", b=2)
        
        R = self.para[token_positions] # shape(..., d_k/2, 2)
        
        x2k_1 = x_reshape[..., 0]
        x2k = x_reshape[..., 1]
        cos = R[..., 0]
        sin = R[..., 1]
        
        out2k_1 = x2k_1 * cos - x2k * sin
        out2k = x2k_1 * sin + x2k * cos
        out = torch.stack([out2k_1, out2k], dim = -1)
        out = out.reshape(*x.shape)
        
        return out