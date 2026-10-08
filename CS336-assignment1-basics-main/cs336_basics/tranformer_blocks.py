import torch
import torch.nn as nn
from cs336_basics.multihead_self_attention import Multihead_Self_Attention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU

class Transformer_Blocks(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int | None = None,
                 rope_theta: float | None = None):
        super().__init__()
        self.multihead_layer = Multihead_Self_Attention(d_model, num_heads, max_seq_len, rope_theta)
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.swiglu = SwiGLU(d_model, d_ff)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y = x + self.multihead_layer(self.rmsnorm1(x), token_positions)
        y = y + self.swiglu(self.rmsnorm2(y))  
        return y