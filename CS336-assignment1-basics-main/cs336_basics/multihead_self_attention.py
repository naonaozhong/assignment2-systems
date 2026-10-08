import torch
import torch.nn as nn 
import math
from jaxtyping import Int
from einops import einsum, rearrange
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding

class Multihead_Self_Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.dk = d_model // self.h
        self.dv = self.dk
        self.std = math.sqrt(1 / (self.d_model))
        
        self.wq = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.h * self.dk, self.d_model),
                0, self.std, -3 * self.std, 3 * self.std
            )
        )
        
        self.wk = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.h * self.dk, self.d_model),
                0, self.std, -3 * self.std, 3 * self.std
            )
        )
        
        self.wv = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.h * self.dv, self.d_model),
                0, self.std, -3 * self.std, 3 * self.std
            )
        )
        
        self.wo = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.d_model, self.h * self.dv),
                0, self.std, -3 * self.std, 3 * self.std
            )
        )
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.dk, max_seq_len)
        else:
            self.rope = None
        
    def forward(self, x: torch.Tensor, token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        q = einsum(self.wq, x, "hdk d_model, ... d_model -> ... hdk")
        k = einsum(self.wk, x, "hdk d_model, ... d_model -> ... hdk")
        v = einsum(self.wv, x, "hdv d_model, ... d_model -> ... hdv")
        
        q = rearrange(q, "... seq_len (h dk) -> ... h seq_len dk", h=self.h)
        k = rearrange(k, "... seq_len (h dk) -> ... h seq_len dk", h=self.h)
        v = rearrange(v, "... seq_len (h dv) -> ... h seq_len dv", h=self.h)
        
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
            
        out = scaled_dot_product_attention(k, q, v, mask)
        
        out = rearrange(out, "... h seq_len dv -> ... seq_len (h dv)")
        
        return einsum(self.wo, out, "d_model hdv, ... seq_len hdv -> ... seq_len d_model")