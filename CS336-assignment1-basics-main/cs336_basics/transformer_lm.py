import torch
import torch.nn as nn

from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.softmax import softmax
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.multihead_self_attention import Multihead_Self_Attention
from cs336_basics.tranformer_blocks import Transformer_Blocks

class Transformer_lm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Transformer_Blocks(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.final_rmsnorm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor):
        token_positions = torch.arange(x.shape[1], device=x.device)
        y = self.embedding(x)
        for layer in self.layers:
            y = layer(y, token_positions)
        y = self.final_rmsnorm(y)
        y = self.linear(y)
        return y