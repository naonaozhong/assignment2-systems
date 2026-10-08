import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embedding: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        
        super().__init__()
        self.num_embedding = num_embedding # Size of the vocabulary
        self.embedding_dim = embedding_dim # Dimensions of the embedding vectors, d_model
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty(self.num_embedding, self.embedding_dim, device=self.device, dtype=self.dtype),
                                        a = -3, b = 3)
        )
        
    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.weight[token_ids]