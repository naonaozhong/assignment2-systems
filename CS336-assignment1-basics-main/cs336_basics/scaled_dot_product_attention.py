import torch
from jaxtyping import Float, Bool
import math
from einops import einsum

def scaled_dot_product_attention(
    keys: Float[torch.Tensor, "batch_size ... seq_len d_k"],
    queries: Float[torch.Tensor, "batch_size ... seq_len d_k"],
    values: Float[torch.Tensor, "batch_size ... seq_len d_v"],
    masks: Bool[torch.Tensor, "seq_len seq_len"] | None = None
    ) -> Float[torch.Tensor, "batch_size ... seq_len d_v"]:
    d_k = keys.shape[-1]
    scores = einsum(queries, keys, "batch_size ... seq_len1 d_k, batch_size ... seq_len2 d_k -> batch_size ... seq_len1 seq_len2") / math.sqrt(d_k)
    if masks is not None:
        scores = scores.masked_fill(~masks, float("-inf"))
    scores = scores.softmax(dim=-1)
    return einsum(scores, values, "batch_size ... seq_len1 seq_len2, batch_size ... seq_len2 d_v -> batch_size ... seq_len1 d_v")