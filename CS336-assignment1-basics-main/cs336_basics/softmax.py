import torch

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    """apply softmax to the 𝑖-th dimension of the input tensor."""
    x_max = x.max(dim = i, keepdim = True).values
    
    x_stable = x - x_max
    x_exp = torch.exp(x_stable)
    return x_exp / x_exp.sum(dim=i, keepdim=True)