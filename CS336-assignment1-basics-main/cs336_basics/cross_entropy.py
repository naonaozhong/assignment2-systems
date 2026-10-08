import torch

def cross_entropy(logits: torch.Tensor, target: torch.Tensor):
    y = logits - logits.max(dim=-1, keepdim=True).values
    loss = -y[torch.arange(0, logits.shape[0]), target]
    y = torch.exp(y)
    loss += torch.log(y.sum(dim=-1))
    loss = loss.mean()
    return loss