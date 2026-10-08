import numpy as np
import torch

def data_loading(x: np.ndarray, batch_size: int, context_len: int, 
                 device: str)-> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(x) - context_len
    if max_start <= 0:
        raise ValueError
    
    starts = np.random.randint(0, max_start, size = batch_size)
    
    inputs = x[starts[:, None] + np.arange(context_len)]
    targets = x[starts[:, None] + np.arange(context_len) + 1]
    
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    
    return inputs_tensor, targets_tensor