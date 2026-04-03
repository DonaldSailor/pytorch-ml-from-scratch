import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x1 = torch.exp(x)
    s = torch.sum(x1)
    return x1.div(s)
