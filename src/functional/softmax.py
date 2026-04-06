import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_x = x.max(dim, keepdim=True).values
    x1 = torch.exp(x-max_x)
    s = torch.sum(x1, dim=dim, keepdim=True)
    return x1.div(s)
