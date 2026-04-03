import torch
import math


def layer_norm(
    x: torch.Tensor, gamma: float, beta: float, eps: float = 1e-5
) -> torch.Tensor:
    u = torch.mean(x, dim=-1, keepdim=True)
    s = torch.var(x, dim=-1, keepdim=True, unbiased=False)

    x = torch.sub(x, u)
    x = torch.div(x, torch.sqrt(torch.add(s, eps)))
    x = torch.mul(x, gamma)

    return torch.add(x, beta)
