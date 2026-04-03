import torch

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    d = x.shape[-1]
    rms = torch.sqrt(torch.mul(torch.sum(torch.square(x), dim=-1, keepdim=True), 1/d)+eps)
    return torch.mul(torch.div(x, rms), weight)
