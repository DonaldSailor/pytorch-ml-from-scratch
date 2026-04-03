import torch


def batch_norm_2d(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
    momentum: float = 0.1,
    training: bool = True,
) -> torch.Tensor:

    if training:
        batch_mean = x.mean(dim=0, keepdim=True)
        batch_var = x.var(dim=0, keepdim=True, unbiased=False)

        running_mean = running_mean.mul(1 - momentum).add(
            momentum * batch_mean.detach()
        )
        running_var = running_var.mul(1 - momentum).add(momentum * batch_var.detach())

        mean = batch_mean
        var = batch_var

    else:
        mean = running_mean
        var = running_var

    x = torch.div(torch.sub(x, mean), torch.sqrt(torch.add(var, eps)))
    return torch.add(torch.mul(x, gamma), beta)
