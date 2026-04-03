import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    x = torch.maximum(x, torch.tensor(0.0, dtype=x.dtype))
    return x


test_tensor = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])
print(relu(test_tensor))
