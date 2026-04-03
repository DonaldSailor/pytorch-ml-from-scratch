import torch
import math


class LinearLayer:
    def __init__(self, in_features: int, out_features: int):
        self.layer = torch.randn((out_features, in_features), requires_grad=True) * (
            1 / math.sqrt(in_features)
        )
        self.bias = torch.zeros((out_features,), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.add(torch.matmul(x, torch.transpose(self.layer, 0, 1)), self.bias)


layer = LinearLayer(8, 4)
print("W shape:", layer.layer.shape)  # should be (4, 8)
print("b shape:", layer.bias.shape)  # should be (4,)

x = torch.randn(2, 8)
y = layer.forward(x)
print("Output shape:", y.shape)  # should be (2, 4)
