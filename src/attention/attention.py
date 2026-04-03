import torch
import math

from functional.softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:

    qk = torch.matmul(Q, K.transpose(-2,-1))
    dem = math.sqrt(K.size(-1))
    x = softmax(qk.div(dem), dim=-1)

    return x.matmul(V)
