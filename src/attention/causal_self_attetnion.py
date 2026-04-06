import torch
import math
from src.functional.softmax import softmax

def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    seq_len = Q.shape[-2]
    mask = ~torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0) #only one unsqueeze, multihead would require another one
    

    qk = torch.matmul(Q, K.transpose(-2,-1))
    dem = qk.div(math.sqrt(K.size(-1)))

    masked = dem.masked_fill(mask, -torch.inf)

    x = softmax(masked, dim=-1)

    return x.matmul(V)
