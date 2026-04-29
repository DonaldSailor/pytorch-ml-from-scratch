import torch
import torch.nn.functional as F
import math

def sliding_window_attention(Q, K, V, window_size):
    _, seq_len, d = Q.shape
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (math.sqrt(d))
    positions = torch.arange(seq_len)
    distances = (positions[None, :] - positions[:, None]).abs()
    
    mask = distances > window_size
    res = F.softmax(scores.masked_fill(mask.unsqueeze(0), float('-inf')))

    return torch.matmul(res, V)

