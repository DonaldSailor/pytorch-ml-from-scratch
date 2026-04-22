import torch
from torch import nn

import math

class KVCacheAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert(d_model%num_heads==0)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        self.d_model = d_model

    def forward(self, x: torch.Tensor, cache: torch.Tensor=None) -> torch.Tensor:
        batch_size, seq, _ = x.shape

        q = self.W_q(x).view(batch_size, seq, self.num_heads, self.d_k).transpose(1,2)
        k = self.W_k(x).view(batch_size, seq, self.num_heads, self.d_k).transpose(1,2)
        v = self.W_v(x).view(batch_size, seq, self.num_heads, self.d_k).transpose(1,2)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        cache_updated = (k, v)
        seq_len_total = k.shape[2]
        scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(self.d_k)

        if seq > 1:
            mask = torch.triu(torch.ones(seq, seq_len_total, dtype=torch.bool), diagonal=seq_len_total-seq)
            scores = scores.masked_fill(mask, float('-inf'))

        att = torch.softmax(scores, dim=-1).matmul(v)

        att_result = self.W_o(att.transpose(1,2).contiguous().view(batch_size, seq, -1))
        

        return att_result, cache_updated
     

torch.manual_seed(0)
attn = KVCacheAttention(d_model=64, num_heads=4)
x = torch.randn(1, 6, 64)

full_out, _ = attn(x)
out1, cache = attn(x[:, :4])
out2, cache = attn(x[:, 4:5], cache=cache)
out3, cache = attn(x[:, 5:6], cache=cache)
inc_out = torch.cat([out1, out2, out3], dim=1)

print('Full shape:', full_out.shape)
print('Match:', torch.allclose(full_out, inc_out, atol=1e-5))
print('Final cache K shape:', cache[0].shape)
     