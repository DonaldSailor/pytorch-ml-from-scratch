import torch
from torch import nn

from src.attention.attention import scaled_dot_product_attention

class GroupQueryAttention:
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int) -> None:
        self.d_k = d_model // num_heads
        self.num_kv_heads = num_kv_heads
        kv_size_dim = num_kv_heads*self.d_k
        self.d_model = d_model
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, kv_size_dim)
        self.W_k = nn.Linear(d_model, kv_size_dim)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x) -> torch.Tensor:
        batch_size, seq, _ = x.shape
        q = self.W_q(x).view(batch_size, seq, self.num_heads, self.d_k).transpose(1,2)
        k = self.W_k(x).view(batch_size, seq, self.num_kv_heads, self.d_k).transpose(1,2)
        v = self.W_v(x).view(batch_size, seq, self.num_kv_heads, self.d_k).transpose(1,2)

        k = torch.repeat_interleave(k, self.num_heads//self.num_kv_heads, dim=1)
        v = torch.repeat_interleave(v, self.num_heads//self.num_kv_heads, dim=1)

        att = scaled_dot_product_attention(q, k, v).transpose(1,2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(att)
