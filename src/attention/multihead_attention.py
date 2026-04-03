import torch
from torch import nn

from attention.attention import scaled_dot_product_attention

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int) -> None:
        assert(d_model%num_heads==0)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        self.d_model = d_model

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        batch_size = Q.shape[0]
        seq_q = Q.shape[1]
        seq_k = K.shape[1]

        q = self.W_q(Q).view(batch_size, seq_q, self.num_heads, self.d_k).transpose(1,2)
        k = self.W_k(K).view(batch_size, seq_k, self.num_heads, self.d_k).transpose(1,2)
        v = self.W_v(V).view(batch_size, seq_k, self.num_heads, self.d_k).transpose(1,2)

        att = scaled_dot_product_attention(q,k,v).transpose(1,2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(att)
     
