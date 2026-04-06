import torch

from torch import nn

class Embedding(nn.Module):
    
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.rand((num_embeddings, embedding_dim)))

    def forward(self, indices: torch.tensor) -> torch.Tensor:
        return self.weights[indices]
