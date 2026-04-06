import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_x = logits.max(dim=-1, keepdim=True).values

    numerator = torch.exp(logits - max_x)
    sum_x = torch.sum(numerator, dim=-1, keepdim=True)
    log_sum_exp = max_x + torch.log(sum_x)
    correct_logits = logits[torch.arange(logits.size(0)), targets].unsqueeze(1)
    loss = -correct_logits + log_sum_exp

    return loss.mean()


# 🧪 Debug
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
print('Loss:', cross_entropy(logits, targets))
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets))
     
