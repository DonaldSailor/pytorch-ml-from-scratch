import torch
import math

def gelu(x):
    return 0.5*x*(1+torch.erf(x/math.sqrt(2)))
     



# 🧪 Debug
x = torch.tensor([-2., -1., 0., 1., 2.])
print('Output:', gelu(x))
print('Ref:   ', torch.nn.functional.gelu(x))
     
