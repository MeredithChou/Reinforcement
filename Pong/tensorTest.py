import torch

x = torch.tensor([[1,2],[9,3]])
print(x)
print(x.unsqueeze(0).dim())
