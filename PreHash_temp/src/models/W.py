import torch

x = torch.zeros([16384,64])
print(x.shape)
print(x)
y = x.sum(dim=1).view([-1])
print(y.shape)