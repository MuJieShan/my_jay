import torch
a=torch.zeros(6,dtype=float)
b=torch.tensor([0,3])
c=torch.tensor([1,1]).to(dtype=a.dtype)
print(a[b].shape,c.shape)
a[b]=c
print(a)
print(a.tolist())
