import kermac
import torch

M = 100
N = 100
K = 64
L = 3

device = torch.device('cuda')
a = torch.randn(L,M,K,device=device)
b = torch.randn(L,N,K,device=device)
out = torch.randn(L,M,N,device=device)

torch_out = torch.cdist(a,b)
print(torch_out)
kermac_out = kermac.cdist(a,b,out=out)
print(kermac_out)
# print(kermac.cdist(a,b))

# tensor_stats_a = kermac.tensor_stats(a)
# print(a.stride())
# print(tensor_stats_a)