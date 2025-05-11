import kermac
import torch

M = 100
N = 20
K = 10

a = torch.randn(K, M).cuda()
b = torch.randn(K, N).cuda()
c = torch.randn(N, M).cuda()

c = kermac.cdist_transposed(a, b)

print(c)