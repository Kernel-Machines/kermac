import kermac
import torch

device = torch.device('cuda')
a = torch.randn(64,5000,5000,device=device)
b = torch.randn(64,5000,16,device=device)

timer = kermac.CudaTimer()

timer.start()
for _ in range(1):
    sol = torch.linalg.solve(a,b)
torch.cuda.synchronize()
ms = timer.stop()
print(ms)
print(sol.shape)
# print(sol)

