import kermac
import torch

N = 10000
K = 6

device = torch.device('cuda')
data = torch.randn(N,K,device=device)

p = torch.tensor([0.5,0.6,0.7,2.0], dtype=torch.float32, device=device)
inner_p = torch.tensor([0.5,0.6,0.7,2.0], dtype=torch.float32, device=device)
outer_p = torch.tensor([0.5,0.6,0.7,2.0], dtype=torch.float32, device=device)
bandwidth = torch.tensor([10.0,20.0], dtype=torch.float32, device=device)
regularization = torch.tensor([1e-3,1e-2,1e-1,1.0], dtype=torch.float32, device=device)

out = kermac.run_kernel(
    kermac.KernelDescriptor(
        inner_operator=kermac.InnerOperator.DIFF,
        inner_power=kermac.PowerType.SQUARE,
        outer_power=kermac.PowerType.SQRT,
        kernel_type=kermac.KernelType.LAPLACE
    ),
    data, data,
    bandwidth=bandwidth,
    regularization=10.0,
    regularization_offset_x=3, 
    regularization_offset_y=1
)
print(out)

bandwidth = torch.tensor(10.0, dtype=torch.float32, device=device)

out = kermac.run_kernel(
    kermac.KernelDescriptor(
        inner_operator=kermac.InnerOperator.DIFF,
        inner_power=kermac.PowerType.POW,
        outer_power=kermac.PowerType.POW,
        kernel_type=kermac.KernelType.LAPLACE
    ),
    data, data,
    p=p,
    # inner_p=inner_p,
    # outer_p=outer_p,
    bandwidth=bandwidth,
    regularization=regularization
)

print(out.shape)
print(out)