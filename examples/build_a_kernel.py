import torch
import kermac

device = torch.device('cuda')
a = torch.randn(10,10000,device=device)
b = torch.randn(10,10000,device=device)
out = torch.randn(10000,10000,device=device)

print('Running euclidean laplace kernel')
kermac.run_kernel(
    kermac.kernel_descriptor_laplace_l2,
    a, b,
    out = out,
    bandwidth=10.0,
    debug=True
)
print(out)

print('Running L1 laplace kernel')
kermac.run_kernel(
    kermac.kernel_descriptor_laplace_l1,
    a, b,
    out = out,
    bandwidth=10.0,
    debug=True
)
print(out)

print('Running L1 norm kernel')
kermac.run_kernel(
    kermac.kernel_descriptor_l1_norm,
    a, b,
    out = out,
    # bandwidth=10.0,
    debug=True
)
print(out)

kernel_descriptor_gaussian_p_norm = \
    kermac.KernelDescriptor(
        inner_operator=kermac.InnerOperator.DIFF,
        inner_power=kermac.PowerType.POW,
        outer_power=kermac.PowerType.POW,
        kernel_type=kermac.KernelType.GAUSSIAN,
    )

print('Running p-power gaussian kernel')
kermac.run_kernel(
    kernel_descriptor_gaussian_p_norm,
    a, b,
    out = out,
    inner_p=1.3,
    outer_p=1.0/1.3,
    bandwidth=10.0,
    debug=True
)
print(out)

print('Running L1 norm kernel again')
kermac.run_kernel(
    kermac.kernel_descriptor_l1_norm,
    a, b,
    out = out,
    # bandwidth=10.0,
    debug=True
)
print(out)

print('torch.cdist')
print(torch.cdist(a.T,b.T,p=1.0).T)

print('Running MMA')
kermac.run_kernel(
    kermac.kernel_descriptor_mma,
    a, b,
    out = out,
    # bandwidth=10.0,
    debug=True
)
print(out)

print((a.T @ b).T)
