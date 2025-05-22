import torch
import kermac

device = torch.device('cuda')
a = torch.randn(10,10000,device=device)
b = torch.randn(10,10000,device=device)
out = torch.randn(10000,10000,device=device)

if False:
    # Meant to name descriptor and reuse
    euclidean_laplace_descriptor = \
        kermac.KernelDescriptor(
            inner_operator=kermac.InnerOperator.DIFF,
            inner_power=kermac.PowerType.SQUARE,
            outer_power=kermac.PowerType.SQRT,
            kernel_type=kermac.KernelType.LAPLACE,
        )

print('Running euclidean laplace kernel')
kermac.run_kernel(
    kermac.KernelDescriptor(
        inner_operator=kermac.InnerOperator.DIFF,
        inner_power=kermac.PowerType.SQUARE,
        outer_power=kermac.PowerType.SQRT,
        kernel_type=kermac.KernelType.LAPLACE,
    ),
    a, b,
    out = out,
    bandwidth=10.0,
    debug=True
)
print(out)

print('Running L1 laplace kernel')
kermac.run_kernel(
    kermac.KernelDescriptor(
        inner_operator=kermac.InnerOperator.DIFF,
        inner_power=kermac.PowerType.ABS,
        outer_power=kermac.PowerType.ABS,
        kernel_type=kermac.KernelType.LAPLACE,
    ),
    a, b,
    out = out,
    bandwidth=10.0,
    debug=True
)
print(out)

print('Running L1 norm kernel')
kermac.run_kernel(
    kermac.KernelDescriptor(
        inner_operator=kermac.InnerOperator.DIFF,
        inner_power=kermac.PowerType.ABS,
        outer_power=kermac.PowerType.ABS,
        kernel_type=kermac.KernelType.NONE,
    ),
    a, b,
    out = out,
    # bandwidth=10.0,
    debug=True
)
print(out)

print('Running p-power gaussian kernel')
kermac.run_kernel(
    kermac.KernelDescriptor(
        inner_operator=kermac.InnerOperator.DIFF,
        inner_power=kermac.PowerType.POW,
        outer_power=kermac.PowerType.POW,
        kernel_type=kermac.KernelType.GAUSSIAN,
    ),
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
    kermac.KernelDescriptor(
        inner_operator=kermac.InnerOperator.DIFF,
        inner_power=kermac.PowerType.ABS,
        outer_power=kermac.PowerType.ABS,
        kernel_type=kermac.KernelType.NONE,
    ),
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
    kermac.KernelDescriptor(
        inner_operator=kermac.InnerOperator.DOT,
        inner_power=kermac.PowerType.NOOP,
        outer_power=kermac.PowerType.NOOP,
        kernel_type=kermac.KernelType.NONE,
    ),
    a, b,
    out = out,
    # bandwidth=10.0,
    debug=True
)
print(out)

print((a.T @ b).T)
