import torch
import kermac

device = torch.device('cuda')
a = torch.randn(10,20000,device=device)
b = torch.randn(10,20000,device=device)
out = torch.randn(20000,20000,device=device)
debug = True
try_to_align = True

# Example of a custom non-predefined kernel
# Because it uses PowerType.POW it will require a `p=` in the argument list for `run_kernel`
# Because it uses a KernelType.GAUSSIAN it will require a `bandwidth=` in the argument list for `run_kernel`
kernel_descriptor_gaussian_p_norm = \
    kermac.KernelDescriptor(
        inner_operator=kermac.InnerOperator.DIFF,
        inner_power=kermac.PowerType.POW,
        outer_power=kermac.PowerType.POW,
        kernel_type=kermac.KernelType.GAUSSIAN,
    )

descriptors = [
    kermac.kernel_descriptor_l1_norm,
    kermac.kernel_descriptor_l2_norm,
    kermac.kernel_descriptor_p_norm,
    kermac.kernel_descriptor_laplace_l1,
    kermac.kernel_descriptor_laplace_l2,
    kermac.kernel_descriptor_mma,
    kernel_descriptor_gaussian_p_norm
]

if debug:
    print('(Kermac Debug) Bulk compiling kernels')
    print('(Kermac Debug) Choosing architecture from cuda device 0')
    print('(Kermac Debug) Pre compiling a bunch of kernels')
kermac.pre_compile_descriptors(
    device=device,
    descriptors=descriptors,
    try_to_align=try_to_align,
    debug=debug
)

print('Running euclidean laplace kernel')
kermac.run_kernel(
    kermac.kernel_descriptor_laplace_l2,
    a, b,
    out = out,
    bandwidth=10.0,
    try_to_align=try_to_align,
    debug=debug
)
print(out)

print('Running L1 laplace kernel')
kermac.run_kernel(
    kermac.kernel_descriptor_laplace_l1,
    a, b,
    out = out,
    bandwidth=10.0,
    try_to_align=try_to_align,
    debug=debug
)
print(out)

print('Running L1 norm kernel')
kermac.run_kernel(
    kermac.kernel_descriptor_l1_norm,
    a, b,
    out = out,
    # bandwidth=10.0,
    try_to_align=try_to_align,
    debug=debug
)
print(out)


print('Running p-power gaussian kernel')
kermac.run_kernel(
    kernel_descriptor_gaussian_p_norm,
    a, b,
    out = out,
    inner_p=1.3,
    outer_p=1.0/1.3,
    bandwidth=10.0,
    try_to_align=try_to_align,
    debug=debug
)
print(out)

print('Running L1 norm kernel again')
kermac.run_kernel(
    kermac.kernel_descriptor_l1_norm,
    a, b,
    out = out,
    # bandwidth=10.0,
    try_to_align=try_to_align,
    debug=debug
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
    try_to_align=try_to_align,
    debug=debug
)
print(out)

print((a.T @ b).T)
