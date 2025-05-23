import torch
import kermac

device = torch.device('cuda')
a = torch.randn(10,20000,device=device)
b = torch.randn(10,20000,device=device)
out = torch.randn(20000,20000,device=device)
debug = True
try_to_align = True

print('Bulk compiling kernels')
print('Choosing architecture from cuda device 0')
arch = kermac.get_compute_capability(device)

device_loaded_function_map = kermac.DeviceLoadedFunctionMap(debug)

print('Pre compiling a bunch of kernels')
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

function_names = []
descriptors = [
    kermac.kernel_descriptor_l1_norm._render_function_name,
    kermac.kernel_descriptor_l2_norm._render_function_name,
    kermac.kernel_descriptor_p_norm._render_function_name,
    kermac.kernel_descriptor_laplace_l1._render_function_name,
    kermac.kernel_descriptor_laplace_l2._render_function_name,
    kermac.kernel_descriptor_mma._render_function_name,
    kernel_descriptor_gaussian_p_norm._render_function_name
]

if try_to_align:
    print('Because `try_to_align` is set, generating full matrix of alignment conditions')
    for descriptor in descriptors:
        for align_A in kermac.Alignment:
            for align_B in kermac.Alignment:
                function_names.append(descriptor(align_A=align_A, align_B=align_B))
else:
    print('Because `try_to_align` is not set, generating only Align_1 conditions')
    for descriptor in descriptors:
        function_names.append(descriptor(align_A=kermac.Alignment.ALIGN_1, align_B=kermac.Alignment.ALIGN_1))

device_loaded_function_map.compile_and_cache_functions(
    arch,
    function_names=function_names,
    debug = debug
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
