import nvmath
import torch
import kermac
import nvmath

import numpy as np

N = 60000
D = 6
C = 10
try_to_align = True
debug = True

device = torch.device('cuda')
a = torch.randn(N,D,device=device,dtype=torch.float32)
b = torch.randn(C,N,device=device,dtype=torch.float32)
out = torch.randn(N,N,device=device,dtype=torch.float32)
factor_info = torch.ones(1,device=device,dtype=torch.int32)
solve_info = torch.ones(1,device=device,dtype=torch.int32)
print(f'factor_info: {factor_info}')
print(f'solve_info: {solve_info}')

kernel_timer = kermac.CudaTimer()
factor_timer = kermac.CudaTimer()
solve_timer = kermac.CudaTimer()

kernel_timer.start()
kermac.run_kernel(
    kermac.kernel_descriptor_laplace_l2,
    a, a,
    out=out,
    bandwidth=10.0,
    try_to_align=try_to_align,
    debug=debug
)
print(f'ms: {kernel_timer.stop()}')

print(nvmath.bindings.cufft.get_version())
cusolver_handle = nvmath.bindings.cusolverDn.create()
cusolver_params = nvmath.bindings.cusolverDn.create_params()
uplo = nvmath.bindings.cublas.FillMode.LOWER
data_type_a = nvmath.CudaDataType.CUDA_R_32F
data_type_b = nvmath.CudaDataType.CUDA_R_32F
compute_type = nvmath.CudaDataType.CUDA_R_32F
# compute_type = nvmath.CudaDataType.CUDA_R_64F

upper = nvmath.bindings.cublas.FillMode.UPPER

# nvmath.bindings.cusolverDn.xpotrf_buffer_size(
#   intptr_t handle, 
#   intptr_t params, 
#   int uplo, 
#   int64_t n, 
#   int data_type_a, 
#   intptr_t a, 
#   int64_t lda, 
#   int compute_type
# )→ tuple[source]
device_bytes, host_bytes = \
    nvmath.bindings.cusolverDn.xpotrf_buffer_size(
        cusolver_handle,
        cusolver_params,
        uplo,
        out.size(0),
        data_type_a,
        out.data_ptr(), out.stride(0),
        compute_type
    )

buffer_on_device = torch.zeros(kermac.ceil_div(device_bytes,4), device=device, dtype=torch.int32)
buffer_on_host = torch.zeros(kermac.ceil_div(host_bytes,4), dtype=torch.int32)

print(f'workspace_device_bytes: {device_bytes}')
print(f'workspace_host_bytes: {host_bytes}')

factor_timer.start()
nvmath.bindings.cusolverDn.xpotrf(
    cusolver_handle,
    cusolver_params,
    uplo,
    out.size(0),
    data_type_a,
    out.data_ptr(), out.stride(0),
    compute_type,
    buffer_on_device.data_ptr(),
    device_bytes,
    buffer_on_host.data_ptr(),
    host_bytes,
    factor_info.data_ptr()
)
print(f'ms:{factor_timer.stop()}')

print(f'factor_info: {factor_info}')

del buffer_on_device
del buffer_on_host

solve_timer.start()
nvmath.bindings.cusolverDn.xpotrs(
    cusolver_handle,
    cusolver_params,
    uplo,
    out.size(0),
    b.size(0),
    data_type_a,
    out.data_ptr(), out.stride(0),
    data_type_b,
    b.data_ptr(), b.stride(0),
    solve_info.data_ptr()
)
print(f'ms:{solve_timer.stop()}')

print(f'solve_info: {solve_info}')
