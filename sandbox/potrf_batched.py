import nvmath
import torch
import kermac
import nvmath

import numpy as np

def get_batch_data_ptrs(tensor):
    """
    Constructs a 1D tensor containing the data_ptr of each matrix in a batched tensor.
    
    Args:
        tensor (torch.Tensor): 3D tensor of shape [batch_size, n, m] on CUDA device.
        
    Returns:
        torch.Tensor: 1D tensor of shape [batch_size] containing data_ptr values (int64).
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on a CUDA device.")
    if tensor.dim() != 3:
        raise ValueError("Tensor must be 3D (batch_size, n, m).")
    
    # Ensure tensor is contiguous for correct pointer arithmetic
    # tensor = tensor.contiguous()
    
    # Get batch size and matrix dimensions
    batch_size, n, m = tensor.shape
    
    # Calculate strides (in bytes) and base data_ptr
    base_ptr = tensor.data_ptr()
    stride = n * m * tensor.element_size()  # Size of one matrix in bytes
    
    # Create a tensor with indices [0, 1, ..., batch_size-1]
    indices = torch.arange(batch_size, dtype=torch.int64, device=tensor.device)
    
    # Compute data_ptr for each matrix: base_ptr + index * stride
    ptrs = base_ptr + indices * stride
    
    return ptrs

def broadcast_matrix_to_batch(matrix, batch_size):
    """
    Copies an NxN matrix to a BxNxN tensor, replicating the matrix across B batches.
    
    Args:
        matrix (torch.Tensor): 2D tensor of shape [N, N].
        batch_size (int): Number of batches (B).
        
    Returns:
        torch.Tensor: 3D tensor of shape [batch_size, N, N] with the matrix copied to each batch.
    """
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be 2D and square (N x N).")
    
    # Ensure matrix is on the correct device and contiguous
    matrix = matrix.contiguous()
    
    # Add a batch dimension and repeat across batch_size
    batched_matrix = matrix.unsqueeze(0).expand(batch_size, *matrix.shape)
    
    # Return a contiguous copy to ensure proper memory layout
    return batched_matrix.contiguous()

N = 100
D = 6
C = 10
P = 2
try_to_align = True
debug = True

device = torch.device('cuda')
a = torch.randn(D,N,device=device,dtype=torch.float32)
b_batched = torch.randn(P,C,N,device=device,dtype=torch.float32)
out = torch.randn(N,N,device=device,dtype=torch.float32)
factor_info = torch.ones(P,device=device,dtype=torch.int32)
solve_info = torch.ones(P,device=device,dtype=torch.int32)
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

out_batched = broadcast_matrix_to_batch(out, P)
print(out)
print(out_batched)
batched_data_pointers = get_batch_data_ptrs(out_batched)

print(nvmath.bindings.cufft.get_version())
cusolver_handle = nvmath.bindings.cusolverDn.create()
cusolver_params = nvmath.bindings.cusolverDn.create_params()
uplo = nvmath.bindings.cublas.FillMode.LOWER
data_type_a = nvmath.CudaDataType.CUDA_R_32F
data_type_b = nvmath.CudaDataType.CUDA_R_32F
compute_type = nvmath.CudaDataType.CUDA_R_32F
# compute_type = nvmath.CudaDataType.CUDA_R_64F

upper = nvmath.bindings.cublas.FillMode.UPPER

# nvmath.bindings.cusolverDn.spotrf_batched(
# intptr_t handle,
# int uplo,
# int n,
# aarray,
# int lda,
# intptr_t info_array,
# int batch_size,
# )[source]
a_array = get_batch_data_ptrs(out_batched)
factor_timer.start()
nvmath.bindings.cusolverDn.spotrf_batched(
    cusolver_handle,
    uplo,
    out_batched.size(1),
    a_array.data_ptr(), out_batched.stride(1),
    factor_info.data_ptr(),
    P
)
print(f'ms:{factor_timer.stop()}')

print(out_batched)

print(factor_info)
# nvmath.bindings.cusolverDn.spotrs_batched(
# intptr_t handle,
# int uplo,
# int n,
# int nrhs,
# a,
# int lda,
# b,
# int ldb,
# intptr_t d_info,
# int batch_size,
# )[source]

print(out_batched.shape)
print(b_batched.shape)
print(out_batched.size(1))
print(b_batched.size(1))
print(out_batched.stride(1))
print(b_batched.stride(1))
solve_timer.start()

print('here')
print(out_batched.size(1))
print(b_batched.size(2))
print(out_batched.stride(1))
print(b_batched.stride(1))
print(P)
b_array = get_batch_data_ptrs(b_batched)
print(b_array)

print('ONLY NRHS=1 is supported!')
nvmath.bindings.cusolverDn.spotrs_batched(
    cusolver_handle,
    uplo,
    np.int32(N),
    np.int32(1),
    a_array.data_ptr(), np.int32(out_batched.stride(1)),
    b_array.data_ptr(), np.int32(b_batched.stride(1)),
    solve_info.data_ptr(),
    np.int32(P)
)
print(f'ms:{solve_timer.stop()}')

print(f'factor_info: {factor_info}')
print(f'solve_info: {solve_info}')
