from .common import cusolverDnHandle

import torch

import nvmath
import kermac
from ..common import Majorness

def solve_lu(
    a : torch.Tensor,
    b : torch.Tensor,
    overwrite_a : bool = False,
    overwrite_b : bool = False,
    check_errors : bool = False,
    debug : bool = False
):
    # Check if inputs are tensors
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be PyTorch tensors")

    # Check dtype
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("a and b must have dtype torch.float32")
    
    # Check CUDA device
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be on a CUDA device")
    
    tensor_device = a.device
    if not all(x.device == tensor_device for x in (a, b)):
        raise ValueError(f"All inputs must be on the same CUDA device: got {[x.device for x in (a, b)]}")
    
    tensor_stats_a = kermac.tensor_stats(a)
    tensor_stats_b = kermac.tensor_stats(b)

    L_a, N_0_a, N_1_a = tensor_stats_a.shape
    L_b, C_b, N_b = tensor_stats_b.shape

    if L_a != L_b:
        raise ValueError(f'a and b tensor must have the same batch mode got{(L_a, L_b)}')
    
    L = L_a
    
    if N_0_a != N_1_a:
        raise ValueError(f'a tensor is not square, got {(N_0_a, N_1_a)}')
    
    N = N_0_a

    if N_b != N:
        raise ValueError(f'b tensor must have {N} columns, got {N_b}')
    
    if tensor_stats_b.majorness == Majorness.COL_MAJOR:
        raise ValueError(f'b tensor must have 1 stride in the rightmost dimension, got {tensor_stats_b.leading_dimension_stride}')
    
    C = C_b

    if not overwrite_a:
        a = a.clone()
    
    if not overwrite_b:
        b = b.clone()

    stride_a = tensor_stats_a.leading_dimension_stride
    stride_b = tensor_stats_b.leading_dimension_stride
    
    cusolver_handle = cusolverDnHandle()

    data_type_a = nvmath.CudaDataType.CUDA_R_32F
    data_type_b = nvmath.CudaDataType.CUDA_R_32F
    compute_type = nvmath.CudaDataType.CUDA_R_32F

    trans = nvmath.bindings.cublas.Operation.N

    device_bytes, host_bytes = \
        nvmath.bindings.cusolverDn.xgetrf_buffer_size(
            cusolver_handle._cusolver_handle,
            cusolver_handle._cusolver_params,
            N,
            N,
            data_type_a,
            a.data_ptr(), 
            stride_a,
            compute_type
        )
    
    if debug:
        print(f'(Kermac Debug) lu_decomposition (problem_size: ({L},{N},{N}), device_bytes_per_batch: {device_bytes})')

    buffer_on_device = torch.zeros(L,kermac.ceil_div(device_bytes,4), device=tensor_device, dtype=torch.int32)
    buffer_on_host = torch.zeros(L,kermac.ceil_div(host_bytes,4), dtype=torch.int32)

    factor_infos = torch.ones(L,device=tensor_device,dtype=torch.int32)
    solve_infos = torch.ones(L,device=tensor_device,dtype=torch.int32)

    ipiv = torch.zeros(L,N,device=tensor_device,dtype=torch.int64)

    primary_stream = torch.cuda.current_stream()
    primary_event = torch.cuda.Event(enable_timing=False)

    # Record an event on the primary stream so other streams don't race past it
    primary_stream.record_event(primary_event)

    streams = [torch.cuda.Stream() for _ in range(L)]
    events = [torch.cuda.Event(enable_timing=False) for _ in range(L)]

    for l in range(L):
        this_stream = streams[l]
        this_event = events[l]
        # Wait for primary_event to finish w.r.t this_stream
        this_stream.wait_event(primary_event)

        nvmath.bindings.cusolverDn.set_stream(cusolver_handle._cusolver_handle, this_stream.cuda_stream)
        nvmath.bindings.cusolverDn.xgetrf(
            cusolver_handle._cusolver_handle,
            cusolver_handle._cusolver_params,
            N,
            N,
            data_type_a,
            a[l].data_ptr(), 
            stride_a,
            ipiv[l].data_ptr(),
            compute_type,
            buffer_on_device[l].data_ptr(),
            device_bytes,
            buffer_on_host[l].data_ptr(),
            host_bytes,
            factor_infos[l].data_ptr()
        )

        nvmath.bindings.cusolverDn.xgetrs(
            cusolver_handle._cusolver_handle,
            cusolver_handle._cusolver_params,
            trans,
            N,
            C,
            data_type_a,
            a[l].data_ptr(), 
            stride_a,
            ipiv[l].data_ptr(),
            data_type_b,
            b[l].data_ptr(), 
            stride_b,
            solve_infos[l].data_ptr()
        )

        this_stream.record_event(this_event)

    for event in events:
        # Now make sure that primary_stream synchronizes with all of the work from streams
        primary_stream.wait_event(event)

    if check_errors:
        primary_stream.synchronize()
        non_zero_errors = []
        for l in range(L):
            factor_info = factor_infos[l]
            solve_info = solve_infos[l]

            if factor_info != 0:
                non_zero_errors.append(('factor_info', l, factor_info))
            # Check if solve_info is non-zero and append to list
            if solve_info != 0:
                non_zero_errors.append(('solve_info', l, solve_info))
        # If there are non-zero errors, raise an exception with the list
        if non_zero_errors:
            raise ValueError(f"Non-zero items found: {non_zero_errors}")

    return b, factor_infos, solve_infos
