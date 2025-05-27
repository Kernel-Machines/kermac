from .common import cusolverDnHandle

import torch

import nvmath
import kermac
from ..common import FillMode, Majorness
from .common import map_fill_mode

def eigh(
    a : torch.Tensor,
    fill_mode : kermac.FillMode = kermac.FillMode.LOWER,
    overwrite_a : bool = False,
    check_errors : bool = False
):
    # Check if inputs are tensors
    if not isinstance(a, torch.Tensor):
        raise TypeError("a must be PyTorch tensors")

    # Check dtype
    if a.dtype != torch.float32:
        raise TypeError("a must have dtype torch.float32")
    
    # Check CUDA device
    if not a.is_cuda:
        raise ValueError("a must be on a CUDA device")
    
    tensor_device = a.device
    
    tensor_stats_a = kermac.tensor_stats(a)

    L_a, N_0_a, N_1_a = tensor_stats_a.shape
    
    L = L_a
    
    if N_0_a != N_1_a:
        raise ValueError(f'a tensor is not square, got {(N_0_a, N_1_a)}')
    
    N = N_0_a
    
    w = torch.zeros(L,N,device=tensor_device)

    if not overwrite_a:
        a = a.clone()

    stride_a = tensor_stats_a.leading_dimension_stride
    
    cusolver_handle = cusolverDnHandle()

    uplo = map_fill_mode(fill_mode)

    data_type_a = nvmath.CudaDataType.CUDA_R_32F
    data_type_w = nvmath.CudaDataType.CUDA_R_32F
    compute_type = nvmath.CudaDataType.CUDA_R_32F

    jobz = nvmath.bindings.cusolver.EigMode.VECTOR

    device_bytes, host_bytes = \
        nvmath.bindings.cusolverDn.xsyevd_buffer_size(
            cusolver_handle._cusolver_handle,
            cusolver_handle._cusolver_params,
            jobz,
            uplo,
            N,
            data_type_a,
            a.data_ptr(), 
            stride_a,
            data_type_w,
            w.data_ptr(),
            compute_type
        )

    buffer_on_device = torch.zeros(L, kermac.ceil_div(device_bytes,4), device=tensor_device, dtype=torch.int32)
    buffer_on_host = torch.zeros(L, kermac.ceil_div(host_bytes,4), dtype=torch.int32)

    infos = torch.ones(L,device=tensor_device,dtype=torch.int32)

    primary_stream = torch.cuda.current_stream()
    primary_event = torch.cuda.Event(enable_timing=False)

    # Record an event on the primary stream so other streams don't race past it
    primary_stream.record_event(primary_event)

    streams = [torch.cuda.Stream() for _ in range(L)]
    events = [torch.cuda.Event(enable_timing=False) for _ in range(L)]
    torch.cuda.synchronize()
    for l in range(L):
        this_stream = streams[l]
        this_event = events[l]
        # Wait for primary_event to finish w.r.t this_stream
        this_stream.wait_event(primary_event)

        nvmath.bindings.cusolverDn.set_stream(cusolver_handle._cusolver_handle, this_stream.cuda_stream)
        nvmath.bindings.cusolverDn.xsyevd(
            cusolver_handle._cusolver_handle,
            cusolver_handle._cusolver_params,
            jobz,
            uplo,
            N,
            data_type_a,
            a[l].data_ptr(), 
            stride_a,
            data_type_w,
            w[l].data_ptr(),
            compute_type,
            buffer_on_device[l].data_ptr(),
            device_bytes,
            buffer_on_host[l].data_ptr(),
            host_bytes,
            infos[l].data_ptr()
        )

        this_stream.record_event(this_event)

    for event in events:
        # Now make sure that primary_stream synchronizes with all of the work from streams
        primary_stream.wait_event(event)

    if check_errors:
        primary_stream.synchronize()
        non_zero_errors = []
        for l in range(L):
            info = infos[l]

            if info != 0:
                non_zero_errors.append(('info', l, info))
        # If there are non-zero errors, raise an exception with the list
        if non_zero_errors:
            raise ValueError(f"Non-zero items found: {non_zero_errors}")
    
    # cusolver uses column-major, need to permute
    return w, a.permute(0,2,1), infos
