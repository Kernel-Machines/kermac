from cuda.core.experimental import Device, LaunchConfig, launch

import torch
import numpy as np

from .paths import *
from .module_cache import *
from .common import *

def cdist_t(
    a : torch.Tensor,             # [K,M] # M-major
    b : torch.Tensor,             # [K,N] # N-major
    out : torch.Tensor = None,      # [N,M] # M-major
    p : float = 2.0,
    skip_epilogue : bool = False,
    debug = False
):
    """
    Computes a cdist on transposed tensors with input validation.
    
    Args:
        a_t (torch.Tensor): Input tensor of shape (K, M), stride 1 in M, dtype float32, on CUDA.
        b_t (torch.Tensor): Input tensor of shape (K, N), stride 1 in N, dtype float32, on CUDA.
        p (float, optional=2.0): p value for the p-norm distance. 
        out (torch.Tensor, optional=None): Output tensor of shape (N, M), stride 1 in M, dtype float32, on CUDA.
        skip_epilogue (bool, optional=False): Avoid the final step of the result where we raise the result to the 1.0/p power.
    
    Returns:
        torch.Tensor: Result tensor (placeholder implementation).
    
    Raises:
        TypeError: If inputs are not PyTorch tensors or have incorrect dtype.
        ValueError: If shapes, strides, dimensions, or CUDA devices are invalid.
    """
    # Check if inputs are tensors
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be PyTorch tensors")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a PyTorch tensor if provided")

    # Check dtype
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("a and b must have dtype torch.float32")
    if out is not None and out.dtype != torch.float32:
        raise TypeError("out must have dtype torch.float32")

    # Check number of dimensions
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("a and b must be 2-dimensional")
    if out is not None and out.dim() != 2:
        raise ValueError("out must be 2-dimensional")

    # Check CUDA device
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be on a CUDA device")
    if out is not None and not out.is_cuda:
        raise ValueError("out must be on a CUDA device")
    
    tensor_device = a.device
    if not all(x.device == tensor_device for x in (a, b)):
        raise ValueError(f"All inputs must be on the same CUDA device: got {[x.device for x in (a, b)]}")
    if out is not None and out.device != tensor_device:
        raise ValueError(f"out must be on the same CUDA device as inputs: got {out.device}, expected {tensor_device}")

    # Get shapes
    K_a, M = a.shape
    K_b, N = b.shape

    # Check shape consistency
    if K_a != K_b:
        raise ValueError(f"K dimensions must match: got {K_a} for a and {K_b} for b")
    
    K = K_a

    # Check strides (stride 1 in last dimension)
    if a.stride(1) != 1:
        raise ValueError("a must have stride 1 in dimension M (last dimension)")
    if b.stride(1) != 1:
        raise ValueError("b must have stride 1 in dimension N (last dimension)")
    if out is not None and out.stride(1) != 1:
        raise ValueError("out must have stride 1 in dimension M (last dimension)")

    # Check output shape if provided
    if out is not None:
        if out.shape != (N, M):
            raise ValueError(f"out must have shape (N={N}, M={M}), got {out.shape}")

    result = torch.zeros((N, M), dtype=torch.float32, device=a.device) if out is None else out

    device_module_map = DeviceModuleMap(debug)

    pt_stream = torch.cuda.current_stream()
    pt_device = pt_stream.device

    if tensor_device != pt_device:
        raise ValueError("cuda stream must be on the same device as the tensors: got {pt_device}, expected {tensor_device}")

    pt_device_id = pt_device.index

    device = Device(pt_device_id)

    device.set_current()
    stream = PyTorchStreamWrapper(pt_stream)

    if p == 1.0:
        norm_type = 'L1'
    elif p == 2.0:
        norm_type = 'L2'
    else:
        norm_type = 'P'

    skip = 'true' if skip_epilogue else 'false'
    function_string = f'cute_norm_m128m128k8p3<NormType::{norm_type},{skip}>'
    module_cubin = device_module_map.get_module(device, function_string, debug=debug)
    
    if debug:
        print(f'(Kermac Debug) Launching kernel: {function_string}')
    kernel = module_cubin.get_kernel(function_string)

    p = np.float32(p) # convert to float32

    bM = 128
    bN = 128
    bK = 8
    bP = 3
    shmem_size = (bM * bK * bP + bN * bK * bP) * 4

    block = 256
    num_blocks_M = ceil_div(M, bM)
    num_blocks_N = ceil_div(N, bN)

    grid = (num_blocks_M, num_blocks_N, 1)
    config = LaunchConfig(grid=grid, block=block, shmem_size=shmem_size)
    ld_a = a.stride(0)
    ld_b = b.stride(0)
    ld_c = result.stride(0)

    kernel_args = (
        np.float32(p),
        M, N, K,
        a.data_ptr(), ld_a,
        b.data_ptr(), ld_b,
        result.data_ptr(), ld_c,
    )

    launch(stream, config, kernel, *kernel_args)

    return result
