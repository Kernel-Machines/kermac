from cuda.core.experimental import Device, LaunchConfig, launch

import torch
import numpy as np

from typing import Union
from .module_cache.module_cache import *
from .common import *

def cdist_grad(
    a : torch.Tensor,           # [K,M]     # M-major # [N,M]   # kernel_matrix
    b : torch.Tensor,           # [N,K]     # K-major # [D,N]   # x
    c : torch.Tensor,           # [O,K]     # K-major # [C,N]   # coefs
    d : torch.Tensor,           # [N,M]     # M-major # [D,M]   # z
    out : torch.Tensor = None,  # [O,N,M]   # M-major # [C,D,M] # grad
    p : Union[float, torch.Tensor] = 2.0,
    debug = False
):
    """
    Computes cdist_grad on transposed tensors with input validation with CUDA.

    If in terms of AGOP.
        a is `kernel_matrix`
        b is `x`
        c is `coefs`
        d is `z`
        out is `grad`

    Computes (efficiently):
    ``` c
    // a[K,M], b[N,K], c[O,K], d[N,M], out[O,N,M]
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int o = 0; o < O; o++) {
                for (int k = 0; k < K; k++) {
                    float diff = d[n,m] - b[n,k];
                    float sign = signum(diff);
                    out[o,n,m] += c[o,k] * a[k,m] * sign * pow(abs(diff), p - 1.0));
                }
            }
        }
    }
    ```
    
    Args:
        a (torch.Tensor): Input tensor of shape (K, M), stride 1 in M, dtype float32, on CUDA.
        b (torch.Tensor): Input tensor of shape (N, K), stride 1 in K, dtype float32, on CUDA.
        c (torch.Tensor): Input tensor of shape (O, K), stride 1 in K, dtype float32, on CUDA.
        d (torch.Tensor): Input tensor of shape (N, M), stride 1 in M, dtype float32, on CUDA.
        out (torch.Tensor, optional=None): Output tensor of shape (O, N, M), stride 1 in M, dtype float32, on CUDA.
        p (float, optional=2.0): p value for the p-norm distance.
        debug (bool, optional=False): Print debug messages.
    
    Returns:
        torch.Tensor: Result tensor.
    
    Raises:
        TypeError: If inputs are not PyTorch tensors or have incorrect dtype.
        ValueError: If shapes, strides, dimensions, or CUDA devices are invalid.
    """

    # Check if inputs are tensors
    if not all(isinstance(x, torch.Tensor) for x in (a, b, c, d)):
        raise TypeError("All inputs must be PyTorch tensors")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a PyTorch tensor if provided")
    
    # Check dtype for a, b, c, d
    if not all(x.dtype == torch.float32 for x in (a, b, c, d)):
        raise TypeError("All inputs must have dtype torch.float32")
    # Check dtype for out, if provided
    if out is not None and out.dtype != torch.float32:
        raise TypeError("out must have dtype torch.float32")
    
    # Check number of dimensions for a, b, c, d
    if not all((x.dim() == 2 or x.dim() == 3) for x in (a, b, c, d)):
        raise ValueError("All inputs must be 2-dimensional or 3-dimensional with a batch mode")
    # Check number of dimensions for out, if provided
    if out is not None and (out.dim() != 3 and out.dim() != 4):
        raise ValueError("out must be 3-dimensional or 4-dimensional with a batch mode")

    # Check CUDA device for a, b, c, d
    if not all(x.is_cuda for x in (a, b, c, d)):
        raise ValueError("All inputs must be on a CUDA device")
    # Check CUDA device for out, if provided
    if out is not None and not out.is_cuda:
        raise ValueError("out must be on a CUDA device")

    tensor_device = a.device
    # Check device consistency for a, b, c, d
    if not all(x.device == tensor_device for x in (a, b, c, d)):
        raise ValueError(f"All inputs must be on the same CUDA device: got {[x.device for x in (a, b, c, d)]}")
    # Check device consistency for out, if provided
    if out is not None and out.device != tensor_device:
        raise ValueError(f"out must be on the same CUDA device as inputs: got {out.device}, expected {tensor_device}")
    
    tensor_stats_a = tensor_stats(a)
    tensor_stats_b = tensor_stats(b)
    tensor_stats_c = tensor_stats(c)
    tensor_stats_d = tensor_stats(d)
    
    _, K_a, M_a = tensor_stats_a.shape
    _, N_b, K_b = tensor_stats_b.shape
    _, O_c, K_c = tensor_stats_c.shape
    _, N_d, M_d = tensor_stats_d.shape

    L = 1
    L = merge_batch_size('p', L, p, expected_dims=0, can_be_none=False)
    L = merge_batch_size('a', L, a, expected_dims=2, can_be_none=False)
    L = merge_batch_size('b', L, b, expected_dims=2, can_be_none=False)
    L = merge_batch_size('c', L, c, expected_dims=2, can_be_none=False)
    L = merge_batch_size('d', L, d, expected_dims=2, can_be_none=False)
    L = merge_batch_size('out', L, out, expected_dims=3, can_be_none=True)

    if out is not None:
        L_e = 1 if out.dim() == 2 else out.size(0)
        if L_e != L and L != 1:
            raise ValueError(f"out must have batch dimension (L={L}), got {(L_e)}")

    # L is decided
    K = K_a
    M = M_a
    N = N_b
    O = O_c

    if isinstance(p, float):
        p = torch.tensor(p, dtype=torch.float32, device=tensor_device)
    else:
        if p.dtype != torch.float32:
            raise TypeError("`p` tensor must have dtype torch.float32")
        if not p.is_cuda or p.device != tensor_device:
            raise ValueError("`inner_p` tensor must be on the same CUDA device as inputs")

    shape_a = (K_a, M_a)
    shape_b = (N_b, K_b)
    shape_c = (O_c, K_c)
    shape_d = (N_d, M_d)

    # Check shapes
    if shape_a != (K, M):
        raise ValueError(f"Expected shape {(K, M)} for a, got {shape_a}")
    if shape_b != (N, K):
        raise ValueError(f"Expected shape {(N, K)} for b, got {shape_b}")
    if shape_c != (O, K):
        raise ValueError(f"Expected shape {(O, K)} for c, got {shape_c}")
    if shape_d != (N, M):
        raise ValueError(f"Expected shape {(N, M)} for d, got {shape_d}")
    if out is not None:
        if L == 1:
            if (out.shape != (O, N, M) and out.shape != (1, O, N, M)):
                raise ValueError(f"Expected shape {(O, N, M)} or {(L, O, N, M)} for out, got {out.shape}")
        else:
            if (out.shape != (L, O, N, M)):
                raise ValueError(f"Expected shape {(L, O, N, M)} for out, got {out.shape}")

    # Check strides (stride 1 in last dimension)
    if a.stride(-1) != 1:
        raise ValueError("a must have stride 1 in last dimension")
    if b.stride(-1) != 1:
        raise ValueError("b must have stride 1 in last dimension")
    if c.stride(-1) != 1:
        raise ValueError("c must have stride 1 in last dimension")
    if d.stride(-1) != 1:
        raise ValueError("d must have stride 1 in last dimension")
    if out is not None and out.stride(-1) != 1:
        raise ValueError("out must have stride 1 in last dimension")
    
    out = torch.zeros((L, O, N, M), dtype=torch.float32, device=tensor_device) if out is None else out

    module_cache = ModuleCache(debug)

    pt_stream = torch.cuda.current_stream()
    pt_device = pt_stream.device
    device = Device(pt_device.index)
    device.set_current()
    stream = PyTorchStreamWrapper(pt_stream)

    if tensor_device != pt_device:
        raise ValueError("cuda stream must be on the same device as the tensors: got {pt_device}, expected {tensor_device}")

    if p.numel() == 1:
        p_value = p.item()
        if p_value == 1.0:
            norm_type = 'L1'
        elif p_value == 2.0:
            norm_type = 'L2'
        else:
            norm_type = 'P'
    else:
        norm_type = 'P'
        
    function_string = f'cute_norm_kernel_gradient<NormType::{norm_type}>'
    kernel = module_cache.get_function(device, function_string, debug=debug)

    if debug:
        print(f'(Kermac Debug) Launching kernel: {function_string}')

    p_tensor = p

    num_blocks_M = ceil_div(M, 32)
    num_blocks_N = ceil_div(N, 32)
    num_blocks_O = ceil_div(O, 32)
    num_blocks_L = L

    grid = (num_blocks_L*num_blocks_M, num_blocks_N, num_blocks_O)
    config = LaunchConfig(grid=grid, block=256)

    ld_a = np.uint64(tensor_stats_a.leading_dimension_stride)
    batch_stride_a = np.uint64(tensor_stats_a.batch_stride)

    ld_b = np.uint64(tensor_stats_b.leading_dimension_stride)
    batch_stride_b = np.uint64(tensor_stats_b.batch_stride)

    ld_c = np.uint64(tensor_stats_c.leading_dimension_stride)
    batch_stride_c = np.uint64(tensor_stats_c.batch_stride)

    ld_d = np.uint64(tensor_stats_d.leading_dimension_stride)
    batch_stride_d = np.uint64(tensor_stats_d.batch_stride)

    ld_e_N = np.uint64(out.stride(-2))
    ld_e_O = np.uint64(out.stride(-3)) # outer-most/slowest-moving/left-most stride
    batch_stride_e = np.uint64(0 if L == 1 else out.stride(-4))

    batch_stride_p = np.uint64(0 if p_tensor.numel() == 1 else 1)

    kernel_args = (
        M, N, O, K, L,
        np.int32(num_blocks_M), # Need this to index num_blocks_L by division
        a.data_ptr(),       ld_a,                   batch_stride_a,
        b.data_ptr(),       ld_b,                   batch_stride_b,
        c.data_ptr(),       ld_c,                   batch_stride_c,
        d.data_ptr(),       ld_d,                   batch_stride_d,
        out.data_ptr(),     ld_e_N,     ld_e_O,     batch_stride_e,
        p_tensor.data_ptr(),                        batch_stride_p
    )

    launch(stream, config, kernel, *kernel_args)

    return out