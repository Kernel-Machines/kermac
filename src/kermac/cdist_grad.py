from cuda.core.experimental import Device, LaunchConfig, launch

import torch
import numpy as np

from .paths import *
from .module_cache import *
from .common import *

def cdist_grad(
    a : torch.Tensor,           # [K,M]     # M-major # [N,M]   # kernel_matrix
    b : torch.Tensor,           # [N,K]     # K-major # [D,N]   # x
    c : torch.Tensor,           # [O,K]     # K-major # [C,N]   # coefs
    d : torch.Tensor,           # [N,M]     # M-major # [D,M]   # z
    out : torch.Tensor = None,  # [O,N,M]   # M-major # [C,D,M] # grad
    p : float = 2.0,
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
            for (int k = 0; k < K; k++) {
                float diff = d[n,m] - b[n,k];
                float sign = signum(diff);
                float v = pow(abs(diff), p - 1.0)) * sign;
                v = a[k,m] * v;
                for (int o = 0; o < O; o++) {
                    out[o,n,m] += c[o,k] * v;
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
        skip_epilogue (bool, optional=False): Avoid the final step of the result where we raise the result to the 1.0/p power.
    
    Returns:
        torch.Tensor: Result tensor (placeholder implementation).
    
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
    if not all(x.dim() == 2 for x in (a, b, c, d)):
        raise ValueError("All inputs must be 2-dimensional")
    # Check number of dimensions for out, if provided
    if out is not None and out.dim() != 3:
        raise ValueError("out must be 3-dimensional")

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
    
    K, M = a.shape
    N, _ = b.shape
    O, _ = c.shape

    # Check shapes
    if a.shape != (K, M):
        raise ValueError(f"Expected shape {(K, M)} for a, got {a.shape}")
    if b.shape != (N, K):
        raise ValueError(f"Expected shape {(N, K)} for b, got {b.shape}")
    if c.shape != (O, K):
        raise ValueError(f"Expected shape {(O, K)} for c, got {c.shape}")
    if d.shape != (N, M):
        raise ValueError(f"Expected shape {(N, M)} for d, got {d.shape}")
    if out is not None and out.shape != (O, N, M):
        raise ValueError(f"Expected shape {(O, N, M)} for out, got {out.shape}")

    # Check strides (stride 1 in last dimension)
    if a.stride(1) != 1:
        raise ValueError("a must have stride 1 in last dimension")
    if b.stride(1) != 1:
        raise ValueError("b must have stride 1 in last dimension")
    if c.stride(1) != 1:
        raise ValueError("c must have stride 1 in last dimension")
    if d.stride(1) != 1:
        raise ValueError("d must have stride 1 in last dimension")
    if out is not None and out.stride(2) != 1:
        raise ValueError("out must have stride 1 in last dimension")
    
    result = torch.zeros((O, N, M), dtype=torch.float32, device=tensor_device) if out is None else out

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
        
    function_string = f'cute_norm_kernel_gradient_m128n16o16k32p2<NormType::{norm_type}>'
    module_cubin = device_module_map.get_module(device, function_string, debug=debug)

    if debug:
        print(f'(Kermac Debug) Launching kernel: {function_string}')
    kernel = module_cubin.get_kernel(function_string)

    p = np.float32(p) # convert to float32

    bM = 128
    bN = 16
    bO = 16

    block = 256
    num_blocks_M = ceil_div(M, bM)
    num_blocks_N = ceil_div(N, bN)
    num_blocks_O = ceil_div(O, bO)

    grid = (num_blocks_M, num_blocks_N, num_blocks_O)
    config = LaunchConfig(grid=grid, block=block)

    ld_a = a.stride(0)
    ld_b = b.stride(0)
    ld_c = c.stride(0)
    ld_d = d.stride(0)
    ld_e_N = result.stride(1) 
    ld_e_O = result.stride(0) # outer-most/slowest-moving/left-most stride

    kernel_args = (
        np.float32(p),
        M, N, O, K,
        a.data_ptr(), ld_a,
        b.data_ptr(), ld_b,
        c.data_ptr(), ld_c,
        d.data_ptr(), ld_d,
        result.data_ptr(), ld_e_N, ld_e_O
    )

    launch(stream, config, kernel, *kernel_args)

    return result