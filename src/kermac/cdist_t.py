from cuda.core.experimental import Device, LaunchConfig, launch

import torch
import numpy as np

from .module_cache import *
from .common import *
from .build_a_kernel import *

def cdist_t(
    a : torch.Tensor,           # [K,M] # M-major
    b : torch.Tensor,           # [K,N] # N-major
    out : torch.Tensor = None,  # [N,M] # M-major
    p : float = 2.0,
    skip_epilogue : bool = False,
    try_to_align : bool = False,
    debug = False
):
    """
    Computes a cdist on transposed tensors with input validation with CUDA.

    Computes (efficiently):
    ``` c
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                out[n,m] += pow(abs(b[k,n] - a[k,m]), p);
            }
            if (!skip_epilogue) {
                out[n,m] = pow(out[n,m], 1.0/p);
            }
        }
    }
    ```
    
    Args:
        a (torch.Tensor): Input tensor of shape (K, M), stride 1 in M, dtype float32, on CUDA.
        b (torch.Tensor): Input tensor of shape (K, N), stride 1 in N, dtype float32, on CUDA.
        out (torch.Tensor, optional=None): Output tensor of shape (N, M), stride 1 in M, dtype float32, on CUDA.
        p (float, optional=2.0): p value for the p-norm distance. 
        skip_epilogue (bool, optional=False): Avoid the final step of the result where we raise the result to the 1.0/p power.
        try_to_align (bool, optional=False): Specialize kernel for if tensor A and B are 16 byte aligned in starting pointer and stride(1)
        debug (bool, optional=False): Print debug messages.
    
    Returns:
        torch.Tensor: Result tensor.
    
    Raises:
        TypeError: If inputs are not PyTorch tensors or have incorrect dtype.
        ValueError: If shapes, strides, dimensions, or CUDA devices are invalid.
    """

    if not skip_epilogue:
        if p == 1.0:
            return run_kernel(
                kernel_descriptor_l1_norm,
                a, b, 
                out = out,
                try_to_align=try_to_align,
                debug=debug
            )
        elif p == 2.0:
            return run_kernel(
                kernel_descriptor_l2_norm,
                a, b,
                out = out,
                try_to_align=try_to_align,
                debug=debug
            )
        else:
            return run_kernel(
                kernel_descriptor_p_norm,
                a, b,
                out = out,
                p = p,
                try_to_align=try_to_align,
                debug=debug
            )
    else:
        if p == 1.0:
            return run_kernel(
                KernelDescriptor(
                    inner_operator=InnerOperator.DIFF,
                    inner_power=PowerType.ABS,
                    outer_power=PowerType.NOOP,
                    kernel_type=None
                ),
                a, b, 
                out = out,
                try_to_align=try_to_align,
                debug=debug
            )
        elif p == 2.0:
            return run_kernel(
                KernelDescriptor(
                    inner_operator=InnerOperator.DIFF,
                    inner_power=PowerType.SQUARE,
                    outer_power=PowerType.NOOP,
                    kernel_type=None
                ),
                a, b,
                out = out,
                try_to_align=try_to_align,
                debug=debug
            )
        else:
            return run_kernel(
                KernelDescriptor(
                    inner_operator=InnerOperator.DIFF,
                    inner_power=PowerType.POW,
                    outer_power=PowerType.NOOP,
                    kernel_type=None
                ),
                a, b,
                out = out,
                p = p,
                try_to_align=try_to_align,
                debug=debug
            )
