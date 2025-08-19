from cuda.core.experimental import Device, LaunchConfig, launch

import torch

from .module_cache.module_cache import *
from .common import *
from .build_a_kernel import *

def cdist(
    a : torch.Tensor,
    b : torch.Tensor,
    out : torch.Tensor = None,
    p : float = 2.0,
    skip_epilogue : bool = False,
    try_to_align : bool = False,
    debug = False
):
    """
    Computes a cdist with input validation with CUDA.

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
        a (torch.Tensor): Input tensor of shape (M, K), dtype float32, on CUDA.
        b (torch.Tensor): Input tensor of shape (N, K), dtype float32, on CUDA.
        out (torch.Tensor, optional=None): Output tensor of shape (M, N), dtype float32, on CUDA.
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

    if p == 1.0:
        descriptor = kernel_descriptor_l1_norm
        # already skipped if p = 1.0
        # if skip_epilogue: descriptor._outer_power = PowerType.NOOP
        return run_kernel(
            descriptor,
            a, b, 
            out=out,
            try_to_align=try_to_align,
            debug=debug
        )
    elif p == 2.0:
        descriptor = kernel_descriptor_l2_norm
        if skip_epilogue: descriptor._outer_power = PowerType.NOOP
        return run_kernel(
            descriptor,
            a, b,
            out=out,
            try_to_align=try_to_align,
            debug=debug
        )
    else:
        descriptor = kernel_descriptor_p_norm
        if skip_epilogue: descriptor._outer_power = PowerType.NOOP
        return run_kernel(
            descriptor,
            a, b,
            out=out,
            p=p,
            try_to_align=try_to_align,
            debug=debug
        )
