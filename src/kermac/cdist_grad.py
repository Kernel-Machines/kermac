from cuda.core.experimental import Device, LaunchConfig, launch

import torch
import numpy as np

from .paths import *
from .module_cache import *
from .common import *

def cdist_grad(
    a : torch.Tensor,           # [K,M] # M-major # [N,M] # kernel_matrix
    b : torch.Tensor,           # [N,K] # K-major # [D,N] # x
    c : torch.Tensor,           # [O,K] # K-major # [C,N] # coefs
    d : torch.Tensor,           # [N,M] # M-major # [D,M] # z
    out : torch.Tensor = None,  # [O,N,M] # M-major # [C,D,M] # grad
    debug = False
):
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
    
    result = torch.zeros((O, N, M), dtype=torch.float32, device=device) if out is None else out

    device_module_map = DeviceModuleMap(debug)

    pt_stream = torch.cuda.current_stream()
    pt_device = pt_stream.device

    if tensor_device != pt_device:
        raise ValueError("cuda stream must be on the same device as the tensors: got {pt_device}, expected {tensor_device}")

    pt_device_id = pt_device.index

    device = Device(pt_device_id)

    device.set_current()
    stream = PyTorchStreamWrapper(pt_stream)