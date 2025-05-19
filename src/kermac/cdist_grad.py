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
    
    