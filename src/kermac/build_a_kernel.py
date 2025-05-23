from cuda.core.experimental import Device, LaunchConfig, launch

from .module_cache import *
from .common import *

from enum import Enum, auto
from typing import Optional

import torch
import numpy as np

# For templates to dictate whether
# an input tensor is aligned to 16 Bytes (4 float elements)
class Alignment(Enum):
    ALIGN_1 = auto()
    ALIGN_4 = auto()

# For templates to dictate the type of
# contraction operation
class InnerOperator(Enum):
    DIFF = auto()
    DOT = auto()

# For templates to dictate the type of
# inner and outer power operation
class PowerType(Enum):
    NOOP = auto()
    ABS = auto()
    SQUARE = auto()
    SQRT = auto()
    POW = auto()

# For templates to dictate the type of
# kernel to apply
class KernelType(Enum):
    NONE = auto()
    LAPLACE = auto()
    GAUSSIAN = auto()

class Symmetry(Enum):
    NonSymmetric = auto()
    Symmetric = auto()

class KernelDescriptor():
    def __init__(
        self,
        inner_operator,
        inner_power,
        outer_power,
        kernel_type,
    ):
        self._inner_operator = inner_operator
        self._inner_power = inner_power
        self._outer_power = outer_power
        self._kernel_type = kernel_type
    
    def _render_function_name(
      self,
      align_A,
      align_B,
    ):
        kernel_name_str = f'cute_build_kernel_m128n128k8p3'
        inner_operator_str = f'InnerOperator::{self._inner_operator.name}'
        inner_power_str = f'PowerType::{self._inner_power.name}'
        outer_power_str = f'PowerType::{self._outer_power.name}'
        kernel_type_str = f'KernelType::{self._kernel_type.name}'
        align_A_str = f'Alignment::{align_A.name}'
        align_B_str = f'Alignment::{align_B.name}'
        function_name = f'{kernel_name_str}<{inner_operator_str},{inner_power_str},{outer_power_str},{kernel_type_str},{align_A_str},{align_B_str}>'
        return function_name

kernel_descriptor_laplace_l1 = \
    KernelDescriptor(
        inner_operator=InnerOperator.DIFF,
        inner_power=PowerType.ABS,
        outer_power=PowerType.NOOP,
        kernel_type=KernelType.LAPLACE,
    )

kernel_descriptor_laplace_l2 = \
    KernelDescriptor(
        inner_operator=InnerOperator.DIFF,
        inner_power=PowerType.SQUARE,
        outer_power=PowerType.SQRT,
        kernel_type=KernelType.LAPLACE,
    )

kernel_descriptor_p_norm = \
    KernelDescriptor(
        inner_operator=InnerOperator.DIFF,
        inner_power=PowerType.POW,
        outer_power=PowerType.POW,
        kernel_type=KernelType.NONE,
    )

kernel_descriptor_l1_norm = \
    KernelDescriptor(
        inner_operator=InnerOperator.DIFF,
        inner_power=PowerType.ABS,
        outer_power=PowerType.NOOP,
        kernel_type=KernelType.NONE,
    )

kernel_descriptor_l2_norm = \
    KernelDescriptor(
        inner_operator=InnerOperator.DIFF,
        inner_power=PowerType.SQUARE,
        outer_power=PowerType.SQRT,
        kernel_type=KernelType.NONE,
    )

kernel_descriptor_mma = \
    KernelDescriptor(
        inner_operator=InnerOperator.DOT,
        inner_power=PowerType.NOOP,
        outer_power=PowerType.NOOP,
        kernel_type=KernelType.NONE,
    )

def run_kernel(
    kernel_descriptor : KernelDescriptor,
    a : torch.Tensor,
    b : torch.Tensor,
    out : torch.Tensor = None,
    p : Optional[float] = None,
    inner_p : Optional[float] = None,
    outer_p : Optional[float] = None,
    bandwidth : Optional[float] = None,
    try_to_align : bool = False,
    debug = False
):
    if p is not None:
        if kernel_descriptor._inner_power is not PowerType.POW and kernel_descriptor._outer_power is not PowerType.POW:
            raise ValueError("`p` is set but kernel doesn't use the value")
        if inner_p is not None:
            raise ValueError("`inner_p` is set but 'p' is also set")
        if outer_p is not None:
            raise ValueError("`outer_p` is set but 'p' is also set")

    if inner_p is not None:
        if kernel_descriptor._inner_power is not PowerType.POW:
            raise ValueError("`inner_p` is set but 'inner_power' is not 'POW")
        
    if outer_p is not None:
        if kernel_descriptor._outer_power is not PowerType.POW:
            raise ValueError("`outer_p` is set but 'outer_power' is not 'POW")

    if p is not None:
        inner_p = p     # if p is set then interpret that we want p
        outer_p = 1.0/p # and recip-p

    if bandwidth is not None:
        if kernel_descriptor._kernel_type is KernelType.NONE:
             raise ValueError("`bandwidth` is set but 'kernel_type' is 'NONE")
        
    if bandwidth is None:
        if kernel_descriptor._kernel_type is not KernelType.NONE:
            raise ValueError("`bandwidth` is not set but 'kernel_type' is not 'NONE")
        
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

    device_function_map = DeviceLoadedFunctionMap(debug)
   

    pt_stream = torch.cuda.current_stream()
    pt_device = pt_stream.device

    if tensor_device != pt_device:
        raise ValueError("cuda stream must be on the same device as the tensors: got {pt_device}, expected {tensor_device}")

    pt_device_id = pt_device.index
    device = Device(pt_device_id)

    # arch = get_compute_capability(device)
    # function_names = [
    #     kernel_descriptor_l1_norm._render_function_name(Alignment.ALIGN_1, Alignment.ALIGN_1),
    #     kernel_descriptor_l2_norm._render_function_name(Alignment.ALIGN_1, Alignment.ALIGN_1)
    # ]
    # device_function_map.pre_compile_and_store_cubin(arch, function_names , debug=debug)
    # exit()
    
    device.set_current()
    stream = PyTorchStreamWrapper(pt_stream)

    align_4_A = Alignment.ALIGN_4 if try_to_align and is_tensor_16_byte_aligned(a) else Alignment.ALIGN_1
    align_4_B = Alignment.ALIGN_4 if try_to_align and is_tensor_16_byte_aligned(b) else Alignment.ALIGN_1

    function_name = kernel_descriptor._render_function_name(align_4_A, align_4_B)

    kernel = device_function_map.get_function(device, function_name, debug=debug)

    if debug:
        print(f'(Kermac Debug) Launching kernel: {function_name}')

    bM = 128
    bN = 128

    block = 256
    num_blocks_M = ceil_div(M, bM)
    num_blocks_N = ceil_div(N, bN)

    grid = (num_blocks_M, num_blocks_N, 1)
    config = LaunchConfig(grid=grid, block=block)
    ld_a = a.stride(0)
    ld_b = b.stride(0)
    ld_c = result.stride(0)

    kernel_args = (
        M, N, K,
        a.data_ptr(), ld_a,
        b.data_ptr(), ld_b,
        result.data_ptr(), ld_c,
        np.float32(inner_p),
        np.float32(outer_p),
        np.float32(bandwidth)
    )

    launch(stream, config, kernel, *kernel_args)

    return result