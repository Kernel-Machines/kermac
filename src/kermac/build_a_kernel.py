from cuda.core.experimental import Device, LaunchConfig, launch

from .module_cache.module_cache import *
from .common import *

from enum import Enum, auto
from typing import Optional, List

import torch
import numpy as np
from itertools import product

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
      majorness_A,
      majorness_B,
      align_A,
      align_B,
    ):
        kernel_name_str = 'cute_build_kernel'
        template_parameters = [
            f'InnerOperator::{self._inner_operator.name}',
            f'PowerType::{self._inner_power.name}',
            f'PowerType::{self._outer_power.name}',
            f'KernelType::{self._kernel_type.name}',
            f'Majorness::{majorness_A.name}',
            f'Majorness::{majorness_B.name}',
            f'Alignment::{align_A.name}',
            f'Alignment::{align_B.name}'
        ]
        function_name = f'{kernel_name_str}<{",".join(template_parameters)}>'
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

def pre_compile_descriptors(
    device,
    descriptors: List[Any],
    try_to_align=False,
    debug=False
):
    function_names = []

    if try_to_align:
        if debug:
            print('(Kermac Debug) Because `try_to_align` is set, generating full matrix of alignment conditions')

    valid_combinations = [
        (Majorness.ROW_MAJOR, Alignment.ALIGN_1),
        (Majorness.COL_MAJOR, Alignment.ALIGN_1),
    ]
    if try_to_align:
        valid_combinations.append((Majorness.COL_MAJOR, Alignment.ALIGN_4))
                                  
    for descriptor in descriptors:
        for (majorness_A, align_A), (majorness_B, align_B) in product(valid_combinations, valid_combinations):
            function_names.append(
                descriptor._render_function_name(
                    majorness_A=majorness_A,
                    majorness_B=majorness_B,
                    align_A=align_A, 
                    align_B=align_B
                )
            )

    module_cache = ModuleCache(debug)
    module_cache.compile_and_cache_functions(
        device=device,
        function_names=function_names,
        debug=debug
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
    if a.dim() not in (2, 3) or b.dim() not in (2, 3):
        raise ValueError("a and b must be 2-dimensional")
    if out is not None and out.dim() not in (2,3):
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

    tensor_stats_a = tensor_stats(a)
    tensor_stats_b = tensor_stats(b)

    # Get shapes
    L_a, M, K_a = tensor_stats_a.shape
    L_b, N, K_b = tensor_stats_b.shape

    # Check shape consistency
    if K_a != K_b:
        raise ValueError(f"K dimensions must match: got {K_a} for a and {K_b} for b")
    
    K = K_a

    # If one tensor has batch size 1 and the other doesn't we can broadcast
    L = max(L_a, L_b)
    if L_a != L and L_a != 1:
        raise ValueError(f"a must have batch dimension (L={L_a}), got {(L)}")
    
    if L_b != L and L_b != 1:
        raise ValueError(f"b must have batch dimension (L={L_b}), got {(L)}")
    
    if out is not None:
        tensor_stats_c = tensor_stats(out)
        L_c, M_c, N_c = tensor_stats_c.shape
        if (M_c, N_c) != (M,N):
            raise ValueError(f"out must have shape (M={M}, N={N}), got {(M_c, N_c)}")
        if L_c != L and L != 1:
            raise ValueError(f"out must have batch dimension (L={L}), got {(L_c)}")
        L = L_c
    else:
        out = torch.zeros((L, M, N), dtype=torch.float32, device=a.device)
        tensor_stats_c = tensor_stats(out)   

    result = out

    module_cache = ModuleCache(debug)
   
    pt_stream = torch.cuda.current_stream()
    pt_device = pt_stream.device
    device = Device(pt_device.index)
    device.set_current()
    stream = PyTorchStreamWrapper(pt_stream)

    if tensor_device != pt_device:
        raise ValueError("cuda stream must be on the same device as the tensors: got {pt_device}, expected {tensor_device}")
    
    if tensor_stats_c.majorness == Majorness.ROW_MAJOR:
        # Swap arguments if output tensor is row major
        # Kernel will dispatch to version with output as col major
        temp_M = M
        M = N
        N = temp_M

        temp_a = a
        a = b
        b = temp_a
        
        temp_tensor_stats_a = tensor_stats_a
        tensor_stats_a = tensor_stats_b
        tensor_stats_b = temp_tensor_stats_a

    
    align_4_A = Alignment.ALIGN_1 if not try_to_align else tensor_stats_a.alignment
    align_4_B = Alignment.ALIGN_1 if not try_to_align else tensor_stats_b.alignment

    bM = 128
    bN = 128

    block = 256

    function_name = kernel_descriptor._render_function_name(tensor_stats_a.majorness, tensor_stats_b.majorness, align_4_A, align_4_B)
    kernel = module_cache.get_function(device, function_name, debug=debug)

    if debug:
        print(f'(Kermac Debug) Launching kernel: {function_name}')

    num_blocks_M = ceil_div(M, bM)
    num_blocks_N = ceil_div(N, bN)
    num_batches = L

    grid = (num_blocks_M, num_blocks_N, num_batches)
    config = LaunchConfig(grid=grid, block=block)
    ld_a = np.uint64(tensor_stats_a.leading_dimension_stride)
    batch_stride_a = np.uint64(tensor_stats_a.batch_stride)

    ld_b = np.uint64(tensor_stats_b.leading_dimension_stride)
    batch_stride_b = np.uint64(tensor_stats_b.batch_stride)

    ld_c = np.uint64(tensor_stats_c.leading_dimension_stride)
    batch_stride_c = np.uint64(tensor_stats_c.batch_stride)

    kernel_args = (
        M, N, K, L,
        a.data_ptr(),       ld_a,    batch_stride_a,
        b.data_ptr(),       ld_b,    batch_stride_b,
        result.data_ptr(),  ld_c,    batch_stride_c,
        np.float32(inner_p),
        np.float32(outer_p),
        np.float32(bandwidth)
    )

    launch(stream, config, kernel, *kernel_args)

    return result