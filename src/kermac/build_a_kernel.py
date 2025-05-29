from cuda.core.experimental import Device, LaunchConfig, launch

from .module_cache.module_cache import *
from .common import *

from enum import Enum, auto
from typing import Optional, List, Union

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
    p :         Optional[Union[float, torch.Tensor]] = None,
    inner_p :   Optional[Union[float, torch.Tensor]] = None,
    outer_p :   Optional[Union[float, torch.Tensor]] = None,
    bandwidth : Optional[Union[float, torch.Tensor]] = None,
    epsilon:    Optional[float] = None,
    regularization: Optional[Union[float, torch.Tensor]] = None,
    regularization_offset_x : int = 0,
    regularization_offset_y : int = 0,
    try_to_align : bool = False,
    debug = False
):
    if kernel_descriptor._inner_power is PowerType.POW:
        if p is None and inner_p is None:
            raise ValueError("'inner_power' 'PowerType' is 'Pow' but 'p' and 'inner_p' is not set")
    else:
        if p is not None or inner_p is not None:
            raise ValueError("'inner_power' 'PowerType' is not 'Pow' but 'p' or 'inner_p' is set")
        
    if kernel_descriptor._outer_power is PowerType.POW:
        if p is None and outer_p is None:
            raise ValueError("'outer_power' 'PowerType' is 'Pow' but 'p' and 'outer_p' is not set")
    else:
        if p is not None or outer_p is not None:
            raise ValueError("'outer_power' 'PowerType' is not 'Pow' but 'p' or 'outer_p' is set")
        
    if kernel_descriptor._kernel_type is KernelType.NONE:
        if bandwidth is not None or epsilon is not None or regularization is not None:
            raise ValueError("'KernelType' is 'None' but 'bandwidth' or 'epsilon' or 'regularization' is set")
    else:
        if bandwidth is None:
            raise ValueError("'KernelType' is not 'None' but 'bandwidth' is not set")
        
    if p is not None and inner_p is not None:
        raise ValueError("'p' is not 'None' but 'inner_p' is also not 'None")
    
    if p is not None and outer_p is not None:
        raise ValueError("'p' is not 'None' but 'outer_p' is also not 'None")

    # batch sizes can always be 1 or L, if we merge two L's that are not equal and both 
    # are not 1 then we have a problem
    L = 1
    L = merge_batch_size('p', L, p, expected_dims=0, can_be_none=True)
    L = merge_batch_size('inner_p', L, inner_p, expected_dims=0, can_be_none=True)
    L = merge_batch_size('outer_p', L, outer_p, expected_dims=0, can_be_none=True)
    L = merge_batch_size('bandwidth', L, bandwidth, expected_dims=0, can_be_none=True)
    L = merge_batch_size('regularization', L, regularization, expected_dims=0, can_be_none=True)
    L = merge_batch_size('a', L, a, expected_dims=2, can_be_none=False)
    L = merge_batch_size('b', L, b, expected_dims=2, can_be_none=False)
    L = merge_batch_size('out', L, out, expected_dims=2, can_be_none=True)
        
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
    _, M, K_a = tensor_stats_a.shape
    _, N, K_b = tensor_stats_b.shape

    # Check shape consistency
    if K_a != K_b:
        raise ValueError(f"K dimensions must match: got {K_a} for a and {K_b} for b")
    
    K = K_a
    
    if out is not None:
        L_c = 1 if out.dim() == 2 else out.size(0)
        tensor_stats_c = tensor_stats(out)
        _, M_c, N_c = tensor_stats_c.shape
        if (M_c, N_c) != (M,N):
            raise ValueError(f"out must have shape (M={M}, N={N}), got {(M_c, N_c)}")
        if L_c != L and L != 1:
            raise ValueError(f"out must have batch dimension (L={L}), got {(L_c)}")
    else:
        out = torch.zeros((L, M, N), dtype=torch.float32, device=tensor_device)
        tensor_stats_c = tensor_stats(out)

    # L is decided now
    if inner_p is not None:
        if isinstance(inner_p, float):
            inner_p = torch.tensor(inner_p, dtype=torch.float32, device=tensor_device)
        else: # must be torch.tensor
            if inner_p.dtype != torch.float32:
                raise TypeError("`inner_p` tensor must have dtype torch.float32")
            if not inner_p.is_cuda or inner_p.device != tensor_device:
                raise ValueError("`inner_p` tensor must be on the same CUDA device as inputs")
            
    if outer_p is not None:
        if isinstance(outer_p, float):
            outer_p = torch.tensor(outer_p, dtype=torch.float32, device=tensor_device)
        else: # must be torch.tensor
            if outer_p.dtype != torch.float32:
                raise TypeError("`outer_p` tensor must have dtype torch.float32")
            if not outer_p.is_cuda or outer_p.device != tensor_device:
                raise ValueError("`outer_p` tensor must be on the same CUDA device as inputs")

    if p is not None:
        if isinstance(p, float):
            inner_p = torch.tensor(p, dtype=torch.float32, device=tensor_device)
        else: # must be torch.tensor
            if p.dtype != torch.float32:
                raise TypeError("`p` tensor must have dtype torch.float32")
            if not p.is_cuda or p.device != tensor_device:
                raise ValueError("`p` tensor must be on the same CUDA device as inputs")
            inner_p = p.contiguous()
        outer_p = 1.0 / inner_p

    if bandwidth is None:
        bandwidth = 0.0
    if isinstance(bandwidth, float):
        bandwidth = torch.tensor(bandwidth, dtype=torch.float32, device=tensor_device)
    else:
        if bandwidth.dtype != torch.float32:
            raise TypeError("`bandwidth` tensor must have dtype torch.float32")
        if not bandwidth.is_cuda or bandwidth.device != tensor_device:
            raise ValueError("`bandwidth` tensor must be on the same CUDA device as inputs")
    
    if regularization is None:
        regularization = 0.0
    if isinstance(regularization, float):
        regularization = torch.tensor(regularization, dtype=torch.float32, device=tensor_device)
    else:
        if regularization.dtype != torch.float32:
            raise TypeError("`regularization` tensor must have dtype torch.float32")
        if not regularization.is_cuda or regularization.device != tensor_device:
            raise ValueError("`regularization` tensor must be on the same CUDA device as inputs")
            
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

    function_name = kernel_descriptor._render_function_name(tensor_stats_a.majorness, tensor_stats_b.majorness, align_4_A, align_4_B)
    kernel = module_cache.get_function(device, function_name, debug=debug)

    if debug:
        print(f'(Kermac Debug) Launching kernel: {function_name}')

    num_blocks_M = ceil_div(M, 128)
    num_blocks_N = ceil_div(N, 128)
    num_batches = L

    grid = (num_blocks_M, num_blocks_N, num_batches)
    config = LaunchConfig(grid=grid, block=256)

    ld_a = np.uint64(tensor_stats_a.leading_dimension_stride)
    batch_stride_a = np.uint64(tensor_stats_a.batch_stride)

    ld_b = np.uint64(tensor_stats_b.leading_dimension_stride)
    batch_stride_b = np.uint64(tensor_stats_b.batch_stride)

    ld_c = np.uint64(tensor_stats_c.leading_dimension_stride)
    batch_stride_c = np.uint64(tensor_stats_c.batch_stride)

    if inner_p is None:
        inner_p = torch.tensor(0.0, dtype=torch.float32, device=tensor_device)

    if outer_p is None:
        outer_p = torch.tensor(0.0, dtype=torch.float32, device=tensor_device)

    batch_stride_inner_p = np.uint64(0 if inner_p.dim() == 0 else 1)
    batch_stride_outer_p = np.uint64(0 if outer_p.dim() == 0 else 1)
    batch_stride_bandwidth = np.uint64(0 if bandwidth.dim() == 0 else 1)
    batch_stride_regularization = np.uint64(0 if regularization.dim() == 0 else 1)

    if epsilon is None:
        epsilon = 1e-5

    kernel_args = (
        M, N, K, L,
        a.data_ptr(),       ld_a,   batch_stride_a,
        b.data_ptr(),       ld_b,   batch_stride_b,
        out.data_ptr(),     ld_c,   batch_stride_c,
        inner_p.data_ptr(),         batch_stride_inner_p,
        outer_p.data_ptr(),         batch_stride_outer_p,
        bandwidth.data_ptr(),       batch_stride_bandwidth,
        regularization.data_ptr(),  batch_stride_regularization,
        regularization_offset_x, 
        regularization_offset_y,
        np.float32(epsilon)
    )

    launch(stream, config, kernel, *kernel_args)

    return out