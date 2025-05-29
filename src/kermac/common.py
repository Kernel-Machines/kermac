import torch
from enum import Enum, auto
from typing import Tuple, Union

class Majorness(Enum):
    COL_MAJOR = auto()
    ROW_MAJOR = auto()
# For templates to dictate whether
# an input tensor is aligned to 16 Bytes (4 float elements)
class Alignment(Enum):
    ALIGN_1 = auto()
    ALIGN_4 = auto()

class FillMode(Enum):
    UPPER = auto()
    LOWER = auto()

class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python

class CudaTimer:
    def __init__(self):
        """Initialize the timer, creating start and end CUDA events and recording the start time."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start(self):
        """Reset the timer by recording a new start time."""
        self.start_event.record()

    def stop(self):
        """Stop the timer, record the end time, and return the elapsed time in milliseconds."""
        self.end_event.record()
        self.end_event.synchronize()  # Ensure events are complete
        return self.start_event.elapsed_time(self.end_event)

def ceil_div(x, d):
    return int((x + d - 1) // d)

def merge_batch_size(
    name : str,
    L, # previous batch size
    a : Union[float, torch.tensor],
    expected_dims : int,
    can_be_none: bool = False,
):
    if a is None:
        if not can_be_none:
            raise ValueError(f'{name} is none, but can_be_none is False')
        return L
    elif isinstance(a, float):
        if expected_dims != 0:
            raise ValueError(f'{name} is a float but expected_dims is {expected_dims}')
        return L
    elif isinstance(a, torch.Tensor):
        if a.dim() == expected_dims:
            return L
        if a.dim() == expected_dims + 1:
            this_L = a.size(0)
            if this_L != 1 and L != 1 and this_L != L:
                raise ValueError(f'{name} batch size of {this_L} does not match with another non-one batch size, {L}')
            return max(L, this_L)
        raise ValueError(f'{name} does have the right number of dimensions, expected_dims is {expected_dims} or {expected_dims+1}, got {a.dim()}')

class TensorStats:
    def __init__(self, tensor: torch.Tensor):
        if tensor.dim() not in (2, 3):
            raise ValueError("Input tensor must be 2 or 3-dimensional")
        
        if tensor.dtype != torch.float32:
            raise TypeError("Input tensor must have dtype torch.float32")
        
        dim = tensor.dim()
        shape = tensor.size()
        strides = tensor.stride()
        
        # Calculate batch information
        self._num_batches = 1 if dim == 2 else shape[0]
        self._batch_stride = 0 if dim == 2 or self._num_batches == 1 else strides[0]

        # Get row/col dimensions (last two dimensions for rank 3)
        num_rows = shape[-2]
        num_cols = shape[-1]
        stride_row = strides[-2]
        stride_col = strides[-1]

        self._shape = (self._num_batches, num_rows, num_cols)
        
        # Determine majorness
        if stride_col == 1 and stride_row >= num_cols:
            self._majorness = Majorness.ROW_MAJOR
        elif stride_row == 1 and stride_col >= num_rows:
            self._majorness = Majorness.COL_MAJOR
        else:
            raise ValueError(f"Tensor has non-standard memory layout: strides={strides}, shape={shape}")
        
        # Calculate alignment
        self._alignment_requirement_bytes = 16
        self._alignment_requirement_elements = 4
        
        self._leading_dimension_stride = max(stride_row, stride_col)
        
        is_starting_pointer_aligned = tensor.data_ptr() % self._alignment_requirement_bytes == 0
        is_leading_dimension_aligned = self._leading_dimension_stride % self._alignment_requirement_elements == 0
        
        if self._majorness == Majorness.ROW_MAJOR:
            self._alignment = Alignment.ALIGN_1
        else:
            self._alignment = Alignment.ALIGN_4 if is_starting_pointer_aligned and is_leading_dimension_aligned else Alignment.ALIGN_1

    # Accessors
    @property
    def shape(self) -> Tuple[int,int,int]:
        return self._shape

    @property
    def majorness(self) -> Majorness:
        return self._majorness
    
    @property
    def alignment(self) -> Alignment:
        return self._alignment
    
    @property
    def num_batches(self) -> int:
        return self._num_batches
    
    @property
    def batch_stride(self) -> int:
        return self._batch_stride
    
    @property
    def leading_dimension_stride(self) -> int:
        return self._leading_dimension_stride

def tensor_stats(tensor: torch.Tensor) -> TensorStats:
    return TensorStats(tensor)