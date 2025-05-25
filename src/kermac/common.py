import torch
from enum import Enum, auto

class Majorness(Enum):
    COL_MAJOR = auto()
    ROW_MAJOR = auto()
# For templates to dictate whether
# an input tensor is aligned to 16 Bytes (4 float elements)
class Alignment(Enum):
    ALIGN_1 = auto()
    ALIGN_4 = auto()

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

def tensor_stats(
    tensor : torch.Tensor
):
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional")

    if tensor.dtype != torch.float32:
        raise TypeError("a must have dtype torch.float32")
    
    stride_row, stride_col = tensor.stride()
    num_rows, num_cols = tensor.size()

    if stride_col == 1 and stride_row >= num_cols:
        majorness = Majorness.ROW_MAJOR
    elif stride_row == 1 and stride_col >= num_rows:
        majorness = Majorness.COL_MAJOR
    else:
        raise ValueError(f"Tensor has non-standard memory layout: strides={tensor.stride()}, shape=({num_rows}, {num_cols})")
    
    if majorness == Majorness.ROW_MAJOR:
        # Don't allow specialization for row major for align 4
        return Majorness.ROW_MAJOR, Alignment.ALIGN_1

    alignment_requirement_bytes = 16
    alignment_requirement_elements = 4

    leading_dimension_index = 0 if majorness == Majorness.ROW_MAJOR else 1
    leading_dimension = tensor.stride(leading_dimension_index)

    is_starting_pointer_aligned = tensor.data_ptr() % alignment_requirement_bytes == 0
    is_leading_dimension_aligned =leading_dimension % alignment_requirement_elements == 0

    alignment = Alignment.ALIGN_4 if is_starting_pointer_aligned and is_leading_dimension_aligned else Alignment.ALIGN_1
    
    return majorness, alignment