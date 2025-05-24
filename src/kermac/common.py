import torch

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

def is_tensor_16_byte_aligned(
    a : torch.Tensor
):
    if a.dtype != torch.float32:
        raise TypeError("a must have dtype torch.float32")
    alignment_requirement_bytes = 16
    alignment_requirement_elements = 4

    if not a.data_ptr() % alignment_requirement_bytes == 0:
        return False
    
    if not a.stride(0) % alignment_requirement_elements == 0:
        return False
    
    return True
