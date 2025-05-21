import torch

def ceil_div(x, d):
    return int((x + d - 1) // d)

class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python

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
