import torch
import os
import hashlib

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


def hash_text_files(directory):
    # Initialize SHA-256 hash object
    hasher = hashlib.sha256()
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        # Sort files for consistent hash across runs
        for file_name in sorted(files):
            # Check if file is a text file (e.g., ends with .txt)
            if file_name.endswith('.cuh'):
                file_path = os.path.join(root, file_name)
                try:
                    # Read file in binary mode
                    with open(file_path, 'rb') as f:
                        # Update hash with file contents
                        while chunk := f.read(8192):  # Read in 8KB chunks
                            hasher.update(chunk)
                except (IOError, PermissionError) as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Return the hexadecimal hash
    return hasher.hexdigest()