from cuda.core.experimental import Device
import torch

import os
import hashlib

def get_compute_capability(device) -> str:
    if isinstance(device, torch.device):
        pt_device_id = device.index
        device = Device(pt_device_id)
    
    arch = "".join(f"{i}" for i in device.compute_capability)
    return arch

def hash_cuda_include_files(directory):
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
