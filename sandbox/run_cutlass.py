import torch
import kermac
import numpy as np

from cuda.core.experimental import Device, LaunchConfig, launch

def run_cute():
    function_name = 'cutlass_little_test'
    pt_device = torch.device('cuda')

    pt_stream = torch.cuda.current_stream()
    pt_device = pt_stream.device
    device = Device(pt_device.index)
    device.set_current()
    stream = kermac.PyTorchStreamWrapper(pt_stream)

    debug = True
    module_cache = kermac.ModuleCache(debug)
    kernel = module_cache.get_function(device, function_name, debug=debug)

    grid = (1)
    config = LaunchConfig(grid=grid, block=256)

    M = int(32)
    K = int(16)
      # Define device (use 'cuda' if GPU is available)
    a = torch.arange((M * K), dtype=torch.float32, device=pt_device)
    a = a.reshape((M, K))  # Reshape to (100, 16)
    # print(a.stride())
    # print(a)
    # print(a.stride(0))
    # a = torch.randn(M,K, device=pt_device)
    ldA = np.uint64(a.stride(0))
    kernel_args = (
        M, K,
        a.data_ptr(), 
        ldA
    )

    launch(stream, config, kernel, *kernel_args)
    torch.cuda.synchronize()

run_cute()