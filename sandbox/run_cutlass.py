import torch
import kermac
import numpy as np

from cuda.core.experimental import Device, LaunchConfig, launch

from kermac import ceil_div

def run_scaled_gemm():
    function_name = 'cute_scaled_gemm<float>'
    pt_device = torch.device('cuda')

    pt_stream = torch.cuda.current_stream()
    pt_device = pt_stream.device
    device = Device(pt_device.index)
    device.set_current()
    stream = kermac.PyTorchStreamWrapper(pt_stream)

    debug = True
    module_cache = kermac.ModuleCache(debug)
    kernel = module_cache.get_function(device, function_name, debug=debug)

    M = 128
    N = 128
    K = 16

    num_blocks_M = ceil_div(M, 128)
    num_blocks_N = ceil_div(N, 128)
    # num_batches = L

    grid = (num_blocks_M, num_blocks_N)
    config = LaunchConfig(grid=grid, block=256)

    a = torch.randn(M,K,device=pt_device)
    b = torch.randn(N,K,device=pt_device)
    d = torch.zeros(M,N,device=pt_device)
    # print(a.stride())
    # print(a)
    # print(a.stride(0))
    # a = torch.randn(M,K, device=pt_device)
    kernel_args = (
        M, N, K,
        a.data_ptr(), np.uint64(a.stride(0)),
        b.data_ptr(), np.uint64(b.stride(0)),
        d.data_ptr(), np.uint64(d.stride(0))
    )

    launch(stream, config, kernel, *kernel_args)
    torch.cuda.synchronize()

    print(d)
    print(b @ a.T)

run_scaled_gemm()
