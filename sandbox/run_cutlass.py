import torch
import kermac
import numpy as np

from cuda.core.experimental import Device, LaunchConfig, launch

from kermac import ceil_div

def run_scaled_gemm():
    
    BOsize = 2

    function_name = f'cute_scaled_gemm<{BOsize}, float>'
    pt_device = torch.device('cuda')

    pt_stream = torch.cuda.current_stream()
    pt_device = pt_stream.device
    device = Device(pt_device.index)
    device.set_current()
    stream = kermac.PyTorchStreamWrapper(pt_stream)

    debug = True
    module_cache = kermac.ModuleCache(debug)
    kernel = module_cache.get_function(device, function_name, debug=debug)
    # module_cache.compile_and_cache_functions(
    #     device,

    # )

    M = 1001
    N = 1001
    O = 33
    K = 17

    num_blocks_M = ceil_div(M, 32)
    num_blocks_N = ceil_div(N, 32)
    num_blocks_O = ceil_div(O, BOsize)
    # num_batches = L

    grid = (num_blocks_M, num_blocks_N, num_blocks_O)
    config = LaunchConfig(grid=grid, block=256)

    a = torch.randn(M,K,device=pt_device)
    b = torch.randn(N,K,device=pt_device)
    c = torch.randn(O,K,device=pt_device)
    d = torch.zeros(O,N,M,device=pt_device)
    # print(a.stride())
    # print(a)
    # print(a.stride(0))
    # a = torch.randn(M,K, device=pt_device)
    kernel_args = (
        M, N, O, K,
        a.data_ptr(), np.uint64(a.stride(0)),
        b.data_ptr(), np.uint64(b.stride(0)),
        c.data_ptr(), np.uint64(c.stride(0)),
        d.data_ptr(), np.uint64(d.stride(1)), np.uint64(d.stride(0))
    )

    launch(stream, config, kernel, *kernel_args)
    torch.cuda.synchronize()

    kermac_out = d
    torch_out = torch.einsum('mk,nk,ok->onm', a, b, c)

    print('')
    print(f'\t\t{torch.max(kermac_out - torch_out).item():.3e}')
    print('')
    # print(torch_out)

run_scaled_gemm()
