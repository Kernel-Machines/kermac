import torch
import kermac
import numpy as np

from cuda.core.experimental import Device, LaunchConfig, launch

from kermac import ceil_div

pt_device = torch.device('cuda')

pt_stream = torch.cuda.current_stream()
pt_device = pt_stream.device
device = Device(pt_device.index)
device.set_current()
stream = kermac.PyTorchStreamWrapper(pt_stream)

debug = True
module_cache = kermac.ModuleCache(debug)

def setting_to_function_name(setting):
    cta_m, cta_n, cta_o = setting
    return f'cute_scaled_gemm<{cta_m},{cta_n},{cta_o}>'

settings = [
    (128,128,1),
    (128,64,1),
    (64,64,1),
    (32,32,1),
    (128,128,2),
    (128,64,2),
    (64,64,2),
    (32,32,2),
    # (32,16,4),    # Wrong tiling
    (32,32,4),
    (64,64,4),
    (128,64,4),
    # (128,128,4),  # Spills
    # (32,16,8),    # Wrong tiling
    (32,32,8),
    (64,64,8),
    # (128,64,8),   # Spills
    # (128,128,8),  # Spills
    # (32,16,16),   # Wrong tiling
    (32,32,16),
    # (128,32,16),  # Spills
    # (128,16,8),   # Wrong tiling
    (128,32,8),
    # (128,16,16),
    # (64,64,16),   # Spills
    # (128,64,16),  # Spills
    # (128,128,16), # Spills
    # (64,64,16),   # Spills
    # (128,64,16),  # Spills
    # (128,16,32),  # Spills
    (32,32,32),
]

function_names = [setting_to_function_name(setting) for setting in settings]
module_cache.compile_and_cache_functions(
    device,
    function_names=function_names,
    debug=debug
)

M = 1002
N = 1001
O = 33
K = 1010

num_iters = 1
run_torch = True

def run_setting(
    setting, 
    a, b, c, d
):
    d.zero_()
    torch.cuda.synchronize()
    cta_m, cta_n, cta_o = setting
    function_name = f'cute_scaled_gemm<{cta_m},{cta_n},{cta_o}>'
    kernel = module_cache.get_function(device, function_name, debug=False)

    num_blocks_M = ceil_div(M, cta_m)
    num_blocks_N = ceil_div(N, cta_n)
    num_blocks_O = ceil_div(O, cta_o)
    # num_batches = L

    grid = (num_blocks_M, num_blocks_N, num_blocks_O)
    config = LaunchConfig(grid=grid, block=256)
    kernel_args = (
        M, N, O, K,
        a.data_ptr(), np.uint64(a.stride(0)),
        b.data_ptr(), np.uint64(b.stride(0)),
        c.data_ptr(), np.uint64(c.stride(0)),
        d.data_ptr(), np.uint64(d.stride(1)), np.uint64(d.stride(0))
    )
    launch(stream, config, kernel, *kernel_args)
    torch.cuda.synchronize()
    timer = kermac.CudaTimer()
    timer.start()
    for _ in range(num_iters):
        launch(stream, config, kernel, *kernel_args)

    ms = timer.stop()
    torch.cuda.synchronize()
    return d, ms

a = torch.randn(M,K,device=pt_device)
b = torch.randn(N,K,device=pt_device)
c = torch.randn(O,K,device=pt_device)
d = torch.zeros(O,N,M,device=pt_device)

if run_torch:
    torch_out = torch.einsum('mk,nk,ok->onm', a, b, c)
    torch.cuda.synchronize()

    timer = kermac.CudaTimer()
    timer.start()
    for _ in range(num_iters):
        torch_out = torch.einsum('mk,nk,ok->onm', a, b, c)
    ms = timer.stop()

    print(f'{ms:.3f}')


for _ in range(100):
    for setting in settings:
        d,ms = run_setting(setting, a, b, c, d)
        diff = 0.0
        if run_torch:
            diff = torch.max(d - torch_out).item()
            # if diff > 1e-3:
            print(f'\t{setting}\t{diff:.3e}\t{ms:.3f}')
            
