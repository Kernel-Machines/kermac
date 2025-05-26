import torch

import kermac

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
    
timer = CudaTimer()

device = torch.device('cuda')
debug = True
p = 2.0

if debug: 
    print('\n(Kermac Debug) Warmup kermac.cdist_grad')
kermac.cdist_grad(
    torch.randn(10,100,device=device),
    torch.randn(32,10,device=device),
    torch.randn(16,10,device=device),
    torch.randn(32,100,device=device),
    p = p,
    debug = debug
)

size_M = 10000 # M
size_D = 768  # N
size_C = 16  # O
size_N = 10000 # K (contraction dimension)



tensor_A = torch.randn(size_N,size_M,device=device) # M-major # M-major 
tensor_B = torch.randn(size_D,size_N,device=device) # N-major # K-major
tensor_C = torch.randn(size_C,size_N,device=device) # N-major # K-major
tensor_D = torch.randn(size_D,size_M,device=device) # M-major # M-Major
# result tensor of mine
tensor_E = torch.zeros(size_C,size_D,size_M,device=device) # M-major # M-major # (O,N,M)

# X-major is which dimension is stride=1
coefs =         tensor_C
kernel_matrix = tensor_A
x =             tensor_B
z =             tensor_D

if True:
    # This is known to be correct, contracts in 'i'
    # difference is that 'cmd' needs to be 'cdm' and 'z' and 'x' are transposed
    torch_grad_og = torch.einsum('li,ij,jd->ljd', coefs, kernel_matrix, z.T) - torch.einsum('li,ij,id->ljd', coefs, kernel_matrix, x.T)

    # This is my implementation layout (input), contracts in 'n', z and x are transposed
    my_grad_input_only = torch.einsum('cn,nm,dm->cmd', coefs, kernel_matrix, z) - torch.einsum('cn,nm,dn->cmd',coefs, kernel_matrix, x)

    assert torch.allclose(torch_grad_og, my_grad_input_only)

    timer.start()
    # This is my implementation layout (input/output), contracts in 'k' majorness is on the right
    my_grad_input_output = torch.einsum('ok,km,nm->onm', tensor_C, tensor_A, tensor_D) - torch.einsum('ok,km,nk->onm', tensor_C, tensor_A, tensor_B)
    print(f"\ttorch.einsum \t{timer.stop():.3f} ms")

    # shuffle mine from 'onm' to 'omn' to match torch_grad_og
    assert torch.allclose(torch_grad_og, my_grad_input_output.permute(0,2,1))


timer.start()
kermac.cdist_grad(
    tensor_A,
    tensor_B,
    tensor_C,
    tensor_D,
    out = tensor_E,
    p = p,
    debug = debug
)
print(f"\tkermac.cdist_grad \t{timer.stop():.3f} ms")
if True:
    # print(tensor_E)
    # print(my_grad_input_output)
    print(torch_grad_og.shape)
    print(tensor_E.permute(0,2,1).shape)

    print(f'Maximum per element difference {torch.max(tensor_E - my_grad_input_output).item():.3e}')
# assert torch.allclose(torch_grad_og, tensor_E.permute(0,2,1), atol=1e-3)

# if True:
#     # want
#     M = 128 # M -> M
#     N = 32  # N -> D
#     O = 10  # O -> C
#     K = 256 # K -> N

#     # Contraction happens in K dimension
#     A = torch.randn(K,M).cuda() # M-major # kernel_matrix
#     B = torch.randn(N,K).cuda() # K-major # x
#     C = torch.randn(O,K).cuda() # K-major # coefs
#     D = torch.randn(N,M).cuda() # M-major # z

#     # This is my implementation layout, contracts in 'k' majorness is on the right
#     torch_grad = torch.einsum('ok,km,nm->onm', C, A, D) - torch.einsum('ok,km,nk->cmd', C, A, B)

