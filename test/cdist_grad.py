import torch

import kermac.cdist_grad

size_M = 128 # M
size_D = 32  # N
size_C = 10  # O
size_N = 256 # K (contraction dimension)

tensor_A = torch.randn(size_N,size_M).cuda() # M-major # M-major 
tensor_B = torch.randn(size_D,size_N).cuda() # N-major # K-major
tensor_C = torch.randn(size_C,size_N).cuda() # N-major # K-major
tensor_D = torch.randn(size_D,size_M).cuda() # M-major # M-Major
# result tensor of mine
tensor_E = torch.randn(size_C,size_D,size_M).cuda() # M-major # M-major # (O,N,M)

# X-major is which dimension is stride=1
coefs =         tensor_C
kernel_matrix = tensor_A
x =             tensor_B
z =             tensor_D

# This is known to be correct, contracts in 'i'
# difference is that 'cmd' needs to be 'cdm' and 'z' and 'x' are transposed
torch_grad_og = torch.einsum('li,ij,jd->ljd', coefs, kernel_matrix, z.T) - torch.einsum('li,ij,id->ljd', coefs, kernel_matrix, x.T)

# This is my implementation layout (input), contracts in 'n', z and x are transposed
my_grad_input_only = torch.einsum('cn,nm,dm->cmd', coefs, kernel_matrix, z) - torch.einsum('cn,nm,dn->cmd',coefs, kernel_matrix, x)

assert torch.allclose(torch_grad_og, my_grad_input_only)

# This is my implementation layout (input/output), contracts in 'k' majorness is on the right
my_grad_input_output = torch.einsum('ok,km,nm->onm', tensor_C, tensor_A, tensor_D) - torch.einsum('ok,km,nk->onm', tensor_C, tensor_A, tensor_B)

# shuffle mine from 'onm' to 'omn' to match torch_grad_og
assert torch.allclose(torch_grad_og, my_grad_input_output.permute(0,2,1))

import kermac

kermac.cdist_grad(
    tensor_A,
    tensor_B,
    tensor_C,
    tensor_D,
    out = tensor_E,
    debug = True
)
# print(tensor_E)
# print(my_grad_input_output)
print(torch_grad_og.shape)
print(tensor_E.permute(0,2,1).shape)

print(torch.max(tensor_E - my_grad_input_output))
assert torch.allclose(torch_grad_og, tensor_E.permute(0,2,1), atol=1e-4)

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

