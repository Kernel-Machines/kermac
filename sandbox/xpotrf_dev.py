import kermac
import kermac.linalg

import torch

N = 7
D = 6
C = 2
L = 1

device = torch.device('cuda')
data = torch.randn(L,N,D,device=device)
labels = torch.randn(L,C,N,device=device)
kernel_matrix = torch.randn(L,N,N,device=device)
# kernel_matrix = torch.zeros(L,N,N,device=device)

kermac.run_kernel(
    kermac.kernel_descriptor_laplace_l2,
    data, data,
    out=kernel_matrix,
    bandwidth=10.0,
    try_to_align=True,
    debug=True
)

kernel_matrix_saved = kernel_matrix.clone()
labels_saved = labels.clone()

sol, factor_info, solve_info = \
    kermac.linalg.solve_cholesky(
        a=kernel_matrix, 
        b=labels,
        overwrite_a=True,
        overwrite_b=True,
        check_errors=True
    )

print(f'factor_info: {factor_info}')
print(f'solve_info: {solve_info}')

print(sol @ kernel_matrix_saved)
print(labels_saved)

torch_sol = torch.linalg.solve(
    kernel_matrix_saved,
    labels_saved.permute(0,2,1)
)

print(torch_sol.permute(0,2,1) @ kernel_matrix_saved)
print(labels_saved)

