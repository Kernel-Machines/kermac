import kermac
import kermac.linalg

import torch

N = 10
D = 6
C = 3
L = 4 # batches

device = torch.device('cuda')
data = torch.randn(L,N,D,device=device)
labels = torch.randn(L,C,N,device=device)

kernel_matrix = torch.zeros(L,N,N,device=device)

kernel_matrix = \
    kermac.run_kernel(
        kernel_descriptor=kermac.kernel_descriptor_laplace_l2,
        a=data,
        b=data,
        out=kernel_matrix,
        bandwidth=10.0
    )

print('***EIGH***')
# copy kernel matrix so eigh can write in-place
kernel_matrix_clobber = kernel_matrix.clone()

eigenvalues, eigenvectors, infos = \
    kermac.linalg.eigh(
        a = kernel_matrix_clobber,
        overwrite_a=True,
        check_errors=False
    )

print(f'eigenvalues:\n{eigenvalues}')
# eigenvectors was written to kernel_matrix_clobber
print(f'eigenvectors:\n{eigenvectors}')
# infos are not synchronized because check_errors is False. Should contain zero for each batch
# if routine was successful
print(f'infos:\n{infos}')

print('***CHOLESKY***')
# copy labels and kernel matrix so solve_cholesky can write in-place
labels_clobber = labels.clone()
kernel_matrix_clobber = kernel_matrix.clone()

sol, factor_infos, solve_infos = \
    kermac.linalg.solve_cholesky(
        a=kernel_matrix_clobber,
        b=labels_clobber,
        overwrite_a=True,
        overwrite_b=True,
        check_errors=False
    )

# sol was written to labels_clobber
# kernel_matrix_clobber was destroyed by cholesky factor
print(f'sol:\n{sol}')
# infos are not synchronized because check_errors is False. 
# factor_infos should contain zero for each batch if cholesky factor was successful
print(f'factor_infos:\n{factor_infos}')
# solve_infos should contain zero for each batch if cholesky solve was successful
print(f'solve_infos:\n{solve_infos}')

print('***LU***')
# copy labels and kernel matrix so solve_lu can write in-place
labels_clobber = labels.clone()
kernel_matrix_clobber = kernel_matrix.clone()

sol, factor_infos, solve_infos = \
    kermac.linalg.solve_lu(
        a=kernel_matrix_clobber,
        b=labels_clobber,
        overwrite_a=True,
        overwrite_b=True,
        check_errors=False
    )

# sol was written to labels_clobber
# kernel_matrix_clobber was destroyed by lu factor
print(f'sol:\n{sol}')
# infos are not synchronized because check_errors is False. 
# factor_infos should contain zero for each batch if lu factor was successful
print(f'factor_infos:\n{factor_infos}')
# solve_infos should contain zero for each batch if lu solve was successful
print(f'solve_infos:\n{solve_infos}')
