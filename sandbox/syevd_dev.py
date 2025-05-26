import kermac
import kermac.linalg

import torch

N = 7
D = 6
L = 2

device = torch.device('cuda')
data = torch.randn(L,N,D,device=device)
kernel_matrix = torch.randn(L,N,N,device=device)

kermac.run_kernel(
    kermac.kernel_descriptor_laplace_l2,
    data, data,
    out=kernel_matrix,
    bandwidth=10.0,
    try_to_align=True,
    debug=True
)

kernel_matrix_saved = kernel_matrix.clone()

eigenvalues, eigenvectors, info = \
    kermac.linalg.eigh(
        a=kernel_matrix, 
        overwrite_a=True,
        check_errors=True
    )

print(f'eigenvalues:\n{eigenvalues}')
print(f'eigenvectors:\n{eigenvectors}')
print(f'info: {info}')

Lambda = torch.diag_embed(eigenvalues)
print(Lambda)

# Reconstruct the original matrix: A = V @ Lambda @ V^T
V = eigenvectors
A_reconstructed = V @ Lambda @ V.permute(0,2,1)

torch_eigenvalues, torch_eigenvectors = torch.linalg.eigh(kernel_matrix_saved)

print(f'torch_eigenvalues:\n{torch_eigenvalues}')
print(f'torch_eigenvectors:\n{torch_eigenvectors}')

# Verify the reconstruction
print("Original matrix A:")
print(kernel_matrix_saved)
print("Reconstructed matrix A:")
print(A_reconstructed)
print("Difference (should be close to zero):")
print(kernel_matrix_saved - A_reconstructed)