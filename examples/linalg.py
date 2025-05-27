import kermac
import kermac.linalg
import torch
import argparse

def parse_args():
    """Parse command-line arguments for matrix dimensions and timing parameters."""
    parser = argparse.ArgumentParser(description="Run kernel and linear algebra operations with timings")
    parser.add_argument('-n','--N', type=int, default=5000, help='Number of rows of data (default: 5000)')
    parser.add_argument('-d','--D', type=int, default=32, help='Number of columns of data (default: 1000)')
    parser.add_argument('-c','--C', type=int, default=16, help='Number of labels')
    parser.add_argument('-l','--L', type=int, default=10, help='Number of batches')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup rounds (default: 2)')
    parser.add_argument('--iters', type=int, default=10, help='Number of iteration rounds (default: 10)')
    args = parser.parse_args()
    return args

def main(N, D, C, L, warmup_rounds, iterations):
    device = torch.device('cuda')
    timer = kermac.CudaTimer()

    # Initialize data
    data = torch.randn(L, N, D, device=device)
    labels = torch.randn(L, C, N, device=device)
    kernel_matrix = torch.zeros(L, N, N, device=device)

    # Warmup for run_kernel
    print(f'Warmup {warmup_rounds} iterations of kermac.run_kernel')
    for _ in range(warmup_rounds):
        kermac.run_kernel(
            kernel_descriptor=kermac.kernel_descriptor_laplace_l2,
            a=data,
            b=data,
            out=kernel_matrix,
            bandwidth=10.0
        )

    # Timed run for run_kernel
    print('Running kermac.run_kernel')
    timer.start()
    for _ in range(iterations):
        kernel_matrix = kermac.run_kernel(
            kernel_descriptor=kermac.kernel_descriptor_laplace_l2,
            a=data,
            b=data,
            out=kernel_matrix,
            bandwidth=10.0
        )
    print(kernel_matrix)
    print(f'Running {iterations} iterations of kermac.run_kernel with size ({L},{N},{D})')
    print(f"\tkermac.run_kernel \t{timer.stop() / iterations:.3f} ms / iteration")

    print('***EIGH***')
    # Warmup for eigh
    for _ in range(warmup_rounds):
        kernel_matrix_clobber = kernel_matrix.clone()
        kermac.linalg.eigh(
            a=kernel_matrix_clobber,
            overwrite_a=True,
            check_errors=False
        )

    # Timed run for eigh
    timer.start()
    for _ in range(iterations):
        kernel_matrix_clobber = kernel_matrix.clone()
        eigenvalues, eigenvectors, infos = kermac.linalg.eigh(
            a=kernel_matrix_clobber,
            overwrite_a=True,
            check_errors=False
        )
    print(f'Running {iterations} iterations of kermac.linalg.eigh with size ({L},{N},{N})')
    print(f"\tkermac.linalg.eigh \t{timer.stop() / iterations:.3f} ms / iteration")
    print(f'eigenvalues:\n{eigenvalues}')
    print(f'eigenvectors:\n{eigenvectors}')
    print(f'infos:\n{infos}')
    print('***CHOLESKY***')
    # Warmup for solve_cholesky
    for _ in range(warmup_rounds):
        labels_clobber = labels.clone()
        kernel_matrix_clobber = kernel_matrix.clone()
        kermac.linalg.solve_cholesky(
            a=kernel_matrix_clobber,
            b=labels_clobber,
            overwrite_a=True,
            overwrite_b=True,
            check_errors=False
        )
    torch.cuda.synchronize()

    # Timed run for solve_cholesky
    timer.start()
    for _ in range(iterations):
        labels_clobber = labels.clone()
        kernel_matrix_clobber = kernel_matrix.clone()
        sol, factor_infos, solve_infos = kermac.linalg.solve_cholesky(
            a=kernel_matrix_clobber,
            b=labels_clobber,
            overwrite_a=True,
            overwrite_b=True,
            check_errors=False
        )
    torch.cuda.synchronize()
    print(f'Running {iterations} iterations of kermac.linalg.solve_cholesky with size ({L},{N},{N}) and labels ({L},{C},{N})')
    print(f"\tkermac.linalg.solve_cholesky \t{timer.stop() / iterations:.3f} ms / iteration")
    print(f'sol:\n{sol}')
    print(f'factor_infos:\n{factor_infos}')
    print(f'solve_infos:\n{solve_infos}')

    print('***LU***')
    # Warmup for solve_lu
    labels_clobber = labels.clone()
    kernel_matrix_clobber = kernel_matrix.clone()
    for _ in range(warmup_rounds):
        kermac.linalg.solve_lu(
            a=kernel_matrix_clobber,
            b=labels_clobber,
            overwrite_a=True,
            overwrite_b=True,
            check_errors=False
        )
    torch.cuda.synchronize()

    # Timed run for solve_lu
    timer.start()
    for _ in range(iterations):
        labels_clobber = labels.clone()
        kernel_matrix_clobber = kernel_matrix.clone()
        sol, factor_infos, solve_infos = kermac.linalg.solve_lu(
            a=kernel_matrix_clobber,
            b=labels_clobber,
            overwrite_a=True,
            overwrite_b=True,
            check_errors=False
        )
    torch.cuda.synchronize()
    print(f'Running {iterations} iterations of kermac.linalg.solve_lu with size ({L},{N},{N}) and labels ({L},{C},{N})')
    print(f"\tkermac.linalg.solve_lu \t{timer.stop() / iterations:.3f} ms / iteration")
    print(f'sol:\n{sol}')
    print(f'factor_infos:\n{factor_infos}')
    print(f'solve_infos:\n{solve_infos}')

if __name__ == '__main__':
    args = parse_args()
    main(args.N, args.D, args.C, args.L, args.warmup, args.iters)
