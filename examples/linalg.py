import kermac
import kermac.linalg
import torch
import argparse

def parse_args():
    """Parse command-line arguments for matrix dimensions and timing parameters."""
    parser = argparse.ArgumentParser(description="Run kernel and linear algebra operations with timings")
    parser.add_argument('-m','--M', type=int, default=5000, help='Number of rows of data (default: 5000)')
    parser.add_argument('-k','--K', type=int, default=32, help='Number of columns of data (default: 1000)')
    parser.add_argument('-c','--C', type=int, default=16, help='Number of labels')
    parser.add_argument('-l','--L', type=int, default=10, help='Number of batches')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Enable debug output (default: False)')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup rounds (default: 2)')
    parser.add_argument('--iters', type=int, default=10, help='Number of iteration rounds (default: 10)')
    parser.add_argument('--skip_eigh', default=False, action='store_true', help='Skip running kermac.linalg.eigh.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    M, K, C, L = args.M, args.K, args.C, args.L
    warmup_rounds = args.warmup
    iterations = args.iters
    debug = args.debug
    skip_eigh = args.skip_eigh

    device = torch.device('cuda')
    timer = kermac.CudaTimer()

    # Initialize data
    data = torch.randn(L, M, K, device=device)
    labels = torch.randn(L, C, M, device=device)
    kernel_matrix = torch.zeros(L, M, M, device=device)

    def infos_all_zero(infos):
        return 'OK' if torch.all(infos == 0) else 'FAIL'

    # Warmup for run_kernel
    print(f'Warmup {warmup_rounds} iterations of kermac.run_kernel')
    for _ in range(warmup_rounds):
        kermac.run_kernel(
            kernel_descriptor=kermac.kernel_descriptor_laplace_l2,
            a=data,
            b=data,
            out=kernel_matrix,
            bandwidth=10.0,
            debug=debug
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
            bandwidth=10.0,
            debug=debug
        )
    print(f'Running {iterations} iterations of kermac.run_kernel with size ({L},{M},{K})')
    print(f"\tkermac.run_kernel \t{timer.stop() / iterations:.3f} ms / iteration")

    if not skip_eigh:
        print('***EIGH***')
        # Warmup for eigh
        print(f'Warmup {warmup_rounds} iterations of kermac.linalg.eigh with size ({L},{M},{M})')
        for _ in range(warmup_rounds):
            kernel_matrix_clobber = kernel_matrix.clone()
            kermac.linalg.eigh(
                a=kernel_matrix_clobber,
                overwrite_a=True,
                check_errors=False
            )
        torch.cuda.synchronize()
        # Timed run for eigh
        print(f'Running {iterations} iterations of kermac.linalg.eigh with size ({L},{M},{M})')
        timer.start()
        for _ in range(iterations):
            kernel_matrix_clobber = kernel_matrix.clone()
            eigenvalues, eigenvectors, infos = kermac.linalg.eigh(
                a=kernel_matrix_clobber,
                overwrite_a=True,
                check_errors=False
            )
        print(f"\tkermac.linalg.eigh \t{timer.stop() / iterations:.3f} ms / iteration")
        # print(f'eigenvalues:\n{eigenvalues}')
        # print(f'eigenvectors:\n{eigenvectors}')
        print(f'\tkermac.linalg.eigh infos: {infos_all_zero(infos)}')
    print('***CHOLESKY***')
    # Warmup for solve_cholesky
    print(f'Warmup {warmup_rounds} iterations of kermac.linalg.solve_cholesky with size ({L},{M},{M}) and labels ({L},{C},{M})')
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

    print(f'Running {iterations} iterations of kermac.linalg.solve_cholesky with size ({L},{M},{M}) and labels ({L},{C},{M})')
    timer.start()
    for _ in range(iterations):
        labels_clobber = labels.clone()
        kernel_matrix_clobber = kernel_matrix.clone()
        sol, factor_infos, solve_infos = kermac.linalg.solve_cholesky(
            a=kernel_matrix_clobber,
            b=labels_clobber,
            overwrite_a=True,
            overwrite_b=True,
            check_errors=False,
            debug=debug
        )
    torch.cuda.synchronize()
   
    print(f"\tkermac.linalg.solve_cholesky \t{timer.stop() / iterations:.3f} ms / iteration")
    # print(f'sol:\n{sol}')
    

    print(f'\tkermac.linalg.solve_cholesky factor_infos: {infos_all_zero(factor_infos)}')
    print(f'\tkermac.linalg.solve_cholesky solve_infos: {infos_all_zero(solve_infos)}')

    print('***LU***')
    # Warmup for solve_lu
    print(f'Warmup {warmup_rounds} iterations of kermac.linalg.solve_lu with size ({L},{M},{M}) and labels ({L},{C},{M})')
    for _ in range(warmup_rounds):
        labels_clobber = labels.clone()
        kernel_matrix_clobber = kernel_matrix.clone()
        kermac.linalg.solve_lu(
            a=kernel_matrix_clobber,
            b=labels_clobber,
            overwrite_a=True,
            overwrite_b=True,
            check_errors=False,
            debug=debug
        )
    torch.cuda.synchronize()

    # Timed run for solve_lu
    print(f'Running {iterations} iterations of kermac.linalg.solve_lu with size ({L},{M},{M}) and labels ({L},{C},{M})')
    timer.start()
    for _ in range(iterations):
        labels_clobber = labels.clone()
        kernel_matrix_clobber = kernel_matrix.clone()
        sol, factor_infos, solve_infos = kermac.linalg.solve_lu(
            a=kernel_matrix_clobber,
            b=labels_clobber,
            overwrite_a=True,
            overwrite_b=True,
            check_errors=False,
            debug=debug
        )
    torch.cuda.synchronize()
    
    print(f"\tkermac.linalg.solve_lu \t{timer.stop() / iterations:.3f} ms / iteration")
    # print(f'sol:\n{sol}')
    print(f'\tkermac.linalg.solve_lu factor_infos: {infos_all_zero(factor_infos)}')
    print(f'\tkermac.linalg.solve_lu solve_infos: {infos_all_zero(solve_infos)}')

if __name__ == '__main__':
    main()
    
