import argparse
import kermac
import torch

def parse_args():
    """Parse command-line arguments for matrix dimensions, p-norm, and flags."""
    parser = argparse.ArgumentParser(description="Run kermac.cdist_t with configurable parameters")
    parser.add_argument('-m', '--M', type=int, default=10000, help='Number of rows in output matrix (default: 10000)')
    parser.add_argument('-n', '--N', type=int, default=10000, help='Number of columns in output matrix (default: 10000)')
    parser.add_argument('-k', '--K', type=int, default=1024, help='Inner dimension of input matrices (default: 1024)')
    parser.add_argument('-l', '--L', type=int, default=2, help='Number of batches (default: 2)')
    parser.add_argument('-p', '--p', type=float, default=1.0, help='p-norm for distance computation (default: 1.0)')
    parser.add_argument('-s', '--skip_epilogue', default=False, action='store_true', help='Skip epilogue in kermac.cdist_t (default: False)')
    parser.add_argument('-a', '--try_align', default=False, action='store_true', help='Specialize kernel if tensors are 4 element aligned')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Enable debug output (default: False)')
    parser.add_argument('--skip_numeric_compare', default=False, action='store_true', help='Skip comparing torch and kermac results. Helps avoid memory errors.')
    parser.add_argument('--skip_torch', default=False, action='store_true', help='Skip running torch version.')
    parser.add_argument('-t','--transposes', type=str, default=None, help='3-character string of "n" or "t" to set a_col_major, b_col_major, c_col_major (e.g., "nnt")')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup rounds (default:2)')
    parser.add_argument('--iters', type=int, default=10, help='Number of iteration rounds (default:10)')

    args = parser.parse_args()

    # If transposes is provided, override a_col_major, b_col_major, c_col_major
    a_col_major = False
    b_col_major = False
    c_col_major = False

    if args.transposes is not None:
        if not isinstance(args.transposes, str) or len(args.transposes) != 3 or not all(c in 'nt' for c in args.transposes):
            parser.error('The --flags argument must be a 3-character string containing only "n" or "t"')
        a_col_major = args.transposes[0] == 'n'
        b_col_major = args.transposes[1] == 't'
        c_col_major = args.transposes[2] == 'n'
        
    return a_col_major, b_col_major, c_col_major, args

def main():
    a_col_major, b_col_major, c_col_major, args = parse_args()
    M, N, K, L, p = args.M, args.N, args.K, args.L, args.p
    skip_epilogue = args.skip_epilogue
    try_align = args.try_align
    debug = args.debug
    skip_torch = args.skip_torch
    warmup_rounds = args.warmup
    iterations = args.iters

    device = torch.device('cuda')
    timer = kermac.CudaTimer()

    a = torch.randn(L,K,M,device=device).permute(0,2,1) if a_col_major else torch.randn(L,M,K,device=device)
    b = torch.randn(L,K,N,device=device).permute(0,2,1) if b_col_major else torch.randn(L,N,K,device=device)
    c = torch.randn(L,N,M,device=device).permute(0,2,1) if c_col_major else torch.randn(L,M,N,device=device)

    kermac_out = c

    print(f'Warmup {warmup_rounds} iterations of kermac.cdist (Might be JIT compiling a kernel)')
    for _ in range(warmup_rounds):
        kermac.cdist(
            a, b, 
            p=p, out=kermac_out,
            skip_epilogue=skip_epilogue,
            try_to_align=try_align,
            debug=debug
        )
    torch.cuda.synchronize()
    if not skip_torch:
        print(f'Warmup {warmup_rounds} iterations of torch.cdist')
        for _ in range(warmup_rounds):
            torch.cdist(
                a, b, p=p
            )
        torch.cuda.synchronize()
    
    print('Running kermac.cdist')
    timer.start()
    for _ in range(iterations):
        kermac_out = kermac.cdist(
            a, b, 
            p=p, out=kermac_out,
            skip_epilogue=skip_epilogue,
            try_to_align=try_align,
            debug=debug
        )
    print(f'Running {iterations} iterations of p-norm={p} with size ({L},{M},{K}) by ({L},{N},{K})')
    print(f"\tkermac.cdist \t{timer.stop() / iterations:.3f} ms / iteration")

    if skip_torch:
        exit()

    timer.start()
    for _ in range(iterations):
        torch_out = torch.cdist(a, b, p=p)
    print(f"\ttorch.cdist \t{timer.stop() / iterations:.3f} ms / iteration")

    if not args.skip_numeric_compare:
        try:
            diff = kermac_out - torch_out
            squared_diff = diff ** 2
            mse = torch.mean(squared_diff)
            rmse = torch.sqrt(mse).item()

            abs_error = torch.abs(diff)
            max_abs_error = torch.max(abs_error).item()
            mean_abs_error = torch.mean(abs_error).item()

            print(f"\nTensor Comparison:")
            print(f"\tRoot Mean Squared Error:\t{rmse:.6e}")
            print(f"\tMax Absolute Error:\t\t{max_abs_error:.6e}")
            print(f"\tMean Absolute Error:\t\t{mean_abs_error:.6e}")
        except Exception as e:
            print(f'Exception: {e}')
            print('\nYou can use argument \'--skip_numeric_compare\' to skip comparison and avoid the slow allocation and eventual exception')


if __name__ == '__main__':
    main()
