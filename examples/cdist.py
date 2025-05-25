import argparse
import kermac
import torch

from kermac.common import tensor_stats, Alignment

def parse_col_major_flags(flag_string):
    # Validate input
    if not isinstance(flag_string, str) or len(flag_string) != 3 or not all(c in 'nt' for c in flag_string):
        raise ValueError("Input must be a 3-character string containing only 'n' or 't'")

    # Parse flags and assign to variables
    a_col_major = flag_string[0] == 'n'
    b_col_major = flag_string[1] == 't'
    c_col_major = flag_string[2] == 'n'

    return a_col_major, b_col_major, c_col_major


def parse_args():
    """Parse command-line arguments for matrix dimensions, p-norm, and flags."""
    parser = argparse.ArgumentParser(description="Run kermac.cdist_t with configurable parameters")
    parser.add_argument('-m', '--M', type=int, default=30000, help='Number of rows in output matrix (default: 30000)')
    parser.add_argument('-n', '--N', type=int, default=30000, help='Number of columns in output matrix (default: 30000)')
    parser.add_argument('-k', '--K', type=int, default=1024, help='Inner dimension of input matrices (default: 1024)')
    parser.add_argument('-p', '--p', type=float, default=1.0, help='p-norm for distance computation (default: 1.0)')
    parser.add_argument('-s', '--skip_epilogue', default=False, action='store_true', help='Skip epilogue in kermac.cdist_t (default: False)')
    parser.add_argument('-a', '--try_align', default=False, action='store_true', help='Specialize kernel if tensors are 4 element aligned')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Enable debug output (default: False)')
    parser.add_argument('--skip_numeric_compare', default=False, action='store_true', help='Skip comparing torch and kermac results. Helps avoid memory errors.')
    parser.add_argument('--skip_torch', default=False, action='store_true', help='Skip running torch version.')
    parser.add_argument('--a_col_major', default=False, action='store_true', help='Make tensor A column-major instead of row-major')
    parser.add_argument('--b_col_major', default=False, action='store_true', help='Make tensor B column-major instead of row-major')
    parser.add_argument('--c_col_major', default=False, action='store_true', help='Make tensor C column-major instead of row-major')
    parser.add_argument('--flags', type=str, default=None, help='3-character string of "n" or "t" to set a_col_major, b_col_major, c_col_major (e.g., "nnt")')

    args = parser.parse_args()

    # If flags is provided, override a_col_major, b_col_major, c_col_major
    if args.flags is not None:
        if not isinstance(args.flags, str) or len(args.flags) != 3 or not all(c in 'nt' for c in args.flags):
            parser.error('The --flags argument must be a 3-character string containing only "n" or "t"')
        args.a_col_major = args.flags[0] == 'n'
        args.b_col_major = args.flags[1] == 't'
        args.c_col_major = args.flags[2] == 'n'

    return args

def main():
    args = parse_args()
    M, N, K, p = args.M, args.N, args.K, args.p
    skip_epilogue = args.skip_epilogue
    try_align = args.try_align
    debug = args.debug
    skip_torch = args.skip_torch
    a_col_major = args.a_col_major
    b_col_major = args.b_col_major
    c_col_major = args.c_col_major
    # debug = True

    device = torch.device('cuda')
    timer = kermac.CudaTimer()

    a = torch.randn(K,M,device=device).T if a_col_major else torch.randn(M,K,device=device)
    b = torch.randn(K,N,device=device).T if b_col_major else torch.randn(N,K,device=device)
    c = torch.randn(N,M,device=device).T if c_col_major else torch.randn(M,N,device=device)

    kermac_out = c

    if debug: 
        print('\n(Kermac Debug) Warmup kermac.cdist')
    kermac.cdist(
        a, b, 
        p=p, out=kermac_out,
        skip_epilogue=skip_epilogue,
        try_to_align=try_align,
        debug=debug
    )
    torch.cdist(
        a, b, p=p
    )

    torch.cuda.synchronize()

    if debug: 
        print('\n(Kermac Debug) Running kermac.cdist_t')
    timer.start()
    kermac_out = kermac.cdist(
        a, b, 
        p=p, out=kermac_out,
        skip_epilogue=skip_epilogue,
        try_to_align=try_align,
        debug=debug
    )
    if debug:
        print('')
    print(f'Running p-norm={p} with size ({M},{K}) by ({N},{K})')
    print(f"\tkermac.cdist_t \t{timer.stop():.3f} ms")

    if skip_torch:
        exit()

    timer.start()
    torch_out = torch.cdist(a, b, p=p)
    print(f"\ttorch.cdist \t{timer.stop():.3f} ms")

    if not args.skip_numeric_compare:
        try:
            diff = kermac_out - torch_out
            squared_diff = diff ** 2
            mse = torch.mean(squared_diff)
            rmse = torch.sqrt(mse).item()

            # Existing error checks with RMSE added
            abs_error = torch.abs(diff)
            max_abs_error = torch.max(abs_error).item()
            mean_abs_error = torch.mean(abs_error).item()
#TODO: torch.allclose is super slow (maybe rewrite?)
            # is_close = torch.allclose(kermac_out, torch_out, rtol=1e-5, atol=1e-8)

            print(f"\nTensor Comparison:")
            print(f"\tRoot Mean Squared Error:\t{rmse:.6e}")
            print(f"\tMax Absolute Error:\t\t{max_abs_error:.6e}")
            print(f"\tMean Absolute Error:\t\t{mean_abs_error:.6e}")
            # print(f"\tTensors are close (within tolerance): {is_close}")
        except Exception as e:
            print(f'Exception: {e}')
            print('\nYou can use argument \'--skip_numeric_compare\' to skip comparison and avoid the slow allocation and eventual exception')


if __name__ == '__main__':
    main()
