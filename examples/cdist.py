import argparse
import kermac
import torch

from kermac.common import is_tensor_16_byte_aligned

def parse_args():
    """Parse command-line arguments for matrix dimensions, p-norm, and flags."""
    parser = argparse.ArgumentParser(description="Run kermac.cdist_t with configurable parameters")
    parser.add_argument('-m','--M', type=int, default=30000, help='Number of rows in output matrix (default: 30000)')
    parser.add_argument('-n','--N', type=int, default=30000, help='Number of columns in output matrix (default: 30000)')
    parser.add_argument('-k','--K', type=int, default=1024, help='Inner dimension of input matrices (default: 1024)')
    parser.add_argument('-p','--p', type=float, default=1.0, help='p-norm for distance computation (default: 1.0)')
    parser.add_argument('-s','--skip_epilogue', default=False, action='store_true', help='Skip epilogue in kermac.cdist_t (default: False)')
    parser.add_argument('-a','--try_align', default=False, action='store_true', help='Specialize kernel if tensors are 4 element aligned')
    parser.add_argument('-d','--debug', default=False, action='store_true', help='Enable debug output (default: True)')
    parser.add_argument('--skip_numeric_compare', default=False, action='store_true', help='Skip comparing torch and kermac results. Helps avoid memory errors.')
    parser.add_argument('--skip_torch', default=False, action='store_true', help='Skip running torch version.')
    return parser.parse_args()

def main():
    args = parse_args()
    M, N, K, p = args.M, args.N, args.K, args.p
    skip_epilogue = args.skip_epilogue
    try_align = args.try_align
    debug = args.debug
    skip_torch = args.skip_torch
    # debug = True

    device = torch.device('cuda')
    timer = kermac.CudaTimer()

    a = torch.randn(K,M,device=device)
    b = torch.randn(K,N,device=device)
    kermac_out = torch.zeros(N,M,device=device)

    # offset a and/or b for warmup explicitly if a/b sizes are unaligned
    # forces compile during warmup for this special case
    alignment_offset_a = 0 if is_tensor_16_byte_aligned(a) else 1
    alignment_offset_b = 0 if is_tensor_16_byte_aligned(b) else 1

    if debug: 
        print('\n(Kermac Debug) Warmup kermac.cdist_t')
    kermac.cdist_t(
        torch.randn(10,100 + alignment_offset_a,device=device), # a
        torch.randn(10,100 + alignment_offset_b,device=device), # b
        p=p,
        skip_epilogue=skip_epilogue,
        try_to_align=try_align,
        debug=debug
    )
    torch.cdist(
        torch.randn(10,100,device=device), # a
        torch.randn(10,100,device=device), # b
        p=p
    )

    torch.cuda.synchronize()

    if debug: 
        print('\n(Kermac Debug) Running kermac.cdist_t')
    timer.start()
    kermac_out = kermac.cdist_t(
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
    torch_out = torch.cdist(a.T, b.T, p=p).T
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
