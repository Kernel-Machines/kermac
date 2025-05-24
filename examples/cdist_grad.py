import argparse
import kermac
import torch

def parse_args():
    """Parse command-line arguments for matrix dimensions, p-norm, and flags."""
    parser = argparse.ArgumentParser(description="Run kermac.cdist_t with configurable parameters")
    parser.add_argument('-m','--M', type=int, default=10000, help='Number of rows in data_z (default: 10000)')
    parser.add_argument('-n','--N', type=int, default=768, help='Number of columns in data (default: 768)')
    parser.add_argument('-o','--O', type=int, default=16, help='Number of channels in coefficents (default: 16)')
    parser.add_argument('-k','--K', type=int, default=10000, help='Number of rows in data_x (default: 10000)')
    parser.add_argument('-p','--p', type=float, default=2.0, help='p-norm for distance computation (default: 2.0)')
    parser.add_argument('-d','--debug', default=False, action='store_true', help='Enable debug output (default: True)')
    parser.add_argument('--skip_torch_einsum', default=False, action='store_true', help='Skip running torch einsum. Helps avoid memory errors.')
    parser.add_argument('--skip_numeric_compare', default=False, action='store_true', help='Skip comparing torch and kermac results. Helps avoid memory errors.')
    return parser.parse_args()

def main():
    args = parse_args()
    M, N, O, K, p = args.M, args.N, args.O, args.K, args.p
    debug = args.debug

    device = torch.device('cuda')
    timer = kermac.CudaTimer()

    size_M = M
    size_D = N
    size_C = O
    size_N = K

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

    torch.cuda.synchronize()

    tensor_A = torch.randn(size_N,size_M,device=device) # M-major # M-major 
    tensor_B = torch.randn(size_D,size_N,device=device) # N-major # K-major
    tensor_C = torch.randn(size_C,size_N,device=device) # N-major # K-major
    tensor_D = torch.randn(size_D,size_M,device=device) # M-major # M-Major
    # result tensor of mine
    tensor_E = torch.zeros(size_C,size_D,size_M,device=device) # M-major # M-major # (O,N,M)

    coefs =         tensor_C
    kernel_matrix = tensor_A
    x =             tensor_B
    z =             tensor_D

    torch.cuda.synchronize()

    if debug: 
        print('\n(Kermac Debug) Running kermac.cdist_grad')
    timer.start()
    kermac_out = kermac.cdist_grad(
        tensor_A,
        tensor_B,
        tensor_C,
        tensor_D,
        out = tensor_E,
        p = p,
        debug = debug
    )
    if debug:
        print('')
    print(f'Running p-norm-gradient={p} with size ({K},{M}) by ({N},{K}) by ({O},{K}) by ({N},{M})')
    print(f"\tkermac.cdist_grad \t\t{timer.stop():.3f} ms")

    if not p == 2.0:
        print('Cannot compare timing or results to torch unless p=2.0')
        exit()
    if args.skip_torch_einsum:
        exit()
    try:
        timer.start()
        torch_out = torch.einsum('li,ij,jd->ljd', coefs, kernel_matrix, z.T) - torch.einsum('li,ij,id->ljd', coefs, kernel_matrix, x.T)
        print(f"\ttorch-einsum-cdist_grad\t\t{timer.stop():.3f} ms")
    except Exception as e:
        print(f'Exception: {e}')
        print('\nYou can use argument \'--skip_torch_einsum\' to skip running torch einsum and avoid the slow allocation and eventual exception')
        exit()

    if not args.skip_numeric_compare:
        try:
            diff = kermac_out.permute(0,2,1) - torch_out
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
