import argparse
import torch
import kermac

def parse_args():
    """Parse command-line arguments for matrix dimensions, p-norm, and flags."""
    parser = argparse.ArgumentParser(description="Run kermac.cdist_t with configurable parameters")
    parser.add_argument('-m','--M', type=int, default=10000, help='Number of rows in output matrix (default: 10000)')
    parser.add_argument('-n','--N', type=int, default=10000, help='Number of columns in output matrix (default: 10000)')
    parser.add_argument('-k','--K', type=int, default=1024, help='Inner dimension of input matrices (default: 1024)')
    parser.add_argument('-a','--try_align', default=False, action='store_true', help='Specialize kernel if tensors are 4 element aligned')
    parser.add_argument('-d','--debug', default=False, action='store_true', help='Enable debug output (default: True)')
    parser.add_argument('-s','--skip_torch', default=False, action='store_true', help='Skip running the PyTorch equivalent')
    return parser.parse_args()

def main():
    args = parse_args()
    M, N, K = args.M, args.N, args.K
    try_to_align = args.try_align
    debug = args.debug
    skip_torch = args.skip_torch

    device = torch.device('cuda')
    a = torch.randn(M,K,device=device)
    b = torch.randn(N,K,device=device)
    out = torch.zeros(M,N,device=device)

    # Example of a custom non-predefined kernel
    # Because it uses PowerType.POW it will require a `p=` in the argument list for `run_kernel`
    # Because it uses a KernelType.GAUSSIAN it will require a `bandwidth=` in the argument list for `run_kernel`
    kernel_descriptor_gaussian_p_norm = \
        kermac.KernelDescriptor(
            inner_operator=kermac.InnerOperator.DIFF,
            inner_power=kermac.PowerType.POW,
            outer_power=kermac.PowerType.POW,
            kernel_type=kermac.KernelType.GAUSSIAN,
        )

    descriptors = [
        kermac.kernel_descriptor_l1_norm,
        kermac.kernel_descriptor_l2_norm,
        kermac.kernel_descriptor_p_norm,
        kermac.kernel_descriptor_laplace_l1,
        kermac.kernel_descriptor_laplace_l2,
        kermac.kernel_descriptor_mma,
        kernel_descriptor_gaussian_p_norm
    ]

    print('Bulk compiling kernels')
    if debug:
        print('(Kermac Debug) Choosing architecture from cuda device 0')
    kermac.pre_compile_descriptors(
        device=device,
        descriptors=descriptors,
        try_to_align=try_to_align,
        debug=debug
    )

    print('Running euclidean laplace kernel')
    kermac.run_kernel(
        kermac.kernel_descriptor_laplace_l2,
        a, b,
        out=out,
        bandwidth=10.0,
        epsilon=1e-5,
        try_to_align=try_to_align,
        debug=debug
    )
    print(out)

    print('Running L1 laplace kernel')
    kermac.run_kernel(
        kermac.kernel_descriptor_laplace_l1,
        a, b,
        out=out,
        bandwidth=10.0,
        try_to_align=try_to_align,
        debug=debug
    )
    print(out)

    print('Running L1 norm kernel')
    kermac.run_kernel(
        kermac.kernel_descriptor_l1_norm,
        a, b,
        out=out,
        # bandwidth=10.0,
        try_to_align=try_to_align,
        debug=debug
    )
    print(out)

    print('Running L2 norm kernel')
    kermac.run_kernel(
        kermac.kernel_descriptor_l2_norm,
        a, b,
        out=out,
        # bandwidth=10.0,
        try_to_align=try_to_align,
        debug=debug
    )
    print(out)

    print('Running p-power gaussian kernel')
    kermac.run_kernel(
        kernel_descriptor_gaussian_p_norm,
        a, b,
        out=out,
        inner_p=1.3,
        outer_p=1.0/1.3,
        bandwidth=10.0,
        try_to_align=try_to_align,
        debug=debug
    )
    print(out)

    print('Running L1 norm kernel again')
    kermac.run_kernel(
        kermac.kernel_descriptor_l1_norm,
        a, b,
        out=out,
        # bandwidth=10.0,
        try_to_align=try_to_align,
        debug=debug
    )
    print('kermac L1')
    print(out)

    if not skip_torch:
        print('torch.cdist L1')
        print(torch.cdist(a,b,p=1.0))

    print('Running MMA')
    kermac.run_kernel(
        kermac.kernel_descriptor_mma,
        a, b,
        out=out,
        # bandwidth=10.0,
        try_to_align=try_to_align,
        debug=debug
    )
    print('kermac MMA')
    print(out)

    if not skip_torch:
        print('torch MMA')
        print((a @ b.T))

if __name__ == '__main__':
    main()
