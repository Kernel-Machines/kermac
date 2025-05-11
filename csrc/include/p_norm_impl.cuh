#pragma once

#include <p_norm.cuh>
#include <kermac_internal_common.cuh>

template <
    class T
>
static
void
cute_p_norm_m128n128k8p2(
    T p_power,
    bool skip_epilogue,
    int m, // M
    int n, // N
    int k, // D
    T const *A, int ldA, // data_M      M,D     m,k
    T const *B, int ldB, // data_N      N,D     n,k
    T *C, int ldC, // norm_matrix M,N     m,n
    cudaStream_t stream
) {
    using namespace cute;

    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    auto prob_shape = make_shape(M,N,K);

    auto dA = make_stride(Int<1>{}, ldA); // (dN, dK) : N-major
    auto dB = make_stride(Int<1>{}, ldB); // (dM, dK) : M-major
    auto dC = make_stride(Int<1>{}, ldC); // (dM, dN) : M-major

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<2>{};

    auto thread_tiler = Layout<Shape<_32,_8>>{}; // M, N

    auto sA = make_layout(make_shape(bM, bK, bP));  // M-major
    auto sB = make_layout(make_shape(bN, bK, bP));  // N-major
    auto sC = make_layout(make_shape(bM, bN));      // M-major

    // Copy_Atom<DefaultCopy, float> s2r_atom_A;
    // Copy_Atom<DefaultCopy, float> s2r_atom_B;

    // Define the thread layouts (static)
    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, T>{},
        Layout<Shape<_32,_8>>{}, // Thr layout 32x8 n-major
        Layout<Shape< _4,_1>>{}  // Val layout  4x1 n-major
    );

    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, T>{},
        Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major
        Layout<Shape< _4,_1>>{}  // Val layout  4x1 m-major
    );

    int smem_size = 
        int(sizeof(
            SharedStorageNorm<
                decltype(sA), 
                decltype(sB), 
                T
            >
        ));

    dim3 dimBlock(size(thread_tiler));
    dim3 dimGrid(
        size(ceil_div(M, bM)),
        size(ceil_div(N, bN))
    );

    if (p_power == 1.0f) {
        if (skip_epilogue) {
            printf("Launching L1 Norm, skipping epilogue\n");
            kernel_cute_p_norm<true, true, true, NormType::L1><<<dimGrid, dimBlock, smem_size, stream>>>(
                prob_shape, cta_tiler, thread_tiler,
                A, dA, sA, copyA,
                B, dB, sB, copyB,
                C, dC, sC,
                p_power
            );
        } else {
            printf("Launching L1 Norm\n");
            kernel_cute_p_norm<true, true, false, NormType::L1><<<dimGrid, dimBlock, smem_size, stream>>>(
                prob_shape, cta_tiler, thread_tiler,
                A, dA, sA, copyA,
                B, dB, sB, copyB,
                C, dC, sC,
                p_power
            );
        }
    } else if (p_power == 2.0f) {
        if (skip_epilogue) {
            printf("Launching L2 Norm, skipping epilogue\n");
            kernel_cute_p_norm<true, true, true, NormType::L2><<<dimGrid, dimBlock, smem_size, stream>>>(
                prob_shape, cta_tiler, thread_tiler,
                A, dA, sA, copyA,
                B, dB, sB, copyB,
                C, dC, sC,
                p_power
            );
        } else {
            printf("Launching L2 Norm\n");
            kernel_cute_p_norm<true, true, false, NormType::L2><<<dimGrid, dimBlock, smem_size, stream>>>(
                prob_shape, cta_tiler, thread_tiler,
                A, dA, sA, copyA,
                B, dB, sB, copyB,
                C, dC, sC,
                p_power
            );
        }
    } else {
        if (skip_epilogue) {
            printf("Launching Norm-P=%0.3f, skipping epilogue\n", p_power);
            kernel_cute_p_norm<true, true, true, NormType::P><<<dimGrid, dimBlock, smem_size, stream>>>(
                prob_shape, cta_tiler, thread_tiler,
                A, dA, sA, copyA,
                B, dB, sB, copyB,
                C, dC, sC,
                p_power
            );
        } else {
            printf("Launching Norm-P=%0.3f\n", p_power);
            kernel_cute_p_norm<true, true, false, NormType::P><<<dimGrid, dimBlock, smem_size, stream>>>(
                prob_shape, cta_tiler, thread_tiler,
                A, dA, sA, copyA,
                B, dB, sB, copyB,
                C, dC, sC,
                p_power
            );
        }
    }
}
