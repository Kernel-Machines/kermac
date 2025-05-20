#pragma once

#include <kermac_internal_common.cuh>
#include <p_norm_gradient.cuh>

template <
    bool predicate_reads,
    bool predicate_writes,
    NormType norm_type,
    class T
>
__device__
__forceinline__
void
cute_p_norm_kernel_gradient_m128n16o16k32p2(
    T p_power,
    int m, // M
    int n, // D
    int o, // C
    int k, // N
    T const *A, int ldA,        // kernel_matrix   M,N     m,k
    T const *B, int ldB,        // data_N          N,D     k,n
    T const *C, int ldC,        // solution        N,C     k,o
    T const *D, int ldD,        // data_M          M,D     m,n
    T *E, int ldE, int ldE_2   // grad            M,D,C   m,n,o
) {
    using namespace cute;

    auto M = u64(m);
    auto N = u64(n);
    auto O = u64(o);
    auto K = u64(k);

    auto LDA = u64(ldA);
    auto LDB = u64(ldB);
    auto LDC = u64(ldC);
    auto LDD = u64(ldD);
    auto LDE = u64(ldE);
    auto LDE_2 = u64(ldE_2);

    auto prob_shape = make_shape(M,N,O,K);

    auto dA = make_stride(Int<1>{}, LDA); // (dM, dK) : M-major
    auto dB = make_stride(LDB, Int<1>{}); // (dN, dK) : K-major
    auto dC = make_stride(LDC, Int<1>{}); // (dO, dK) : K-major
    auto dD = make_stride(Int<1>{}, LDD); // (dM, dN) : M-major
    auto dE = make_stride(Int<1>{}, LDE, LDE_2); // (dM, dN, dO) : M-major

    auto bM = Int<128>{};
    auto bN = Int<16>{};
    auto bO = Int<16>{};
    auto bK = Int<32>{};
    auto cta_tiler = make_shape(bM, bN, bO, bK);
    auto bP = Int<2>{};

    auto thread_tiler = Layout<Shape<_32, _8, _1>>{}; // M, N, O

    auto sA = make_layout(make_shape(bM, bK, bP)); // M-major
    
    auto sB_atom = make_layout(
        make_shape(bN, bK),
        make_stride(Int<1>{}, bN)
    ); // (n,k) -> smem_idx; padded n-major
    auto sB = tile_to_shape(sB_atom, make_shape(bN, bK, bP)); // N-major

    auto sC_atom = make_layout(
        make_shape(bO, bK),
        make_stride(Int<1>{}, bO)
    ); // (o,k) -> smem_idx; padded o-major
    auto sC = tile_to_shape(sC_atom, make_shape(bO, bK, bP)); // O-major

    auto sD = make_layout(make_shape(bM, bN)); // M-major
    auto sE = make_layout(make_shape(bM, bN, bO)); // M-major

    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
        Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major
        Layout<Shape< _1,_1>>{}  // Val layout  4x1 m-major
    );

    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
        Layout<Shape<_8,_32>, Stride<_32,_1>>{}, // Thr layout 8x32 k-major
        Layout<Shape< _1,_1>>{} // Val layout  1x1
    );

    TiledCopy copyC = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
        Layout<Shape<_8,_32>, Stride<_32,_1>>{}, // Thr layout 8x32 k-major
        Layout<Shape< _1,_1>>{} // Val layout  1x1 
    );

#if 0
    int smem_size = 
        int(sizeof(
            SharedStorageNormGradient<
                decltype(sA), 
                decltype(sB), 
                decltype(sC), 
                T
            >
        ));

    dim3 dimBlock(size(thread_tiler));
    dim3 dimGrid(
        size(ceil_div(M, bM)),
        size(ceil_div(N, bN)),
        size(ceil_div(O, bO))
    );
#endif

#if 0
    printf("smem_size: %d\n", smem_size);
#endif
    kernel_cute_p_norm_kernel_gradient<
        predicate_reads, 
        predicate_writes,
        norm_type
    > (
        prob_shape, cta_tiler, thread_tiler,
        A, dA, sA, copyA,
        B, dB, sB, copyB,
        C, dC, sC, copyC,
        D, dD, sD,
        E, dE, sE,
        p_power
    );
}

template <
    NormType norm_type
>
__global__
__launch_bounds__(256)
void
cute_norm_kernel_gradient_m128n16o16k32p2(
    float p_power,
    int m, // M
    int n, // D
    int o, // C
    int k, // N
    float const *A, int ldA,        // kernel_matrix   M,N     m,k
    float const *B, int ldB,        // data_N          N,D     k,n
    float const *C, int ldC,        // solution        N,C     k,o
    float const *D, int ldD,        // data_M          M,D     m,n
    float *E, int ldE, int ldE_2
) {
    if constexpr (norm_type == NormType::L1) {
        cute_p_norm_kernel_gradient_m128n16o16k32p2<true, true, NormType::L1>(
            c_zero<float>,
            m, n, o, k,
            A, ldA, 
            B, ldB,
            C, ldC,
            D, ldD,
            E, ldE, ldE_2
        );
    } else if constexpr (norm_type == NormType::L2) {
        cute_p_norm_kernel_gradient_m128n16o16k32p2<true, true, NormType::L2>(
            c_zero<float>,
            m, n, o, k,
            A, ldA, 
            B, ldB,
            C, ldC,
            D, ldD,
            E, ldE, ldE_2
        );
    } else {
        float p_power_grad = p_power-c_one<float>;
        cute_p_norm_kernel_gradient_m128n16o16k32p2<true, true, NormType::P>(
            p_power_grad,
            m, n, o, k,
            A, ldA, 
            B, ldB,
            C, ldC,
            D, ldD,
            E, ldE, ldE_2
        );
    }
}