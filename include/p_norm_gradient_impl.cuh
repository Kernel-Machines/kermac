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
cute_p_norm_kernel_gradient(
    int m, int n, int o, int k, int l,
    T const *A, u64 ldA,                u64 batch_stride_a, // kernel_matrix   L,M,N     l,m,k
    T const *B, u64 ldB,                u64 batch_stride_b, // data_N          L,N,D     l,k,n
    T const *C, u64 ldC,                u64 batch_stride_c, // solution        L,N,C     l,k,o
    T const *D, u64 ldD,                u64 batch_stride_d, // data_M          L,M,D     l,m,n
    T *E,       u64 ldE_N, u64 ldE_O,   u64 batch_stride_e, // grad            L,M,D,C   l,m,n,o
    int num_blocks_M,
    T p_power
) {
    using namespace cute;

    auto M = u64(m);
    auto N = u64(n);
    auto O = u64(o);
    auto K = u64(k);
    auto L = u64(l);

    auto prob_shape = make_shape(M,N,O,K,L);

    auto dA = make_stride(Int<1>{}, ldA, batch_stride_a);           // (dM, dK) : M-major
    auto dB = make_stride(ldB, Int<1>{}, batch_stride_b);           // (dN, dK) : K-major
    auto dC = make_stride(ldC, Int<1>{}, batch_stride_c);           // (dO, dK) : K-major
    auto dD = make_stride(Int<1>{}, ldD, batch_stride_d);           // (dM, dN) : M-major
    auto dE = make_stride(Int<1>{}, ldE_N, ldE_O, batch_stride_e);  // (dM, dN, dO) : M-major

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
        num_blocks_M,
        p_power
    );
}

template <
    NormType norm_type
>
__global__
__launch_bounds__(256)
void
cute_norm_kernel_gradient(
    int m, int n, int o, int k, int l,
    float const *A,     u64 ldA,              u64 batch_stride_a,  // kernel_matrix   M,N     m,k
    float const *B,     u64 ldB,              u64 batch_stride_b,  // data_N          N,D     k,n
    float const *C,     u64 ldC,              u64 batch_stride_c,  // solution        N,C     k,o
    float const *D,     u64 ldD,              u64 batch_stride_d,  // data_M          M,D     m,n
    float *E,           u64 ldE_N, u64 ldE_O, u64 batch_stride_e,
    int num_blocks_M,
    float p_power_grad
) {
    cute_p_norm_kernel_gradient<
        true, true, 
        norm_type
    >(
        m, n, o, k, l,
        A, ldA,             batch_stride_a,
        B, ldB,             batch_stride_b,
        C, ldC,             batch_stride_c,
        D, ldD,             batch_stride_d,
        E, ldE_N,   ldE_O,  batch_stride_e,
        num_blocks_M,
        p_power_grad
    );
}