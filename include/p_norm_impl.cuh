#pragma once

#include <kermac_internal_common.cuh>
#include <p_norm.cuh>

template<
    // bool align_4_M,
    // bool align_4_N,
    bool predicate_reads,
    bool predicate_writes,
    bool skip_epilogue,
    NormType norm_type,
    class T
>
__device__
__forceinline__
void
cuda_p_norm_m128n128k8p3(
    T p_power,
    int m, int n, int k,
    T const *A, int ldA,
    T const *B, int ldB,
    T *C, int ldC
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
    auto bP = Int<3>{};

    auto thread_tiler = Layout<Shape<_32,_8>>{}; // M, N

    auto sA = make_layout(make_shape(bM, bK, bP));  // M-major
    auto sB = make_layout(make_shape(bN, bK, bP));  // N-major
    auto sC = make_layout(make_shape(bM, bN));      // M-major

    // Copy_Atom<DefaultCopy, float> s2r_atom_A;
    // Copy_Atom<DefaultCopy, float> s2r_atom_B;

    ///TODO: 
    // Can make these copies uint128_t if i can check for tensor alignment and include the check on tensor view
    // Data pointer can be shifted up by x elements
    // Leading dimension needs to be a multiple of 4 even if shape isn't a multiple of 4
    // Hard to explain

    // Moving pipeline from 2 to 3 completely hides benefit of uint128_t copy size
    
    // Define the thread layouts (static)
    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
        Layout<Shape<_32,_8>>{}, // Thr layout 32x8 n-major
        Layout<Shape< _1,_1>>{}  // Val layout  4x1 n-major
    );

    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
        Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major
        Layout<Shape< _1,_1>>{}  // Val layout  4x1 m-major
    );
    
#if 0
    int smem_size = 
        int(sizeof(
            SharedStorageNorm<
                decltype(sA), 
                decltype(sB), 
                T
            >
        ));
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("smem: %d\n", smem_size);
        printf("p_power: %f\n", p_power);
    }
    return;
#endif

    kernel_cute_p_norm<
        predicate_reads,
        predicate_writes,
        skip_epilogue,
        norm_type
    > (
        prob_shape, cta_tiler, thread_tiler, 
        A, dA, sA, copyA,
        B, dB, sB, copyB,
        C, dC, sC, p_power
    );
}

template <
    NormType norm_type
>
__global__
__launch_bounds__(256)
void
cute_norm_m128m128k8p3(
    float p_power,
    int m, int n, int k,
    float const *A, int ldA, // M,K m-major
    float const *B, int ldB, // N,K n-major
    float       *C, int ldC  // M,N m-major
) {
    if constexpr (norm_type == NormType::P) {
        cuda_p_norm_m128n128k8p3<true, true, false, norm_type>(
            p_power, m, n, k, A, ldA, B, ldB, C, ldC
        );
    } else {
        cuda_p_norm_m128n128k8p3<true, true, false, norm_type>(
            c_zero<f32>, m, n, k, A, ldA, B, ldB, C, ldC
        );
    }
}
