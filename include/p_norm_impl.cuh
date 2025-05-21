#pragma once

#include <kermac_internal_common.cuh>
#include <p_norm.cuh>

template<
    bool predicate_reads,
    bool predicate_writes,
    NormType norm_type,
    bool skip_epilogue,
    bool align_4_A = false,
    bool align_4_B = false,
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

    auto M = u64(m);
    auto N = u64(n);
    auto K = u64(k);

    auto LDA = u64(ldA);
    auto LDB = u64(ldB);
    auto LDC = u64(ldC);

    auto prob_shape = make_shape(M,N,K);

    auto dA = make_stride(Int<1>{}, LDA); // (dN, dK) : N-major
    auto dB = make_stride(Int<1>{}, LDB); // (dM, dK) : M-major
    auto dC = make_stride(Int<1>{}, LDC); // (dM, dN) : M-major

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};

    auto thread_tiler = Layout<Shape<_16,_16>>{}; // M, N

    auto sA = make_layout(make_shape(bM, bK, bP));  // M-major
    auto sB = make_layout(make_shape(bN, bK, bP));  // N-major
    auto sC = make_layout(make_shape(bM, bN));      // M-major

    // Copy_Atom<DefaultCopy, float> s2r_atom_A;
    // Copy_Atom<DefaultCopy, float> s2r_atom_B;
    
    // Define the thread layouts (static)
    // Dispatch to align_4 is tensors are aligned by 16 bytes
    auto copyA = [] {
        if constexpr (align_4_A) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, T>{},
                Layout<Shape<_32, _8>>{},
                Layout<Shape<_4, _1>>{}
            );
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
                Layout<Shape<_32, _8>>{},
                Layout<Shape<_1, _1>>{}
            );
        }
    }();

    auto copyB = [] {
        if constexpr (align_4_B) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, T>{},
                Layout<Shape<_32, _8>>{},
                Layout<Shape<_4, _1>>{}
            );
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
                Layout<Shape<_32, _8>>{},
                Layout<Shape<_1, _1>>{}
            );
        }
    }();
    
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
    NormType norm_type,
    bool skip_epilogue,
    bool align_4_A = false,
    bool align_4_B = false
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
        cuda_p_norm_m128n128k8p3<true, true, norm_type, skip_epilogue, align_4_A, align_4_B>(
            p_power, m, n, k, A, ldA, B, ldB, C, ldC
        );
    } else {
        cuda_p_norm_m128n128k8p3<true, true, norm_type, skip_epilogue, align_4_A, align_4_B>(
            c_zero<float>, m, n, k, A, ldA, B, ldB, C, ldC
        );
    }
}
