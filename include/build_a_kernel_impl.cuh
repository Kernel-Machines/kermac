#pragma once

#include <kermac_internal_common.cuh>
#include <build_a_kernel.cuh>

template<
    bool predicate_reads,
    bool predicate_writes,
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B,
    class T
>
__device__
__forceinline__
void
cute_build_kernel_m128n128k8p3_impl(
    int m, int n, int k,
    T const *A, int ldA,
    T const *B, int ldB,
    T *C, int ldC,
    T p_power_inner, 
    T p_power_outer,
    T bandwidth
) {
    using namespace cute;

    auto M = u64(m);
    auto N = u64(n);
    auto K = u64(k);

    auto LDA = u64(ldA);
    auto LDB = u64(ldB);
    auto LDC = u64(ldC);

    auto prob_shape = make_shape(M,N,K);

    auto dA = make_stride(Int<1>{}, LDA); // (dM, dK) : M-major
    auto dB = make_stride(Int<1>{}, LDB); // (dN, dK) : N-major
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
        if constexpr (align_A == Alignment::ALIGN_4) {
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
        if constexpr (align_B == Alignment::ALIGN_4) {
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

    kernel_cute_build_kernel<
        predicate_reads,
        predicate_writes,
        inner_operator,
        inner_power,
        outer_power,
        kernel_type
    > (
        prob_shape, cta_tiler, thread_tiler, 
        A, dA, sA, copyA,
        B, dB, sB, copyB,
        C, dC, sC, 
        p_power_inner,
        p_power_outer,
        bandwidth
    );
}

///TODO: Can turn off predication in certain cases
// check for divisibility by cta size
template <
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__global__
__launch_bounds__(256)
void
cute_build_kernel_m128n128k8p3(
    int m, int n, int k,
    float const *A, int ldA, // M,K m-major
    float const *B, int ldB, // N,K n-major
    float       *C, int ldC,  // M,N m-major
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    cute_build_kernel_m128n128k8p3_impl<
        true, true,
        inner_operator,
        inner_power,
        outer_power,
        kernel_type,
        align_A,
        align_B
    >(
        m, n, k, 
        A, ldA, 
        B, ldB, 
        C, ldC, 
        p_power_inner,
        p_power_outer,
        bandwidth
    );
}