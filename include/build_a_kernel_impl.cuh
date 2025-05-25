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
    Majorness majorness_A,
    Majorness majorness_B,
    Alignment align_A,
    Alignment align_B,
    class T
>
__device__
__forceinline__
void
cute_build_kernel(
    int m, int n, int k,
    T const *A, int ldA,
    T const *B, int ldB,
    T *C, int ldC,
    T p_power_inner, 
    T p_power_outer,
    T bandwidth
) {
    // Don't support ALIGN_4 specialization for ROW_MAJOR tensors
    static_assert(majorness_A != Majorness::ROW_MAJOR || align_A != Alignment::ALIGN_4);
    static_assert(majorness_B != Majorness::ROW_MAJOR || align_B != Alignment::ALIGN_4);

    using namespace cute;

    auto M = u64(m);
    auto N = u64(n);
    auto K = u64(k);

    auto LDA = u64(ldA);
    auto LDB = u64(ldB);
    auto LDC = u64(ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};

    auto prob_shape = make_shape(M,N,K);
    auto dA = [LDA] { 
        if constexpr(majorness_A == Majorness::COL_MAJOR) {
            return make_stride(Int<1>{}, LDA);
        } else {
            return make_stride(LDA, Int<1>{});
        }
    }();
    auto dB = [LDB] { 
        if constexpr (majorness_B == Majorness::COL_MAJOR) {
            return make_stride(Int<1>{}, LDB);
        } else {
            return make_stride(LDB, Int<1>{});
        }
    }();
    auto dC = make_stride(Int<1>{}, LDC);

    auto sA = [bM,bK,bP] {
        if constexpr (majorness_A == Majorness::COL_MAJOR) {
            return make_layout(make_shape(bM, bK, bP));
        } else {
            auto atom = make_layout(
                make_shape(bM,bK),
                make_stride(Int<1>{}, bM+Int<4>{})
            );
            return tile_to_shape(atom, make_shape(bM, bK, bP));
        }
    }();
    auto sB = [bN,bK,bP] {
        if constexpr (majorness_B == Majorness::COL_MAJOR) {
            return make_layout(make_shape(bN, bK, bP));
        } else {
            auto atom = make_layout(
                make_shape(bN,bK),
                make_stride(Int<1>{}, bN+Int<4>{})
            );
            return tile_to_shape(atom, make_shape(bN, bK, bP));
        }
    }();
    
    auto sC = make_layout(make_shape(bM, bN));
    
    auto copyA = [] {
        if constexpr (majorness_A == Majorness::COL_MAJOR) {
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
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
                Layout<Shape<_32,_8>, Stride<_8,_1>>{},
                Layout<Shape< _1,_1>>{}
            );
        }
    }();

    auto copyB = [] {
        if constexpr (majorness_B == Majorness::COL_MAJOR) {
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
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
                Layout<Shape<_32,_8>, Stride<_8,_1>>{},
                Layout<Shape< _1,_1>>{}
            );
        }
    }();

    auto thread_tiler = Layout<Shape<_16,_16>>{}; // M, N

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

template <
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Majorness majorness_A,
    Majorness majorness_B,
    Alignment align_A,
    Alignment align_B
>
__global__
__launch_bounds__(256)
void
cute_build_kernel(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    cute_build_kernel<
        true, true,
        inner_operator, 
        inner_power,
        outer_power, 
        kernel_type,
        majorness_A, 
        majorness_B,
        align_A, 
        align_B
    >(
        m,n,k,
        A, ldA,
        B, ldB,
        C, ldC,
        p_power_inner,
        p_power_outer,
        bandwidth
    );
}