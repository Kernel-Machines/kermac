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
    Alignment align_B
>
__device__
__forceinline__
void
cute_build_kernel_nn(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    using namespace cute;
    using T = float;

    auto M = u64(m);
    auto N = u64(n);
    auto K = u64(k);

    auto LDA = u64(ldA);
    auto LDB = u64(ldB);
    auto LDC = u64(ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<32>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};

    auto prob_shape = make_shape(M,N,K);
    auto dA = make_stride(Int<1>{}, LDA);
    auto dB = make_stride(LDB, Int<1>{});
    auto dC = make_stride(Int<1>{}, LDC);

    auto sB_atom = make_layout(
        make_shape(bN,bK),
        make_stride(Int<1>{}, bN+Int<4>{})
    );

    auto sA = make_layout(make_shape(bM, bK, bP));
    auto sB = tile_to_shape(sB_atom, make_shape(bN, bK, bP));
    auto sC = make_layout(make_shape(bM, bN));
    
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

    auto copyB = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        Layout<Shape<_8,_32>, Stride<_32,_1>>{},
        Layout<Shape< _1,_1>>{}
    );

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

template<
    bool predicate_reads,
    bool predicate_writes,
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__device__
__forceinline__
void
cute_build_kernel_nt(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    using namespace cute;
    using T = float;

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
    auto dA = make_stride(Int<1>{}, LDA);
    auto dB = make_stride(Int<1>{}, LDB);
    auto dC = make_stride(Int<1>{}, LDC);

    auto sA = make_layout(make_shape(bM, bK, bP));
    auto sB = make_layout(make_shape(bN, bK, bP));
    auto sC = make_layout(make_shape(bM, bN));
    
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

template<
    bool predicate_reads,
    bool predicate_writes,
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__device__
__forceinline__
void
cute_build_kernel_tn(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    using namespace cute;
    using T = float;

    auto M = u64(m);
    auto N = u64(n);
    auto K = u64(k);

    auto LDA = u64(ldA);
    auto LDB = u64(ldB);
    auto LDC = u64(ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<32>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};

    auto prob_shape = make_shape(M,N,K);
    auto dA = make_stride(LDA, Int<1>{});
    auto dB = make_stride(LDB, Int<1>{});
    auto dC = make_stride(Int<1>{}, LDC);

    auto sA_atom = make_layout(
        make_shape(bN,bK),
        make_stride(Int<1>{}, bN+Int<4>{})
    );

    auto sB_atom = make_layout(
        make_shape(bN,bK),
        make_stride(Int<1>{}, bN+Int<4>{})
    );

    auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(sB_atom, make_shape(bN, bK, bP));
    auto sC = make_layout(make_shape(bM, bN));
    
    auto copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        Layout<Shape<_8,_32>, Stride<_32,_1>>{},
        Layout<Shape< _1,_1>>{}
    );

    auto copyB = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        Layout<Shape<_8,_32>, Stride<_32,_1>>{},
        Layout<Shape< _1,_1>>{}
    );

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

template<
    bool predicate_reads,
    bool predicate_writes,
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__device__
__forceinline__
void
cute_build_kernel_tt(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    using namespace cute;
    using T = float;

    auto M = u64(m);
    auto N = u64(n);
    auto K = u64(k);

    auto LDA = u64(ldA);
    auto LDB = u64(ldB);
    auto LDC = u64(ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<32>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};

    auto prob_shape = make_shape(M,N,K);
    auto dA = make_stride(LDA, Int<1>{});
    auto dB = make_stride(Int<1>{}, LDB);
    auto dC = make_stride(Int<1>{}, LDC);

    auto sA_atom = make_layout(
        make_shape(bM,bK),
        make_stride(Int<1>{}, bM+Int<4>{})
    );

    auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));
    auto sB = make_layout(make_shape(bN, bK, bP));
    auto sC = make_layout(make_shape(bM, bN));
    
    auto copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        Layout<Shape<_8,_32>, Stride<_32,_1>>{},
        Layout<Shape< _1,_1>>{}
    );

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
#if 0
template<
    bool predicate_reads,
    bool predicate_writes,
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
cute_build_kernel_ntt(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    using namespace cute;
    using T = float;

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
    auto dA = make_stride(Int<1>{}, LDA);
    auto dB = make_stride(Int<1>{}, LDB);
    auto dC = make_stride(LDC, Int<1>{});

    auto sA = make_layout(make_shape(bM, bK, bP));
    auto sB = make_layout(make_shape(bN, bK, bP));
    auto sC = make_layout(make_shape(bM, bN));
    
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
#endif
#if 0
    auto dA = [LDA] {
        if constexpr (majorness_A == Majorness::COL_MAJOR) {
            return make_stride(Int<1>{}, LDA); // (dM, dK) : M-major
        } else {
            return make_stride(LDA, Int<1>{}); // (dM, dK) : K-major
        }
    }();

    auto dB = [LDB] {
        if constexpr (majorness_B == Majorness::COL_MAJOR) {
            return make_stride(Int<1>{}, LDB); // (dN, dK) : N-major
        } else {
            return make_stride(LDB, Int<1>{}); // (dN, dK) : K-major
        }
    }();

    // auto dC = make_stride(Int<1>{}, LDC); // (dM, dN) : N-major
    auto dC = make_stride(LDC,Int<1>{}); // (dM, dN) : N-major

    #if 0
    auto dA = make_stride(Int<1>{}, LDA); // (dM, dK) : M-major
    auto dB = make_stride(Int<1>{}, LDB); // (dN, dK) : N-major
    #endif

    
    auto thread_tiler = Layout<Shape<_16,_16>>{}; // M, N

    auto sA_atom = [bM,bK] {
        if constexpr (majorness_A == Majorness::ROW_MAJOR) {
            return make_layout(
                make_shape(bM, bK),
                make_stride(Int<1>{}, bM+Int<4>{}) // Pad by 4 because of bank conflicts
            );
        } else {
            return make_layout(make_shape(bM, bK));
        }
    }();

    auto sB_atom = [bN,bK] {
        if constexpr (majorness_B == Majorness::ROW_MAJOR) {
            return make_layout(
                make_shape(bN,bK),
                make_stride(Int<1>{}, bN+Int<4>{}) // Pad by 4 because of bank conflicts
            );
        } else {
            return make_layout(make_shape(bN, bK));
        }
    }();

    auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(sB_atom, make_shape(bN, bK, bP));
    auto sC = make_layout(make_shape(bM, bN));

    // Copy_Atom<DefaultCopy, float> s2r_atom_A;
    // Copy_Atom<DefaultCopy, float> s2r_atom_B;
    
    // Define the thread layouts (static)
    // Dispatch to align_4 is tensors are aligned by 16 bytes
    auto copyA = [] {
        if constexpr (majorness_A == Majorness::ROW_MAJOR) {
            return make_tiled_copy(
                Copy_Atom<UniversalCopy<T>, T>{},
                Layout<Shape<_8,_32>, Stride<_32,_1>>{},
                Layout<Shape< _1,_1>>{}
            );
        } else if constexpr (align_A == Alignment::ALIGN_4) {
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
        if constexpr (majorness_B == Majorness::ROW_MAJOR) {
            return make_tiled_copy(
                Copy_Atom<UniversalCopy<T>, T>{},
                Layout<Shape<_8,_32>, Stride<_32,_1>>{},
                Layout<Shape< _1,_1>>{}
            );
        } else if constexpr (align_B == Alignment::ALIGN_4) {
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
#endif
