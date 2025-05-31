#pragma once

#include <cute/tensor.hpp>

template <
    class ProblemShape, class CtaTiler, class ThreadTiler,
    class AStride, class ASmemLayout, class TiledCopyA,
    class BStride, class BSmemLayout, class TiledCopyB,
    class CStride, class CSmemLayout, class TiledCopyC,
    class DStride, class DSmemLayout,
    class T
>
__device__
__forceinline__
void
kernel_cute_scaled_gemm(
    ProblemShape shape_MNOK, CtaTiler cta_tiler, ThreadTiler thread_tiler,
    T const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    T const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    T const *C, CStride dC, CSmemLayout sC_layout, TiledCopyC copy_c,
    T       *D, DStride dD, DSmemLayout
) {
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNOK) == Int<4>{}); // (M, N, O, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<4>{}); // (BLK_M, BLK_N, BLK_O, BLK_K)
    CUTE_STATIC_ASSERT_V(rank(thread_tiler) == Int<3>{}); // (THR_M, THR_N, THR_O)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(thread_tiler)); // NumThreads
    // CUTE_STATIC_ASSERT_V(size(copy_b) == size(thread_tiler)); // NumThreads
    // CUTE_STATIC_ASSERT_V(size(copy_c) == size(thread_tiler)); // NumThreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);
    static_assert(is_static<DSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(DSmemLayout{}) == size<0>(cta_tiler));  // BLK_M

    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(DSmemLayout{}) == size<1>(cta_tiler));  // BLK_N

    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<2>(cta_tiler));  // BLK_O
    CUTE_STATIC_ASSERT_V(size<2>(DSmemLayout{}) == size<2>(cta_tiler));  // BLK_O

    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<3>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<3>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<3>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0,3>(shape_MNOK), dA));       // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,3>(shape_MNOK), dB));       // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<2,3>(shape_MNOK), dC));       // dC strides for shape OK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1,2>(shape_MNOK), dD));       // dD strides for shape MNO

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,3>(shape_MNOK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,3>(shape_MNOK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<2,3>(shape_MNOK), dC); // (N,K)
    Tensor mD = make_tensor(make_gmem_ptr(D), select<0,1,2>(shape_MNOK), dD); // (M,N)

    auto bidx = blockIdx.x;
    auto bidy = blockIdx.y;
    auto bidz = blockIdx.z;

    auto cta_coord = make_coord(bidx, bidy, bidz, _); // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1,  X,  X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1,  X, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step< X,  X, _1, _1>{}); // (BLK_O,BLK_K,k)
    Tensor gD = local_tile(mD, cta_tiler, cta_coord, Step<_1, _1, _1,  X>{}); // (BLK_M,BLK_N,BLK_O)

    auto m_max_coord = size<0>(shape_MNOK) - size<0>(gA) * bidx;  // M - BLK_M * m_coord
    auto n_max_coord = size<1>(shape_MNOK) - size<0>(gB) * bidy;  // N - BLK_N * n_coord
    auto o_max_coord = size<2>(shape_MNOK) - size<0>(gC) * bidz;  // N - BLK_N * n_coord
    auto k_residue   = size<3>(shape_MNOK) - size<1>(gA) * size<2>(gA); // K - BLK_K * k_coord_max

    // Need to get the tile count before the offsetting in gA, gB of the k_residue
    int k_tile_count = 0;
    {
        ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
        Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
        // Total count of tiles
        k_tile_count = size<3>(tAgA);
    }

    // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
    // This aligns the tensor with BLK_K for all but the 0th k_tile
    gA = cute::domain_offset(make_coord(0, k_residue, 0), gA);
    gB = cute::domain_offset(make_coord(0, k_residue, 0), gB);
    gC = cute::domain_offset(make_coord(0, k_residue, 0), gC);

    __shared__ T smem_a[cosize_v<ASmemLayout>];
    __shared__ T smem_b[cosize_v<BSmemLayout>];
    __shared__ T smem_c[cosize_v<CSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smem_a), sA_layout);   // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem_b), sB_layout);   // (BLK_N,BLK_K,PIPE)
    Tensor sC = make_tensor(make_smem_ptr(smem_c), sC_layout);   // (BLK_N,BLK_K,PIPE)

    // Tiled copy setups
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    ThrCopy thr_copy_c = copy_c.get_slice(threadIdx.x);
    Tensor tCgC = thr_copy_c.partition_S(gC); // (CPY,CPY_N,CPY_K,k)
    Tensor tCsC = thr_copy_c.partition_D(sC); // (CPY,CPY_N,CPY_K,PIPE)

    // Pipe size
    auto K_PIPE_MAX = size<3>(tAsA);
    // Current tile index in gmem to read from
    int k_tile_next = 0;
    
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsC)); // CPY_O
    CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<2>(tCsC)); // CPY_K
    CUTE_STATIC_ASSERT_V(K_PIPE_MAX == size<3>(tAsA)); // PIPE A
    CUTE_STATIC_ASSERT_V(K_PIPE_MAX == size<3>(tBsB)); // PIPE B
    CUTE_STATIC_ASSERT_V(K_PIPE_MAX == size<3>(tCsC)); // PIPE C

    // Partition the tensors
    Tensor tDsA = local_partition(sA, thread_tiler, threadIdx.x, Step<_1,  X,  X>{}); // (THR_M,THR_K,PIPE)
    Tensor tDsB = local_partition(sB, thread_tiler, threadIdx.x, Step< X, _1,  X>{}); // (THR_N,THR_K,PIPE)
    Tensor tDsC = local_partition(sC, thread_tiler, threadIdx.x, Step< X,  X, _1>{}); // (THR_N,THR_K,PIPE)
    Tensor tDgD = local_partition(gD, thread_tiler, threadIdx.x, Step<_1, _1, _1>{}); // (THR_M,THR_N)

    Tensor tDrA = make_fragment_like(tDsA(_,_,0)); // (THR_M,THR_K)
    Tensor tDrB = make_fragment_like(tDsB(_,_,0)); // (THR_N,THR_K)
    Tensor tDrC = make_fragment_like(tDsC(_,_,0)); // (THR_O,THR_K)
    Tensor tDrD = make_fragment_like(tDgD);        // (THR_M,THR_N)

    // Create coordinate tensors for the problem for predication
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (M,K) -> (m,k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (N,K) -> (n,k)
    Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC))); // (O,K) -> (o,k)
    Tensor cD = make_identity_tensor(make_shape(size<0>(gD), size<1>(gD), size<2>(gD))); // (M,N,O) -> (m,n,o)

    // Partition coordinate tensors for predication
    Tensor tAcA = thr_copy_a.partition_S(cA);
    Tensor tBcB = thr_copy_b.partition_S(cB);
    Tensor tCcC = thr_copy_c.partition_S(cC);
    Tensor tDcD = local_partition(cD, thread_tiler, threadIdx.x, Step<_1, _1, _1>{});

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX-1;

    // Pipe slice
    Tensor tDsA_p = tDsA(_,_,smem_pipe_read);
    Tensor tDsB_p = tDsB(_,_,smem_pipe_read);
    Tensor tDsC_p = tDsC(_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<1>(tDrA);

    Tensor tApA = make_tensor<bool>(
        make_shape(size<1>(tAsA), size<2>(tAsA)),
        make_stride(Int<1>{}, Int<0>{})
    );
    Tensor tBpB = make_tensor<bool>(
        make_shape(size<1>(tBsB), size<2>(tBsB)),
        make_stride(Int<1>{}, Int<0>{})
    );
    Tensor tCpC = make_tensor<bool>(
        make_shape(size<1>(tCsC), size<2>(tCsC)),
        make_stride(Int<1>{}, Int<0>{})
    );

    // Generate the in-bounds/out-of-bounds coordinates for each tensor as a bool predicate
    CUTE_UNROLL
    for (int m = 0; m < size<0>(tApA); m++) {
        tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
    }
    CUTE_UNROLL
    for (int n = 0; n < size<0>(tBpB); n++) {
        tBpB(n,0) = get<0>(tBcB(0,n,0)) < n_max_coord && threadIdx.x < size(copy_b);
    }
    CUTE_UNROLL
    for (int o = 0; o < size<0>(tCpC); o++) {
        tCpC(o,0) = get<0>(tCcC(0,o,0)) < o_max_coord && threadIdx.x < size(copy_c);
    }

    // Print all tensor shapes/data here before anything functionally happens such as copies

    // Clear the smem tiles to account for predicated off loads 
    clear(tAsA);
    clear(tBsB);
    clear(tCsC);

    // Start async loads for 0th k-tile, where we take care of the k-residue
    // We already shifted the global memory coordinate over to account for the k-residue
    {
        constexpr int k_pipe = 0;

        Tensor tAgAk = tAgA(_,_,_,k_tile_next);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(tAsA); ++k) {
            if (get<1>(tAcA(0,0,k)) >= -k_residue) { // blk_k coord < residue_k (gA shifted)
                copy_if(copy_a, tApA(_,k), tAgAk(_,_,k), tAsA(_,_,k,k_pipe));
            }
        }
        Tensor tBgBk = tBgB(_,_,_,k_tile_next);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(tBsB); ++k) {
            if (get<1>(tBcB(0,0,k)) >= -k_residue) { // blk_k coord < residue_k (gB shifted)
                copy_if(copy_b, tBpB(_,k), tBgBk(_,_,k), tBsB(_,_,k,k_pipe));
            }
        }
        Tensor tCgCk = tCgC(_,_,_,k_tile_next);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(tCsC); ++k) {
            if (get<1>(tCcC(0,0,k)) >= -k_residue) { // blk_k coord < residue_k (gC shifted)
                copy_if(copy_c, tCpC(_,k), tCgCk(_,_,k), tCsC(_,_,k,k_pipe));
            }
        }
        cp_async_fence();
        __syncthreads();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    // Start async loads for 1st k-tile onwards, no k-residue handling needed
    // Do this for all but the last pipe. Each mainloop iter will schedule a pipeline copy.
    CUTE_UNROLL
    for (int k_pipe = 1; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
        if (k_tile_count <= 0) {
            clear(tApA);
            clear(tBpB);
            clear(tCpC);
        }
        copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        copy_if(copy_b, tBpB, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        copy_if(copy_c, tCpC, tCgC(_,_,_,k_tile_next), tCsC(_,_,_,k_pipe));
        cp_async_fence();
        __syncthreads();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    // Clear accumulators
    clear(tDrD);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
    
        // Prefetch the first rmem from the first k-tile
        copy(tDsA_p(_,Int<0>{}), tDrA(_,Int<0>{}));
        copy(tDsB_p(_,Int<0>{}), tDrB(_,Int<0>{}));
        copy(tDsC_p(_,Int<0>{}), tDrC(_,Int<0>{}));
    }

    // Main LOOP!
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX-1)) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                // Slice the smem_pipe_read smem
                tDsA_p = tDsA(_,_,smem_pipe_read);
                tDsB_p = tDsB(_,_,smem_pipe_read);
                tDsC_p = tDsC(_,_,smem_pipe_read);

                // Commit the smem for smem_pipe_read
                cp_async_wait<K_PIPE_MAX-2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
            copy(tDsA_p(_,k_block_next), tDrA(_,k_block_next));
            copy(tDsB_p(_,k_block_next), tDrB(_,k_block_next));
            copy(tDsC_p(_,k_block_next), tDrC(_,k_block_next));
            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0) {
                // Set all predicates to false if we are going to overshoot bounds
                if (k_tile_count <= 0) {
                    clear(tApA);
                    clear(tBpB);
                    clear(tCpC);
                }
                copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
                copy_if(copy_b, tBpB, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
                copy_if(copy_c, tCpC, tCgC(_,_,_,k_tile_next), tCsC(_,_,_,smem_pipe_write));

                cp_async_fence();
                __syncthreads();

                // Advance the gmem tile
                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile_next; }

                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
            }

            CUTE_UNROLL
            for (int m = 0; m < size<0>(tDrD); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(tDrD); n++) {
                    CUTE_UNROLL
                    for (int o = 0; o < size<2>(tDrD); o++) {
                        T v = tDrC(o,k_block) * tDrB(n,k_block) * tDrA(m,k_block);
                        tDrD(m,n,o) += v;
                    }
                }
            }
        }
    }

    // Write accumulators
    CUTE_UNROLL
    for (int i = 0; i < size(tDrD); i++) {
        if (elem_less(tDcD(i), make_coord(m_max_coord,n_max_coord,o_max_coord))) {
            tDgD(i) = tDrD(i);
        }
    }
}

template<
    int cta_m,
    int cta_n,
    int cta_o
>
__global__
__launch_bounds__(256)
void
cute_scaled_gemm(
    int m, int n, int o, int k,
    float const *A, size_t ldA,
    float const *B, size_t ldB,
    float const *C, size_t ldC,
    float       *D, size_t ldD_N, size_t ldD_O
) {
    static_assert(cta_m == 32 || cta_m == 64 || cta_m == 128);
    static_assert(cta_n == 16 || cta_n == 32 || cta_n == 64 || cta_n == 128);
    static_assert(cta_o == 1 || cta_o == 2 || cta_o == 4 || cta_o == 8 || cta_o == 16 || cta_o == 32);

    using namespace cute;
    using T = float;

    auto M = size_t(m);
    auto N = size_t(n);
    auto O = size_t(o);
    auto K = size_t(k);

    auto bM = Int<cta_m>{};
    auto bN = Int<cta_n>{};
    auto bO = Int<cta_o>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bO, bK);
    auto bP = Int<3>{};

    auto prob_shape = make_shape(M,N,O,K);
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(ldC, Int<1>{});
    auto dD = make_stride(Int<1>{}, ldD_N, ldD_O);

    auto sA_atom = make_layout(
        make_shape(bM, bK),
        make_stride(Int<1>{}, bM+Int<4>{})
    );
    auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));

    auto sB_atom = make_layout(
        make_shape(bN, bK),
        make_stride(Int<1>{}, bN+Int<4>{})
    );
    auto sB = tile_to_shape(sB_atom, make_shape(bN, bK, bP));

    auto o_stride = [bO,bK] {
        if constexpr(cta_o == 32 || cta_o == 16) {
            return make_stride(Int<1>{}, bO+Int<4>{});
        } else if constexpr (cta_o == 8) { // ?
            return make_stride(Int<8>{}, Int<1>{});
        } else if constexpr (cta_o == 4) { // ?
            return make_stride(Int<8>{}, Int<1>{});
        } else if constexpr (cta_o == 2 || cta_o == 1) {
            return make_stride(Int<8>{}, Int<1>{});
        }
    }();
    

    auto sC_atom = make_layout(
        make_shape(bO, bK), o_stride
    );
    auto sC = tile_to_shape(sC_atom, make_shape(bO, bK, bP));

    auto sD = make_layout(make_shape(bM, bN, bO));

    auto copyA = 
        make_tiled_copy(
            Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<T>, T>{},
            Layout<Shape<_32,_8>, Stride<_8,_1>>{},
            Layout<Shape<_1,_1>>{}
        );
    auto copyB = [] {
        if constexpr (cta_n >= 32) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<T>, T>{},
                Layout<Shape<_32,_8>, Stride<_8,_1>>{},
                Layout<Shape<_1,_1>>{}
            );
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<T>, T>{},
                Layout<Shape<decltype(bN),_8>, Stride<_8,_1>>{},
                Layout<Shape<_1,_1>>{}
            );
        }
    }();

    auto copyC =
        make_tiled_copy(
            Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<T>, T>{},
            Layout<Shape<decltype(bO),_8>, Stride<_8,_1>>{},
            Layout<Shape<_1,_1>>{}
        );

    auto thread_tiler = Layout<Shape<_16,_16,_1>>{}; // M, N, O

    // if (thread0()) {
    //     print(cta_m); print("\n");
    //     print(cta_n); print("\n");
    //     print(cta_o); print("\n");
    //     print(size(copyB)); print("\n");
    //     print(size(copyC)); print("\n");
    // }

    kernel_cute_scaled_gemm(
        prob_shape, cta_tiler, thread_tiler, 
        A, dA, sA, copyA,
        B, dB, sB, copyB,
        C, dC, sC, copyC,
        D, dD, sD
    );
}
