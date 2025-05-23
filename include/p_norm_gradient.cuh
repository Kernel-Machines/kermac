#pragma once

#include <kermac_internal_common.cuh>
#include <cute/tensor.hpp>

#if 0
template <
class SmemLayoutA,
class SmemLayoutB,
class SmemLayoutC,
class T
>
struct SharedStorageNormGradient
{
    alignas(16) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutA>> A;
    alignas(16) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutB>> B;
    alignas(16) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutC>> C;
};
#endif

/// TODO: Need to make UniversalCopy smem to rmem available for different cta sizes
template <
    bool predicate_reads,
    bool predicate_writes,
    NormType norm_type,
    class ProblemShape, class CtaTiler, class ThreadTiler,
    class AStride, class ASmemLayout, class TiledCopyA,
    class BStride, class BSmemLayout, class TiledCopyB,
    class CStride, class CSmemLayout, class TiledCopyC,
    class DStride, class DSmemLayout,
    class EStride, class ESmemLayout,
    class T
>
__device__
__forceinline__
void
kernel_cute_p_norm_kernel_gradient(
    ProblemShape shape_MNOK, CtaTiler cta_tiler, ThreadTiler thread_tiler,
    T const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    T const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    T const *C, CStride dC, CSmemLayout sC_layout, TiledCopyC copy_c,
    T const *D, DStride dD, DSmemLayout sD_layout,
    T       *E, EStride dE, ESmemLayout,
    T p_power
) {
    static_assert(norm_type == NormType::L1 || norm_type == NormType::L2 || norm_type == NormType::P);

    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNOK) == Int<4>{}); // (M, N, O, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<4>{}); // (BLK_M, BLK_N, BLK_O, BLK_K)
    CUTE_STATIC_ASSERT_V(rank(thread_tiler) == Int<3>{}); // (THR_M, THR_N, THR_O)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(thread_tiler)); // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(thread_tiler)); // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_c) == size(thread_tiler)); // NumThreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);
    static_assert(is_static<DSmemLayout>::value);
    static_assert(is_static<ESmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(DSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(ESmemLayout{}) == size<0>(cta_tiler));  // BLK_M

    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(DSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ESmemLayout{}) == size<1>(cta_tiler));  // BLK_N

    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<2>(cta_tiler));  // BLK_O
    CUTE_STATIC_ASSERT_V(size<2>(ESmemLayout{}) == size<2>(cta_tiler));  // BLK_O

    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<3>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<3>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<3>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0,3>(shape_MNOK), dA));         // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,3>(shape_MNOK), dB));         // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<2,3>(shape_MNOK), dC));         // dC strides for shape OK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNOK), dD));         // dD strides for shape MN
    CUTE_STATIC_ASSERT_V(congruent(select<0,1,2>(shape_MNOK), dE));       // dE strides for shape MNO

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,3>(shape_MNOK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,3>(shape_MNOK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<2,3>(shape_MNOK), dC); // (O,K)
    Tensor mD = make_tensor(make_gmem_ptr(D), select<0,1>(shape_MNOK), dD); // (M,N)
    Tensor mE = make_tensor(make_gmem_ptr(E), select<0,1,2>(shape_MNOK), dE); // (M,N,O)

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, blockIdx.z, _); // (m,n,o,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1,  X,  X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1,  X, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step< X,  X, _1, _1>{}); // (BLK_O,BLK_K,k)
    Tensor gD = local_tile(mD, cta_tiler, cta_coord, Step<_1, _1,  X,  X>{}); // (BLK_M,BLK_N)
    Tensor gE = local_tile(mE, cta_tiler, cta_coord, Step<_1, _1, _1,  X>{}); // (BLK_M,BLK_N,BLK_O)

    auto m_max_coord = size<0>(shape_MNOK) - size<0>(gA) * blockIdx.x;  // M - BLK_M * m_coord
    auto n_max_coord = size<1>(shape_MNOK) - size<0>(gB) * blockIdx.y;  // N - BLK_N * n_coord
    auto o_max_coord = size<2>(shape_MNOK) - size<0>(gC) * blockIdx.z;  // O - BLK_O * o_coord
    auto k_residue   = size<3>(shape_MNOK) - size<1>(gA) * size<2>(gA); // K - BLK_K * k_coord_max

    // Need to get the tile count before the offsetting in gA, gB, gC of the k_residue
    int k_tile_count = 0;
    {
        ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
        Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
        // Total count of tiles
        k_tile_count = size<3>(tAgA);
    }

    if constexpr (predicate_reads) {
        // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
        // This aligns the tensor with BLK_K for all but the 0th k_tile
        gA = cute::domain_offset(make_coord(0, k_residue, 0), gA);
        gB = cute::domain_offset(make_coord(0, k_residue, 0), gB);
        gC = cute::domain_offset(make_coord(0, k_residue, 0), gC);
    }
    
#if 1
    alignas(16) __shared__ T smem_a[cosize_v<ASmemLayout>];
    alignas(16) __shared__ T smem_b[cosize_v<BSmemLayout>];
    alignas(16) __shared__ T smem_c[cosize_v<CSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smem_a), sA_layout);   // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem_b), sB_layout);   // (BLK_N,BLK_K,PIPE)
    Tensor sC = make_tensor(make_smem_ptr(smem_c), sC_layout);   // (BLK_N,BLK_K,PIPE)
#else
    // Shared memory buffers
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorageNormGradient<ASmemLayout, BSmemLayout, CSmemLayout, T>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)
    Tensor sC = make_tensor(make_smem_ptr(smem.C.begin()), sC_layout);   // (BLK_O,BLK_K,PIPE)
#endif

    // Tiled copy setups
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    ThrCopy thr_copy_c = copy_c.get_slice(threadIdx.x);
    Tensor tCgC = thr_copy_c.partition_S(gC); // (CPY,CPY_O,CPY_K,k)
    Tensor tCsC = thr_copy_c.partition_D(sC); // (CPY,CPY_O,CPY_K,PIPE)

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
    Tensor tEsA = local_partition(sA, thread_tiler, threadIdx.x, Step<_1,  X,  X>{}); // (THR_M,THR_K,PIPE)
    Tensor tEsB = local_partition(sB, thread_tiler, threadIdx.x, Step< X, _1,  X>{}); // (THR_N,THR_K,PIPE)
    Tensor tEsC = local_partition(sC, thread_tiler, threadIdx.x, Step< X,  X, _1>{}); // (THR_O,THR_K,PIPE)
    Tensor tEgD = local_partition(gD, thread_tiler, threadIdx.x, Step<_1, _1,  X>{}); // (THR_M,THR_N)
    Tensor tEgE = local_partition(gE, thread_tiler, threadIdx.x, Step<_1, _1, _1>{}); // (THR_M,THR_N,THR_O)
    
    Tensor tErA = make_fragment_like(tEsA(_,_,0)); // (THR_M,THR_K)
    Tensor tErB = make_fragment_like(tEsB(_,_,0)); // (THR_N,THR_K)
    Tensor tErC = make_fragment_like(tEsC(_,_,0)); // (THR_O,THR_K)
    Tensor tErD = make_fragment_like(tEgD);        // (THR_M,THR_N)
    Tensor tErE = make_fragment_like(tEgE);        // (THR_M,THR_N,THR_O)

    // Create coordinate tensors for the problem for predication
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (M,K) -> (m,k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (N,K) -> (n,k)
    Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC))); // (O,K) -> (n,k)
    Tensor cD = make_identity_tensor(make_shape(size<0>(gD), size<1>(gD))); // (M,N) -> (m,n)
    Tensor cE = make_identity_tensor(make_shape(size<0>(gE), size<1>(gE), size<2>(gE))); // (M,N,O) -> (m,n,o)
    
    // Partition coordinate tensors for predication
    Tensor tAcA = thr_copy_a.partition_S(cA);
    Tensor tBcB = thr_copy_b.partition_S(cB);
    Tensor tCcC = thr_copy_c.partition_S(cC);
    Tensor tEcD = local_partition(cD, thread_tiler, threadIdx.x, Step<_1, _1,  X>{});
    Tensor tEcE = local_partition(cE, thread_tiler, threadIdx.x, Step<_1, _1, _1>{});

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX-1;

    // Pipe slice
    Tensor tEsA_p = tEsA(_,_,smem_pipe_read);
    Tensor tEsB_p = tEsB(_,_,smem_pipe_read);
    Tensor tEsC_p = tEsC(_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<1>(tErA);

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
    Tensor tEpD = make_tensor<bool>(
        make_shape(size<0>(tEcD), size<1>(tEcD))
    );

    if constexpr (predicate_reads) {
        // Generate the in-bounds/out-of-bounds coordinates for each tensor as a bool predicate
        CUTE_UNROLL
        for (int m = 0; m < size<0>(tApA); m++) {
            tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
        }
        CUTE_UNROLL
        for (int n = 0; n < size<0>(tBpB); n++) {
            tBpB(n,0) = get<0>(tBcB(0,n,0)) < n_max_coord;
        }
        CUTE_UNROLL
        for (int o = 0; o < size<0>(tCpC); o++) {
            tCpC(o,0) = get<0>(tCcC(0,o,0)) < o_max_coord;
        }
        CUTE_UNROLL
        for (int i = 0; i < size(tEcD); i++) {
            tEpD(i) = elem_less(tEcD(i), make_coord(m_max_coord,n_max_coord));
        }
    }

    // Print all tensor shapes/data here before anything functionally happens such as copies

    if constexpr (predicate_reads) {
        // Clear the smem tiles to account for predicated off loads 
        clear(tAsA);
        clear(tBsB);
        clear(tCsC);

        // Start async loads for 0th k-tile, where we take care of the k-residue
        // We already shifted the global memory coordinate over to account for the k-residue
        {
            constexpr int k_pipe = 0;
    
            Tensor tAgAk = tAgA(_,_,_,k_tile_next);
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < size<2>(tAsA); ++k) {
                if (get<1>(tAcA(0,0,k)) >= -k_residue) { // blk_k coord < residue_k (gA shifted)
                    copy_if(copy_a, tApA(_,k), tAgAk(_,_,k), tAsA(_,_,k,k_pipe));
                }
            }
            Tensor tBgBk = tBgB(_,_,_,k_tile_next);
            for (int k = 0; k < size<2>(tBsB); ++k) {
                if (get<1>(tBcB(0,0,k)) >= -k_residue) { // blk_k coord < residue_k (gB shifted)
                    copy_if(copy_b, tBpB(_,k), tBgBk(_,_,k), tBsB(_,_,k,k_pipe));
                }
            }
            Tensor tCgCk = tCgC(_,_,_,k_tile_next);
            for (int k = 0; k < size<2>(tCsC); ++k) {
                if (get<1>(tCcC(0,0,k)) >= -k_residue) { // blk_k coord < residue_k (gC shifted)
                    copy_if(copy_c, tCpC(_,k), tCgCk(_,_,k), tCsC(_,_,k,k_pipe));
                }
            }
            cp_async_fence();
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
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }
        }
        
        // Copy D direct gmem to rmem. It stays static for whole mainloop.
        // The bigger this is the more work it generates to excuse gmem latency.
        copy_if(tEpD, tEgD, tErD);
    } else { // No predicated reads
        // Start async loads for 0th k-tile onwards, no k-residue handling needed
        // Do this for all but the last pipe. Each mainloop iter will schedule a copy.
        CUTE_UNROLL
        for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
            copy(copy_c, tCgC(_,_,_,k_tile_next), tCsC(_,_,_,k_pipe));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }
        }

        // Copy D direct gmem to rmem. It stays static for whole mainloop.
        // The bigger this is the more work it generates to excuse gmem latency.
        copy(tEgD, tErD);
    }

    // Clear accumulators
    clear(tErE);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
    
        // Prefetch the first rmem from the first k-tile
        copy(tEsA_p(_,Int<0>{}), tErA(_,Int<0>{}));
        copy(tEsB_p(_,Int<0>{}), tErB(_,Int<0>{}));
        copy(tEsC_p(_,Int<0>{}), tErC(_,Int<0>{}));
    }

    // Main LOOP!
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX-1)) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                // Slice the smem_pipe_read smem
                tEsA_p = tEsA(_,_,smem_pipe_read);
                tEsB_p = tEsB(_,_,smem_pipe_read);
                tEsC_p = tEsC(_,_,smem_pipe_read);

                // Commit the smem for smem_pipe_read
                cp_async_wait<K_PIPE_MAX-2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
            copy(tEsA_p(_,k_block_next), tErA(_,k_block_next));
            copy(tEsB_p(_,k_block_next), tErB(_,k_block_next));
            copy(tEsC_p(_,k_block_next), tErC(_,k_block_next));
            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0) {
                if constexpr (predicate_reads) {
                    // Set all predicates to false if we are going to overshoot bounds
                    if (k_tile_count <= 0) {
                        clear(tApA);
                        clear(tBpB);
                        clear(tCpC);
                    }
                    copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
                    copy_if(copy_b, tBpB, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
                    copy_if(copy_c, tCpC, tCgC(_,_,_,k_tile_next), tCsC(_,_,_,smem_pipe_write));
                } else { // No predicated reads
                    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
                    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
                    copy(copy_c, tCgC(_,_,_,k_tile_next), tCsC(_,_,_,smem_pipe_write));
                }
                cp_async_fence();

                // Advance the gmem tile
                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile_next; }

                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
            }

            // Apply c(o,k) * a(m,k) * (d(m,n) - b(n,k))
            // Can apply fractional powers here
            // Trick here is that diff stays constant while looping over C.
            // If expensive fractional power is applied to diff, this result can be 
            // reused in the core for each of the O accumulator for C.
            CUTE_UNROLL
            for (int m = 0; m < size<0>(tErE); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(tErE); n++) {
                    T diff = tErD(m,n) - tErB(n,k_block);
                    T sign = _signum(diff);
                    if constexpr (norm_type == NormType::L1) {
                        diff = sign;
                    } else if constexpr (norm_type == NormType::L2) {
                        diff = diff;
                    } else {
                        diff = _abs(diff);
                        diff = _pow(diff, p_power);
                        diff = diff * sign;
                    }
                    diff = tErA(m,k_block) * diff;
                    CUTE_UNROLL
                    for (int o = 0; o < size<2>(tErE); o++) {
                        tErE(m,n,o) += tErC(o,k_block) * diff;
                    }
                }
            }
        }
    }

    // Write accumulators
    if constexpr (predicate_writes) {
        CUTE_UNROLL
        for (int i = 0; i < size(tErE); i++) {
            if (elem_less(tEcE(i), make_coord(m_max_coord,n_max_coord,o_max_coord))) {
                tEgE(i) = tErE(i);
            }
        }
    } else {
        CUTE_UNROLL
        for (int i = 0; i < size(tErE); i++) {
            tEgE(i) = tErE(i);
        }
    }
}
