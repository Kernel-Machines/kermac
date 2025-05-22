#pragma once

#pragma once

#include <kermac_internal_common.cuh>
#include <cute/tensor.hpp>

#if 0
template <
class SmemLayoutA,
class SmemLayoutB,
class T
>
struct SharedStorageNorm
{
    alignas(16) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutA>> A;
    alignas(16) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutB>> B;
};
#endif

template <
    bool predicate_reads,
    bool predicate_writes,
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    class ProblemShape, class CtaTiler, class ThreadTiler,
    class AStride, class ASmemLayout, class TiledCopyA,
    class BStride, class BSmemLayout, class TiledCopyB,
    class CStride, class CSmemLayout,
    class T
>
__device__
__forceinline__
void
kernel_cute_build_kernel(
    ProblemShape shape_MNK, CtaTiler cta_tiler, ThreadTiler thread_tiler,
    T const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    T const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    T       *C, CStride dC, CSmemLayout,
    T p_power_inner, 
    T p_power_outer,
    T bandwidth
) {
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)
    CUTE_STATIC_ASSERT_V(rank(thread_tiler) == Int<2>{}); // (THR_M, THR_N)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(thread_tiler)); // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(thread_tiler)); // NumThreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M

    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N

    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));       // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));       // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));       // dC strides for shape MNO

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1,  X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1,  X>{}); // (BLK_M,BLK_N)

    auto m_max_coord = size<0>(shape_MNK) - size<0>(gA) * blockIdx.x;  // M - BLK_M * m_coord
    auto n_max_coord = size<1>(shape_MNK) - size<0>(gB) * blockIdx.y;  // N - BLK_N * n_coord
    auto k_residue   = size<2>(shape_MNK) - size<1>(gA) * size<2>(gA); // K - BLK_K * k_coord_max

    // Need to get the tile count before the offsetting in gA, gB of the k_residue
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
    }

#if 1
    alignas(16) __shared__ T smem_a[cosize_v<ASmemLayout>];
    alignas(16) __shared__ T smem_b[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smem_a), sA_layout);   // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem_b), sB_layout);   // (BLK_N,BLK_K,PIPE)

#else
    // Shared memory buffers
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorageNorm<ASmemLayout, BSmemLayout, T>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)
#endif

    // Tiled copy setups
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    // Pipe size
    auto K_PIPE_MAX = size<3>(tAsA);
    // Current tile index in gmem to read from
    int k_tile_next = 0;
    
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K
    CUTE_STATIC_ASSERT_V(K_PIPE_MAX == size<3>(tAsA)); // PIPE A
    CUTE_STATIC_ASSERT_V(K_PIPE_MAX == size<3>(tBsB)); // PIPE B

    // Partition the tensors
    Tensor tCsA = local_partition(sA, thread_tiler, threadIdx.x, Step<_1,  X>{}); // (THR_M,THR_K,PIPE)
    Tensor tCsB = local_partition(sB, thread_tiler, threadIdx.x, Step< X, _1>{}); // (THR_N,THR_K,PIPE)
    Tensor tCgC = local_partition(gC, thread_tiler, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)

    Tensor tCrA = make_fragment_like(tCsA(_,_,0)); // (THR_M,THR_K)
    Tensor tCrB = make_fragment_like(tCsB(_,_,0)); // (THR_N,THR_K)
    Tensor tCrC = make_fragment_like(tCgC);        // (THR_M,THR_N)

    // Create coordinate tensors for the problem for predication
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (M,K) -> (m,k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (N,K) -> (n,k)
    Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC))); // (M,N) -> (m,n)

    // Partition coordinate tensors for predication
    Tensor tAcA = thr_copy_a.partition_S(cA);
    Tensor tBcB = thr_copy_b.partition_S(cB);
    Tensor tCcC = local_partition(cC, thread_tiler, threadIdx.x, Step<_1, _1>{});

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX-1;

    // Pipe slice
    Tensor tCsA_p = tCsA(_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<1>(tCrA);

    Tensor tApA = make_tensor<bool>(
        make_shape(size<1>(tAsA), size<2>(tAsA)),
        make_stride(Int<1>{}, Int<0>{})
    );
    Tensor tBpB = make_tensor<bool>(
        make_shape(size<1>(tBsB), size<2>(tBsB)),
        make_stride(Int<1>{}, Int<0>{})
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
    }

    // Print all tensor shapes/data here before anything functionally happens such as copies

    if constexpr (predicate_reads) {
        // Clear the smem tiles to account for predicated off loads 
        clear(tAsA);
        clear(tBsB);

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
            }
            copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
            copy_if(copy_b, tBpB, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }
        }
    } else { // No predicated reads
        // Start async loads for 0th k-tile onwards, no k-residue handling needed
        // Do this for all but the last pipe. Each mainloop iter will schedule a copy.
        CUTE_UNROLL
        for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }
        }
    }

    // Clear accumulators
    clear(tCrC);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
    
        // Prefetch the first rmem from the first k-tile
        copy(tCsA_p(_,Int<0>{}), tCrA(_,Int<0>{}));
        copy(tCsB_p(_,Int<0>{}), tCrB(_,Int<0>{}));
    }

    // Main LOOP!
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX-1)) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                // Slice the smem_pipe_read smem
                tCsA_p = tCsA(_,_,smem_pipe_read);
                tCsB_p = tCsB(_,_,smem_pipe_read);

                // Commit the smem for smem_pipe_read
                cp_async_wait<K_PIPE_MAX-2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
            copy(tCsA_p(_,k_block_next), tCrA(_,k_block_next));
            copy(tCsB_p(_,k_block_next), tCrB(_,k_block_next));
            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0) {
                if constexpr (predicate_reads) {
                    // Set all predicates to false if we are going to overshoot bounds
                    if (k_tile_count <= 0) {
                        clear(tApA);
                        clear(tBpB);
                    }
                    copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
                    copy_if(copy_b, tBpB, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
                } else { // No predicated reads
                    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
                    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
                }
                cp_async_fence();

                // Advance the gmem tile
                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile_next; }

                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
            }

            // Apply inner
            CUTE_UNROLL
            for (int m = 0; m < size<0>(tCrC); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(tCrC); n++) {
                    T diff;
                    if constexpr (inner_operator == InnerOperator::DIFF) {
                        diff = tCrB(n,k_block) - tCrA(m,k_block);
                    } else if constexpr (inner_operator == InnerOperator::DOT) {
                        diff = tCrB(n,k_block) * tCrA(m,k_block);
                    } else {
                        diff = _nan<T>();
                    }

                    if constexpr (inner_power == PowerType::NOOP) {
                        diff = diff;
                    } else if constexpr (inner_power == PowerType::ABS) {
                        diff = _abs(diff);
                    } else if constexpr (inner_power == PowerType::SQUARE) {
                        diff = diff * diff;
                    } else if constexpr (inner_power == PowerType::SQRT) {
                        diff = _sqrt(diff);
                    } else if constexpr (inner_power == PowerType::POW) {
                        diff = _abs(diff);
                        diff = _pow(diff, p_power_inner);
                    } else {
                        diff = _nan<T>();
                    }

                    tCrC(m,n) += diff;
                }
            }
        }
    }

    // Apply outer
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); i++) {
        T accum = tCrC(i);
        if constexpr (outer_power == PowerType::NOOP) {
            accum = accum;
        } else if constexpr (outer_power == PowerType::ABS) {
            accum = _abs(accum);
        } else if constexpr (outer_power == PowerType::SQUARE) {
            accum = accum * accum;
        } else if constexpr (outer_power == PowerType::SQRT) {
            accum = _sqrt(accum);
        } else if constexpr (outer_power == PowerType::POW) {
            accum = _pow(accum, p_power_outer);    
        } else {
            accum = _nan<T>();
        }
        tCrC(i) = accum;
    }

    // Apply Kernel
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); i++) {
        T accum = tCrC(i);
        if constexpr (kernel_type == KernelType::NONE) {
            accum = accum;
        } else if constexpr (kernel_type == KernelType::LAPLACE) {
            ///TODO: can either pass this gamma in or do it at the time after the inflight memory requests.
            // Division is expensive, but for batch mode with bandwidth it should stay bandwidth and not gamma.
            //
            T gamma = T(1.0) / bandwidth;
            accum = accum * -gamma;
            accum = _exp(accum);
        } else if constexpr (kernel_type == KernelType::GAUSSIAN) {
            T gamma = T(1.0) / (2 * bandwidth * bandwidth);
            accum = accum * -gamma;
            accum = _exp(accum);
        } else {
            accum = _nan<T>();
        }
        tCrC(i) = accum;
    }

    // Write accumulators
    if constexpr (predicate_writes) {
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); i++) {
            if (elem_less(tCcC(i), make_coord(m_max_coord,n_max_coord))) {
                tCgC(i) = tCrC(i);
            }
        }
    } else {
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); i++) {
            tCgC(i) = tCrC(i);
        }
    }
}
