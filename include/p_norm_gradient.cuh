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
    NormType norm_type,
    class ProblemShape, class CtaTiler, class ThreadTiler,
    class AStride, class ASmemLayout, class TiledCopyA,
    class BStride, class BSmemLayout, class TiledCopyB,
    class CStride, class CSmemLayout, class TiledCopyC,
    class DStride, class DSmemLayout,
    class EStride, class ESmemLayout,
    class PStride,
    class T
>
__device__
__forceinline__
void
kernel_cute_p_norm_kernel_gradient(
    ProblemShape shape_MNOKL, CtaTiler cta_tiler, ThreadTiler thread_tiler,
    i32 num_blocks_M,
    T const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    T const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    T const *C, CStride dC, CSmemLayout sC_layout, TiledCopyC copy_c,
    T const *D, DStride dD, DSmemLayout sD_layout,
    T       *E, EStride dE, ESmemLayout,
    T       *P, PStride dP
) {
    static_assert(norm_type == NormType::L1 || norm_type == NormType::L2 || norm_type == NormType::P);

    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNOKL) == Int<5>{}); // (M, N, O, K, L)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<4>{}); // (BLK_M, BLK_N, BLK_O, BLK_K)
    CUTE_STATIC_ASSERT_V(rank(thread_tiler) == Int<3>{}); // (THR_M, THR_N, THR_O)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(thread_tiler)); // NumThreads
    // CUTE_STATIC_ASSERT_V(size(copy_b) == size(thread_tiler)); // NumThreads
    // CUTE_STATIC_ASSERT_V(size(copy_c) == size(thread_tiler)); // NumThreads

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

    CUTE_STATIC_ASSERT_V(congruent(select<0,3,4>(shape_MNOKL), dA));         // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,3,4>(shape_MNOKL), dB));         // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<2,3,4>(shape_MNOKL), dC));         // dC strides for shape OK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1,4>(shape_MNOKL), dD));         // dD strides for shape MN
    CUTE_STATIC_ASSERT_V(congruent(select<0,1,2,4>(shape_MNOKL), dE));       // dE strides for shape MNO
    
    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,3,4>(shape_MNOKL), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,3,4>(shape_MNOKL), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<2,3,4>(shape_MNOKL), dC); // (O,K)
    Tensor mD = make_tensor(make_gmem_ptr(D), select<0,1,4>(shape_MNOKL), dD); // (M,N)
    Tensor mE = make_tensor(make_gmem_ptr(E), select<0,1,2,4>(shape_MNOKL), dE); // (M,N,O)

    Tensor mP = make_tensor(make_gmem_ptr(P), select<4>(shape_MNOKL), dP);

    auto bidx = blockIdx.x % num_blocks_M;
    auto bidy = blockIdx.y;
    auto bidz = blockIdx.z;
    auto bidw = blockIdx.x / num_blocks_M;

    T p_power_grad = T(0);

    if constexpr (norm_type == NormType::P) {
        p_power_grad = mP(bidw) - 1.0;
    }

    auto cta_coord = make_coord(bidx, bidy, bidz, _); // (m,n,o,k)
    Tensor gA = local_tile(mA(_,_,bidw), cta_tiler, cta_coord, Step<_1,  X,  X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB(_,_,bidw), cta_tiler, cta_coord, Step< X, _1,  X, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC(_,_,bidw), cta_tiler, cta_coord, Step< X,  X, _1, _1>{}); // (BLK_O,BLK_K,k)
    Tensor gD = local_tile(mD(_,_,bidw), cta_tiler, cta_coord, Step<_1, _1,  X,  X>{}); // (BLK_M,BLK_N)
    Tensor gE = local_tile(mE(_,_,_,bidw), cta_tiler, cta_coord, Step<_1, _1, _1,  X>{}); // (BLK_M,BLK_N,BLK_O)

    auto m_max_coord = size<0>(shape_MNOKL) - size<0>(gA) * bidx;  // M - BLK_M * m_coord
    auto n_max_coord = size<1>(shape_MNOKL) - size<0>(gB) * bidy;  // N - BLK_N * n_coord
    auto o_max_coord = size<2>(shape_MNOKL) - size<0>(gC) * bidz;  // O - BLK_O * o_coord
    auto k_residue   = size<3>(shape_MNOKL) - size<1>(gA) * size<2>(gA); // K - BLK_K * k_coord_max

    // Need to get the tile count before the offsetting in gA, gB, gC of the k_residue
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
                        diff = _pow(diff, p_power_grad);
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
    CUTE_UNROLL
    for (int i = 0; i < size(tErE); i++) {
        if (elem_less(tEcE(i), make_coord(m_max_coord,n_max_coord,o_max_coord))) {
            tEgE(i) = tErE(i);
        }
    }
}

template <
    NormType norm_type
>
__global__
__launch_bounds__(256)
void
cute_norm_kernel_gradient(
    i32 m, i32 n, i32 o, i32 k, i32 l,
    i32 num_blocks_M,
    f32 const *A, u64 ldA,                u64 batch_stride_a, // kernel_matrix   L,M,N     l,m,k
    f32 const *B, u64 ldB,                u64 batch_stride_b, // data_N          L,N,D     l,k,n
    f32 const *C, u64 ldC,                u64 batch_stride_c, // solution        L,N,C     l,k,o
    f32 const *D, u64 ldD,                u64 batch_stride_d, // data_M          L,M,D     l,m,n
    f32 *E,       u64 ldE_N, u64 ldE_O,   u64 batch_stride_e, // grad            L,M,D,C   l,m,n,o
    f32 *P,                               u64 batch_stride_p
) {
    using namespace cute;
    using T = f32;

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
    auto dP = make_stride(batch_stride_p);

    auto bM = Int<32>{};
    auto bN = Int<32>{};
    auto bO = Int<32>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bO, bK);
    auto bP = Int<2>{};

    auto thread_tiler = Layout<Shape<_16, _16, _1>>{}; // M, N, O

    auto sA = make_layout(make_shape(bM, bK, bP)); // M-major
    
    auto sB_atom = make_layout(
        make_shape(bN, bK),
        make_stride(Int<1>{}, bN+Int<4>{})
    ); // (n,k) -> smem_idx; padded n-major
    auto sB = tile_to_shape(sB_atom, make_shape(bN, bK, bP)); // N-major

    auto sC_atom = make_layout(
        make_shape(bO, bK),
        make_stride(Int<1>{}, bO+Int<4>{})
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
        Layout<Shape<_32,_8>, Stride<_8,_1>>{}, // Thr layout 8x32 k-major
        Layout<Shape< _1,_1>>{} // Val layout  1x1
    );

    TiledCopy copyC = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
        Layout<Shape<_32,_8>, Stride<_8,_1>>{}, // Thr layout 8x32 k-major
        Layout<Shape< _1,_1>>{} // Val layout  1x1 
    );

    kernel_cute_p_norm_kernel_gradient<
        norm_type
    > (
        prob_shape, cta_tiler, thread_tiler,
        num_blocks_M,
        A, dA, sA, copyA,
        B, dB, sB, copyB,
        C, dC, sC, copyC,
        D, dD, sD,
        E, dE, sE,
        P, dP
    );
}
