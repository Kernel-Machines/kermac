#pragma once

#include <kermac_internal_common.cuh>
#include <cute/tensor.hpp>

__global__
__launch_bounds__(256)
void
cutlass_little_test(
    int m, int k,
    f32 const *A, u64 ldA
) {
    using namespace cute;
    using T = f32;

    auto M = u64(m);
    auto K = u64(k);

    auto prob_shape = make_shape(M,K);

    auto dA = make_stride(ldA, Int<1>{});

    auto bM = Int<2>{};
    auto bK = Int<8>{};

    auto cta_tiler = make_shape(bM, bK);

    auto sA_atom = make_layout(
        make_shape(bM,bK),
        make_stride(Int<1>{},bM+Int<4>{})
    );
    auto sA_layout = tile_to_shape(sA_atom, make_shape(bM,bK));

    TiledCopy copy_a = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<T>, T>{},
        // Layout<Shape<decltype(bM),decltype(bK)>, Stride<decltype(bK),_1>>{}, // Thr layout 8x32 k-major
        make_layout(make_shape(bM,bK), make_stride(bK,Int<1>{})),
        Layout<Shape< _1,_1>>{} // Val layout  1x1 
    );

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,1>(prob_shape), dA);

    auto cta_coord = make_coord(blockIdx.x, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, _1>{});

    auto smem_layout = make_layout(
        make_shape(Int<4>{},Int<16>{}),
        make_stride(Int<1>{},Int<4>{})
    );
    alignas(16) __shared__ T smem_a[cosize_v<decltype(smem_layout)>];
    Tensor sA = make_tensor(make_smem_ptr(smem_a), sA_layout);
    Tensor sMem = make_tensor(make_smem_ptr(smem_a), smem_layout);

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    Tensor tApA = make_tensor<bool>(
        make_shape(size<1>(tAsA), size<2>(tAsA)),
        make_stride(Int<1>{}, size<1>(tAsA))
    );

    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
    Tensor tAcA = thr_copy_a.partition_S(cA);

    CUTE_UNROLL
    for (int m = 0; m < size<0>(tApA); m++) {
        CUTE_UNROLL
        for (int k = 0; k < size<1>(tApA); k++) {
            tApA(m,k) = get<0>(tAcA(0,m,k)) < bM && get<1>(tAcA(0,m,k)) < bK;
        }
    }
    // Tensor tACA = thr_copy_a.partition_D(cA);
    // auto thread_tiler = Layout<Shape<_1>>{};

    // Tensor tEsA = local_partition(sA, thread_tiler, threadIdx.x, Step<_1>{});
    // Tensor tErA = make_fragment_like(tEsA);

    if (thread(16)) {
        print("tAgA : "); print(tAgA); print("\n");
        print("tAsA : "); print(tAsA); print("\n");
        print("tApA : "); print(tApA); print("\n");
        print("tAcA : "); print(tAcA); print("\n");
        print("tAcA : "); print_tensor(tAcA);
        print("tApA : "); print_tensor(tApA);
    }
    clear(sMem);
    copy_if(copy_a, tApA, tAgA(_,_,_,0), tAsA(_,_,_));
    // copy(copy_a, tAgA(_,_,_,0), tAsA(_,_,_));

    cp_async_fence();

    cp_async_wait<0>();
    __syncthreads();

    if (thread(0)) {
        print_tensor(sA); print("\n");
        print_tensor(sMem); print("\n");
        print_tensor(tAsA); print("\n");
    }
}