#include "kermac.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <stdio.h>
#include <cute/tensor.hpp>

void tensor_stats(torch::Tensor a) {
    printf("num_dims: %d\n", a.dim());
    for (int i = 0; i < a.dim(); i++) {
        printf("size %d: %d\n", i, a.size(i));
        printf("stride %d: %d\n", i, a.stride(i));
    }

    using namespace cute;

    auto bM = Int<128>{};
    auto bN = Int<16>{};

    auto sD = make_layout(make_shape(bM, bN));

    print(sD); print("\n");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    using T = float;
}