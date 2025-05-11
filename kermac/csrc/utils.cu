#include "kermac.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <stdio.h>

void tensor_stats(torch::Tensor a) {
    printf("num_dims: %d\n", a.dim());
    for (int i = 0; i < a.dim(); i++) {
        printf("size %d: %d\n", i, a.size(i));
        printf("stride %d: %d\n", i, a.stride(i));
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    using T = float;
}