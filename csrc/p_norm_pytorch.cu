#include <kermac_pytorch.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <stdio.h>
#include <p_norm_impl.cuh>

void _p_norm_pytorch(
    float p_power,
    bool skip_epilogue,
    int M, int N, int K,
    torch::Tensor a_t, // [K, M] => M,K : M-Major
    torch::Tensor b_t, // [K, N] => N,K : N-Major
    torch::Tensor c    // [N, M] => M,N : M-Major
) {
    TORCH_CHECK(
        a_t.is_cuda(), 
        "Tensor (a_t) must be on CUDA device, but got device: ", a_t.device()
    );

    TORCH_CHECK(
        b_t.is_cuda(), 
        "Tensor (b_t) must be on CUDA device, but got device: ", b_t.device()
    );

    TORCH_CHECK(
        c.is_cuda(), 
        "Tensor (c) must be on CUDA device, but got device: ", c.device()
    );
                
    TORCH_CHECK(
        a_t.dtype() == torch::kFloat32, 
        "Tensor (a_t) must have float32 dtype, but got ", a_t.dtype()
    );

    TORCH_CHECK(
        b_t.dtype() == torch::kFloat32, 
        "Tensor (b_t) must have float32 dtype, but got ", b_t.dtype()
    );

    TORCH_CHECK(
        c.dtype() == torch::kFloat32, 
        "Tensor (c) must have float32 dtype, but got ", c.dtype()
    );
    
    TORCH_CHECK(a_t.dim() == 2, "Tensor must be 2-dimensional, got ", a_t.dim(), " dimensions");
    TORCH_CHECK(b_t.dim() == 2, "Tensor must be 2-dimensional, got ", b_t.dim(), " dimensions");
    TORCH_CHECK(c.dim() == 2, "Tensor must be 2-dimensional, got ", c.dim(), " dimensions");

    // Check if the tensor has the expected dimensions
    TORCH_CHECK(
        a_t.size(0) == K && a_t.size(1) == M,
        "Tensor (a_t) must have dimensions [", K, ", ", M,"], got [", a_t.size(0), ", ", a_t.size(1), "]"
    );

    TORCH_CHECK(
        b_t.size(0) == K && b_t.size(1) == N,
        "Tensor (b_t) must have dimensions [", K, ", ", N,"], got [", b_t.size(0), ", ", b_t.size(1), "]"
    );

    TORCH_CHECK(
        c.size(0) == N && c.size(1) == M,
        "Tensor (c) must have dimensions [", N, ", ", M,"], got [", c.size(0), ", ", c.size(1), "]"
    );

    TORCH_CHECK(
        a_t.stride(1) == 1,
        "Tensor (a_t) must have stride 1 in the rightmost dimension, got ", a_t.stride(1),
        ", is the tensor transposed incorrectly or not contiguous?"
    );

    TORCH_CHECK(
        b_t.stride(1) == 1,
        "Tensor (b_t) must have stride 1 in the rightmost dimension, got ", b_t.stride(1),
        ", is the tensor transposed incorrectly or not contiguous?"
    );

    TORCH_CHECK(
        c.stride(1) == 1,
        "Tensor (c) must have stride 1 in the rightmost dimension, got ", c.stride(1),
        ", is the tensor transposed incorrectly or not contiguous?"
    );

    float* a_t_data_ptr = a_t.data_ptr<float>();
    float* b_t_data_ptr = b_t.data_ptr<float>();
    float* c_data_ptr = c.data_ptr<float>();

    int ldA = a_t.stride(0);
    int ldB = b_t.stride(0);
    int ldC = c.stride(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cute_p_norm_m128n128k8p2(
        p_power,
        skip_epilogue,
        M, N, K,
        a_t_data_ptr, ldA,
        b_t_data_ptr, ldB,
        c_data_ptr, ldC,
        stream
    );
}