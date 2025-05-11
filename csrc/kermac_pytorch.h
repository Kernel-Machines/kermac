#pragma once

#include <torch/extension.h>

void _p_norm_pytorch(
    float p_power,
    bool skip_epilogue,
    int M, int N, int K,
    torch::Tensor a_t, // [K, M] => M,K : M-Major
    torch::Tensor b_t, // [K, N] => N,K : N-Major
    torch::Tensor c    // [N, M] => M,N : M-Major
);