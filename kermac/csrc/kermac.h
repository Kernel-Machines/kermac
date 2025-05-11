#pragma once

#include <torch/extension.h>

void tensor_stats(torch::Tensor a);

void tensor_p_norm(
    int M, int N, int K,
    torch::Tensor a_t, // [K, M] => M,K : M-Major
    torch::Tensor b_t, // [K, N] => N,K : N-Major
    torch::Tensor c    // [N, M] => M,N : M-Major
);