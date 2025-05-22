#pragma once

#include <kermac_internal_common.cuh>

__device__
__forceinline__
double 
fast_sqrt(
    double theta
) {
  #if defined(__CUDA_ARCH__)
  return ::sqrt(theta);
  #else
  return std::sqrt(theta);
  #endif
}

// Triangular indexing
// idx -> lower triangle consecutive
template <class T>
__global__
void
syrk_test(
) {
    i32 tid = threadIdx.x;
    i32 macro_id = blockIdx.x;
    i32 macro_row = ceil(fast_sqrt((2*macro_id) + 2.25) - 0.5) - 1;
    i32 macro_col = macro_id - (((macro_row+1) * macro_row)/2);
}