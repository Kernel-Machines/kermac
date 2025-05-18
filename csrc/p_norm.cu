#include <kermac_internal_common.cuh>
#include <p_norm_impl.cuh>

extern "C"
__global__
__launch_bounds__(256)
void
cute_l1_norm_m128n128k8p3(
    float p_power,
    int m, int n, int k,
    float const *A, int ldA, // M,K m-major
    float const *B, int ldB, // N,K n-major
    float       *C, int ldC  // M,N m-major
) {
    cuda_p_norm_m128n128k8p3<true, true, false, NormType::L1>(
        c_zero<f32>, m, n, k, A, ldA, B, ldB, C, ldC
    );
}

extern "C"
__global__
__launch_bounds__(256)
void
cute_l2_norm_m128n128k8p3(
    float p_power,
    int m, int n, int k,
    float const *A, int ldA, // M,K m-major
    float const *B, int ldB, // N,K n-major
    float       *C, int ldC  // M,N m-major
) {
    cuda_p_norm_m128n128k8p3<true, true, false, NormType::L2>(
        c_zero<f32>, m, n, k, A, ldA, B, ldB, C, ldC
    );
}

extern "C"
__global__
__launch_bounds__(256)
void
cute_p_norm_m128n128k8p3(
    float p_power,
    int m, int n, int k,
    float const *A, int ldA, // M,K m-major
    float const *B, int ldB, // N,K n-major
    float       *C, int ldC  // M,N m-major
) {
    cuda_p_norm_m128n128k8p3<true, true, false, NormType::P>(
        p_power, m, n, k, A, ldA, B, ldB, C, ldC
    );
}

extern "C"
__global__
__launch_bounds__(256)
void
cute_l1_norm_m128n128k8p3_skip_epilogue(
    float p_power,
    int m, int n, int k,
    float const *A, int ldA, // M,K m-major
    float const *B, int ldB, // N,K n-major
    float       *C, int ldC  // M,N m-major
) {
    cuda_p_norm_m128n128k8p3<true, true, true, NormType::L1>(
        c_zero<f32>, m, n, k, A, ldA, B, ldB, C, ldC
    );
}

extern "C"
__global__
__launch_bounds__(256)
void
cute_l2_norm_m128n128k8p3_skip_epilogue(
    float p_power,
    int m, int n, int k,
    float const *A, int ldA, // M,K m-major
    float const *B, int ldB, // N,K n-major
    float       *C, int ldC  // M,N m-major
) {
    cuda_p_norm_m128n128k8p3<true, true, true, NormType::L2>(
        c_zero<f32>, m, n, k, A, ldA, B, ldB, C, ldC
    );
}

extern "C"
__global__
__launch_bounds__(256)
void
cute_p_norm_m128n128k8p3_skip_epilogue(
    float p_power,
    int m, int n, int k,
    float const *A, int ldA, // M,K m-major
    float const *B, int ldB, // N,K n-major
    float       *C, int ldC  // M,N m-major
) {
    cuda_p_norm_m128n128k8p3<true, true, true, NormType::P>(
        p_power, m, n, k, A, ldA, B, ldB, C, ldC
    );
}
