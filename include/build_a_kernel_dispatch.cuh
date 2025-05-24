#pragma once
#include <kermac_internal_common.cuh>
#include <build_a_kernel_impl.cuh>

template <
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__global__
__launch_bounds__(256)
void
cute_build_kernel_nnn(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    cute_build_kernel_nn<
        true, true,
        inner_operator, inner_power,
        outer_power, 
        kernel_type,
        align_A, align_B
    >(
        m,n,k,
        A, ldA,
        B, ldB,
        C, ldC,
        p_power_inner,
        p_power_outer,
        bandwidth
    );
}

template <
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__global__
__launch_bounds__(256)
void
cute_build_kernel_ntn(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    cute_build_kernel_nt<
        true, true,
        inner_operator, inner_power,
        outer_power, 
        kernel_type,
        align_A, align_B
    >(
        m,n,k,
        A, ldA,
        B, ldB,
        C, ldC,
        p_power_inner,
        p_power_outer,
        bandwidth
    );
}

template <
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__global__
__launch_bounds__(256)
void
cute_build_kernel_tnn(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    cute_build_kernel_tn<
        true, true,
        inner_operator, inner_power,
        outer_power, 
        kernel_type,
        align_A, align_B
    >(
        m,n,k,
        A, ldA,
        B, ldB,
        C, ldC,
        p_power_inner,
        p_power_outer,
        bandwidth
    );
}

template <
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__global__
__launch_bounds__(256)
void
cute_build_kernel_ttn(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    cute_build_kernel_tt<
        true, true,
        inner_operator, inner_power,
        outer_power, 
        kernel_type,
        align_A, align_B
    >(
        m,n,k,
        A, ldA,
        B, ldB,
        C, ldC,
        p_power_inner,
        p_power_outer,
        bandwidth
    );
}

template <
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__global__
__launch_bounds__(256)
void
cute_build_kernel_nnt(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    cute_build_kernel_tt<
        true, true,
        inner_operator, inner_power,
        outer_power, 
        kernel_type,
        align_B, align_A
    >(
        m,n,k,
        B, ldB,
        A, ldA,
        C, ldC,
        p_power_inner,
        p_power_outer,
        bandwidth
    );
}

template <
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__global__
__launch_bounds__(256)
void
cute_build_kernel_ntt(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    cute_build_kernel_nt<
        true, true,
        inner_operator, inner_power,
        outer_power, 
        kernel_type,
        align_B, align_A
    >(
        m,n,k,
        B, ldB,
        A, ldA,
        C, ldC,
        p_power_inner,
        p_power_outer,
        bandwidth
    );
}

template <
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__global__
__launch_bounds__(256)
void
cute_build_kernel_tnt(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    cute_build_kernel_tn<
        true, true,
        inner_operator, inner_power,
        outer_power, 
        kernel_type,
        align_B, align_A
    >(
        m,n,k,
        B, ldB,
        A, ldA,
        C, ldC,
        p_power_inner,
        p_power_outer,
        bandwidth
    );
}

template <
    InnerOperator inner_operator,
    PowerType inner_power,
    PowerType outer_power,
    KernelType kernel_type,
    Alignment align_A,
    Alignment align_B
>
__global__
__launch_bounds__(256)
void
cute_build_kernel_ttt(
    int m, int n, int k,
    float const *A, int ldA,
    float const *B, int ldB,
    float *C, int ldC,
    float p_power_inner, 
    float p_power_outer,
    float bandwidth
) {
    cute_build_kernel_nn<
        true, true,
        inner_operator, inner_power,
        outer_power, 
        kernel_type,
        align_B, align_A
    >(
        m,n,k,
        B, ldB,
        A, ldA,
        C, ldC,
        p_power_inner,
        p_power_outer,
        bandwidth
    );
}
