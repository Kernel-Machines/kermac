#pragma once

#include <kermac_internal_common.cuh>

// Gradient helper for the "expadd" kernel family:
//
// Given (batched) tensors:
//   A: (K, M)   [pairwise scalar, user-provided]
//   B: (N, K)   [x, transposed: features x points]
//   C: (O, K)   [coeffs]
//   D: (N, M)   [z, transposed: features x points]
//   S: (K, M)   where S(k,m) = sum_{n=1..N} exp( |D(n,m)-B(n,k)|^p * scale )
// Produces:
//   E: (O, N, M) where
//     E(o,n,m) = sum_k C(o,k) * A(k,m) * q*(c0 + S(k,m))^(q-1)
//                      * exp(|diff|^p * scale) * (p*scale) * sign(diff) * |diff|^(p-1)
//
// Notes:
// - This kernel is intentionally "simple CUDA" (not CUTE tiled) to keep implementation
//   straightforward. It relies on the caller to provide S, computed efficiently via
//   `cdist_expadd`.

__device__ __forceinline__ f32 _signum_f32(f32 x) {
    return (f32)((x > 0.f) - (x < 0.f));
}

__device__ __forceinline__ f32 _pow_f32(f32 x, f32 p) {
    // kermac_internal_common.cuh provides _pow, but keep this local for clarity.
    return _pow(x, p);
}

__global__ __launch_bounds__(256)
void expadd_norm_kernel_gradient(
    i32 m, i32 n, i32 o, i32 k, i32 l,
    i32 num_blocks_M,
    f32 const *A, u64 ldA,                u64 batch_stride_a, // (K,M)
    f32 const *B, u64 ldB,                u64 batch_stride_b, // (N,K)
    f32 const *C, u64 ldC,                u64 batch_stride_c, // (O,K)
    f32 const *D, u64 ldD,                u64 batch_stride_d, // (N,M)
    f32 const *S, u64 ldS,                u64 batch_stride_s, // (K,M)
    f32       *E, u64 ldE_N, u64 ldE_O,   u64 batch_stride_e, // (O,N,M) in python layout: (L,O,N,M)
    f32 const *P,                            u64 batch_stride_p, // p (scalar or batched)
    f32 const *Q,                            u64 batch_stride_q, // q (scalar or batched)
    f32 const *Scale,                        u64 batch_stride_scale, // scale = 1/(L^p) (scalar or batched)
    f32 c0,
    f32 epsilon
) {
    // Map blocks: blockIdx.x packs (L, M) like cdist_grad does.
    i32 bid_m = (i32)(blockIdx.x % num_blocks_M);
    i32 bid_l = (i32)(blockIdx.x / num_blocks_M);
    i32 bid_n_tile = (i32)blockIdx.y;
    i32 bid_o = (i32)blockIdx.z;

    // One m per block for simplicity
    i32 mm = bid_m;
    i32 oo = bid_o;
    i32 nn = bid_n_tile * (i32)blockDim.x + (i32)threadIdx.x;

    if (bid_l >= l || mm >= m || oo >= o || nn >= n) {
        return;
    }

    // Batch offsets (broadcasting supported via batch_stride_* possibly being 0)
    u64 offA = (u64)bid_l * batch_stride_a;
    u64 offB = (u64)bid_l * batch_stride_b;
    u64 offC = (u64)bid_l * batch_stride_c;
    u64 offD = (u64)bid_l * batch_stride_d;
    u64 offS = (u64)bid_l * batch_stride_s;
    u64 offE = (u64)bid_l * batch_stride_e;

    f32 p = *(P + (u64)bid_l * batch_stride_p);
    f32 q = *(Q + (u64)bid_l * batch_stride_q);
    f32 scale = *(Scale + (u64)bid_l * batch_stride_scale);

    // Read z and stream over k
    f32 z_nm = *(D + offD + (u64)nn * (u64)ldD + (u64)mm);

    f32 acc = 0.f;

    // Precompute constant p*scale
    f32 p_scale = p * scale;
    f32 p_minus_1 = p - 1.f;
    f32 q_minus_1 = q - 1.f;

    for (i32 kk = 0; kk < k; ++kk) {
        f32 x_nk = *(B + offB + (u64)nn * (u64)ldB + (u64)kk);
        f32 diff = z_nm - x_nk;
        f32 sgn = _signum_f32(diff);
        if (sgn == 0.f) {
            continue;
        }

        f32 ad = _abs(diff);
        // Clamp for stability when p<1 (and generally avoids NaNs)
        ad = ad < epsilon ? epsilon : ad;

        f32 ad_p = _pow_f32(ad, p);
        f32 exp_term = _exp(ad_p * scale);

        f32 ad_p_minus_1 = (p_minus_1 == 0.f) ? 1.f : _pow_f32(ad, p_minus_1);
        f32 per_dim = exp_term * p_scale * sgn * ad_p_minus_1;

        f32 a_km = *(A + offA + (u64)kk * (u64)ldA + (u64)mm);
        f32 s_km = *(S + offS + (u64)kk * (u64)ldS + (u64)mm);

        // q*(c0 + S)^(q-1)
        f32 base = c0 + s_km;
        // base should be >= c0 + N*exp(0) >= 0 if c0>=0, but don't assume.
        // Clamp to epsilon to avoid pow on <=0 for fractional q.
        base = base < epsilon ? epsilon : base;
        f32 outer = (q_minus_1 == 0.f) ? q : (q * _pow_f32(base, q_minus_1));

        f32 c_ok = *(C + offC + (u64)oo * (u64)ldC + (u64)kk);

        acc += c_ok * a_km * outer * per_dim;
    }

    // Store E in (O,N,M) with python-provided strides
    *(E + offE + (u64)oo * (u64)ldE_O + (u64)nn * (u64)ldE_N + (u64)mm) = acc;
}


