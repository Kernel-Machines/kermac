from typing import Optional, Union

import numpy as np
import torch
from cuda.core.experimental import Device, LaunchConfig, launch

from .cdist_expadd import cdist_expadd
from .common import PyTorchStreamWrapper, ceil_div, merge_batch_size, tensor_stats
from .module_cache.module_cache import ModuleCache


def cdist_grad_expadd(
    a: torch.Tensor,  # (K,M) or (L,K,M)  float32 CUDA, stride 1 in M
    b: torch.Tensor,  # (N,K) or (L,N,K)  float32 CUDA, stride 1 in K  (x^T)
    c: torch.Tensor,  # (O,K) or (L,O,K)  float32 CUDA, stride 1 in K  (coefs)
    d: torch.Tensor,  # (N,M) or (L,N,M)  float32 CUDA, stride 1 in M  (z^T)
    *,
    out: Optional[torch.Tensor] = None,  # (O,N,M) or (L,O,N,M)
    s_km: Optional[torch.Tensor] = None,  # (K,M) or (L,K,M) precomputed S (can be scaled)
    p: Union[float, torch.Tensor] = 2.0,
    q: Union[float, torch.Tensor] = 1.0,
    c0: float = 0.0,
    lengthscale: Optional[float] = None,
    scale: Optional[Union[float, torch.Tensor]] = None,
    eps: float = 1e-8,
    try_to_align: bool = False,  # kept for API symmetry; currently unused
    debug: bool = False,
):
    """
    Efficient gradient accumulation helper for the "expadd" kernel family.

    Computes (conceptually):

        S[k,m] = sum_{n} exp( |d[n,m] - b[n,k]|^p * scale )

        out[o,n,m] = sum_{k} c[o,k] * a[k,m] * q * (c0 + S[k,m])^(q-1)
                          * exp(|diff|^p * scale) * (p*scale) * sign(diff) * |diff|^(p-1)

    Notes:
    - `S` is computed once via `kermac.cdist_expadd` (fast fused kernel), then a
      dedicated CUDA kernel performs the (k)-accumulation for each (o,n,m).
    - If you already have `S` (for example you need to scale it before the outer power),
      pass it as `s_km` to avoid recomputation. `s_km` must be laid out like `a`: (K,M).
    - Inputs follow the same transposed conventions as `cdist_grad`.
    """

    if not all(isinstance(x, torch.Tensor) for x in (a, b, c, d)):
        raise TypeError("All inputs must be PyTorch tensors")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a PyTorch tensor if provided")

    if not all(x.dtype == torch.float32 for x in (a, b, c, d)):
        raise TypeError("All inputs must have dtype torch.float32")
    if out is not None and out.dtype != torch.float32:
        raise TypeError("out must have dtype torch.float32")

    if not all((x.dim() == 2 or x.dim() == 3) for x in (a, b, c, d)):
        raise ValueError("All inputs must be 2D or 3D (batched)")
    if out is not None and (out.dim() != 3 and out.dim() != 4):
        raise ValueError("out must be 3D or 4D (batched)")

    if not all(x.is_cuda for x in (a, b, c, d)):
        raise ValueError("All inputs must be on a CUDA device")
    if out is not None and not out.is_cuda:
        raise ValueError("out must be on a CUDA device")

    tensor_device = a.device
    if not all(x.device == tensor_device for x in (a, b, c, d)):
        raise ValueError("All inputs must be on the same CUDA device")
    if out is not None and out.device != tensor_device:
        raise ValueError("out must be on the same CUDA device as inputs")

    # batch size merge (same pattern as cdist_grad)
    L = 1
    L = merge_batch_size("p", L, p, expected_dims=0, can_be_none=False)
    L = merge_batch_size("q", L, q, expected_dims=0, can_be_none=False)
    L = merge_batch_size("scale", L, scale, expected_dims=0, can_be_none=True)
    L = merge_batch_size("s_km", L, s_km, expected_dims=2, can_be_none=True)
    L = merge_batch_size("a", L, a, expected_dims=2, can_be_none=False)
    L = merge_batch_size("b", L, b, expected_dims=2, can_be_none=False)
    L = merge_batch_size("c", L, c, expected_dims=2, can_be_none=False)
    L = merge_batch_size("d", L, d, expected_dims=2, can_be_none=False)
    L = merge_batch_size("out", L, out, expected_dims=3, can_be_none=True)

    if scale is None:
        if lengthscale is None:
            raise ValueError("Provide either `scale` (recommended) or `lengthscale`.")
        if isinstance(p, torch.Tensor):
            ls = torch.tensor(float(lengthscale), dtype=torch.float32, device=tensor_device)
            scale = 1.0 / torch.pow(ls, p)
        else:
            scale = 1.0 / (float(lengthscale) ** float(p))

    # normalize p/q/scale to CUDA float32 tensors (optionally batched)
    def _to_cuda_f32(x, name: str) -> torch.Tensor:
        if isinstance(x, float):
            return torch.tensor(x, dtype=torch.float32, device=tensor_device)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"`{name}` must be a float or torch.Tensor")
        if x.dtype != torch.float32:
            raise TypeError(f"`{name}` tensor must have dtype torch.float32")
        if not x.is_cuda or x.device != tensor_device:
            raise ValueError(f"`{name}` tensor must be on the same CUDA device as inputs")
        return x

    p_t = _to_cuda_f32(p, "p")
    q_t = _to_cuda_f32(q, "q")
    scale_t = _to_cuda_f32(scale, "scale")

    stats_a = tensor_stats(a)
    stats_b = tensor_stats(b)
    stats_c = tensor_stats(c)
    stats_d = tensor_stats(d)

    _, K_a, M_a = stats_a.shape
    _, N_b, K_b = stats_b.shape
    _, O_c, K_c = stats_c.shape
    _, N_d, M_d = stats_d.shape

    if (K_a, M_a) != (K_b, M_d):
        raise ValueError(f"Incompatible shapes: a is (K={K_a},M={M_a}), b is (N={N_b},K={K_b}), d is (N={N_d},M={M_d})")
    if K_c != K_a:
        raise ValueError(f"c has K={K_c} but expected {K_a}")
    if N_d != N_b:
        raise ValueError(f"d has N={N_d} but expected {N_b}")

    K = K_a
    M = M_a
    N = N_b
    O = O_c

    # strides: require last dim contiguous (matches cdist_grad expectations)
    if a.stride(-1) != 1:
        raise ValueError("a must have stride 1 in last dimension")
    if b.stride(-1) != 1:
        raise ValueError("b must have stride 1 in last dimension")
    if c.stride(-1) != 1:
        raise ValueError("c must have stride 1 in last dimension")
    if d.stride(-1) != 1:
        raise ValueError("d must have stride 1 in last dimension")
    if out is not None and out.stride(-1) != 1:
        raise ValueError("out must have stride 1 in last dimension")

    out = torch.zeros((L, O, N, M), dtype=torch.float32, device=tensor_device) if out is None else out

    if s_km is None:
        # Compute S once: use non-transposed views (points x features)
        # b: (L,N,K) -> x: (L,K,N)
        # d: (L,N,M) -> z: (L,M,N)
        x = b.transpose(-2, -1).contiguous()
        z = d.transpose(-2, -1).contiguous()

        # S_mk: (L,M,K) or (M,K)
        S_mk = cdist_expadd(z, x, p=p_t, scale=scale_t, try_to_align=try_to_align, debug=debug)
        # Make S_km: (L,K,M) to match a's (K,M) indexing
        S_km = S_mk.transpose(-2, -1).contiguous()
    else:
        if not isinstance(s_km, torch.Tensor):
            raise TypeError("s_km must be a torch.Tensor if provided")
        if s_km.dtype != torch.float32:
            raise TypeError("s_km must have dtype torch.float32")
        if not s_km.is_cuda or s_km.device != tensor_device:
            raise ValueError("s_km must be on the same CUDA device as inputs")
        if s_km.dim() not in (2, 3):
            raise ValueError("s_km must be 2D or 3D (batched)")
        # Ensure last dim contiguous
        if s_km.stride(-1) != 1:
            s_km = s_km.contiguous()
        S_km = s_km

    stats_s = tensor_stats(S_km)

    # CUDA launch plumbing (same pattern as cdist_grad)
    pt_stream = torch.cuda.current_stream()
    pt_device = pt_stream.device
    if tensor_device != pt_device:
        raise ValueError("cuda stream must be on the same device as the tensors")

    device = Device(pt_device.index)
    device.set_current()
    stream = PyTorchStreamWrapper(pt_stream)

    module_cache = ModuleCache(debug)
    function_string = "expadd_norm_kernel_gradient"
    kernel = module_cache.get_function(device, function_string, debug=debug)

    # grid: (L*M, ceil(N/256), O)
    num_blocks_M = M
    grid = (L * num_blocks_M, ceil_div(N, 256), O)
    config = LaunchConfig(grid=grid, block=256)

    ld_a = np.uint64(stats_a.leading_dimension_stride)
    bs_a = np.uint64(stats_a.batch_stride)

    ld_b = np.uint64(stats_b.leading_dimension_stride)
    bs_b = np.uint64(stats_b.batch_stride)

    ld_c = np.uint64(stats_c.leading_dimension_stride)
    bs_c = np.uint64(stats_c.batch_stride)

    ld_d = np.uint64(stats_d.leading_dimension_stride)
    bs_d = np.uint64(stats_d.batch_stride)

    ld_s = np.uint64(stats_s.leading_dimension_stride)
    bs_s = np.uint64(stats_s.batch_stride)

    ld_e_N = np.uint64(out.stride(-2))
    ld_e_O = np.uint64(out.stride(-3))
    bs_e = np.uint64(0 if L == 1 else out.stride(-4))

    bs_p = np.uint64(0 if p_t.numel() == 1 else 1)
    bs_q = np.uint64(0 if q_t.numel() == 1 else 1)
    bs_scale = np.uint64(0 if scale_t.numel() == 1 else 1)

    kernel_args = (
        M,
        N,
        O,
        K,
        L,
        np.int32(num_blocks_M),
        a.data_ptr(),
        ld_a,
        bs_a,
        b.data_ptr(),
        ld_b,
        bs_b,
        c.data_ptr(),
        ld_c,
        bs_c,
        d.data_ptr(),
        ld_d,
        bs_d,
        S_km.data_ptr(),
        ld_s,
        bs_s,
        out.data_ptr(),
        ld_e_N,
        ld_e_O,
        bs_e,
        p_t.data_ptr(),
        bs_p,
        q_t.data_ptr(),
        bs_q,
        scale_t.data_ptr(),
        bs_scale,
        np.float32(float(c0)),
        np.float32(float(eps)),
    )

    launch(stream, config, kernel, *kernel_args)
    return out


