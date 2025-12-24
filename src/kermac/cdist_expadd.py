import torch
from typing import Optional, Union

from .build_a_kernel import (
    KernelDescriptor,
    InnerOperator,
    PowerType,
    KernelType,
    run_kernel,
)


def cdist_expadd(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    p: Union[float, torch.Tensor] = 2.0,
    lengthscale: Optional[float] = None,
    scale: Optional[Union[float, torch.Tensor]] = None,
    try_to_align: bool = False,
    debug: bool = False,
):
    """
    Computes:

        out[m,n] = sum_{k=1..K} exp(|a[m,k] - b[n,k]|^p / lengthscale^p)

    Notes:
    - Internally uses a fused kernel that computes sum_k (exp(...) - 1) to avoid K-padding
      adding exp(0)=1; this wrapper adds back +K so the returned value matches the formula.
    - Supports batches/broadcasting like `run_kernel`.

    Args:
        a: (M,K) or (L,M,K) float32 CUDA
        b: (N,K) or (L,N,K) float32 CUDA
        out: optional output tensor shaped like `run_kernel` expects (L,M,N) or (M,N)
        p: float or tensor (optionally batched) for the inner power
        lengthscale: L in the formula; required if `scale` is not provided
        scale: optional precomputed scale = 1/(lengthscale^p). If provided, overrides `lengthscale`.
        try_to_align/debug: forwarded to `run_kernel`
    """
    if scale is None:
        if lengthscale is None:
            raise ValueError("Provide either `scale` (recommended) or `lengthscale`.")
        if isinstance(p, torch.Tensor):
            ls = torch.tensor(float(lengthscale), dtype=torch.float32, device=p.device)
            scale = 1.0 / torch.pow(ls, p)
        else:
            scale = 1.0 / (float(lengthscale) ** float(p))

    descriptor = KernelDescriptor(
        inner_operator=InnerOperator.DIFF,
        inner_power=PowerType.EXP_POW_SCALED,
        outer_power=PowerType.NOOP,
        kernel_type=KernelType.NONE,
    )

    out = run_kernel(
        descriptor,
        a,
        b,
        out=out,
        p=p,
        bandwidth=scale,  # interpreted as `scale` for EXP_POW_SCALED
        try_to_align=try_to_align,
        debug=debug,
    )

    K = a.shape[-1]
    out.add_(K)
    return out


