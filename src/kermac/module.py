import torch

from ._cuda_extension import _p_norm_pytorch

def cdist_transposed(a_t, b_t, p=2.0, out=None, skip_epilogue=False):
    """
    Computes a cdist on transposed tensors with input validation.
    
    Args:
        a_t (torch.Tensor): Input tensor of shape (K, M), stride 1 in M, dtype float32, on CUDA.
        b_t (torch.Tensor): Input tensor of shape (K, N), stride 1 in N, dtype float32, on CUDA.
        p (float, optional=2.0): p value for the p-norm distance. 
        out (torch.Tensor, optional=None): Output tensor of shape (N, M), stride 1 in M, dtype float32, on CUDA.
        skip_epilogue (bool, optional=False): Avoid the final step of the result where we raise the result to the 1.0/p power.
    
    Returns:
        torch.Tensor: Result tensor (placeholder implementation).
    
    Raises:
        TypeError: If inputs are not PyTorch tensors or have incorrect dtype.
        ValueError: If shapes, strides, dimensions, or CUDA devices are invalid.
    """
    # Check if inputs are tensors
    if not isinstance(a_t, torch.Tensor) or not isinstance(b_t, torch.Tensor):
        raise TypeError("a_t and b_t must be PyTorch tensors")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a PyTorch tensor if provided")

    # Check dtype
    if a_t.dtype != torch.float32 or b_t.dtype != torch.float32:
        raise TypeError("a_t and b_t must have dtype torch.float32")
    if out is not None and out.dtype != torch.float32:
        raise TypeError("out must have dtype torch.float32")

    # Check number of dimensions
    if a_t.dim() != 2 or b_t.dim() != 2:
        raise ValueError("a_t and b_t must be 2-dimensional")
    if out is not None and out.dim() != 2:
        raise ValueError("out must be 2-dimensional")

    # Check CUDA device
    if not a_t.is_cuda or not b_t.is_cuda:
        raise ValueError("a_t and b_t must be on a CUDA device")
    if out is not None and not out.is_cuda:
        raise ValueError("out must be on a CUDA device")
    if a_t.device != b_t.device:
        raise ValueError(f"a_t and b_t must be on the same CUDA device: got {a_t.device} and {b_t.device}")
    if out is not None and out.device != a_t.device:
        raise ValueError(f"out must be on the same CUDA device as inputs: got {out.device}, expected {a_t.device}")

    # Get shapes
    K_a, M = a_t.shape
    K_b, N = b_t.shape

    # Check shape consistency
    if K_a != K_b:
        raise ValueError(f"K dimensions must match: got {K_a} for a_t and {K_b} for b_t")
    
    K = K_a

    # Check strides (stride 1 in last dimension)
    if a_t.stride(-1) != 1:
        raise ValueError("a_t must have stride 1 in dimension M (last dimension)")
    if b_t.stride(-1) != 1:
        raise ValueError("b_t must have stride 1 in dimension N (last dimension)")
    if out is not None and out.stride(-1) != 1:
        raise ValueError("out must have stride 1 in dimension M (last dimension)")

    # Check output shape if provided
    if out is not None:
        if out.shape != (N, M):
            raise ValueError(f"out must have shape (N={N}, M={M}), got {out.shape}")

    # Placeholder for actual computation (e.g., cdist or other operation)
    # Replace with your actual logic
    result = torch.zeros((N, M), dtype=torch.float32, device=a_t.device) if out is None else out
    _p_norm_pytorch(
        p,
        skip_epilogue,
        M, N, K,
        a_t, b_t, 
        result
    )
    return result
