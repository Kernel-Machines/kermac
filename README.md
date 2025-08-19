# KERMAC

Kermac is a collection of fused CUDA kernels meant for fast and memory efficient computation for kernel methods. Kermac makes heavy use of JIT (Just-in-time) compilation to generate custom CUDA kernels on demand. These compiled kernels are stored in a cache database so the JIT costs are only incurred once. Using `debug=True` in most kermac routines will print information related to the compilation and caching of these JIT CUDA kernels.

Kermac supports only Nvidia cards with capability of `sm_80` or higher. This includes:
* Server cards like A10, A100, H100, B100
* Consumer cards like RTX 30xx, RTX 40xx, RTX 50xx

Kermac relies on [**cuda-core**](https://nvidia.github.io/cuda-python/cuda-core/latest/) for JIT compilation which is supported for cuda toolkits 11.8 and 12.x. Because of cuda-core and nvmath packages no C++ compilation or wheel system is needed to install this library.

# Installation

### CUDA 12
``` bash
pip install "kermac[cu12] @ git+https://github.com/Kernel-Machines/kermac"
```

### CUDA 11
``` bash
pip install "kermac[cu11] @ git+https://github.com/Kernel-Machines/kermac"
```

### Linalg
linalg functionality depends on nvmath-python. This isn't a required dependency. To run linalg routines please do:
``` bash
pip install nvmath-python[cu12]
```
or
``` bash
pip install nvmath-python[cu11]
```

# Examples
From a fresh environment you can do:
## [`cdist.py`](examples/cdist.py)

``` bash
wget https://raw.githubusercontent.com/Kernel-Machines/kermac/refs/heads/master/examples/cdist.py
python cdist.py -d -p 1.0
```
## [`cdist_grad.py`](examples/cdist_grad.py)
``` bash
wget https://raw.githubusercontent.com/Kernel-Machines/kermac/refs/heads/master/examples/cdist_grad.py
python cdist_grad.py -d
```
## [`build_a_kernel.py`](examples/build_a_kernel.py)
Running `build_a_kernel.py` will batch compile quite a few different kernels on first run. Expect around 20 seconds of JIT compiling.
``` bash
wget https://raw.githubusercontent.com/Kernel-Machines/kermac/refs/heads/master/examples/build_a_kernel.py
python build_a_kernel.py -d
```
## [`linalg.py`](examples/linalg.py)
``` bash
wget https://raw.githubusercontent.com/Kernel-Machines/kermac/refs/heads/master/examples/linalg.py
python linalg.py
```
## Function: `linalg.solve_cholesky`
Solves a symmetric system of equations like [`torch.linalg.cholesky`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.cholesky.html). Wraps `xpotrf` and `xpotrs` from [`nvmath.bindings.cusolverDn`](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cusolver.html). This implementation is special because it doesn't synchronize with the cpu on a failed cholesky factor. Additionally this routine can write the factorization in-place to the input matrix. In some cases this avoids a full 2x increase in memory usage. It launches a separate cuda-stream for each of the batches passed in. It does require a bit of workspace memory allocation for each stream. It synchronizes the cuda-streams against the current stream at the end of the routine.

## Function: `linalg.solve_lu`
Solves a symmetric system of equations like [`torch.linalg.solve`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve.html).
Wraps `xgetrf` and `xgetrs` from [`nvmath.bindings.cusolverDn`](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cusolver.html). This implementation is special because it doesn't synchronize with the cpu on a failed LU decomposition. Additionally this routine can write the factorization in-place to the input matrix. In some cases this avoids a full 2x increase in memory usage. It launches a separate cuda-stream for each of the batches passed in. It does require a bit of workspace memory allocation for each stream. It synchronizes the cuda-streams against the current stream at the end of the routine.

## Function: `linalg.eigh`
Computes eigenvalues and eigenvectors of a symmetric matrix like [`torch.linalg.eigh`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.html)
Wraps `xsyevd` from [`nvmath.bindings.cusolverDn`](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cusolver.html). This implementation is special because it doesn't synchronize with the cpu on a failed eigenvalue decomposition. Additionally this routine can write the eigenvector decomposition in-place to the input matrix. In some cases this avoids a full 2x increase in memory usage. It launches a separate cuda-stream for each of the batches passed in. It does require a bit of workspace memory allocation for each stream. It synchronizes the cuda-streams against the current stream at the end of the routine.

## Function: `cdist`
An implementation of [**`torch.cdist`**](https://docs.pytorch.org/docs/stable/generated/torch.cdist.html). Computes fractional norms. Supports batches and broadcasting. Aside from the `out` tensor in the `out=None` case does not allocate.

Computes:

$out_{n,m} = \left( \sum_{k=1}^{K} |b_{k,n} - a_{k,m}|^p \right)^{\frac{1}{p}}$

If instead `skip_epilogue` is set it computes:

$out_{n,m} = \sum_{k=1}^{K} |b_{k,n} - a_{k,m}|^p$

Or expressed in **c-style** it efficiently computes:
``` c
// a[K,M], b[K,N], out[N,M]
for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            out[n,m] += pow(abs(b[k,n] - a[k,m]), p);
        }
        if (!skip_epilogue) {
            out[n,m] = pow(out[n,m], 1.0/p);
        }
    }
}
```

It has special code paths for $p=1.0$ and $p=2.0$ to avoid fractional power instructions.
### `kermac.cdist` vs `torch.cdist`
with problem size $[M,N,K]$ = $[30000,30000,1024]$

| GPU / p-norm | Speed-up (×) | kermac.cdist (ms) | torch.cdist (ms) |
|:-------------|-------------:|--------------------:|-----------------:|
| **GH200 · p = 1.0**      | **29.1×** | 82  | 2,389 |
| **GH200 · p = 1.3**      | **9.6×**  | 453 | 4,360 |
| **GH200 · p = 2.0**      | **5.2×**  | 79  | 406  |
| **H100-PCIe · p = 1.0**  | **27.0×** | 108 | 2,907 |
| **H100-PCIe · p = 1.3**  | **9.4×**  | 592 | 5,591 |
| **H100-PCIe · p = 2.0**  | **3.3×**  | 104 | 346  |
| **A100 · p = 1.0**       | **15.4×** | 251 | 3,878 |
| **A100 · p = 1.3**       | **9.4×**  | 873 | 8,230 |
| **A100 · p = 2.0**       | **0.9×**  | 325 | 301  |
| **RTX 4090 · p = 1.0**   | **52.6×** | 76  | 4,021 |
| **RTX 4090 · p = 1.3**   | **11.8×** | 350 | 4,141 |
| **RTX 4090 · p = 2.0**   | **3.4×**  | 77  | 262  |

## Function: `run_kernel`
This is a more customizable version of `kermac.cdist`, `kermac.cdist` is written on top of this. `run_kernel` allows a descriptor as one of it's arguments that can create fully fused kernel functions. You can specify the inner-norm type (`abs(x)`, `x*x`, or `pow(x,p)`), the outer-norm type (`x`, `sqrt(x)`, or `pow(x,1/p`)) and finally a laplace or gaussian epilogue. On first run a fully fused kernel will be JIT compiled and cached for future use. This function also allows broadcasting and batching of it's input tensors. See [`build_a_kernel.py`](examples/build_a_kernel.py) for various examples of usage. It also allows batching and broadcasting of it's hyperparameters such as `p`, `bandwidth`, and `regularization`. See [`broadcast_kernel.py`](examples/broadcast_kernel.py) for examples of batching and broadcasting hyperparameters

## Function: `cdist_grad`
Computes the gradient of `cdist` in the style like:

$out_{o,n,m} = \sum_{k=1}^{K} c_{o,k}a_{k,m}\mathrm{sgn}\left(d_{n,m}-b_{n,k}\right)\left|d_{n,m}-b_{n,k}\right|^{p-1}$

Or expressed in c-style it efficiently computes:
``` c
// a[K,M], b[N,K], c[O,K], d[N,M], out[O,N,M]
for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
        for (int o = 0; o < O; o++) {
            for (int k = 0; k < K; k++) {
                float diff = d[n,m] - b[n,k];
                out[o,n,m] += c[o,k] * a[k,m] * signum(diff) * pow(abs(diff), p - 1.0));
            }
        }
    }
}
```
Aside from the `out` tensor in the `out=None` case **DOES NOT ALLOCATE**
It has special code paths for $p=1.0$ and $p=2.0$ to avoid fractional power instructions.

It's supposed to be used like:
* $a_{k,m}$ is `kernel_matrix`
* $b_{n,k}$ is `data_x`
* $c_{o,k}$ is `coefficients`
* $d_{n,m}$ is `data_z`
* $out_{o,n,k}$ is `gradient`

### Tensors must satisfy
``` python
# Given tensors a,b,c,d,out and sizes M,N,O,K
# K is the contracted mode
assert a.shape == torch.Size([K,M])
assert b.shape == torch.Size([N,K])
assert c.shape == torch.Size([O,K])
assert d.shape == torch.Size([N,M])
assert out.shape == torch.Size([O,N,M])

assert a.stride(1) == 1
assert b.stride(1) == 1
assert c.stride(1) == 1
assert d.stride(1) == 1
assert out.stride(1) == 1

out = kermac.cdist_grad(a,b,c,d,out=out) # OK
```

### Views are OK
