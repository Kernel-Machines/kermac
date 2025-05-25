![function](media/frame179_2min_crop.jpg)
# kermac
**Pytorch** routines for (**Ker**)nel (**Mac**)hines

## These routines are very fast

However they only support **sm_80** or higher **Nvidia** cards. This includes:
* Server cards like **A10**, **A100**, **H100**, **B100**
* Consumer **RTX 30xx**, **RTX 40xx**, **RTX 50xx**

## Install

### CUDA 12
``` bash
pip install "kermac[cu12] @ git+https://github.com/Kernel-Machines/kermac"
```

### CUDA 11
``` bash
pip install "kermac[cu11] @ git+https://github.com/Kernel-Machines/kermac"
```

### Check Install
From a fresh environment you can do:
``` bash
wget https://raw.githubusercontent.com/Kernel-Machines/kermac/refs/heads/master/examples/cdist.py
python cdist.py -d -p 1.0
```
or from a python repl you can do:
``` python
import kermac
import torch

device = torch.device('cuda')
a = torch.randn(100,10,device=device)
b = torch.randn(100,10,device=device)
c = kermac.cdist(a,b)
print(c)
```

### Examples
To run `kermac.cdist`:
``` bash
wget https://raw.githubusercontent.com/Kernel-Machines/kermac/refs/heads/master/examples/cdist.py
python cdist.py -d -p 1.0
```
To run `kermac.cdist_grad`:
``` bash
wget https://raw.githubusercontent.com/Kernel-Machines/kermac/refs/heads/master/examples/cdist_grad.py
python cdist_grad.py -d -p 2.0
# For some reason running the script a second time after compiling kernel is much faster
python cdist_grad.py -d -p 2.0
```
To wipe out the compiled cubin cache you can do:
``` bash
rm -rf ~/.cache/kermac
```

## Function: cdist
An implementation of [**`torch.cdist`**](https://docs.pytorch.org/docs/stable/generated/torch.cdist.html). Computes fractional norms. Supports batches and broadcasting. Aside from the `out` tensor in the `out=None` case **DOES NOT ALLOCATE**

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

## Function: cdist_grad
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
As explained with `cdist_t`.

# Just-In-Time (JIT)
This library just-in-time (JIT) compiles it's cuda kernels using Nvidia's [**cuda-core**](https://nvidia.github.io/cuda-python/cuda-core/latest/) package. The first run of a given configuration compiles the kernel and stores it in a cache database on disk. The next run for the same configuration should be fast. Using the debug flag like in:
``` bash
python cdist.py -p 1.0 -d
```
``` bash
python examples/cdist.py -p 1.0 -d
```
or when calling a function like:
``` python
cdist(a,b,p=1.0,debug=True)
```
Will print information related to the compilation of kernel functions.