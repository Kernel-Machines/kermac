# kermac
Pytorch routines for (Ker)nel (Mac)hines.

### These routines are very fast

However they only support **sm_80** or higher Nvidia cards. This includes:
* Server cards like A100 and up.
* Geforce cards like 3 series and up.

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
Either run `examples/main.py` or in the repl you can do:
``` python
import kermac
import torch

a = torch.randn(10,100).cuda()  # Because of cdist_t this will reflect a tensor with 100 rows and 10 columns
b = torch.randn(10,20).cuda()   # Because of cdist_t this will reflect a tensor with 20 rows and 10 columns
c = kermac.cdist_t(a,b)         # this will return a tensor with 100 rows and 20 columns shaped like (20,100)
print(c)
```

# kermac.cdist_t
A reimplementation of [`torch.cdist`](https://docs.pytorch.org/docs/stable/generated/torch.cdist.html). Computes fractional norms. Requires tensors to be transposed w.r.t. input tensors in `torch.cdist`. Does not support batches yet.

Has special code paths for $p=1.0$ and $p=2.0$ to avoid fractional power instructions.
### `kermac.cdist_t` vs `torch.cdist`
with problem size $[M,N,K]$ = $[30000,30000,1024]$

| GPU / p-norm | Speed-up (×) | kermac.cdist_t (ms) | torch.cdist (ms) |
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
### Tensors must satisfy
``` python
# Given tensors a,b,c and sizes M,N,K
# K is the contracted mode
assert a.shape == torch.Size([K,M])
assert b.shape == torch.Size([K,N])
assert c.shape == torch.Size([N,M])

assert a.stride(1) == 1
assert b.stride(1) == 1
assert c.stride(1) == 1

c = kermac.cdist_t(a,b,out=c) # OK
```
### Views are OK
```python
import torch
import kermac

M, N, K = (100, 20, 10)

x = torch.randn(K,M).cuda()

a = x[1:5,:N]                 # shape [4,20]
print(a.is_contiguous())      # prints 'False'

b = x[2:6,1:N+11]             # shape [4,30]
print(b.is_contiguous())      # prints 'False'

y = torch.zeros(40,30).cuda()
c = y[10:40, 10:30]           # shape [30,20]

c = kermac.cdist_t(a,b,out=c) # OK
```

<!-- ### GH200
```
Running p-norm=1.0 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  82.073 ms
        torch.cdist     2388.529 ms
Running p-norm=1.3 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  453.440 ms
        torch.cdist     4360.252 ms
Running p-norm=2.0 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  78.744 ms
        torch.cdist     405.845 ms
```
### H100 - PCIe
```
Running p-norm=1.0 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  107.582 ms
        torch.cdist     2906.757 ms
Running p-norm=1.3 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  592.267 ms
        torch.cdist     5591.462 ms
Running p-norm=2.0 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  103.502 ms
        torch.cdist     346.187 ms
```
### A100
A100s have low simt but high tensor core. p=2.0 uses tensor cores in that special case.
```
Running p-norm=1.0 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  250.978 ms
        torch.cdist     3877.834 ms
Running p-norm=1.3 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  872.534 ms
        torch.cdist     8229.695 ms
Running p-norm=2.0 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  325.448 ms
        torch.cdist     301.412 ms
```
### RTX 4090
```
Running p-norm=1.0 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  76.474 ms
        torch.cdist     4020.806 ms
Running p-norm=1.3 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  350.319 ms
        torch.cdist     4140.799 ms
Running p-norm=2.0 with size (30000,1024) by (30000,1024)
        kermac.cdist_t  77.172 ms
        torch.cdist     261.975 ms
``` -->

<!-- ### Tensor Alignment
``` python
import torch

# Aligned tensors can give a performance boost here

alignment_requirement_elements = 4 # 4 Floats
alignment_requirement_bytes = alignment_requirement_elements * 8 # 32 bytes

a = torch.randn(10,100).cuda()
# Check if tensor starts on a multiple of 4 elements
assert(a.data_ptr() % alignment_requirement_bytes == 0)         # Pass, PyTorch tensors are aligned by default
# Check if the tensor stride is a multiple of 4 elements
# It's ok for the shape in dim 0 to not be a multiple of 4 elements
assert(a.stride(0) % alignment_requirement_elements == 0)       # Pass, stride(0) is 100

# view of a in dim 0
a_view = a[:,4:] # [10,96], starting 4 elements over.
assert(a_view.data_ptr() % alignment_requirement_bytes == 0)    # Pass, a_view starts 4 elements shifted over
assert(a_view.stride(0) % alignment_requirement_elements == 0)  # Pass, a.stride(0) equals a_view.stride(0)

# view of a in dim 1 and dim 0
a_view = a[1:3,4:] # [2,96], starting 4 elements over still
assert(a_view.data_ptr() % alignment_requirement_bytes == 0)    # Pass, views in dim(1) don't affect starting pointer
assert(a_view.stride(0) % alignment_requirement_elements == 0)  # Pass, views in dim(1) don't affect stride(0)

a_view = a[:,1:97] # [10,96], starting 1 element over
assert(a_view.data_ptr() % alignment_requirement_bytes == 0)    # Fail, a_view starts 1 element over
assert(a_view.stride(0) % alignment_requirement_elements == 0)  # Pass, a.stride(0) == a_view.stride(0)

a = torch.randn(10,97).cuda()
assert(a.data_ptr() % alignment_requirement_bytes == 0)         # Pass
assert(a.stride(0) % alignment_requirement_elements == 0)       # Fail, stride(0) is 97, not divisible by 4

``` -->