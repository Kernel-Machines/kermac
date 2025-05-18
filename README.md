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
`kermac.cdist_t` can beat `torch.cdist` both operating in cuda by up to:
* **60x** for **p=1.0**
* **12x** for **fractional-p**, i.e. **p=1.3**
* **3x** for **p=2.0** (`torch.cdist` uses gemm trick for **p=2.0**)

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