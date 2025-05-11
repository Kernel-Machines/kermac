# (Ker)nel (Mac)hines

Only supports sm80 or higher Nvidia cards. This includes:
* Server cards like A100 and up.
* Geforce cards like 3 series and up.

# Install

You can install using the latest wheel with:

`pip install https://github.com/Kernel-Machines/kermac/releases/download/v0.1.0/kermac-0.1-cp312-cp312-linux_x86_64.whl`

This avoids a long build time. If this doesn't work due to cuda versions or python/pytorch versions please let me know. You can instead install with the following which requires a decent compile time and python dev headers if you get an error:

* `pip install git+https://github.com/Kernel-Machines/kermac.git` 

* `pip install git+https://github.com/Kernel-Machines/kermac.git -vvv` to see what bullshit pip is doing for 2 whole minutes.

# cdist_transposed
A cdist implementation that computes cdist efficiently with a fractional p-value in a tiled and asynchronous copy manner. Requires `cp.async` that is in cards sm80 or higher. 

Due to complexity of memory accesses the tensors are required to be `stride==1` in the non-contracted dimension. This is known as `column-major`, `M-major` or in Pytorch as `[K,M]`. Therefore the tensors passed in to `cdist_transpose` should be of sizes `[K,M]`, `[K,N]` and optionally `[N,M]` for the result. The tensor does not have to be `torch.is_contiguous` but the right-most stride has to be 1. 

`p=1.0` and `p=2.0` have special implementations. If `k` is the contracted dimension:
* `p=1.0` performs a simple `abs(a_k - b_k)` with no epilogue operation.
* `p=2.0` does not use matrix multiply so torch.cdist might be faster in some cases. `p=2.0` does `(a_k - b_k) * (a_k - b_k)` directly and performs a `sqrt` on the result.

For other p-values we do `pow(abs(a_k - b_k), p)` while contracting. We apply `pow(accum, 1.0/p)` for the final result.

`skip_epilogue` can be used to avoid performing `pow(accum, 1.0/p)` on the final result.

### Nerdy note
The difference in speed between p=1.0 and p=1.1 can be assumed to be directly from the extra clock cycles for `pow(v,p)`. `pow` is implemented with `lg2`, `mul` and `ex2` in ptx assembly. `mul`, `add`, `subtract`, etc.. is 128 results per clock cycle, `lg2` and `ex2` are both 16 results per clock cycle. Check out [godbolt](https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:cuda,selection:(endColumn:1,endLineNumber:11,positionColumn:1,positionLineNumber:11,selectionStartColumn:1,selectionStartLineNumber:11,startColumn:1,startLineNumber:11),source:'%0A//+Type+your+code+here,+or+load+an+example.%0A__global__+void+pow(float*+array,+int+n)+%7B%0A++++int+tid+%3D+blockDim.x+*+blockIdx.x+%2B+threadIdx.x%3B%0A++++if+(tid+%3C+n)+%7B%0A++++++++float+v+%3D++array%5Btid%5D%3B%0A++++++++array%5Btid%5D+%3D+__powf(v,+1.3f)%3B%0A++++%7D%0A%7D%0A%0A'),l:'5',n:'0',o:'CUDA+C%2B%2B+source+%231',t:'0')),k:41.815823605706875,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:nvcc125u1,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:cuda,libs:!(),options:'--use_fast_math+-O3',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+NVCC+12.5.1+(Editor+%231)',t:'0')),k:24.850843060959793,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:device,i:(compilerName:'NVCC+12.5.1',device:PTX,editorid:1,fontScale:14,fontUsePx:'0',j:1,selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),treeid:0),l:'5',n:'0',o:'Device+Viewer+NVCC+12.5.1+(Editor+%231,+Compiler+%231)',t:'0')),k:33.33333333333333,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4).

### Results
On a RTX 4090 

#### (L1 Norm)
```
~/kermac$ python examples/main.py 
Running p-norm=1.0 with size (30000,1000) by (30000,1000)
Generating random data..
Generated random data in: 3980.548 ms
Launching L1 Norm
kermac.cdist_transposed elapsed time:   67.840 ms
torch.cdist elapsed time:               3919.742 ms
kermac.cdist_transposed:
tensor([[1104.2902, 1102.2158, 1084.7244,  ..., 1158.2500, 1099.5819,
         1092.7471],
        [1114.5688, 1136.5596, 1122.0276,  ..., 1136.2686, 1103.1516,
         1139.1567],
        [1079.3171, 1114.2296, 1132.8383,  ..., 1121.4536, 1113.2092,
         1109.5905],
        ...,
        [1088.6182, 1096.4436, 1089.1270,  ..., 1176.3712, 1103.2915,
         1109.7858],
        [1126.6702, 1113.8507, 1120.6118,  ..., 1148.6765, 1135.8652,
         1147.1234],
        [1113.4796, 1099.7776, 1080.5443,  ..., 1112.8654, 1095.8372,
         1121.5585]], device='cuda:0')
torch.cdist:
tensor([[1104.2891, 1102.2161, 1084.7249,  ..., 1158.2507, 1099.5811,
         1092.7467],
        [1114.5692, 1136.5591, 1122.0278,  ..., 1136.2681, 1103.1512,
         1139.1577],
        [1079.3167, 1114.2290, 1132.8389,  ..., 1121.4536, 1113.2091,
         1109.5908],
        ...,
        [1088.6173, 1096.4435, 1089.1274,  ..., 1176.3716, 1103.2920,
         1109.7864],
        [1126.6700, 1113.8508, 1120.6116,  ..., 1148.6766, 1135.8644,
         1147.1238],
        [1113.4790, 1099.7776, 1080.5442,  ..., 1112.8655, 1095.8372,
         1121.5587]], device='cuda:0')
```
#### (L2 Norm)
```
~/kermac$ python examples/main.py 
Running p-norm=2.0 with size (30000,1000) by (30000,1000)
Generating random data..
Generated random data in: 3996.537 ms
Launching L2 Norm
kermac.cdist_transposed elapsed time:   78.399 ms
torch.cdist elapsed time:               212.221 ms
kermac.cdist_transposed:
tensor([[45.9527, 44.2694, 45.8102,  ..., 44.9583, 43.6784, 45.6395],
        [45.2251, 44.8852, 45.5869,  ..., 45.2605, 44.6036, 45.0743],
        [45.0368, 44.7698, 45.6845,  ..., 45.4862, 44.9845, 45.7555],
        ...,
        [46.2763, 45.4796, 45.4163,  ..., 45.5491, 44.7861, 46.2175],
        [45.7084, 45.2250, 42.8998,  ..., 45.0993, 46.5036, 46.3731],
        [44.9687, 43.8122, 45.5294,  ..., 45.2108, 43.4662, 44.9389]],
       device='cuda:0')
torch.cdist:
tensor([[45.9527, 44.2694, 45.8102,  ..., 44.9583, 43.6784, 45.6395],
        [45.2251, 44.8852, 45.5869,  ..., 45.2605, 44.6036, 45.0743],
        [45.0368, 44.7698, 45.6845,  ..., 45.4862, 44.9845, 45.7555],
        ...,
        [46.2763, 45.4796, 45.4163,  ..., 45.5492, 44.7861, 46.2175],
        [45.7084, 45.2250, 42.8998,  ..., 45.0993, 46.5036, 46.3731],
        [44.9687, 43.8122, 45.5293,  ..., 45.2108, 43.4662, 44.9389]],
       device='cuda:0')
```
#### (P-Norm: 1.3f)
```
~/kermac$ python examples/main.py 
Running p-norm=1.3 with size (30000,1000) by (30000,1000)
Generating random data..
Generated random data in: 4001.529 ms
Launching Norm-P=1.300
kermac.cdist_transposed elapsed time:   334.805 ms
torch.cdist elapsed time:               4144.665 ms
kermac.cdist_transposed:
tensor([[248.7032, 259.8393, 241.4097,  ..., 248.2398, 261.0034, 241.5130],
        [245.4600, 242.3709, 243.5946,  ..., 249.2112, 245.0326, 244.2303],
        [245.1459, 247.6284, 243.0163,  ..., 250.0242, 238.4540, 239.3967],
        ...,
        [249.5395, 249.6548, 247.7663,  ..., 254.2753, 248.5217, 245.7377],
        [259.2698, 252.7882, 249.9454,  ..., 246.9260, 255.8708, 242.5159],
        [250.6359, 245.9745, 245.7898,  ..., 246.2731, 248.6370, 236.3426]],
       device='cuda:0')
torch.cdist:
tensor([[248.7032, 259.8394, 241.4099,  ..., 248.2400, 261.0034, 241.5131],
        [245.4601, 242.3709, 243.5948,  ..., 249.2113, 245.0327, 244.2302],
        [245.1461, 247.6285, 243.0164,  ..., 250.0242, 238.4543, 239.3968],
        ...,
        [249.5395, 249.6548, 247.7664,  ..., 254.2753, 248.5219, 245.7379],
        [259.2698, 252.7882, 249.9453,  ..., 246.9260, 255.8709, 242.5160],
        [250.6360, 245.9746, 245.7898,  ..., 246.2731, 248.6369, 236.3427]],
       device='cuda:0')
```
