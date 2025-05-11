# (Ker)nel (Mac)hines

Only supports sm80 or higher Nvidia cards. This includes:
* Server cards like A100 and up.
* Geforce cards like 3 series and up.

# cdist_transposed
A cdist implementation that computes cdist efficiently with a fractional p-value in a tiled and asynchronous copy manner. Requires `cp.async` that is in cards sm80 or higher. 

Due to complexity of memory accesses the tensors are required to be `stride==1` in the non-contracted dimension. This is known as `column-major`, `M-major` or in Pytorch as `[K,M]`. Therefore the tensors passed in to `cdist_transpose` should be of sizes `[K,M]`, `[K,N]` and optionally `[N,M]` for the result. The tensor does not have to be `torch.is_contiguous` but the right-most stride has to be 1. 

`p=1.0` and `p=2.0` have special implementations. If `k` is the contracted dimension:
* `p=1.0` performs a simple `abs(a_k - b_k)` with no epilogue operation.
* `p=2.0` does not use matrix multiply so torch.cdist might be faster in some cases. `p=2.0` does `(a_k - b_k) * (a_k - b_k)` directly and performs a `sqrt` on the result.

For other p-values we do `pow(abs(a_k - b_k), p)` while contracting. We apply `pow(accum, 1.0/p)` for the final result.

### Nerdy note
The difference in speed between p=1.0 and p=1.1 can be assumed to be directly from the extra clock cycles for `pow(v,p)`. `pow` is implemented with `lg2`, `mul` and `ex2` in ptx assembly. `mul`, `add`, `subtract`, etc.. is 128 results per clock cycle, `lg2` and `ex2` are both 16 results per clock cycle. See godbolt: [__powf](https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:cuda,selection:(endColumn:1,endLineNumber:11,positionColumn:1,positionLineNumber:11,selectionStartColumn:1,selectionStartLineNumber:11,startColumn:1,startLineNumber:11),source:'%0A//+Type+your+code+here,+or+load+an+example.%0A__global__+void+pow(float*+array,+int+n)+%7B%0A++++int+tid+%3D+blockDim.x+*+blockIdx.x+%2B+threadIdx.x%3B%0A++++if+(tid+%3C+n)+%7B%0A++++++++float+v+%3D++array%5Btid%5D%3B%0A++++++++array%5Btid%5D+%3D+__powf(v,+1.3f)%3B%0A++++%7D%0A%7D%0A%0A'),l:'5',n:'0',o:'CUDA+C%2B%2B+source+%231',t:'0')),k:41.815823605706875,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:nvcc125u1,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:cuda,libs:!(),options:'--use_fast_math+-O3',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+NVCC+12.5.1+(Editor+%231)',t:'0')),k:24.850843060959793,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:device,i:(compilerName:'NVCC+12.5.1',device:PTX,editorid:1,fontScale:14,fontUsePx:'0',j:1,selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),treeid:0),l:'5',n:'0',o:'Device+Viewer+NVCC+12.5.1+(Editor+%231,+Compiler+%231)',t:'0')),k:33.33333333333333,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4).

### Results
On a RTX 4090:
```
~/kermac$ python examples/main.py 
Running p-norm=1.0 with size (30000,1000) by (30000,1000)
Generating random data..
Generated random data in: 4051.359 ms
Launching L1 Norm
kermac.cdist_transposed elapsed time:   68.385 ms
torch.cdist elapsed time:               3919.779 ms
kermac.cdist_transposed:
tensor([[1176.7207, 1163.4862, 1130.6797,  ..., 1166.4866, 1126.9427,
         1136.0476],
        [1168.8375, 1112.7660, 1131.2385,  ..., 1136.4324, 1151.6187,
         1134.0973],
        [1128.8818, 1103.9948, 1118.8202,  ..., 1124.3627, 1119.5331,
         1104.5433],
        ...,
        [1137.7081, 1127.3167, 1102.1724,  ..., 1102.9048, 1110.6974,
         1121.7233],
        [1132.7302, 1124.1478, 1089.7920,  ..., 1142.3284, 1126.9645,
         1103.2474],
        [1126.5073, 1127.1073, 1160.0742,  ..., 1135.0912, 1143.3641,
         1128.6945]], device='cuda:0')
torch.cdist:
tensor([[1176.7213, 1163.4857, 1130.6792,  ..., 1166.4872, 1126.9426,
         1136.0464],
        [1168.8367, 1112.7654, 1131.2379,  ..., 1136.4326, 1151.6174,
         1134.0964],
        [1128.8813, 1103.9946, 1118.8197,  ..., 1124.3632, 1119.5332,
         1104.5442],
        ...,
        [1137.7083, 1127.3175, 1102.1732,  ..., 1102.9041, 1110.6965,
         1121.7233],
        [1132.7302, 1124.1467, 1089.7927,  ..., 1142.3281, 1126.9646,
         1103.2469],
        [1126.5068, 1127.1073, 1160.0743,  ..., 1135.0909, 1143.3643,
         1128.6951]], device='cuda:0')
```