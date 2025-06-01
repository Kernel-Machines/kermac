# Notes

* Wonky stuff happens when tiling is equal to cta size in the respective dimension.
* Wonky stuff happens when so much of the requests are predicated that the warps don't see anything to fence on.

## ncu banks
```
ncu --set full --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum --kernel-name regex:.*_build_kernel_.* python examples/cdist.py -a -d --skip_torch -m 1024 -n 1024
```

```
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum --kernel-name regex:.*_scaled_gemm.* python sandbox/run_cutlass.py
```