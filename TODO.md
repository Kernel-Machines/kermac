# TODO
* copy atom for smem to registers with retile (allows non-vectorized accesses to avoid bank conflicts when K < 32 K-major)
* cta size and tiling shmoo for different C/O sizes in coefs. 16 is too large for many cases
* batch modes
* build-a-bear
* replace my `_sqrt`, `_exp`, etc.. with `--use_fast_math` option in JIT
* with inner dot product, can use samples, samples_m, centers, centers_m in L2 norm and accumulate (x @ m) @ x in registers locally
* L1_M requires sqrt(M) to do regular norm
