# TODO
* copy atom for smem to registers with retile (allows non-vectorized accesses to avoid bank conflicts when K < 32 K-major)
* cta size and tiling shmoo for different C/O sizes in coefs. 16 is too large for many cases
* batch modes
* build-a-bear
* replace my `_sqrt`, `_exp`, etc.. with `--use_fast_math` option in JIT
* with inner dot product, can use samples, samples_m, centers, centers_m in L2 norm and accumulate (x @ m) @ x in registers locally
* L1_M requires sqrt(M) to do regular norm
* database should be able to bulk compile functions mapping hash of cubin to cubin binary
* for each function in the bulk compile a function -> hash of cubin should be generated for a separate database.
* align_1 align_4 for build_a_kernel

* Function name -> (lowered_name, cubin_hash)
* cubin_hash -> cubin

* Figure out how to use multiple cores to compile different kernel functions
