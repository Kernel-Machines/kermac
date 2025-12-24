import unittest
import torch
import kermac


class TestExpPowScaled(unittest.TestCase):
    def setUp(self):
        if kermac is None:
            raise unittest.SkipTest("kermac (and/or its CUDA Python deps) not available in this environment")
        self.device = torch.device("cuda")
        self.atol = 5e-4
        self.rtol = 5e-4

    def test_sum_exp_abs_pow_scaled_matches_torch(self):
        # Small sizes to keep the reference cheap
        M, N, K = 7, 5, 12
        p = 1.2
        L = 1.3

        x = torch.randn(M, K, device=self.device, dtype=torch.float32)
        z = torch.randn(N, K, device=self.device, dtype=torch.float32)

        scale = 1.0 / (L ** p)

        descriptor = kermac.KernelDescriptor(
            inner_operator=kermac.InnerOperator.DIFF,
            inner_power=kermac.PowerType.EXP_POW_SCALED,
            outer_power=kermac.PowerType.NOOP,
            kernel_type=kermac.KernelType.NONE,
        )

        fused = kermac.run_kernel(
            descriptor,
            x,
            z,
            p=p,
            bandwidth=scale,  # interpreted as `scale` for EXP_POW_SCALED
        ).squeeze(0)

        # Kernel computes sum_i (exp(|xi-zi|^p * scale) - 1) to avoid K-padding adding exp(0)=1.
        # Add back +K to recover the true sum_i exp(|xi-zi|^p * scale).
        fused = fused + K
        print("fused", fused)

        ref = torch.exp((x[:, None, :] - z[None, :, :]).abs().pow(p) * scale).sum(dim=-1)
        print("ref", ref)

        torch.testing.assert_close(fused, ref, atol=self.atol, rtol=self.rtol)


if __name__ == "__main__":
    unittest.main()


