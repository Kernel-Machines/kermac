import unittest
import torch
import kermac


class TestCDistExpAdd(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.atol = 5e-4
        self.rtol = 5e-4

    def test_cdist_expadd_matches_reference_with_padding(self):
        # Choose K not divisible by 8 to exercise K-padding behavior
        M, N, K = 32, 17, 29
        p = 0.8
        L = 1.3
        scale = 1.0 / (L ** p)

        x = torch.randn(M, K, device=self.device, dtype=torch.float32)
        z = torch.randn(N, K, device=self.device, dtype=torch.float32)

        out = kermac.cdist_expadd(x, z, p=p, scale=scale).squeeze(0)
        ref = torch.exp((x[:, None, :] - z[None, :, :]).abs().pow(p) * scale).sum(dim=-1)

        torch.testing.assert_close(out, ref, atol=self.atol, rtol=self.rtol)


if __name__ == "__main__":
    unittest.main()


