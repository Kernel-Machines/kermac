import unittest
import torch
import kermac


class TestCDistGradExpAdd(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.atol = 2e-3
        self.rtol = 2e-3

    def test_cdist_grad_expadd_matches_reference_small(self):
        # Keep tiny so the reference is cheap.
        Lb = 1
        O = 2
        N = 7   # features
        K = 5   # x points
        M = 4   # z points

        p = 1.3
        q = 1.2
        c0 = 0.7
        lengthscale = 1.4
        scale = 1.0 / (lengthscale ** p)

        a = torch.randn(Lb, K, M, device=self.device, dtype=torch.float32).contiguous()
        b = torch.randn(Lb, N, K, device=self.device, dtype=torch.float32).contiguous()
        c = torch.randn(Lb, O, K, device=self.device, dtype=torch.float32).contiguous()
        d = torch.randn(Lb, N, M, device=self.device, dtype=torch.float32).contiguous()

        out = kermac.cdist_grad_expadd(
            a,
            b,
            c,
            d,
            p=p,
            q=q,
            c0=c0,
            scale=scale,
        )

        # Reference
        # x: (K,N), z: (M,N)
        x = b.transpose(-2, -1)  # (L,K,N)
        z = d.transpose(-2, -1)  # (L,M,N)
        diff = z[:, :, None, :] - x[:, None, :, :]  # (L,M,K,N) = z_m - x_k
        ad = diff.abs()

        exp_term = torch.exp(ad.pow(p) * scale)  # (L,M,K,N)
        S_mk = exp_term.sum(dim=-1)  # (L,M,K)
        outer = q * (c0 + S_mk).pow(q - 1.0)  # (L,M,K)

        sign = diff.sign()
        per_dim = exp_term * (p * scale) * sign * ad.clamp_min(1e-8).pow(p - 1.0)  # (L,M,K,N)

        # out[l,o,n,m] = sum_k c[l,o,k] * a[l,k,m] * outer[l,m,k] * per_dim[l,m,k,n]
        # Make shapes explicit to avoid subscript mistakes:
        #   outer_lkm: (L,K,M)
        #   per_dim_lkmn: (L,K,M,N)
        outer_lkm = outer.permute(0, 2, 1).contiguous()
        per_dim_lkmn = per_dim.permute(0, 2, 1, 3).contiguous()
        w_lkm = a * outer_lkm
        ref = torch.einsum("lok,lkm,lkmn->lonm", c, w_lkm, per_dim_lkmn)

        torch.testing.assert_close(out, ref, atol=self.atol, rtol=self.rtol)


if __name__ == "__main__":
    unittest.main()


