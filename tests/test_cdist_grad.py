import unittest
import torch
from itertools import product
import kermac

class TestCDistGrad(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda')
        self.M = 1000   # Number of rows of AGOP
        self.N = 1000   # Number of dimension
        self.O = 16     # Number of classes
        self.K = 64     # Contraction dimension
        self.L = 4      # Number of batches
        self.p_values = [2.0] # Can only test this case easily. 
        self.atol = 1e-4    # Absolute tolerance for numerical comparison
        self.rtol = 1e-5    # Relative tolerance for numerical comparison
        self.debug = False

        self.a = torch.randn(self.L, self.K, self.M, device=self.device)
        self.b = torch.randn(self.L, self.N, self.K, device=self.device)
        self.c = torch.randn(self.L, self.O, self.K, device=self.device)
        self.d = torch.randn(self.L, self.N, self.M, device=self.device)
        self.e = torch.randn(self.L, self.O, self.N, self.M, device=self.device)

    def _compare_outputs(self, kermac_out, torch_out):
        """Compare kermac.cdist and torch.cdist outputs."""
        diff = kermac_out - torch_out
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse).item()
        max_abs_error = torch.max(torch.abs(diff)).item()
        return rmse, max_abs_error

    def test_cdist_grad(self):
        kermac_out = kermac.cdist_grad(
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
            out=self.e,
            p=2.0
        )

        coefs = self.c
        kernel_matrix = self.a
        x = self.b.permute(0,2,1)
        z = self.d.permute(0,2,1)
        torch_left = torch.einsum('bli,bij,bjd->bljd', coefs, kernel_matrix, z)
        torch_right = torch.einsum('bli,bij,bid->bljd', coefs, kernel_matrix, x)
        torch_out = torch_left - torch_right

        kermac_out = kermac_out.permute(0,1,3,2)
        # Compare outputs
        rmse, max_abs_error = self._compare_outputs(kermac_out, torch_out)

        # Assert numerical closeness
        self.assertLess(rmse, self.atol,
            f"RMSE too high for cdist_grad")
        self.assertLess(max_abs_error, self.atol,
            f"Max absolute error too high for cdist_grad")

if __name__ == "__main__":
    unittest.main()