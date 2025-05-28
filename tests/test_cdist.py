import unittest
import torch
from itertools import product
import kermac

class TestCDist(unittest.TestCase):
    def setUp(self):
        """Set up test parameters and device."""
        self.device = torch.device('cuda')
        self.M = 1000  # Smaller sizes for faster tests
        self.N = 1000
        self.K = 64
        self.L = 2
        self.p_values = [1.0, 1.3, 2.0]  # p-norm values to test
        self.try_to_align_values = [False, True]  # try_to_align values to test
        self.atol = 1e-4  # Absolute tolerance for numerical comparison
        self.rtol = 1e-5  # Relative tolerance for numerical comparison
        self.debug = False

        descriptors = [
            kermac.kernel_descriptor_l1_norm,
            kermac.kernel_descriptor_l2_norm,
            kermac.kernel_descriptor_p_norm,
        ]
        print('Bulk compiling kernels')
        kermac.pre_compile_descriptors(
            device=self.device,
            descriptors=descriptors,
            try_to_align=True,
            debug=self.debug
        )

    def _create_tensors(self, a_col_major, b_col_major, c_col_major):
        """Create input and output tensors based on transpose flags."""
        a = torch.randn(self.L, self.K, self.M, device=self.device).permute(0,2,1) if a_col_major else torch.randn(self.L, self.M, self.K, device=self.device)
        b = torch.randn(self.L, self.K, self.N, device=self.device).permute(0,2,1) if b_col_major else torch.randn(self.L, self.N, self.K, device=self.device)
        c = torch.randn(self.L, self.N, self.M, device=self.device).permute(0,2,1) if c_col_major else torch.randn(self.L, self.M, self.N, device=self.device)
        return a, b, c

    def _compare_outputs(self, kermac_out, torch_out):
        """Compare kermac.cdist and torch.cdist outputs."""
        diff = kermac_out - torch_out
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse).item()
        max_abs_error = torch.max(torch.abs(diff)).item()
        return rmse, max_abs_error

    def test_cdist_transposes(self):
        """Test kermac.cdist against torch.cdist for all transpose, p, and try_to_align combinations."""
        transpose_combinations = list(product([False, True], repeat=3))
        amount_of_combinations = len(transpose_combinations) * len(self.p_values) * len(self.try_to_align_values)
        print(f'Running {amount_of_combinations} combinations of kermac.cdist settings')
        print(f'\t{len(transpose_combinations)} transpose configurations')
        print(f'\t{len(self.p_values)} p-value configurations (1.0, 1.3, 2.0)')
        print(f'\t{len(self.try_to_align_values)} alignment configurations')
        
        for (a_col_major, b_col_major, c_col_major), p, try_to_align in product(transpose_combinations, self.p_values, self.try_to_align_values):
            with self.subTest(
                a_col_major=a_col_major,
                b_col_major=b_col_major,
                c_col_major=c_col_major,
                p=p,
                try_to_align=try_to_align
            ):
                # Create tensors
                a, b, c = self._create_tensors(a_col_major, b_col_major, c_col_major)

                # Run kermac.cdist
                kermac_out = kermac.cdist(
                    a, b, p=p, out=c,
                    skip_epilogue=False, try_to_align=try_to_align, debug=self.debug
                )

                # Run torch.cdist
                torch_out = torch.cdist(a, b, p=p)

                # Compare outputs
                rmse, max_abs_error = self._compare_outputs(kermac_out, torch_out)

                # Assert numerical closeness
                self.assertLess(rmse, self.atol,
                    f"RMSE too high for transposes a_col_major={a_col_major}, "
                    f"b_col_major={b_col_major}, c_col_major={c_col_major}, "
                    f"p={p}, try_to_align={try_to_align}")
                self.assertLess(max_abs_error, self.atol,
                    f"Max absolute error too high for transposes a_col_major={a_col_major}, "
                    f"b_col_major={b_col_major}, c_col_major={c_col_major}, "
                    f"p={p}, try_to_align={try_to_align}")
                        
if __name__ == "__main__":
    unittest.main()