import kermac
import kermac.linalg
import torch
import unittest

class TestSolveLU(unittest.TestCase):
    def setUp(self):
        # Set up test parameters and data
        N = 7
        D = 6
        C = 2
        self.L = 100
        self.device = torch.device('cuda')
        
        # Generate random test data
        data = torch.randn(self.L, N, D, device=self.device)
        self.labels = torch.randn(self.L, C, N, device=self.device)
        self.kernel_matrix = torch.randn(self.L, N, N, device=self.device)
        
        # Compute kernel matrix
        kermac.run_kernel(
            kermac.kernel_descriptor_laplace_l2,
            data, data,
            out=self.kernel_matrix,
            bandwidth=10.0,
            try_to_align=True,
            debug=False
        )

    def test_solve_lu(self):
        kernel_matrix_clobber = self.kernel_matrix.clone()
        labels_clobber = self.labels.clone()
        # Run kermac.linalg.solve_lu
        sol, factor_infos, solve_infos = kermac.linalg.solve_lu(
            a=kernel_matrix_clobber,
            b=labels_clobber,
            overwrite_a=True,
            overwrite_b=True,
            check_errors=True
        )

        torch.testing.assert_close(
            factor_infos,
            torch.zeros(self.L, device=self.device, dtype=torch.int32),
            msg="factor_infos contains non-zero elements"
        )

        torch.testing.assert_close(
            solve_infos,
            torch.zeros(self.L, device=self.device, dtype=torch.int32),
            msg="solve_infos contains non-zero elements"
        )
        
        # Compute reconstructed labels
        reconstructed = sol @ self.kernel_matrix
        
        # Compare with original labels
        torch.testing.assert_close(
            reconstructed,
            self.labels,
            rtol=1e-5,
            atol=1e-5,
            msg="kermac.linalg.solve_lu solution does not reconstruct original labels"
        )

        reconstructed = labels_clobber @ self.kernel_matrix

         # The labels should be overwritten with the solution
        torch.testing.assert_close(
            reconstructed,
            self.labels,
            rtol=1e-5,
            atol=1e-5,
            msg="kermac.linalg.solve_lu solution does not reconstruct original labels"
        )
        
        # Compare with torch.linalg.solve
        torch_sol = torch.linalg.solve(
            self.kernel_matrix,
            self.labels.permute(0, 2, 1)
        ).permute(0, 2, 1)
        
        torch_reconstructed = torch_sol @ self.kernel_matrix
        
        # Compare kermac solution with torch solution
        torch.testing.assert_close(
            sol,
            torch_sol,
            rtol=1e-5,
            atol=1e-5,
            msg="kermac.linalg.solve_lu solution differs from torch.linalg.solve"
        )
        
        # Verify reconstructed labels from torch solution
        torch.testing.assert_close(
            torch_reconstructed,
            self.labels,
            rtol=1e-5,
            atol=1e-5,
            msg="torch.linalg.solve solution does not reconstruct original labels"
        )
    
    def test_solve_cholesky(self):
        for uplo in kermac.FillMode:
            kernel_matrix_clobber = self.kernel_matrix.clone()
            labels_clobber = self.labels.clone()
            # Run kermac.linalg.solve_choleksy
            sol, factor_infos, solve_infos = kermac.linalg.solve_cholesky(
                a=kernel_matrix_clobber,
                b=labels_clobber,
                fill_mode=uplo,
                overwrite_a=True,
                overwrite_b=True,
                check_errors=True
            )

            torch.testing.assert_close(
                factor_infos,
                torch.zeros(self.L, device=self.device, dtype=torch.int32),
                msg="factor_infos contains non-zero elements"
            )

            torch.testing.assert_close(
                solve_infos,
                torch.zeros(self.L, device=self.device, dtype=torch.int32),
                msg="solve_infos contains non-zero elements"
            )
            
            # Compute reconstructed labels
            reconstructed = sol @ self.kernel_matrix
            
            # Compare with original labels
            torch.testing.assert_close(
                reconstructed,
                self.labels,
                rtol=1e-5,
                atol=1e-5,
                msg="kermac.linalg.solve_lu solution does not reconstruct original labels"
            )

            reconstructed = labels_clobber @ self.kernel_matrix

            # The labels should be overwritten with the solution
            torch.testing.assert_close(
                reconstructed,
                self.labels,
                rtol=1e-5,
                atol=1e-5,
                msg="kermac.linalg.solve_lu solution does not reconstruct original labels"
            )
            
            # Compare with torch.linalg.solve
            torch_sol = torch.linalg.solve(
                self.kernel_matrix,
                self.labels.permute(0, 2, 1)
            ).permute(0, 2, 1)
            
            torch_reconstructed = torch_sol @ self.kernel_matrix
            
            # Compare kermac solution with torch solution
            torch.testing.assert_close(
                sol,
                torch_sol,
                rtol=1e-5,
                atol=1e-5,
                msg="kermac.linalg.solve_lu solution differs from torch.linalg.solve"
            )
            
            # Verify reconstructed labels from torch solution
            torch.testing.assert_close(
                torch_reconstructed,
                self.labels,
                rtol=1e-5,
                atol=1e-5,
                msg="torch.linalg.solve solution does not reconstruct original labels"
            )

if __name__ == '__main__':
    unittest.main()