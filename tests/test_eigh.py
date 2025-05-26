import kermac
import kermac.linalg
import torch
import unittest

class TestEigh(unittest.TestCase):
    def setUp(self):
        # Set up test parameters and data
        self.N = 7
        self.D = 6
        self.L = 10
        self.device = torch.device('cuda')
        
        # Generate random test data
        self.data = torch.randn(self.L, self.N, self.D, device=self.device)
        self.kernel_matrix = torch.randn(self.L, self.N, self.N, device=self.device)
        
        # Compute kernel matrix
        kermac.run_kernel(
            kermac.kernel_descriptor_laplace_l2,
            self.data, self.data,
            out=self.kernel_matrix,
            bandwidth=10.0,
            try_to_align=True,
            debug=False
        )

    def test_eigh(self):
        for uplo in kermac.FillMode:
            kernel_matrix_clobber = self.kernel_matrix.clone()

            eigenvalues, eigenvectors, infos = kermac.linalg.eigh(
                a=kernel_matrix_clobber,
                fill_mode=uplo,
                overwrite_a=True,
                check_errors=True
            )
            
            # Assert info is a vector of size L with all zeros and int32 dtype
            self.assertEqual(infos.shape, (self.L,), f"info shape is {infos.shape}, expected ({self.L},)")
            self.assertEqual(infos.dtype, torch.int32, f"info dtype is {infos.dtype}, expected torch.int32")
            torch.testing.assert_close(
                infos,
                torch.zeros(self.L, device=self.device, dtype=torch.int32),
                msg="info contains non-zero elements"
            )
            
            # Verify shapes and dtypes
            self.assertEqual(eigenvalues.shape, (self.L, self.N), f"eigenvalues shape is {eigenvalues.shape}, expected ({self.L}, {self.N})")
            self.assertEqual(eigenvectors.shape, (self.L, self.N, self.N), f"eigenvectors shape is {eigenvectors.shape}, expected ({self.L}, {self.N}, {self.N})")
            self.assertEqual(eigenvalues.dtype, torch.float32, f"eigenvalues dtype is {eigenvalues.dtype}, expected torch.float32")
            self.assertEqual(eigenvectors.dtype, torch.float32, f"eigenvectors dtype is {eigenvectors.dtype}, expected torch.float32")
            
            # Reconstruct the original matrix: A = V @ Lambda @ V^T
            Lambda = torch.diag_embed(eigenvalues)
            V = eigenvectors
            A_reconstructed = V @ Lambda @ V.permute(0, 2, 1)
            
            # Compare reconstructed matrix with original
            torch.testing.assert_close(
                A_reconstructed,
                self.kernel_matrix,
                rtol=1e-5,
                atol=1e-5,
                msg="kermac.linalg.eigh reconstructed matrix does not match original kernel matrix"
            )
            
            # Compare with torch.linalg.eigh
            torch_eigenvalues, torch_eigenvectors = torch.linalg.eigh(self.kernel_matrix)
            
            # Compare eigenvalues
            torch.testing.assert_close(
                eigenvalues,
                torch_eigenvalues,
                rtol=1e-5,
                atol=1e-5,
                msg="kermac.linalg.eigh eigenvalues differ from torch.linalg.eigh"
            )
            
            # Compare eigenvectors (accounting for possible sign flips)
            # Eigenvectors can have opposite signs, so check V @ V^T against torch_V @ torch_V^T
            VVT = eigenvectors @ eigenvectors.permute(0, 2, 1)
            torch_VVT = torch_eigenvectors @ torch_eigenvectors.permute(0, 2, 1)
            torch.testing.assert_close(
                VVT,
                torch_VVT,
                rtol=1e-5,
                atol=1e-5,
                msg="kermac.linalg.eigh eigenvectors differ from torch.linalg.eigh (V @ V^T mismatch)"
            )
            
            # Verify torch.linalg.eigh reconstruction
            torch_Lambda = torch.diag_embed(torch_eigenvalues)
            torch_A_reconstructed = torch_eigenvectors @ torch_Lambda @ torch_eigenvectors.permute(0, 2, 1)
            torch.testing.assert_close(
                torch_A_reconstructed,
                self.kernel_matrix,
                rtol=1e-5,
                atol=1e-5,
                msg="torch.linalg.eigh reconstructed matrix does not match original kernel matrix"
            )

if __name__ == '__main__':
    unittest.main()