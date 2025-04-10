import unittest
import numpy as np
import torch
from nfnull import NFNull

class TestNFNull(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate some test data
        self.normal_data = np.random.normal(0, 1, 1000)
        self.t_data = np.random.standard_t(df=3, size=1000)
    
    def test_nsf_model(self):
        """Test that the NSF model can fit normal data and generate samples"""
        # Create and fit the model
        model = NFNull(
            x=self.normal_data,
            flow='NSF',
            transforms=2,
            bins=8,
            hidden_features=(32, 32),
            grid_points=1000
        )
        
        # Fit with fewer iterations for testing
        model.fit_pdf(n_iter=1500, verbose=False)
        
        # Test that we can generate samples
        samples = model.sample(n=100)
        self.assertEqual(len(samples), 100)
        
        # Test log probability calculation
        log_prob = model.log_prob(torch.tensor([[0.0]]))
        self.assertIsInstance(log_prob.item(), float)
        
        # Test p-value calculation with specific expectations
        # For a standard normal, P(X > 0) should be close to 0.5
        p_val_greater = model.p_value(0.0, greater_than=True, n=1000)
        self.assertGreater(p_val_greater, 0.47)
        self.assertLess(p_val_greater, 0.53)
        # P(X < -1.96) should be close to 0.025
        p_val_less = model.p_value(-1.96, greater_than=False, n=1000)
        self.assertGreater(p_val_less, 0.01)
        self.assertLess(p_val_less, 0.05)
    
    def test_tnsf_model(self):
        """Test that the TNSF model can fit heavy-tailed data and generate samples"""
        # Create and fit the model
        model = NFNull(
            x=self.t_data,
            flow='TNSF',
            transforms=2,
            bins=8,
            hidden_features=(32, 32),
            grid_points=1000,
            nu=3.0  # Match degrees of freedom with the data
        )
        
        # Fit with fewer iterations for testing
        model.fit_pdf(n_iter=500, verbose=False)
        
        # Test that we can generate samples
        samples = model.sample(n=100)
        self.assertEqual(len(samples), 100)
        
        # Test log probability calculation
        log_prob = model.log_prob(torch.tensor([[0.0]]))
        self.assertIsInstance(log_prob.item(), float)
        
        # Test p-value calculation with specific expectations
        # For a t-distribution with df=3, P(X > 0) should be close to 0.5
        p_val_greater = model.p_value(0.0, greater_than=True, n=1000)
        self.assertGreater(p_val_greater, 0.47)
        self.assertLess(p_val_greater, 0.53)
        
        # For t(3), P(X < -3.18) should be close to 0.025
        p_val_less = model.p_value(-3.18, greater_than=False, n=1000)
        self.assertGreater(p_val_less, 0.01)
        self.assertLess(p_val_less, 0.05)
        
if __name__ == '__main__':
    unittest.main() 