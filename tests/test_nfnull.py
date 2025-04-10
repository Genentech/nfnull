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
            hidden_features=(64, 64), 
            bins=4,            
        )
        
        # Fit with fewer iterations for testing
        model.fit_pdf(n_iter=1000, verbose=False)
        
        # Test that we can generate samples
        samples = model.sample(n=100)
        self.assertEqual(len(samples), 100)
        
        # Test log probability calculation
        log_prob = model.log_prob(torch.tensor([[0.0]]))
        self.assertIsInstance(log_prob.item(), float)
        
        # Test p-value calculation with specific expectations
        # For a standard normal, P(X > 0) should be close to 0.5
        p_val_greater = np.round(model.p_value(0.0, greater_than=True, n=1000), 2)
        self.assertGreaterEqual(p_val_greater, 0.47)
        self.assertLessEqual(p_val_greater, 0.53)
        # P(X < -1.96) should be close to 0.025
        p_val_less = np.round(model.p_value(-1.96, greater_than=False, n=1000), 2)
        self.assertGreaterEqual(p_val_less, 0.01)
        self.assertLessEqual(p_val_less, 0.05)
    
    def test_tnsf_model(self):
        """Test that the TNSF model can fit heavy-tailed data and generate samples"""
        # Create and fit the model
        model = NFNull(
            x=self.t_data,
            flow='TNSF',
            transforms=2, 
            hidden_features=(64, 64), 
            bins=4,             
            nu=3.0  # Match degrees of freedom with the data
        )
        
        # Fit with fewer iterations for testing
        model.fit_pdf(n_iter=1000, verbose=False)
        
        # Test that we can generate samples
        samples = model.sample(n=100)
        self.assertEqual(len(samples), 100)
        
        # Test log probability calculation
        log_prob = model.log_prob(torch.tensor([[0.0]]))
        self.assertIsInstance(log_prob.item(), float)
        
        # Test p-value calculation with specific expectations
        # For a t-distribution with df=3, P(X > 0) should be close to 0.5
        p_val_greater = np.round(model.p_value(0.0, greater_than=True, n=1000), 2)
        self.assertGreaterEqual(p_val_greater, 0.47)
        self.assertLessEqual(p_val_greater, 0.53)
        
        # For t(3), P(X < -3.18) should be close to 0.025
        p_val_less = np.round(model.p_value(-3.18, greater_than=False, n=1000), 2)
        self.assertGreaterEqual(p_val_less, 0.01)
        self.assertLessEqual(p_val_less, 0.05)
    
    def test_nsf_model_with_context(self):
        """Test that the NSF model can handle conditional density estimation"""
        # Generate test data with context
        n_samples = 1000
        context = np.random.normal(0, 1, (n_samples, 2))  # 2D context
        x = context[:, 0] + 0.5 * context[:, 1] + np.random.normal(0, 0.1, n_samples)
        
        # Create and fit the model with context
        model = NFNull(
            x=x,
            flow='NSF',
            transforms=2, 
            hidden_features=(64, 64), 
            bins=4,             
            context=2  # Specify context dimension
        )
        
        # Fit with fewer iterations for testing
        model.fit_pdf(n_iter=1000, verbose=False, context=context)
        
        # Test conditional sampling
        test_context = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        samples = model.sample(n=100, context=test_context)
        self.assertEqual(len(samples), 100)
        
        # Test conditional log probability
        log_prob = model.log_prob(torch.tensor([[0.0]]), context=test_context)
        self.assertIsInstance(log_prob.item(), float)
        
        # Test conditional p-value calculation
        test_context = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        p_val = np.round(model.p_value(1.2, greater_than=True, n=1000, context=test_context), 2)
        self.assertGreaterEqual(p_val, 0.01)
        self.assertLessEqual(p_val, 0.05)
    
    def test_tnsf_model_with_context(self):
        """Test that the TNSF model can handle conditional density estimation"""
        # Generate test data with context
        n_samples = 1000
        context = np.random.normal(0, 1, (n_samples, 2))  # 2D context
        x = context[:, 0] + 0.5 * context[:, 1] + np.random.standard_t(df=3, size=n_samples)
        
        # Create and fit the model with context
        model = NFNull(
            x=x,
            flow='TNSF',
            transforms=2, 
            hidden_features=(64, 64), 
            bins=4,             
            context=2,  # Specify context dimension
            nu=3.0
        )
        
        # Fit with fewer iterations for testing
        model.fit_pdf(n_iter=1000, verbose=False, context=context)
        
        # Test conditional sampling
        test_context = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        samples = model.sample(n=100, context=test_context)
        self.assertEqual(len(samples), 100)
        
        # Test conditional log probability
        log_prob = model.log_prob(torch.tensor([[0.0]]), context=test_context)
        self.assertIsInstance(log_prob.item(), float)
        
        # Test conditional p-value calculation
        test_context = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        p_val = np.round(model.p_value(5.0, greater_than=True, n=1000, context=test_context), 2)
        self.assertGreaterEqual(p_val, 0.01)
        self.assertLessEqual(p_val, 0.05)

if __name__ == '__main__':
    unittest.main() 