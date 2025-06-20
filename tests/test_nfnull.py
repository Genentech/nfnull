import unittest
import numpy as np
import torch
from nfnull import NFNull
from sklearn.datasets import make_swiss_roll

class TestNFNull(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate some test data
        self.normal_data = np.random.normal(0, 1, 1000)
        self.t_data = np.random.standard_t(df=3, size=1000)
        
        # Generate 2D multivariate normal data with heteroskedasticity
        mean = [2.0, -1.0]
        cov = [[2.0, 0.8], [0.8, 0.5]]  # Different variances and correlation
        self.mv_normal_data = np.random.multivariate_normal(mean, cov, 1000)
        print(f"Multivariate normal data shape: {self.mv_normal_data.shape}")
        
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
        model.fit_pdf(n_iter=1000, verbose=True)
        
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
        model.fit_pdf(n_iter=1000, verbose=True)
        
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

    def test_multivariate_nsf_model(self):
        """Test NSF model with 2D multivariate normal data"""
        # Create and fit the model
        model = NFNull(
            x=self.mv_normal_data,
            features=2,
            flow='NSF',
            transforms=2,
            passes=2,  # Need passes > 0 for multivariate coupling layers
            prescaled=False
        )
        
        # Fit with fewer iterations for testing
        model.fit_pdf(n_iter=1000, verbose=False, patience=10)
        
        # Test that we can generate samples with correct shape
        samples = model.sample(n=100)
        self.assertEqual(samples.shape, (100, 2))
        
        # Test log probability calculation for 2D point
        test_point = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
        log_prob = model.log_prob(test_point)
        self.assertIsInstance(log_prob.item(), float)
        
        # Test multivariate p-value calculation
        # Point near the mean should have moderate p-value
        p_val_mean = model.p_value([2.0, -1.0], greater_than=True, n=10000)
        self.assertGreaterEqual(p_val_mean, 0.1)
        self.assertLessEqual(p_val_mean, 0.9)
        
        # Point far from mean should have high p-value (low probability)
        p_val_extreme = model.p_value([10.0, 10.0], greater_than=True, n=10000)
        self.assertLessEqual(p_val_extreme, 0.001)

    def test_multivariate_tnsf_model(self):
        """Test TNSF model with 2D multivariate heavy-tailed data"""
        # Generate 2D data with t-distribution components
        df = 3
        n_samples = 1000
        t_data_2d = np.column_stack([
            np.random.standard_t(df, n_samples),
            np.random.standard_t(df, n_samples) * 0.5 + 0.3 * np.random.standard_t(df, n_samples)
        ])
        
        # Create and fit the model
        model = NFNull(
            x=t_data_2d,
            features=2,
            flow='TNSF',
            transforms=2,
            hidden_features=(64, 64, 64, 64),
            bins=8,
            nu=3.0,
            prescaled=False
        )
        
        # Fit with fewer iterations for testing
        model.fit_pdf(n_iter=1000, verbose=False, patience=10)
        
        # Test that we can generate samples with correct shape
        samples = model.sample(n=100)
        self.assertEqual(samples.shape, (100, 2))
        
        # Test multivariate p-value for origin
        p_val_origin = model.p_value([0.0, 0.0], greater_than=True, n=10000)
        self.assertGreaterEqual(p_val_origin, 0.1)
        self.assertLessEqual(p_val_origin, 0.9)
        
        # Test extreme point
        p_val_extreme = model.p_value([5.0, 5.0], greater_than=True, n=10000)
        self.assertLessEqual(p_val_extreme, 0.01)

    def test_swiss_roll_3d(self):
        """Test NFNull with 3D Swiss roll manifold data"""
        # Generate Swiss roll data
        X, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
        
        # Create and fit the model
        model = NFNull(
            x=X,
            features=3,
            flow='NSF',
            transforms=3,
            hidden_features=(128, 128, 128, 128),
            bins=8,
            passes=2,  # Need passes > 0 for multivariate coupling layers
            prescaled=False
        )
        
        # Fit with fewer iterations for testing
        model.fit_pdf(n_iter=1000, verbose=False, patience=20)
        
        # Test that we can generate samples with correct shape
        samples = model.sample(n=100)
        self.assertEqual(samples.shape, (100, 3))
        
        # Test multivariate p-value
        # Point on the manifold should have lower p-value than point in empty space
        on_manifold = [0.0, 5.0, 0.0]  # Typical Swiss roll point
        off_manifold = [15.0, 15.0, 15.0]  # Point in empty space
        
        p_val_on = model.p_value(on_manifold, greater_than=True, n=10000)
        p_val_off = model.p_value(off_manifold, greater_than=True, n=10000)
        
        self.assertLessEqual(p_val_off, p_val_on)
        self.assertLessEqual(p_val_off, 0.001)

if __name__ == '__main__':
    unittest.main() 