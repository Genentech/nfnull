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
        model.fit_pdf(n_iter=2000, verbose=False, context=context)
        
        # Test conditional sampling
        test_context = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        samples = model.sample(n=100, context=test_context)
        self.assertEqual(len(samples), 100)
        
        # Test conditional log probability
        log_prob = model.log_prob(torch.tensor([[0.0]]), context=test_context)
        self.assertIsInstance(log_prob.item(), float)
        
        # Test conditional p-value calculation
        # For context [1.0, 0.0], expected mean ≈ 1.0
        # Test a point clearly in the tail vs a point near the center
        test_context = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        
        # P(X > 1.0) should be close to 0.5 (near the expected mean)
        p_val_center = model.p_value(1.0, greater_than=True, n=10000, context=test_context)
        self.assertGreater(p_val_center, 0.3)
        self.assertLess(p_val_center, 0.7)
        
        # P(X > 1.5) should be much smaller (in the tail)
        p_val_tail = model.p_value(1.5, greater_than=True, n=10000, context=test_context)
        self.assertLess(p_val_tail, 0.1)
    
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
        model.fit_pdf(n_iter=2000, verbose=False, context=context)
        
        # Test conditional sampling
        test_context = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        samples = model.sample(n=100, context=test_context)
        self.assertEqual(len(samples), 100)
        
        # Test conditional log probability
        log_prob = model.log_prob(torch.tensor([[0.0]]), context=test_context)
        self.assertIsInstance(log_prob.item(), float)
        
        # Test conditional p-value calculation
        # For context [1.0, 0.0], expected mean ≈ 1.0
        test_context = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        
        # P(X > 1.0) should be close to 0.5 (near expected mean)
        p_val_center = model.p_value(1.0, greater_than=True, n=10000, context=test_context)
        self.assertGreater(p_val_center, 0.3)
        self.assertLess(p_val_center, 0.7)
        
        # P(X > 3.0) should be much smaller (t-distribution tail)
        p_val_tail = model.p_value(3.0, greater_than=True, n=10000, context=test_context)
        self.assertLess(p_val_tail, 0.1)

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

    def test_batched_context_operations(self):
        """Test batched sampling and p-value computation with multiple contexts"""
        # Generate synthetic data with clear context dependency
        n_samples = 1000
        n_groups = 3
        
        # Create contexts that shift the mean: [-1, 0, 1]
        contexts = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
        
        # Generate training data where x depends on context
        all_x = []
        all_contexts = []
        for i, ctx in enumerate(contexts):
            # Data centered around the context value with some noise
            group_x = ctx[0] + np.random.normal(0, 0.5, n_samples)
            all_x.append(group_x)
            all_contexts.extend([ctx] * n_samples)
        
        x_train = np.concatenate(all_x)
        context_train = np.array(all_contexts)
        
        # Create and fit the model
        model = NFNull(
            x=x_train,
            flow='NSF',
            transforms=2,
            hidden_features=(64, 64),
            bins=4,
            context=1,  # 1D context
            features=1
        )
        
        # Fit the model
        model.fit_pdf(n_iter=500, verbose=False, context=context_train, patience=10)
        
        # Test 1: Batched sampling shape and distribution
        batched_contexts = torch.tensor(contexts, dtype=torch.float32)
        n_samples_test = 1000
        
        samples = model.sample(n=n_samples_test, context=batched_contexts)
        
        # Check shape: should be [n_samples_test, n_groups]
        expected_shape = (n_samples_test, n_groups)
        self.assertEqual(samples.shape, expected_shape, 
                        f"Expected shape {expected_shape}, got {samples.shape}")
        
        # Check that different groups have different means (approximately)
        group_means = np.mean(samples, axis=0)
        self.assertLess(group_means[0], group_means[1], 
                       "Group 0 mean should be less than Group 1 mean")
        self.assertLess(group_means[1], group_means[2], 
                       "Group 1 mean should be less than Group 2 mean")
        
        # Check approximate means are close to expected context values
        np.testing.assert_allclose(group_means, contexts.flatten(), atol=0.3,
                                  err_msg="Group means should approximate context values")
        
        # Test 2: Batched p-values with scalar threshold
        scalar_threshold = 0.0
        p_values_scalar = model.p_value(scalar_threshold, greater_than=True, 
                                       n=10000, context=batched_contexts)
        
        # Should return array of p-values, one per group
        self.assertEqual(len(p_values_scalar), n_groups,
                        f"Expected {n_groups} p-values, got {len(p_values_scalar)}")
        
        # P(X > 0) should be different for each group due to different means
        # Group 0 (mean=-1): P(X > 0) should be small
        # Group 1 (mean=0): P(X > 0) should be ~0.5  
        # Group 2 (mean=1): P(X > 0) should be large
        self.assertLess(p_values_scalar[0], 0.3, "Group 0: P(X > 0) should be small")
        self.assertGreater(p_values_scalar[0], 0.01, "Group 0: P(X > 0) should be > 0.01")
        
        self.assertGreater(p_values_scalar[1], 0.4, "Group 1: P(X > 0) should be ~0.5")
        self.assertLess(p_values_scalar[1], 0.6, "Group 1: P(X > 0) should be ~0.5")
        
        self.assertGreater(p_values_scalar[2], 0.7, "Group 2: P(X > 0) should be large")
        self.assertLess(p_values_scalar[2], 0.99, "Group 2: P(X > 0) should be < 0.99")
        
        # Test 3: Batched p-values with array of thresholds
        array_thresholds = np.array([-1.0, 0.0, 1.0])  # Different threshold per group
        p_values_array = model.p_value(array_thresholds, greater_than=True,
                                      n=10000, context=batched_contexts)
        
        self.assertEqual(len(p_values_array), n_groups,
                        f"Expected {n_groups} p-values, got {len(p_values_array)}")
        
        # Each group evaluated at its context mean should give ~0.5
        for i, p_val in enumerate(p_values_array):
            self.assertGreater(p_val, 0.3, f"Group {i}: P(X > mean) should be ~0.5")
            self.assertLess(p_val, 0.7, f"Group {i}: P(X > mean) should be ~0.5")
        
        # Test 4: Compare batched vs single context results
        # Single context results should match corresponding elements of batched results
        for i, single_context in enumerate(batched_contexts):
            single_context_reshaped = single_context.reshape(1, -1)
            
            # Single context sampling
            single_samples = model.sample(n=n_samples_test, context=single_context_reshaped)
            
            # Compare means (should be similar)
            single_mean = np.mean(single_samples)
            batched_mean = np.mean(samples[:, i])
            np.testing.assert_allclose(single_mean, batched_mean, atol=0.2,
                                      err_msg=f"Group {i} means should be similar")
            
            # Single context p-value
            single_p_val = model.p_value(scalar_threshold, greater_than=True,
                                        n=10000, context=single_context_reshaped)
            
            # Compare p-values (should be similar)
            np.testing.assert_allclose(single_p_val, p_values_scalar[i], atol=0.05,
                                      err_msg=f"Group {i} p-values should be similar")
        
        # Test 5: Error handling for mismatched array lengths
        with self.assertRaises(ValueError):
            # Wrong number of thresholds
            model.p_value([0.0, 1.0], context=batched_contexts)  # 2 thresholds, 3 groups

    def test_batched_context_multivariate(self):
        """Test batched operations with multivariate data and context"""
        # Generate 2D data with 2D context dependency
        n_samples = 800
        n_groups = 2
        features = 2
        
        # Contexts that affect both dimensions differently
        contexts = np.array([[1.0, 0.0], [-1.0, 0.5]], dtype=np.float32)
        
        # Generate training data
        all_x = []
        all_contexts = []
        for i, ctx in enumerate(contexts):
            # 2D data where each dimension is affected by context
            group_x = np.column_stack([
                ctx[0] + np.random.normal(0, 0.3, n_samples),  # x1 affected by ctx[0]
                ctx[1] + np.random.normal(0, 0.3, n_samples)   # x2 affected by ctx[1]
            ])
            all_x.append(group_x)
            all_contexts.extend([ctx] * n_samples)
        
        x_train = np.vstack(all_x)
        context_train = np.array(all_contexts)
        
        # Create and fit model
        model = NFNull(
            x=x_train,
            features=2,
            flow='NSF',
            transforms=2,
            hidden_features=(64, 64),
            bins=4,
            context=2,  # 2D context
            passes=1,
            prescaled=False
        )
        
        model.fit_pdf(n_iter=500, verbose=False, context=context_train, patience=10)
        
        # Test batched sampling
        batched_contexts = torch.tensor(contexts, dtype=torch.float32)
        n_samples_test = 500
        
        samples = model.sample(n=n_samples_test, context=batched_contexts)
        
        # Check shape: [n_samples_test, n_groups, features]
        expected_shape = (n_samples_test, n_groups, features)
        self.assertEqual(samples.shape, expected_shape,
                        f"Expected shape {expected_shape}, got {samples.shape}")
        
        # Check group means are approximately correct
        group_means = np.mean(samples, axis=0)  # Shape: [n_groups, features]
        np.testing.assert_allclose(group_means, contexts, atol=0.3,
                                  err_msg="Group means should approximate context values")
        
        # Test batched multivariate p-values
        # Scalar threshold (same point for all groups)
        threshold_point = [0.0, 0.0]
        p_values = model.p_value(threshold_point, greater_than=True,
                                n=10000, context=batched_contexts)
        
        self.assertEqual(len(p_values), n_groups,
                        f"Expected {n_groups} p-values, got {len(p_values)}")
        
        # Different groups should give different p-values
        self.assertNotAlmostEqual(p_values[0], p_values[1], places=2,
                                 msg="Different groups should have different p-values")
        
        # Test with different threshold per group
        thresholds_array = np.array([[1.0, 0.0], [-1.0, 0.5]])  # One per group
        p_values_array = model.p_value(thresholds_array, greater_than=True,
                                      n=10000, context=batched_contexts)
        
        # Each group evaluated at its expected mean should give moderate p-values
        for p_val in p_values_array:
            self.assertGreater(p_val, 0.1, "P-value should be reasonable")
            self.assertLess(p_val, 0.9, "P-value should be reasonable")

if __name__ == '__main__':
    unittest.main() 