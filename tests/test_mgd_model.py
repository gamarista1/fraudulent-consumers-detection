import unittest
import numpy as np
import pandas as pd
from modules.mgd_model import MGDAnomaly

class TestMGDAnomaly(unittest.TestCase):
    """Tests for the MGDAnomaly class."""
    
    def setUp(self):
        """Set up test cases."""
        # Create synthetic data with known anomalies
        np.random.seed(42)
        
        # Generate normal data (multivariate normal distribution)
        self.n_samples = 1000
        self.n_features = 5
        self.mean = np.zeros(self.n_features)
        self.cov = np.eye(self.n_features)  # Identity covariance matrix
        
        self.normal_data = np.random.multivariate_normal(
            self.mean, self.cov, size=self.n_samples - 10
        )
        
        # Generate anomalies
        self.anomalies = np.random.multivariate_normal(
            self.mean + 5, self.cov, size=10
        )
        
        # Combine normal data and anomalies
        self.X = np.vstack([self.normal_data, self.anomalies])
        
        # Labels (for validation)
        self.y_true = np.zeros(self.n_samples, dtype=bool)
        self.y_true[-10:] = True  # Last 10 samples are anomalies
        
        # Create DataFrame (for interface compatibility)
        self.X_df = pd.DataFrame(
            self.X, 
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        
        # Initialize model
        self.model = MGDAnomaly()
    
    def test_init(self):
        """Test initialization."""
        self.assertFalse(self.model.fitted)
        self.assertIsNone(self.model.mean)
    
    def test_fit(self):
        """Test fit method."""
        # Fit the model
        self.model.fit(self.X_df)
        
        # Check if fitted
        self.assertTrue(self.model.fitted)
        
        # Check if mean is computed correctly
        np.testing.assert_allclose(
            self.model.mean, 
            np.zeros(self.n_features),
            atol=0.1
        )
    
    def test_score_samples(self):
        """Test score_samples method."""
        # Fit the model
        self.model.fit(self.X_df)
        
        # Compute scores
        scores = self.model.score_samples(self.X_df)
        
        # Check shape
        self.assertEqual(scores.shape, (self.n_samples,))
        
        # Check if anomalies have higher scores
        normal_scores = scores[~self.y_true]
        anomaly_scores = scores[self.y_true]
        
        self.assertTrue(np.mean(anomaly_scores) > np.mean(normal_scores))
    
    def test_predict(self):
        """Test predict method."""