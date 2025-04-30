import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance

class MGDAnomaly:
    """
    Multivariate Gaussian Distribution for anomaly detection.
    
    This class implements the anomaly detection approach described in the paper
    'A Novel Features-Based Multivariate Gaussian Distribution Method for the 
    Fraudulent Consumers Detection in the Power Utilities of Developing Countries.'
    
    The method uses Mahalanobis distance to identify anomalous consumption patterns,
    taking into account the multidimensional nature of the data.
    """
    
    def __init__(self):
        """Initialize the MGD anomaly detector."""
        self.scaler = StandardScaler()
        self.cov_estimator = EmpiricalCovariance(assume_centered=False)
        self.mean = None
        self.fitted = False
    
    def fit(self, X):
        """
        Fit the model with the data X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.cov_estimator.fit(X_scaled)
        self.mean = np.mean(X_scaled, axis=0)
        self.fitted = True
        return self
    
    def score_samples(self, X):
        """
        Compute the Mahalanobis distances of the samples.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        mahal_dist : ndarray of shape (n_samples,)
            Mahalanobis distances of the samples.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        
        X_scaled = self.scaler.transform(X)
        mahal_dist = []
        
        for i in range(X_scaled.shape[0]):
            x = X_scaled[i, :]
            delta = x - self.mean
            # Calculate Mahalanobis distance using precision matrix (inverse of covariance)
            mahal_dist.append(np.sqrt(np.dot(np.dot(delta, self.cov_estimator.precision_), delta.T)))
        
        return np.array(mahal_dist)
    
    def predict(self, X, threshold):
        """
        Predict if observations are anomalies based on a threshold.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        threshold : float
            The threshold above which samples are considered anomalies.
            
        Returns:
        --------
        anomalies : ndarray of shape (n_samples,)
            Boolean array indicating if samples are anomalies.
        scores : ndarray of shape (n_samples,)
            Anomaly scores (Mahalanobis distances).
        """
        scores = self.score_samples(X)
        return scores > threshold, scores