import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler

class MGDAnomaly:
    """
    Multivariate Gaussian Distribution model for anomaly detection.
    
    This class implements the Mahalanobis distance-based anomaly detection
    method described in the paper "A Novel Features-Based Multivariate Gaussian 
    Distribution Method for the Fraudulent Consumers Detection in the Power 
    Utilities of Developing Countries."
    
    The model identifies anomalies by calculating the statistical distance
    of each point from the center of the multivariate distribution.
    """
    
    def __init__(self):
        """Initialize the MGD model with default parameters."""
        self.scaler = StandardScaler()
        self.cov_estimator = EmpiricalCovariance()
        self.fitted = False
        self.mu = None
    
    def fit(self, X):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Training data of shape (n_samples, n_features)
        
        Returns:
        --------
        self : object
            Returns self
        """
        # Check if we have data
        if X is None or len(X) == 0:
            self.fitted = False
            return self
            
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = X
            
        # Handle potential NaN values
        if np.isnan(X_data).any():
            # Replace NaN with column means
            col_means = np.nanmean(X_data, axis=0)
            inds = np.where(np.isnan(X_data))
            X_data[inds] = np.take(col_means, inds[1])
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_data)
        
        # Estimate mean and covariance
        self.mu = np.mean(X_scaled, axis=0)
        self.cov_estimator.fit(X_scaled)
        
        self.fitted = True
        return self
    
    def score_samples(self, X):
        """
        Calculate anomaly scores for each sample.
        
        The score is the Mahalanobis distance from the center of the
        distribution, which measures how many standard deviations away
        a point is from the mean of the distribution.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Data to score, of shape (n_samples, n_features)
        
        Returns:
        --------
        scores : ndarray
            Anomaly scores for each sample
        """
        if not self.fitted:
            return np.array([])
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = X
            
        # Handle potential NaN values
        if np.isnan(X_data).any():
            # Replace NaN with column means
            col_means = np.nanmean(X_data, axis=0)
            inds = np.where(np.isnan(X_data))
            X_data[inds] = np.take(col_means, inds[1])
        
        # Standardize features
        X_scaled = self.scaler.transform(X_data)
        
        # Calculate Mahalanobis distance
        precision = self.cov_estimator.precision_
        scores = np.zeros(X_scaled.shape[0])
        
        for i, x in enumerate(X_scaled):
            # Calculate centered vector (x - mu)
            centered_x = x - self.mu
            
            # Calculate Mahalanobis distance
            # sqrt((x - mu)^T * precision * (x - mu))
            scores[i] = np.sqrt(np.dot(np.dot(centered_x, precision), centered_x))
        
        return scores
    
    def predict(self, X, threshold=None):
        """
        Predict if samples are anomalies.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Data to predict, of shape (n_samples, n_features)
        threshold : float, optional
            Threshold for anomaly detection. If None, a default threshold
            is used based on the distribution of scores.
        
        Returns:
        --------
        predictions : ndarray
            1 for anomalies, 0 for normal samples
        """
        scores = self.score_samples(X)
        
        if len(scores) == 0:
            return np.array([])
        
        if threshold is None:
            # Default threshold is mean + 3*std
            threshold = np.mean(scores) + 3 * np.std(scores)
        
        return (scores > threshold).astype(int)
    
    def get_feature_importance(self):
        """
        Calculate feature importance based on the covariance matrix.
        
        Returns:
        --------
        importance : ndarray
            Importance score for each feature
        """
        if not self.fitted:
            return np.array([])
        
        # Feature importance is related to the diagonal elements of the covariance matrix
        importance = np.abs(np.diag(self.cov_estimator.covariance_))
        return importance