import numpy as np
from math import cos, sin, atan2, sqrt

class KalmanFilter:
    def __init__(self, init_state, init_covariance, process_noise, measurement_noise):
        """
        Initialize Kalman Filter for robot localization
        
        Parameters:
        -----------
        init_state : numpy.ndarray
            Initial state vector [x, y, theta]
        init_covariance : numpy.ndarray
            Initial covariance matrix (3x3)
        process_noise : numpy.ndarray
            Process noise covariance matrix (3x3)
        measurement_noise : numpy.ndarray
            Measurement noise covariance matrix (3x3)
        """
        # State: [x, y, theta]
        self.mu = init_state
        self.sigma = init_covariance
        
        # Noise covariances
        self.R = process_noise      # Motion model noise
        self.Q = measurement_noise  # Measurement model noise
        
        # Identity matrix for convenience
        self.I = np.eye(3)
    
    def predict(self, u, dt):
        """
        Prediction step of the Kalman filter
        
        Parameters:
        -----------
        u : numpy.ndarray
            Control input [v, omega]
        dt : float
            Time step
        """
        # State transition matrix A (identity for this model)
        A = self.I
        
        # Control matrix B (depends on current orientation)
        theta = self.mu[2]
        B = np.array([
            [dt * cos(theta), 0],
            [dt * sin(theta), 0],
            [0, dt]
        ])
        
        # Prediction equations
        self.mu = A @ self.mu + B @ u
        self.sigma = A @ self.sigma @ A.T + self.R
        
        return self.mu, self.sigma
    
    def update(self, z):
        """
        Update step of the Kalman filter
        
        Parameters:
        -----------
        z : numpy.ndarray
            Measurement vector [x, y, theta]
        """
        # Measurement matrix C (identity for direct state measurement)
        C = self.I
        
        # Kalman gain
        K = self.sigma @ C.T @ np.linalg.inv(C @ self.sigma @ C.T + self.Q)
        
        # Update equations
        self.mu = self.mu + K @ (z - C @ self.mu)
        self.sigma = (self.I - K @ C) @ self.sigma
        
        return self.mu, self.sigma